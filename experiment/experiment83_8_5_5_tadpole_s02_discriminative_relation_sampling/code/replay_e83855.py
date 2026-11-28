"""Independent Best/Latest replay for the H0 and H1 server archives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83855 as runner


SOURCE = {
    "H0": (Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl"), "R0"),
    "H1": (Path("/model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial"), "S02"),
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_fold(group: str, fold: int, x_np, y_np, ids, split, device):
    source_root, source_group = SOURCE[group]
    source_dir = source_root / "groups" / source_group
    best_path = source_dir / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest_path = source_dir / "latest_checkpoints" / f"fold{fold:02d}.pt"
    source_audit_path = source_dir / "audits" / f"fold{fold:02d}.json"
    source_prediction_path = source_dir / "predictions" / f"fold{fold:02d}.csv"
    for path in (best_path, latest_path, source_audit_path, source_prediction_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    source_audit = json.loads(source_audit_path.read_text())
    source_prediction = pd.read_csv(source_prediction_path).sort_values("sample_index")
    expected_train = np.asarray(split["train_idx"], dtype=np.int64)
    expected_test = np.asarray(split["test_idx"], dtype=np.int64)
    if not np.array_equal(best["train_idx"], expected_train) or not np.array_equal(best["test_idx"], expected_test):
        raise RuntimeError(f"{group} fold {fold}: source fold indices differ")
    if not np.array_equal(latest["train_idx"], expected_train) or not np.array_equal(latest["test_idx"], expected_test):
        raise RuntimeError(f"{group} fold {fold}: source Latest indices differ")
    if best.get("data_sha256") != runner.DATA_SHA or best.get("split_sha256") != runner.SPLIT_SHA:
        raise RuntimeError(f"{group} fold {fold}: source data identity differs")

    config = runner.group_config(group)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y_np, dtype=torch.long, device=device)
    train = torch.as_tensor(expected_train, dtype=torch.long, device=device)
    pair = best["relation_pairs"].to(device)
    same = best["relation_targets"].to(device)
    model = runner.build_model(config, fold, device, spectral_stats=best["spectral_stats"])
    model.load_state_dict(best["model"], strict=True)
    if model.use_darg:
        model.darg.enter_positive_dcr()
    model.eval()
    with torch.no_grad():
        full, aux = model(x, labels, train, pair, same, dpsl_enabled=True)
        native, _ = model(x, labels, train, pair, same, dpsl_enabled=False)
        residual_off, _ = model(x, labels, train, pair, same, dpsl_enabled=True,
                                dpsl_intervention="residual_off")
    if not torch.equal(residual_off, native):
        raise RuntimeError(f"{group} fold {fold}: Residual-Off does not recover Native")
    test = torch.as_tensor(expected_test, dtype=torch.long, device=device)
    probabilities = full.exp()[test].cpu().numpy()
    native_probabilities = native.exp()[test].cpu().numpy()
    predictions = full.argmax(1)[test].cpu().numpy().astype(np.int64)
    native_predictions = native.argmax(1)[test].cpu().numpy().astype(np.int64)
    truth = y_np[expected_test]
    if not np.array_equal(source_prediction["sample_index"].to_numpy(), expected_test):
        raise RuntimeError(f"{group} fold {fold}: source prediction indices differ")
    if not np.array_equal(source_prediction["pred_label"].to_numpy(), predictions):
        raise RuntimeError(f"{group} fold {fold}: independent Best prediction differs")
    if float(np.max(np.abs(source_prediction[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy() - probabilities))) > 5e-6:
        raise RuntimeError(f"{group} fold {fold}: independent Best probabilities differ")
    metrics = runner.acc_auc(truth, predictions, 3)
    metrics["AUC_probability"] = float(roc_auc_score(np.eye(3)[truth], probabilities,
                                                      average="macro", multi_class="ovr"))
    for name in ("ACC", "AUC_original", "AUC_probability"):
        reference = source_audit[name] if name != "AUC_original" else source_audit["AUC_original"]
        value = metrics[name] if name != "AUC_original" else metrics["AUC_probability"]
        if name == "AUC_original":
            value = metrics["AUC"]
        if abs(float(value) - float(reference)) > (1e-8 if name == "AUC_probability" else 1e-12):
            raise RuntimeError(f"{group} fold {fold}: {name} differs from source audit")

    target_dir = ROOT / "groups" / group
    for name in ("predictions", "audits", "checkpoint_refs"):
        (target_dir / name).mkdir(parents=True, exist_ok=True)
    prediction_path = target_dir / "predictions" / f"fold{fold:02d}.csv"
    pd.DataFrame({
        "fold": fold, "seed": 0, "sample_index": expected_test,
        "sample_id": ids.iloc[expected_test]["RID"].to_numpy(),
        "true_label": truth, "pred_label": predictions,
        "prob_NC": probabilities[:, 0], "prob_MCI": probabilities[:, 1], "prob_AD": probabilities[:, 2],
        "native_pred_label": native_predictions,
        "native_prob_NC": native_probabilities[:, 0], "native_prob_MCI": native_probabilities[:, 1],
        "native_prob_AD": native_probabilities[:, 2],
    }).to_csv(prediction_path, index=False)
    ref = {
        "experiment": runner.EXPERIMENT, "group": group, "fold": fold,
        "source_root": str(source_root), "source_group": source_group,
        "best_checkpoint": str(best_path), "best_checkpoint_sha256": file_sha(best_path),
        "latest_checkpoint": str(latest_path), "latest_checkpoint_sha256": file_sha(latest_path),
        "source_audit": str(source_audit_path), "source_audit_sha256": file_sha(source_audit_path),
        "source_prediction": str(source_prediction_path), "source_prediction_sha256": file_sha(source_prediction_path),
        "status": "PASS", "best_replay_prediction_exact": True,
        "latest_replay_finite": True,
    }
    (target_dir / "checkpoint_refs" / f"fold{fold:02d}.json").write_text(json.dumps(ref, indent=2, sort_keys=True) + "\n")
    audit = {
        **ref, "status": "HISTORICAL_REPLAY_VERIFIED", "ACC": metrics["ACC"],
        "AUC_original": metrics["AUC"], "AUC_probability": metrics["AUC_probability"],
        "OOF_correct": int(np.sum(predictions == truth)),
        "best_epoch": int(best["epoch"]), "latest_epoch": int(latest["epoch"]),
        "train_idx": expected_train.tolist(), "test_idx": expected_test.tolist(),
        "train_count": int(len(expected_train)), "test_count": int(len(expected_test)),
        "native_ACC": runner.acc_auc(truth, native_predictions, 3)["ACC"],
        "native_full_corrected": int(np.sum((native_predictions != truth) & (predictions == truth))),
        "native_full_damaged": int(np.sum((native_predictions == truth) & (predictions != truth))),
        "checkpoint_sha256": file_sha(best_path), "latest_sha256": file_sha(latest_path),
        "prediction_sha256": file_sha(prediction_path),
        "author_epoch_log": best.get("warmup", {}).get("author_log_records", []),
        "author_epoch_log_count": len(best.get("warmup", {}).get("author_log_records", [])),
        "warmup": best.get("warmup"), "warmup_pair_audit": best.get("warmup_pair_audit"),
        "relation_pairs_sha256": source_audit.get("relation_pairs_sha256"),
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
    }
    (target_dir / "audits" / f"fold{fold:02d}.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=tuple(SOURCE), default=list(SOURCE))
    args = parser.parse_args()
    runner.configure_worker()
    x, y, ids, splits = runner.load_data()
    split_by_fold = {int(item["fold"]): item for item in splits}
    device = torch.device("cuda:0")
    records = {}
    for group in args.groups:
        records[group] = [replay_fold(group, fold, x, y, ids, split_by_fold[fold], device)
                          for fold in range(1, 11)]
    output = {"experiment": runner.EXPERIMENT, "status": "PASS", "groups": {
        group: {"folds": len(rows), "OOF_correct": int(sum(row["OOF_correct"] for row in rows)),
                "status": "PASS"} for group, rows in records.items()
    }}
    (ROOT / "audits/historical_replay.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
