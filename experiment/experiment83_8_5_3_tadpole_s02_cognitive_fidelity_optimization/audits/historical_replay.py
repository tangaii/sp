"""Independently replay H0 and H1 from the sealed server archive."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial"
BASELINE_ARCHIVE = ROOT.parent / "experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl"
SOURCES = {"H0": "B0", "H1": "S02"}


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def logged_evaluation(latest_model, cam, x, labels, train, pairs, targets):
    """Recover the pre-EMA evaluation corresponding to a post-EMA Latest."""
    from scmr_runtime import scmr_components, source_config

    statistics = latest_model.dpsl.statistics
    current = statistics.snapshot()
    config = source_config()["evidence"]
    alpha = config["alpha"]
    _, _, evidence = scmr_components(len(x), cam.shape[1])
    pre = statistics.snapshot()
    for name, moment in zip(("mean", "second"), statistics.observation(cam)):
        pre[name] = (current[name] - alpha * moment) / (1 - alpha)
    n = pre["counts"][..., None].double()
    variance = (pre["second"] - pre["mean"].square()).clamp_min(0) * (n / (n - config["ddof"]))
    pre["stable"] = evidence.stabilize(variance, pre["counts"])[1]
    pre["updates"] -= 1
    try:
        statistics.restore(pre)
        with torch.no_grad():
            scores, _ = latest_model(x, labels, train, pairs, targets)
        statistics.advance(statistics.observation(cam))
        for name in ("mean", "second", "stable"):
            torch.testing.assert_close(statistics.ema[name], current[name], atol=1e-12, rtol=1e-12)
        return scores
    finally:
        statistics.restore(current)


def run():
    sys.path.insert(0, str(ARCHIVE / "code"))
    sys.path.insert(0, str(ARCHIVE / "audits"))
    import checkpoint_replay as archive_audit
    import run_e8385 as runner

    runner.configure_worker()
    x, y, ids, splits = runner.load_data()
    device = torch.device("cuda:0")
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, device=device)
    source_audit = ARCHIVE / "audits/checkpoint_replay.json"
    records = {(r["group"], r["fold"]): r for r in json.loads(source_audit.read_text())["records"]}
    output = []
    for key, group in SOURCES.items():
        folder = ARCHIVE / "groups" / group
        oof = []
        for fold, split in enumerate(splits, 1):
            file = f"fold{fold:02d}"
            paths = {
                "best_sha256": folder / "best_checkpoints" / f"{file}.pt",
                "latest_sha256": folder / "latest_checkpoints" / f"{file}.pt",
                "prediction_sha256": folder / "predictions" / f"{file}.csv",
                "fold_audit_sha256": folder / "audits" / f"{file}.json",
            }
            for field, path in paths.items():
                if sha(path) != records[group, fold][field]:
                    raise RuntimeError(f"{key}/{fold}: archived {field} does not match {path}")
            if key == "H0":
                for field in ("best_sha256", "latest_sha256", "prediction_sha256"):
                    path = paths[field]
                    upstream = BASELINE_ARCHIVE / "groups/R0" / path.parent.name / path.name
                    if sha(upstream) != records[group, fold][field]:
                        raise RuntimeError(f"{key}/{fold}: original R0 {field} differs")
            result = archive_audit.replay_fold(group, fold, x, y, ids, split, device,
                                               next_step=False)
            best = torch.load(paths["best_sha256"], map_location="cpu", weights_only=False)
            latest = torch.load(paths["latest_sha256"], map_location="cpu", weights_only=False)
            assert latest["best_sha256"] == result["best_sha256"]
            assert latest["best_epoch"] == best["epoch"]
            rows = np.asarray(split["test_idx"])
            train = np.asarray(split["train_idx"])
            for payload in (best, latest):
                pairs = payload["relation_pairs"].numpy()
                assert np.isin(pairs, train).all()
                np.testing.assert_array_equal(payload["relation_targets"].numpy(),
                                              y[pairs[:, 0]] == y[pairs[:, 1]])
            latest_result = archive_audit.inference(group, fold, latest, x_t, y_t, split, device)
            _, latest_model, latest_logp, _, _, latest_aux = latest_result
            eval_logp = logged_evaluation(
                latest_model, latest_aux["cam"], x_t, y_t, torch.as_tensor(train, device=device),
                latest["relation_pairs"].to(device), latest["relation_targets"].to(device))
            logged_pred = eval_logp.argmax(1).cpu().numpy()
            latest_pred = latest_logp.argmax(1).cpu().numpy()
            latest_metrics = runner.acc_auc(y[rows], latest_pred[rows], 3)
            latest_saved = latest["author_epoch_log_records"][-1]
            for indices, prefix in ((rows, "val"), (train, "trn")):
                metrics = runner.acc_auc(y[indices], logged_pred[indices], 3)
                assert abs(metrics["ACC"] - latest_saved[f"{prefix}_acc"]) <= 1e-12
                assert abs(metrics["AUC"] - latest_saved[f"{prefix}_auc"]) <= 1e-12
            latest_metrics["AUC_probability"] = float(roc_auc_score(
                np.eye(3)[y[rows]], latest_logp.exp().cpu().numpy()[rows], average="macro"))
            first_best = next(r["epoch"] for r in latest["history"]
                              if r["test_acc"] == max(q["test_acc"] for q in latest["history"]))
            assert first_best == best["epoch"]
            assert sha(paths["best_sha256"]) == result["best_sha256"]
            assert sha(paths["latest_sha256"]) == result["latest_sha256"]
            result.update(group=key, archive_group=group, best_path=str(paths["best_sha256"]),
                          latest_path=str(paths["latest_sha256"]),
                          latest_metrics=latest_metrics, latest_pre_ema_history_metrics_match=True,
                          latest_pre_vs_post_ema_test_flips=int(np.sum(
                              logged_pred[rows] != latest_pred[rows])))
            latest_dir = ROOT / "groups" / key / "latest_predictions"
            latest_dir.mkdir(parents=True, exist_ok=True)
            latest_probs = latest_logp.exp().cpu().numpy()
            pd.DataFrame({
                "sample_index": rows, "true_label": y[rows], "pred_label": latest_pred[rows],
                **{f"prob_{c}": latest_probs[rows, i] for i, c in enumerate(("NC", "MCI", "AD"))},
            }).to_csv(latest_dir / f"{file}.csv", index=False)
            output.append(result)
            pred = pd.read_csv(paths["prediction_sha256"])
            oof.extend(zip(pred["sample_index"], pred["true_label"] == pred["pred_label"]))
            print(f"{key} fold {fold:02d} Best/Latest PASS", flush=True)
        assert sorted(idx for idx, _ in oof) == list(range(598))
        assert sum(ok for _, ok in oof) == {"H0": 560, "H1": 559}[key]
        expected = {"H0": (93.6441, 2.2192, 94.4265, 2.1648, 95.8290, 2.3408),
                    "H1": (93.4746, 3.0327, 94.4340, 3.1326, 96.0281, 2.6419)}[key]
        actual = []
        for metric in ("ACC", "AUC_original", "AUC_probability"):
            values = np.array([r["best_metrics"][metric] for r in output if r["group"] == key])
            actual.extend((round(100 * values.mean(), 4), round(100 * values.std(), 4)))
        assert tuple(actual) == expected, (key, actual, expected)
    (ROOT / "audits/historical_replay.json").write_text(json.dumps({
        "status": "PASS", "folds": len(output), "e8352_checkpoint_audit_sha256": sha(source_audit),
        "records": output,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    run()
