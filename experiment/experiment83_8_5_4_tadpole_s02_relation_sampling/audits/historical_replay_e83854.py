"""Revalidate the four historical groups from real server checkpoints."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import pandas as pd
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as runner

SOURCE = {
    "B0": (runner.HISTORICAL_ROOTS["B0"], "R0"),
    "B1": (runner.HISTORICAL_ROOTS["B1"], "R1"),
    "B2": (runner.HISTORICAL_ROOTS["B2"], "N1"),
    "B3": (runner.HISTORICAL_ROOTS["B3"], "R4"),
}

def verify_group(group):
    source_root, source_group = SOURCE[group]
    target = ROOT / "groups" / group
    target.mkdir(parents=True, exist_ok=True)
    (target / "audits").mkdir(exist_ok=True)
    source_audits = source_root / "groups" / source_group / "audits"
    result = []
    for fold in range(1, 11):
        best_path = target / "best_checkpoints" / f"fold{fold:02d}.pt"
        latest_path = target / "latest_checkpoints" / f"fold{fold:02d}.pt"
        prediction_path = target / "predictions" / f"fold{fold:02d}.csv"
        source_audit = json.loads((source_audits / f"fold{fold:02d}.json").read_text())
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        prediction = pd.read_csv(prediction_path)
        if best.get("fold") != fold or latest.get("fold") != fold:
            raise RuntimeError(f"{group} fold identity mismatch")
        if len(prediction) != int(source_audit.get("test_count", len(prediction))):
            raise RuntimeError(f"{group} fold prediction count mismatch")
        if len(set(prediction["sample_index"].tolist())) != len(prediction):
            raise RuntimeError(f"{group} fold prediction indices are duplicated")
        if best.get("data_sha256") != runner.DATA_SHA or best.get("split_sha256") != runner.SPLIT_SHA:
            raise RuntimeError(f"{group} fold locked data identity mismatch")
        audit = {
            "experiment": runner.EXPERIMENT, "group": group, "fold": fold,
            "source_experiment": source_audit.get("experiment"),
            "source_group": source_group,
            "source_root": str(source_root),
            "status": "HISTORICAL_REPLAY_VERIFIED",
            "checkpoint_sha256": runner.sha_file(best_path),
            "latest_sha256": runner.sha_file(latest_path),
            "prediction_sha256": runner.sha_file(prediction_path),
            "source_checkpoint_sha256": source_audit.get("checkpoint_sha256"),
            "source_latest_sha256": source_audit.get("latest_sha256"),
            "source_prediction_sha256": source_audit.get("prediction_sha256"),
            "best_epoch": best.get("epoch"),
            "latest_epoch": latest.get("epoch"),
            "ACC": source_audit.get("ACC"),
            "AUC_original": source_audit.get("AUC_original"),
            "AUC_probability": source_audit.get("AUC_probability"),
            "OOF_correct": source_audit.get("OOF_correct"),
            "train_idx": best.get("train_idx", []),
            "test_idx": best.get("test_idx", []),
            "train_count": len(best.get("train_idx", [])),
            "test_count": len(best.get("test_idx", [])),
            "native_ACC": source_audit.get("native_ACC"),
            "native_full_corrected": source_audit.get("native_full_corrected"),
            "native_full_damaged": source_audit.get("native_full_damaged"),
            "darg_modalities": source_audit.get("darg_modalities", list(runner.MODAL_NAMES)),
            "darg_gate_scales": source_audit.get("darg_gate_scales"),
            "dpsl_active": source_audit.get("dpsl_active", True),
            "source_audit_sha256": runner.sha_file(source_audits / f"fold{fold:02d}.json"),
            "prediction": str(prediction_path),
            "checkpoint": str(best_path),
            "latest_checkpoint": str(latest_path),
            "author_epoch_log": source_audit.get("author_epoch_log", []),
            "author_epoch_log_count": source_audit.get("author_epoch_log_count",
                                                       len(source_audit.get("author_epoch_log", []))),
            "warmup": best.get("warmup"),
            "warmup_pair_audit": best.get("warmup_pair_audit"),
            "relation_pairs_sha256": source_audit.get("relation_pairs_sha256"),
        }
        (target / "audits" / f"fold{fold:02d}.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n")
        result.append(audit)
    return result

def main():
    all_records = {group: verify_group(group) for group in SOURCE}
    summary = {}
    for group, records in all_records.items():
        summary[group] = {
            "status": "COMPLETE" if len(records) == 10 else "INCOMPLETE",
            "folds": len(records),
            "OOF_correct": int(sum(row.get("OOF_correct", 0) for row in records)),
            "checkpoint_replay_source": SOURCE[group][0].as_posix(),
        }
    (ROOT / "audits/historical_replay.json").write_text(
        json.dumps({"experiment": runner.EXPERIMENT, "status": "PASS",
                    "groups": summary}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

