"""Audit the reused Z0 files against the sealed E83.8.3-Y3 artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
Y3_ROOT = ROOT.parent / "experiment83_8_3_tadpole_n3_darg_dpsl_optimization"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    historical_seal = json.loads((Y3_ROOT / "audits/training_seal.json").read_text())
    for fold in range(1, 11):
        checks = {}
        for relative in (f"audits/fold{fold:02d}.json", f"predictions/fold{fold:02d}.csv",
                         f"best_checkpoints/fold{fold:02d}.pt", f"latest_checkpoints/fold{fold:02d}.pt"):
            left = ROOT / "groups/Z0" / relative
            right = Y3_ROOT / "groups/Y3" / relative
            checks[relative] = sha(left) == sha(right)
            if not checks[relative]:
                raise RuntimeError(f"Z0 fold {fold} differs from Y3 for {relative}")
        left_audit = json.loads((ROOT / "groups/Z0/audits" / f"fold{fold:02d}.json").read_text())
        right_audit = json.loads((Y3_ROOT / "groups/Y3/audits" / f"fold{fold:02d}.json").read_text())
        if left_audit["ACC"] != right_audit["ACC"] or left_audit["AUC_original"] != right_audit["AUC_original"]:
            raise RuntimeError(f"Z0 fold {fold} metrics differ from Y3")
        rows.append({"fold": fold, "files_exact": True, "metrics_exact": True,
                     "best_epoch": int(left_audit["best_epoch"]),
                     "latest_epoch": int(left_audit["last_epoch"]),
                     "best_sha256": sha(ROOT / "groups/Z0/best_checkpoints" / f"fold{fold:02d}.pt"),
                     "latest_sha256": sha(ROOT / "groups/Z0/latest_checkpoints" / f"fold{fold:02d}.pt"),
                     "prediction_sha256": sha(ROOT / "groups/Z0/predictions" / f"fold{fold:02d}.csv")})
    left_metrics = pd.read_csv(ROOT / "groups/Z0/fold_metrics.csv")
    right_metrics = pd.read_csv(Y3_ROOT / "groups/Y3/fold_metrics.csv")
    if left_metrics.to_csv(index=False) != right_metrics.to_csv(index=False):
        raise RuntimeError("Z0 fold metrics table differs from Y3")
    report = {
        "experiment": "Experiment 83.8.4", "status": "PASS_EXACT", "group": "Z0",
        "reference_experiment": "Experiment 83.8.3", "reference_group": "Y3",
        "reference_root": str(Y3_ROOT), "historical_y3_implementation_sha256": historical_seal["implementation_sha256"],
        "folds_passed": 10, "best_latest_prediction_files_exact": True,
        "folds": rows,
    }
    target = ROOT / "audits/z0_y3_full_regression.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "folds_passed": report["folds_passed"], "path": str(target)}, indent=2))


if __name__ == "__main__":
    main()
