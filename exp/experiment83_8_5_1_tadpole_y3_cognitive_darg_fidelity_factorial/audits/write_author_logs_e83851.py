"""Rebuild the author-format logs from sealed formal fold histories."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as runner
from metrics import acc_auc


def sensitivity_specificity(truth, prediction):
    matrix = confusion_matrix(truth, prediction, labels=[0, 1, 2]).astype(np.float64)
    sensitivity, specificity = [], []
    total = matrix.sum()
    for class_id in range(3):
        true_positive = matrix[class_id, class_id]
        false_negative = matrix[class_id].sum() - true_positive
        false_positive = matrix[:, class_id].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive
        sensitivity.append(float(true_positive / (true_positive + false_negative + 1e-8)))
        specificity.append(float(true_negative / (true_negative + false_positive + 1e-8)))
    return sensitivity, specificity


def render_group(group, labels):
    sensitivity_rows, specificity_rows, fold_metrics = [], [], []
    lines, folds = [], []
    for fold in range(1, 11):
        folder = ROOT / "groups" / group
        audit = json.loads((folder / "audits" / f"fold{fold:02d}.json").read_text())
        frame = pd.read_csv(folder / "predictions" / f"fold{fold:02d}.csv")
        if audit.get("status") != "TRAINED":
            raise RuntimeError(f"{group} fold {fold} is not formally complete")
        records = audit["author_epoch_log"]
        if (len(records) != audit["author_epoch_log_count"]
                or [item["epoch"] for item in records] != list(range(1, audit["last_epoch"] + 1))):
            raise RuntimeError(f"{group} fold {fold} epoch records are incomplete")

        train_idx = np.asarray(audit["train_idx"], dtype=np.int64)
        test_idx = np.asarray(audit["test_idx"], dtype=np.int64)
        counter = Counter(int(value) for value in labels[train_idx])
        start_line = len(lines) + 1
        lines.append(str(counter))
        for record in records:
            epoch_display = int(record["epoch"]) - 1
            lines.append(
                f"{epoch_display:3d}/450 trn-loss-MF: {record['trn_loss_mf_1']:.4f}  "
                f"trn-loss-MF: {record['trn_loss_mf_2']:.4f} trn-loss-MF: {record['trn_loss_mf_3']:.4f} "
                f"trn-auc: {record['trn_auc']:.4f}  acc: {record['trn_acc']:.4f} "
                f"val-auc: {record['val_auc']:.4f}  acc: {record['val_acc']:.4f}"
            )
        epoch_end_line = len(lines)
        lines.append(f"time: {audit['train_seconds']:.4f} sec.")

        truth = frame["true_label"].to_numpy(dtype=np.int64)
        prediction = frame["pred_label"].to_numpy(dtype=np.int64)
        metrics = acc_auc(truth, prediction, 3)
        sensitivity, specificity = sensitivity_specificity(truth, prediction)
        sensitivity_rows.append(sensitivity)
        specificity_rows.append(specificity)
        fold_metrics.append(metrics)
        lines.append(f"tst-auc: {metrics['AUC']:.4f}  acc: {metrics['ACC']:.4f}")
        lines.append(f"sens: {sensitivity_rows}")
        lines.append(f"spes: {specificity_rows}")
        if fold < 10:
            lines.extend((f"({len(train_idx)},)", f"({len(test_idx)},)"))
        folds.append({
            "fold": fold,
            "start_line": start_line,
            "end_line": len(lines),
            "epoch_start_line": start_line + 1,
            "epoch_end_line": epoch_end_line,
            "epoch_count": len(records),
            "best_epoch": int(audit["best_epoch"]),
            "checkpoint_sha256": audit["checkpoint_sha256"],
            "latest_sha256": audit["latest_sha256"],
            "epoch_history_sha256": hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        })

    sensitivity = np.asarray(sensitivity_rows)
    specificity = np.asarray(specificity_rows)
    acc = np.asarray([metric["ACC"] for metric in fold_metrics])
    auc = np.asarray([metric["AUC"] for metric in fold_metrics])
    lines.append(f"SEN_mean: {np.array2string(sensitivity.mean(0), precision=8)} SEN_std: {np.array2string(sensitivity.std(0), precision=8)}")
    lines.append(f"SPE_mean: {np.array2string(specificity.mean(0), precision=8)} SPE_std: {np.array2string(specificity.std(0), precision=8)}")
    lines.append(f"val={acc.mean():.2%} tst_acc={acc.mean():.2%} tst_auc={auc.mean():.2%}")
    lines.append(f"tst_acc_std={acc.std(ddof=0):.4f} tst_auc_std={auc.std(ddof=0):.4f}")
    lines.append(f"({acc.mean()}, {acc.mean()})")
    text = "\n".join(lines) + "\n"
    path = ROOT / "logs" / f"{group}_result_Tadpole.txt"
    runner.atomic_write(path, lambda output: output.write(text.encode()))
    return {"path": str(path.relative_to(ROOT)), "sha256": runner.sha_file(path), "folds": folds}


def main():
    runner.require_training_gate(list(runner.TRAIN_GROUPS))
    _, labels, _, _ = runner.load_data()
    groups = {group: render_group(group, labels) for group in runner.TRAIN_GROUPS}
    runner.write_json(ROOT / "audits/log_manifest.json", {
        "experiment": runner.EXPERIMENT,
        "origin": "Rebuilt only from actual per-epoch formal training records and saved fold predictions",
        "rng_used": False,
        "groups": groups,
    })
    print(json.dumps({group: value["sha256"] for group, value in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
