"""Render the author-format training logs from the sealed fold histories."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from run_e8385 import EXPERIMENT, FORMAL_GROUPS, atomic_write, load_data, sha_file, write_json
from metrics import acc_auc


def sensitivity_specificity(truth, prediction):
    matrix = confusion_matrix(truth, prediction, labels=[0, 1, 2]).astype(np.float64)
    sensitivity = []
    specificity = []
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
    group_dir = ROOT / "groups" / group
    counters = []
    sens_values = []
    spec_values = []
    fold_metrics = []
    lines = []
    manifest = []
    for fold in range(1, 11):
        audit_path = group_dir / "audits" / f"fold{fold:02d}.json"
        prediction_path = group_dir / "predictions" / f"fold{fold:02d}.csv"
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != "TRAINED":
            raise RuntimeError(f"Missing completed audit for {group} fold {fold}")
        frame = pd.read_csv(prediction_path)
        train_idx = np.asarray(audit["train_idx"], dtype=np.int64)
        test_idx = np.asarray(audit["test_idx"], dtype=np.int64)
        if len(audit["author_epoch_log"]) != audit["author_epoch_log_count"]:
            raise RuntimeError(f"Epoch history count differs for {group} fold {fold}")
        if [item["epoch"] for item in audit["author_epoch_log"]] != list(range(1, audit["last_epoch"] + 1)):
            raise RuntimeError(f"Epoch history is not contiguous for {group} fold {fold}")
        counter = Counter(int(value) for value in labels[train_idx])
        start_line = len(lines) + 1
        counters.append(counter)
        lines.append(str(counter))
        for record in audit["author_epoch_log"]:
            epoch_display = int(record["epoch"]) - 1
            lines.append(
                f"{epoch_display:3d}/{450:d} trn-loss-MF: {record['trn_loss_mf_1']:.4f}  "
                f"trn-loss-MF: {record['trn_loss_mf_2']:.4f} trn-loss-MF: {record['trn_loss_mf_3']:.4f} "
                f"trn-auc: {record['trn_auc']:.4f}  acc: {record['trn_acc']:.4f} "
                f"val-auc: {record['val_auc']:.4f}  acc: {record['val_acc']:.4f}"
            )
        stop_line = len(lines)
        lines.append(f"time: {audit['train_seconds']:.4f} sec.")
        fold_truth = frame["true_label"].to_numpy(dtype=np.int64)
        fold_prediction = frame["pred_label"].to_numpy(dtype=np.int64)
        metrics = acc_auc(fold_truth, fold_prediction, 3)
        sens, spec = sensitivity_specificity(fold_truth, fold_prediction)
        sens_values.append(sens)
        spec_values.append(spec)
        fold_metrics.append(metrics)
        lines.append(f"tst-auc: {metrics['AUC']:.4f}  acc: {metrics['ACC']:.4f}")
        lines.append(f"sens: {sens_values}")
        lines.append(f"spes: {spec_values}")
        if fold < 10:
            lines.append(f"({len(train_idx)},)")
            lines.append(f"({len(test_idx)},)")
        manifest.append({
            "fold": fold,
            "start_line": start_line,
            "end_line": len(lines),
            "epoch_start_line": start_line + 1,
            "epoch_end_line": stop_line,
            "checkpoint": audit["checkpoint"],
            "checkpoint_sha256": audit["checkpoint_sha256"],
            "latest_checkpoint": audit["latest_checkpoint"],
            "epoch_count": len(audit["author_epoch_log"]),
            "actual_latest_epoch": audit["last_epoch"],
            "log_record_sha256": hashlib.sha256(
                json.dumps(audit["author_epoch_log"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        })
    sensitivity = np.asarray(sens_values)
    specificity = np.asarray(spec_values)
    lines.append(f"SEN_mean: {np.array2string(sensitivity.mean(0), precision=8)} SEN_std: {np.array2string(sensitivity.std(0), precision=8)}")
    lines.append(f"SPE_mean: {np.array2string(specificity.mean(0), precision=8)} SPE_std: {np.array2string(specificity.std(0), precision=8)}")
    acc_values = np.asarray([metric["ACC"] for metric in fold_metrics])
    auc_values = np.asarray([metric["AUC"] for metric in fold_metrics])
    mean_acc, mean_auc = float(acc_values.mean()), float(auc_values.mean())
    lines.append(f"val={mean_acc:.2%} tst_acc={mean_acc:.2%} tst_auc={mean_auc:.2%}")
    lines.append(f"tst_acc_std={acc_values.std(ddof=0):.4f} tst_auc_std={auc_values.std(ddof=0):.4f}")
    lines.append(f"({mean_acc}, {mean_acc})")
    text = "\n".join(lines) + "\n"
    log_path = group_dir / "logs" / "result_Tadpole.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if group == "R0":
        if log_path.read_bytes() != text.encode():
            raise RuntimeError("Inherited R0 author log differs from its real Y3 epoch history")
    else:
        atomic_write(log_path, lambda output: output.write(text.encode()))
    return manifest, log_path, sha_file(log_path)


def main():
    _, labels, _, _ = load_data()
    all_folds = {}
    for group in FORMAL_GROUPS:
        manifest, path, digest = render_group(group, labels)
        all_folds[group] = {
            "log": str(path.relative_to(ROOT)), "sha256": digest, "folds": manifest,
            "origin": "byte-identical inherited Y3 log" if group == "R0" else "formal fold epoch records",
        }
        print(f"WROTE {group} {path} {digest}", flush=True)
    target = ROOT / "audits" / "log_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, {"experiment": EXPERIMENT, "groups": all_folds})


if __name__ == "__main__":
    main()
