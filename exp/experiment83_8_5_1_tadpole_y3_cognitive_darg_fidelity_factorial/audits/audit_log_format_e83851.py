"""Audit reconstructed fold logs against the author template and saved records."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as runner
from metrics import acc_auc

AUTHOR_TEMPLATE = Path("/data/adodas/SPromptGL-main/result_Tadpole.txt")
AUTHOR_TEMPLATE_SHA = "5c80d16204e0882e5e264e42952c714397c2d31f07835b9e60d6dafd9488e3b7"
AUTHOR_TEMPLATE_GIT_BLOB = "0446670764286afbeb86573dadfdbb50ae6af15e"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def epoch_line(record):
    return (
        f"{int(record['epoch']) - 1:3d}/450 trn-loss-MF: {record['trn_loss_mf_1']:.4f}  "
        f"trn-loss-MF: {record['trn_loss_mf_2']:.4f} trn-loss-MF: {record['trn_loss_mf_3']:.4f} "
        f"trn-auc: {record['trn_auc']:.4f}  acc: {record['trn_acc']:.4f} "
        f"val-auc: {record['val_auc']:.4f}  acc: {record['val_acc']:.4f}"
    )


def expected_lines(group, labels):
    lines, folds = [], []
    sensitivity_rows, specificity_rows, fold_metrics = [], [], []
    for fold in range(1, 11):
        folder = ROOT / "groups" / group
        audit = json.loads((folder / "audits" / f"fold{fold:02d}.json").read_text())
        frame = pd.read_csv(folder / "predictions" / f"fold{fold:02d}.csv")
        if audit.get("status") != "TRAINED":
            raise RuntimeError(f"{group} fold {fold} lacks formal training audit")
        records = audit["author_epoch_log"]
        if (len(records) != audit["author_epoch_log_count"]
                or [int(item["epoch"]) for item in records] != list(range(1, audit["last_epoch"] + 1))):
            raise RuntimeError(f"{group} fold {fold} has a missing epoch record")
        train_idx = np.asarray(audit["train_idx"], dtype=np.int64)
        test_idx = np.asarray(audit["test_idx"], dtype=np.int64)
        start_line = len(lines) + 1
        lines.append(str(Counter(int(value) for value in labels[train_idx])))
        lines.extend(epoch_line(item) for item in records)
        epoch_end = len(lines)
        lines.append(f"time: {audit['train_seconds']:.4f} sec.")
        truth = frame["true_label"].to_numpy(dtype=np.int64)
        prediction = frame["pred_label"].to_numpy(dtype=np.int64)
        metric = acc_auc(truth, prediction, 3)
        sensitivity, specificity = sensitivity_specificity(truth, prediction)
        sensitivity_rows.append(sensitivity)
        specificity_rows.append(specificity)
        fold_metrics.append(metric)
        lines.append(f"tst-auc: {metric['AUC']:.4f}  acc: {metric['ACC']:.4f}")
        lines.append(f"sens: {sensitivity_rows}")
        lines.append(f"spes: {specificity_rows}")
        if fold < 10:
            lines.extend((f"({len(train_idx)},)", f"({len(test_idx)},)"))
        folds.append({
            "fold": fold,
            "start_line": start_line,
            "epoch_start_line": start_line + 1,
            "epoch_end_line": epoch_end,
            "end_line": len(lines),
            "epoch_count": len(records),
        })

    sensitivity = np.asarray(sensitivity_rows)
    specificity = np.asarray(specificity_rows)
    acc = np.asarray([item["ACC"] for item in fold_metrics])
    auc = np.asarray([item["AUC"] for item in fold_metrics])
    lines.append(f"SEN_mean: {np.array2string(sensitivity.mean(0), precision=8)} SEN_std: {np.array2string(sensitivity.std(0), precision=8)}")
    lines.append(f"SPE_mean: {np.array2string(specificity.mean(0), precision=8)} SPE_std: {np.array2string(specificity.std(0), precision=8)}")
    lines.append(f"val={acc.mean():.2%} tst_acc={acc.mean():.2%} tst_auc={auc.mean():.2%}")
    lines.append(f"tst_acc_std={acc.std(ddof=0):.4f} tst_auc_std={auc.std(ddof=0):.4f}")
    lines.append(f"({acc.mean()}, {acc.mean()})")
    return lines, folds


def canonical_epoch(line):
    line = re.sub(r"^\s*\d+/\d+", "<epoch>/<max>", line)
    return re.sub(r"\d+\.\d{4}", "<value>", line)


def main():
    runner.require_training_gate(list(runner.TRAIN_GROUPS))
    _, labels, _, _ = runner.load_data()
    if not AUTHOR_TEMPLATE.is_file() or sha(AUTHOR_TEMPLATE) != AUTHOR_TEMPLATE_SHA:
        raise RuntimeError("The local author result_Tadpole.txt template identity changed")
    template = AUTHOR_TEMPLATE.read_text(encoding="utf-8").splitlines()
    template_epoch = next(line for line in template if re.match(r"^\s*\d+/\d+ trn-loss-MF:", line))
    fields = re.findall(r"trn-loss-MF|trn-auc|val-auc|acc", template_epoch)
    expected_fields = ["trn-loss-MF", "trn-loss-MF", "trn-loss-MF", "trn-auc", "acc", "val-auc", "acc"]
    if fields != expected_fields or template_epoch.count("trn-loss-MF:") != 3:
        raise RuntimeError("Author template epoch fields or order differ")

    manifest = json.loads((ROOT / "audits/log_manifest.json").read_text())
    audit = {
        "experiment": runner.EXPERIMENT,
        "status": "PASS",
        "author_template": str(AUTHOR_TEMPLATE),
        "template_sha256": AUTHOR_TEMPLATE_SHA,
        "template_git_blob": AUTHOR_TEMPLATE_GIT_BLOB,
        "reference_epoch_fields": ["epoch/450", *fields],
        "loss_field_count": 3,
        "rng_used_by_log_generation": False,
        "source": "Per-epoch records and saved fold predictions from the formal training checkpoints",
        "groups": {},
    }
    for group in runner.TRAIN_GROUPS:
        path = ROOT / "logs" / f"{group}_result_Tadpole.txt"
        actual = path.read_text(encoding="utf-8").splitlines()
        expected, folds = expected_lines(group, labels)
        if actual != expected:
            mismatch = next((i for i, (left, right) in enumerate(zip(actual, expected), 1)
                             if left != right), min(len(actual), len(expected)) + 1)
            raise RuntimeError(f"{group} author log differs from recorded results at line {mismatch}")
        epoch_lines = [line for line in actual if " trn-loss-MF:" in line]
        if (len(folds) != 10 or any(line.count("trn-loss-MF:") != 3 for line in epoch_lines)
                or any(not re.match(r"^\s*\d+/450 trn-loss-MF:", line) for line in epoch_lines)
                or any(canonical_epoch(line) != canonical_epoch(template_epoch) for line in epoch_lines)):
            raise RuntimeError(f"{group} field formatting, precision, or fold count differs from template")
        entry = manifest["groups"][group]
        if entry["sha256"] != sha(path) or len(entry["folds"]) != 10:
            raise RuntimeError(f"{group} log manifest SHA or fold count differs")
        for index, fold in enumerate(entry["folds"]):
            wanted = folds[index]
            if any(fold[key] != wanted[key] for key in wanted):
                raise RuntimeError(f"{group} fold {index + 1} log boundaries differ")
        audit["groups"][group] = {
            "status": "PASS",
            "log_sha256": sha(path),
            "line_count": len(actual),
            "fold_count": len(folds),
            "epoch_counts": [item["epoch_count"] for item in folds],
            "line_by_line_match_to_real_epoch_and_prediction_records": True,
            "author_field_order_spacing_precision_match": True,
        }
    audit["audit_program_sha256"] = sha(Path(__file__))
    runner.write_json(ROOT / "audits/log_format_audit.json", audit)
    print(json.dumps({"status": audit["status"], "groups": list(audit["groups"])}, indent=2))


if __name__ == "__main__":
    main()
