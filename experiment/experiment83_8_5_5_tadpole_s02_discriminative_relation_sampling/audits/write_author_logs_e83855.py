"""Write paper-format TADPOLE logs from real epoch records or historical logs."""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83855 as runner

TEMPLATE = Path("/data/adodas/SPromptGL-main/result_Tadpole.txt")
HISTORICAL_LOGS = {
    "H0": Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl/groups/R0/logs/result_Tadpole.txt"),
    "H1": Path("/model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial/logs/S02_result_Tadpole.txt"),
}
EPOCH_PATTERN = re.compile(
    r"^\s*\d+/450 trn-loss-MF: [-0-9.]+  trn-loss-MF: [-0-9.]+ "
    r"trn-loss-MF: [-0-9.]+ trn-auc: [-0-9.]+  acc: [-0-9.]+ "
    r"val-auc: [-0-9.]+  acc: [-0-9.]+$"
)


def sha_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sensitivity_specificity(truth, predicted):
    matrix = confusion_matrix(truth, predicted, labels=[0, 1, 2])
    sens, spes = [], []
    total = matrix.sum()
    for index in range(3):
        tp = matrix[index, index]
        fn = matrix[index].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = total - tp - fn - fp
        sens.append(float(tp / (tp + fn + 1e-8)))
        spes.append(float(tn / (tn + fp + 1e-8)))
    return sens, spes


def write_group(group, y, splits):
    directory = ROOT / "groups" / group
    output = ROOT / "logs" / f"{group}_result_Tadpole.txt"
    lines = []
    fold_sens, fold_spes = [], []
    fold_acc, fold_auc = [], []
    first = True
    for fold in range(1, 11):
        split = splits[fold]
        train = np.asarray(split["train_idx"], dtype=np.int64)
        test = np.asarray(split["test_idx"], dtype=np.int64)
        audit = json.loads((directory / "audits" / f"fold{fold:02d}.json").read_text())
        prediction = pd.read_csv(
            directory / "predictions" / f"fold{fold:02d}.csv").sort_values("sample_index")
        if (not np.array_equal(prediction["sample_index"].to_numpy(), test)
                or len(prediction) != len(test)):
            raise RuntimeError(f"{group} fold {fold}: prediction rows do not match split")
        if first:
            first = False
        counter = Counter(int(value) for value in y[train])
        lines.append(f"{counter}\n")
        epoch_records = audit.get("author_epoch_log", [])
        if len(epoch_records) != int(audit.get("author_epoch_log_count", -1)):
            raise RuntimeError(f"{group} fold {fold}: missing real author epoch records")
        if not epoch_records:
            raise RuntimeError(f"{group} fold {fold}: author epoch log is empty")
        if [int(row["epoch"]) for row in epoch_records] != list(
                range(1, len(epoch_records) + 1)):
            raise RuntimeError(f"{group} fold {fold}: author epoch sequence is incomplete")
        for row in epoch_records:
            lines.append(
                f"{int(row['epoch']) - 1:3d}/450 "
                f"trn-loss-MF: {float(row['trn_loss_mf_1']):.4f}  "
                f"trn-loss-MF: {float(row['trn_loss_mf_2']):.4f} "
                f"trn-loss-MF: {float(row['trn_loss_mf_3']):.4f} "
                f"trn-auc: {float(row['trn_auc']):.4f}  "
                f"acc: {float(row['trn_acc']):.4f} "
                f"val-auc: {float(row['val_auc']):.4f}  "
                f"acc: {float(row['val_acc']):.4f}\n"
            )
        predicted = prediction["pred_label"].to_numpy(dtype=np.int64)
        truth = prediction["true_label"].to_numpy(dtype=np.int64)
        sens, spes = sensitivity_specificity(truth, predicted)
        fold_sens.append(sens)
        fold_spes.append(spes)
        acc = float(audit["ACC"])
        auc = float(audit["AUC_original"])
        if abs(float(np.mean(predicted == truth)) - acc) > 1e-12:
            raise RuntimeError(f"{group} fold {fold}: author log ACC differs from prediction")
        lines.extend([
            f"time: {float(audit['train_seconds']):.4f} sec.\n",
            f"tst-auc: {auc:.4f}  acc: {acc:.4f}\n",
            f"sens: {fold_sens}\n",
            f"spes: {fold_spes}\n",
        ])
        if fold < 10:
            lines.extend([f"({len(train)},)\n", f"({len(test)},)\n"])
        fold_acc.append(acc)
        fold_auc.append(auc)
    acc_values = np.asarray(fold_acc, dtype=np.float64)
    auc_values = np.asarray(fold_auc, dtype=np.float64)
    sens_values = np.asarray(fold_sens, dtype=np.float64)
    spes_values = np.asarray(fold_spes, dtype=np.float64)
    acc_mean, acc_std = float(acc_values.mean()), float(acc_values.std(ddof=0))
    auc_mean, auc_std = float(auc_values.mean()), float(auc_values.std(ddof=0))
    lines.extend([
        f"SEN_mean: {sens_values.mean(axis=0)} SEN_std: {sens_values.std(axis=0)}\n",
        f"SPE_mean: {spes_values.mean(axis=0)} SPE_std: {spes_values.std(axis=0)}\n",
        f"val={acc_mean * 100:.2f}% tst_acc={acc_mean * 100:.2f}% tst_auc={auc_mean * 100:.2f}%\n",
        f"tst_acc_std={acc_std:.4f} tst_auc_std={auc_std:.4f}\n",
        f"({acc_mean}, {acc_mean})\n",
    ])
    output.write_text("".join(lines))
    return {
        "group": group, "status": "PASS", "file": str(output),
        "sha256": sha_file(output), "source": "real saved author_epoch_log and fold predictions",
        "epochs_by_fold": [
            int(json.loads((directory / "audits" / f"fold{fold:02d}.json").read_text())
                ["author_epoch_log_count"]) for fold in range(1, 11)],
        "folds": 10, "epoch_denominator": 450,
        "trn_loss_mf_fields_per_epoch": 3,
        "field_order": [
            "epoch/450", "trn-loss-MF", "trn-loss-MF", "trn-loss-MF",
            "trn-auc", "acc", "val-auc", "acc",
        ],
    }


def verify_log(path):
    text = path.read_text()
    lines = text.splitlines()
    epoch_lines = [line for line in lines if "/450 trn-loss-MF:" in line]
    if not epoch_lines or any(not EPOCH_PATTERN.fullmatch(line) for line in epoch_lines):
        raise RuntimeError(f"{path.name}: author epoch field layout differs")
    if any(line.count("trn-loss-MF:") != 3 for line in epoch_lines):
        raise RuntimeError(f"{path.name}: expected exactly three trn-loss-MF fields")
    if len([line for line in lines if line.startswith("time: ")]) != 10:
        raise RuntimeError(f"{path.name}: expected ten fold time records")
    if len([line for line in lines if line.startswith("tst-auc: ")]) != 10:
        raise RuntimeError(f"{path.name}: expected ten fold test summaries")
    if len([line for line in lines if line.startswith("sens: ")]) != 10:
        raise RuntimeError(f"{path.name}: expected ten sensitivity fold records")
    if len([line for line in lines if line.startswith("spes: ")]) != 10:
        raise RuntimeError(f"{path.name}: expected ten specificity fold records")
    if len([line for line in lines if line.startswith("Counter(")]) != 10:
        raise RuntimeError(f"{path.name}: expected ten fold label-count records")
    if len([line for line in lines if re.fullmatch(r"\(\d+,\)", line)]) != 18:
        raise RuntimeError(f"{path.name}: expected nine train/test fold-boundary pairs")
    allowed_prefixes = (
        "Counter(", "time: ", "tst-auc: ", "sens: ", "spes: ",
        "SEN_mean: ", "SPE_mean: ", "val=", "tst_acc_std=",
    )
    for line in lines:
        if EPOCH_PATTERN.fullmatch(line) or re.fullmatch(r"\(\d+,\)", line):
            continue
        if line.startswith(allowed_prefixes):
            continue
        if re.fullmatch(r"\([-0-9.eE+, ]+\)", line):
            continue
        raise RuntimeError(f"{path.name}: unexpected author-log line {line[:80]!r}")
    return {
        "status": "PASS", "sha256": sha_file(path),
        "epoch_line_count": len(epoch_lines),
        "counter_count": 10, "train_test_boundary_count": 18,
        "three_trn_loss_fields": True, "author_field_order": True,
        "epoch_denominator_450": True,
    }


def main():
    if not TEMPLATE.is_file():
        raise FileNotFoundError(TEMPLATE)
    template_sha = sha_file(TEMPLATE)
    expected_template_sha = json.loads((ROOT / "source_manifest.json").read_text())[
        "author_repository"]["result_Tadpole_sha256"]
    if template_sha != expected_template_sha:
        raise RuntimeError("Read-only result_Tadpole.txt template SHA differs from manifest")
    _, y, _, raw_splits = runner.load_data()
    splits = {int(item["fold"]): item for item in raw_splits}
    groups = {}
    for group in runner.FORMAL_GROUPS:
        output = ROOT / "logs" / f"{group}_result_Tadpole.txt"
        if group in HISTORICAL_LOGS:
            source = HISTORICAL_LOGS[group]
            if not source.is_file():
                raise FileNotFoundError(f"Missing real historical author log: {source}")
            shutil.copyfile(source, output)
            groups[group] = {
                "group": group, "status": "PASS", "file": str(output),
                "sha256": sha_file(output), "source": str(source),
                "source_sha256": sha_file(source), "epochs_reconstructed": False,
            }
        else:
            groups[group] = write_group(group, y, splits)
            groups[group]["epochs_reconstructed"] = True
    verification = {group: verify_log(ROOT / "logs" / f"{group}_result_Tadpole.txt")
                    for group in runner.FORMAL_GROUPS}
    report = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "template": str(TEMPLATE), "template_sha256": template_sha,
        "template_git_blob": json.loads((ROOT / "source_manifest.json").read_text())
            ["author_repository"]["result_Tadpole_git_blob"],
        "groups": groups, "verification": verification,
        "format_contract": {
            "epoch_denominator": 450, "loss_fields": 3,
            "epoch_field_order": [
                "epoch/450", "trn-loss-MF", "trn-loss-MF", "trn-loss-MF",
                "trn-auc", "acc", "val-auc", "acc",
            ],
            "fold_fields": [
                "time", "tst-auc", "acc", "sens", "spes",
                "train_shape", "test_shape", "next_fold_Counter",
            ],
            "custom_fields_added": False, "per_epoch_decimal_precision": 4,
            "time_decimal_precision": 4, "fold_metric_decimal_precision": 4,
        },
        "script_sha256": sha_file(Path(__file__)),
    }
    (ROOT / "audits/log_format_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"],
                      "groups": {name: value["status"] for name, value in verification.items()}},
                     indent=2))


if __name__ == "__main__":
    main()
