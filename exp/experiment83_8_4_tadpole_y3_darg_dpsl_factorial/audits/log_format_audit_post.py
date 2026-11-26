"""Post-seal audit of author-format E83.8.4 logs against fold records."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/common"))
import run_e8384 as runner
from metrics import acc_auc


def sha256(path):
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


def expected_epoch(record):
    return (
        f"{int(record['epoch']) - 1:3d}/450 trn-loss-MF: {record['trn_loss_mf_1']:.4f}  "
        f"trn-loss-MF: {record['trn_loss_mf_2']:.4f} trn-loss-MF: {record['trn_loss_mf_3']:.4f} "
        f"trn-auc: {record['trn_auc']:.4f}  acc: {record['trn_acc']:.4f} "
        f"val-auc: {record['val_auc']:.4f}  acc: {record['val_acc']:.4f}"
    )


def line_kind(line):
    if re.fullmatch(r"Counter\(\{[0-9:, ]+\}\)", line):
        return "class_counter"
    if re.fullmatch(r"\s*\d+/\d+ trn-loss-MF: \d+\.\d{4}  trn-loss-MF: \d+\.\d{4} trn-loss-MF: \d+\.\d{4} trn-auc: \d+\.\d{4}  acc: \d+\.\d{4} val-auc: \d+\.\d{4}  acc: \d+\.\d{4}", line):
        return "epoch"
    if re.fullmatch(r"time: \d+\.\d{4} sec\.", line):
        return "time"
    if re.fullmatch(r"tst-auc: \d+\.\d{4}  acc: \d+\.\d{4}", line):
        return "test_metrics"
    if line.startswith("sens: ["):
        return "sensitivity"
    if line.startswith("spes: ["):
        return "specificity"
    if re.fullmatch(r"\(\d+,\)", line):
        return "sample_count"
    if line.startswith("SEN_mean:"):
        return "sensitivity_summary"
    if line.startswith("SPE_mean:"):
        return "specificity_summary"
    if re.fullmatch(r"val=\d+\.\d{2}% tst_acc=\d+\.\d{2}% tst_auc=\d+\.\d{2}%", line):
        return "aggregate_metrics"
    if re.fullmatch(r"tst_acc_std=\d+\.\d{4} tst_auc_std=\d+\.\d{4}", line):
        return "aggregate_std"
    if re.fullmatch(r"\([0-9.eE+-]+, [0-9.eE+-]+\)", line):
        return "aggregate_pair"
    raise RuntimeError(f"Unrecognized author log line type: {line[:120]}")


def fold_blocks(kinds):
    starts = [index for index, kind in enumerate(kinds) if kind == "class_counter"]
    if len(starts) != 10:
        raise RuntimeError(f"Expected ten author fold boundaries, got {len(starts)}")
    summary_starts = [index for index, kind in enumerate(kinds) if kind == "sensitivity_summary"]
    if len(summary_starts) != 1 or summary_starts[0] <= starts[-1]:
        raise RuntimeError("The final summary boundary is missing or misplaced")
    ends = starts[1:] + [summary_starts[0]]
    return [kinds[start:end] for start, end in zip(starts, ends)]


def canonical_epoch(line):
    line = re.sub(r"^\s*\d+/\d+", "<epoch>/<max>", line)
    return re.sub(r"\d+\.\d{4}", "<value>", line)


def expected_group(group, labels, manifest_group):
    expected = []
    fold_records = []
    cumulative_sens, cumulative_spec = [], []
    for fold in range(1, 11):
        audit_path = ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json"
        prediction_path = ROOT / "groups" / group / "predictions" / f"fold{fold:02d}.csv"
        audit = json.loads(audit_path.read_text())
        frame = pd.read_csv(prediction_path)
        if audit.get("status") != "TRAINED":
            raise RuntimeError(f"{group} fold {fold} is incomplete")
        train_idx = np.asarray(audit["train_idx"], dtype=np.int64)
        test_idx = np.asarray(audit["test_idx"], dtype=np.int64)
        records = audit.get("author_epoch_log", [])
        if len(records) != audit.get("author_epoch_log_count"):
            raise RuntimeError(f"{group} fold {fold} epoch history count differs")
        if [int(item["epoch"]) for item in records] != list(range(1, audit["last_epoch"] + 1)):
            raise RuntimeError(f"{group} fold {fold} epoch history is incomplete")
        if audit["last_epoch"] < 450 and audit["stop_wait"] <= 200:
            raise RuntimeError(f"{group} fold {fold} stopped before its locked patience condition")
        expected.append(str(Counter(int(value) for value in labels[train_idx])))
        expected.extend(expected_epoch(item) for item in records)
        expected.append(f"time: {audit['train_seconds']:.4f} sec.")
        truth = frame["true_label"].to_numpy(dtype=np.int64)
        prediction = frame["pred_label"].to_numpy(dtype=np.int64)
        metrics = acc_auc(truth, prediction, 3)
        probabilities = frame[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy(dtype=np.float64)
        from sklearn.metrics import roc_auc_score
        auc_probability = float(roc_auc_score(np.eye(3)[truth], probabilities,
                                              average="macro", multi_class="ovr"))
        best = torch.load(ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt", map_location="cpu", weights_only=False)
        best_metrics = best["metrics_at_best"]
        if (metrics["ACC"] != audit["ACC"] or metrics["AUC"] != audit["AUC_original"]
                or auc_probability != audit["AUC_probability"]
                or metrics != {key: best_metrics[key] for key in ("ACC", "AUC")}
                or float(auc_probability) != float(best_metrics["AUC_probability"])):
            raise RuntimeError(f"{group} fold {fold} test metrics differ from Best replay records")
        expected.append(f"tst-auc: {metrics['AUC']:.4f}  acc: {metrics['ACC']:.4f}")
        sens, spec = sensitivity_specificity(truth, prediction)
        cumulative_sens.append(sens)
        cumulative_spec.append(spec)
        expected.append(f"sens: {cumulative_sens}")
        expected.append(f"spes: {cumulative_spec}")
        if fold < 10:
            expected.append(f"({len(train_idx)},)")
            expected.append(f"({len(test_idx)},)")
        entry = manifest_group["folds"][fold - 1]
        if (entry["fold"] != fold or entry["epoch_count"] != len(records)
                or entry["actual_latest_epoch"] != audit["last_epoch"]
                or entry["checkpoint_sha256"] != audit["checkpoint_sha256"]
                or entry["latest_checkpoint"] != audit["latest_checkpoint"]):
            raise RuntimeError(f"{group} fold {fold} manifest identity differs")
        fold_records.append((audit, metrics, sens, spec, len(records)))

    sensitivity = np.asarray(cumulative_sens)
    specificity = np.asarray(cumulative_spec)
    expected.append(f"SEN_mean: {np.array2string(sensitivity.mean(0), precision=8)} "
                    f"SEN_std: {np.array2string(sensitivity.std(0), precision=8)}")
    expected.append(f"SPE_mean: {np.array2string(specificity.mean(0), precision=8)} "
                    f"SPE_std: {np.array2string(specificity.std(0), precision=8)}")
    acc_values = np.asarray([item[1]["ACC"] for item in fold_records])
    auc_values = np.asarray([item[1]["AUC"] for item in fold_records])
    expected.append(f"val={acc_values.mean():.2%} tst_acc={acc_values.mean():.2%} tst_auc={auc_values.mean():.2%}")
    expected.append(f"tst_acc_std={acc_values.std(ddof=0):.4f} tst_auc_std={auc_values.std(ddof=0):.4f}")
    expected.append(f"({acc_values.mean()}, {acc_values.mean()})")
    return expected, fold_records


def main():
    _, labels, _, _ = runner.load_data()
    manifest_path = ROOT / "audits/log_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment") != "Experiment 83.8.4":
        raise RuntimeError("Author log manifest identity differs")
    seal = json.loads((ROOT / "audits/training_seal.json").read_text())
    if seal["files"] != runner.current_implementation_hashes():
        raise RuntimeError("Post-seal source identity differs")
    template = (ROOT / "sources/author/result_Tadpole.txt").read_text(encoding="utf-8").splitlines()
    reference_epoch = next(line for line in template if re.match(r"^\s*\d+/\d+ trn-loss-MF:", line))
    if reference_epoch.count("trn-loss-MF:") != 3:
        raise RuntimeError("The author template has an unexpected loss-field count")
    reference_fields = re.findall(r"trn-loss-MF|trn-auc|val-auc|acc", reference_epoch)
    expected_fields = ["trn-loss-MF", "trn-loss-MF", "trn-loss-MF",
                       "trn-auc", "acc", "val-auc", "acc"]
    if reference_fields != expected_fields:
        raise RuntimeError(f"Author epoch field order differs: {reference_fields}")
    template_kinds = [line_kind(line) for line in template]
    template_blocks = fold_blocks(template_kinds)
    template_suffixes = [block[1 + block.count("epoch"):] for block in template_blocks]
    template_summary = template_kinds[sum(len(block) for block in template_blocks):]
    audit = {
        "experiment": "Experiment 83.8.4",
        "status": "PASS",
        "template_sha256": sha256(ROOT / "sources/author/result_Tadpole.txt"),
        "reference_epoch_fields": ["epoch/450", *reference_fields],
        "loss_field_count": 3,
        "template_fold_count": len(template_blocks),
        "template_epoch_count_per_fold": [block.count("epoch") for block in template_blocks],
        "template_fold_suffixes": template_suffixes,
        "template_final_summary": template_summary,
        "groups": {},
        "audit_program": "audits/log_format_audit_post.py",
        "audit_program_sha256": sha256(ROOT / "audits/log_format_audit_post.py"),
    }
    for group in runner.FORMAL_GROUPS:
        entry = manifest["groups"][group]
        log_path = ROOT / "logs" / group / "result_Tadpole.txt"
        lines = log_path.read_text(encoding="utf-8").splitlines()
        expected, fold_records = expected_group(group, labels, entry)
        actual_kinds = [line_kind(line) for line in lines]
        wanted_kinds = [line_kind(line) for line in expected]
        actual_blocks = fold_blocks(actual_kinds)
        wanted_blocks = fold_blocks(wanted_kinds)
        if len(actual_blocks) != 10 or len(wanted_blocks) != 10:
            raise RuntimeError(f"{group} log does not have ten fold blocks")
        for fold_index, (actual_block, wanted_block) in enumerate(zip(actual_blocks, wanted_blocks), 1):
            actual_suffix = actual_block[1 + actual_block[1:].count("epoch"):]
            wanted_suffix = wanted_block[1 + wanted_block[1:].count("epoch"):]
            reference_suffix = template_suffixes[fold_index - 1]
            if (actual_suffix != reference_suffix or wanted_suffix != reference_suffix
                    or actual_block.count("epoch") != wanted_block.count("epoch")):
                raise RuntimeError(f"{group} fold {fold_index} structure differs from the author template")
        actual_summary = actual_kinds[sum(len(block) for block in actual_blocks):]
        wanted_summary = wanted_kinds[sum(len(block) for block in wanted_blocks):]
        if actual_summary != template_summary or wanted_summary != template_summary:
            raise RuntimeError(f"{group} final summary structure differs from the author template")
        actual_epoch_lines = [line for line in lines if line_kind(line) == "epoch"]
        if any(canonical_epoch(line) != canonical_epoch(reference_epoch) for line in actual_epoch_lines):
            raise RuntimeError(f"{group} epoch spacing, separators or field precision differ from template")
        if lines != expected:
            mismatch = next((index for index, (actual, wanted) in enumerate(zip(lines, expected), 1)
                             if actual != wanted), min(len(lines), len(expected)) + 1)
            raise RuntimeError(f"{group} log differs from real fold records at line {mismatch}")
        if len(lines) != len(expected) or len(entry["folds"]) != 10:
            raise RuntimeError(f"{group} log does not contain ten complete folds")
        if sha256(log_path) != entry["sha256"]:
            raise RuntimeError(f"{group} log SHA differs from its manifest")
        for index, fold in enumerate(entry["folds"]):
            if (fold["start_line"] < 1 or fold["end_line"] > len(lines)
                    or fold["epoch_start_line"] != fold["start_line"] + 1
                    or fold["epoch_end_line"] - fold["epoch_start_line"] + 1 != fold["epoch_count"]):
                raise RuntimeError(f"{group} fold {index + 1} line boundaries are invalid")
        audit["groups"][group] = {
            "status": "PASS", "log_sha256": sha256(log_path), "line_count": len(lines),
            "folds": 10, "epoch_counts": [item[4] for item in fold_records],
            "line_types_match_template": True,
            "checkpoint_replayed_metrics_match": True,
        }
    target = ROOT / "audits/log_format_audit.json"
    target.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "groups": list(audit["groups"]),
                      "path": str(target)}, indent=2))


if __name__ == "__main__":
    main()
