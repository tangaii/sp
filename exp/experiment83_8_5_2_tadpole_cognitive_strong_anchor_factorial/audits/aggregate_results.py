"""Build locked OOF metrics, group summaries and preregistered paired contrasts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as runner

CONTRASTS = (
    ("S09", "B0"), ("S09", "B2"), ("S09", "S08"), ("S09", "S05"),
    ("S02", "B3"), ("S01", "S03"), ("S04", "S06"), ("S05", "S07"),
)
DIAGNOSTIC_CONTRASTS = (
    ("S09", "C02"), ("C01", "S08"), ("C03", "S09"),
)
METRICS = ("ACC", "AUC_original", "AUC_probability")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 8352


def probability_auc(y, probability):
    return float(roc_auc_score(np.eye(3, dtype=np.float64)[y], probability,
                               average="macro", multi_class="ovr"))


def hard_auc(y, prediction):
    return float(roc_auc_score(np.eye(3, dtype=np.float64)[y],
                               np.eye(3, dtype=np.float64)[prediction],
                               average="macro", multi_class="ovr"))


def holm_adjust(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def stratified_paired_bootstrap(y, left, right, *, seed, replicates):
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == class_id) for class_id in range(3)]
    deltas = {metric: np.empty(replicates, dtype=np.float64) for metric in METRICS}
    for iteration in range(replicates):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in classes
        ])
        truth = y[sampled]
        left_pred = left["full_pred"][sampled]
        right_pred = right["full_pred"][sampled]
        deltas["ACC"][iteration] = np.mean(left_pred == truth) - np.mean(right_pred == truth)
        deltas["AUC_original"][iteration] = (
            hard_auc(truth, left_pred) - hard_auc(truth, right_pred))
        deltas["AUC_probability"][iteration] = (
            probability_auc(truth, left["full_prob"][sampled])
            - probability_auc(truth, right["full_prob"][sampled]))
    return {
        metric: np.quantile(values, [0.025, 0.975]).tolist()
        for metric, values in deltas.items()
    }


def read_group(group, y, ids, splits):
    folder = ROOT / "groups" / group
    predictions, folds = [], []
    for fold in range(1, 11):
        audit_path = folder / "audits" / f"fold{fold:02d}.json"
        audit = json.loads(audit_path.read_text())
        prediction = pd.read_csv(
            folder / "predictions" / f"fold{fold:02d}.csv").sort_values("sample_index")
        split = splits[fold]
        expected_test = np.asarray(split["test_idx"], dtype=np.int64)
        if len(prediction) != len(expected_test) or not np.array_equal(
                prediction["sample_index"].to_numpy(), expected_test):
            raise RuntimeError(f"{group} fold {fold}: OOF test rows differ from locked split")
        if (not np.array_equal(prediction["true_label"].to_numpy(), y[expected_test])
                or not np.array_equal(
                    prediction["sample_id"].to_numpy(),
                    ids.iloc[expected_test]["RID"].to_numpy())):
            raise RuntimeError(f"{group} fold {fold}: OOF identity or label mismatch")
        if "native_pred_label" not in prediction:
            raise RuntimeError(f"{group} fold {fold}: Native predictions are missing")
        for row in prediction.to_dict(orient="records"):
            sample_index = int(row["sample_index"])
            predictions.append({
                "group": group,
                "RID": str(row["sample_id"]),
                "fold": int(row["fold"]),
                "sample_index": sample_index,
                "true_label": int(row["true_label"]),
                "native_prediction": int(row["native_pred_label"]),
                "full_prediction": int(row["pred_label"]),
                "native_prob_NC": float(row["native_prob_NC"]),
                "native_prob_MCI": float(row["native_prob_MCI"]),
                "native_prob_AD": float(row["native_prob_AD"]),
                "full_prob_NC": float(row["prob_NC"]),
                "full_prob_MCI": float(row["prob_MCI"]),
                "full_prob_AD": float(row["prob_AD"]),
                "best_checkpoint_sha256": audit.get("checkpoint_sha256"),
                "latest_checkpoint_sha256": audit.get("latest_sha256"),
                "prediction_sha256": audit.get("prediction_sha256"),
            })
        truth = y[expected_test]
        full_prediction = prediction["pred_label"].to_numpy(dtype=np.int64)
        full_probability = prediction[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy()
        folds.append({
            "group": group, "fold": fold,
            "train_count": len(split["train_idx"]), "test_count": len(expected_test),
            "best_epoch": int(audit["best_epoch"]),
            "latest_epoch": int(audit["latest_epoch"]),
            "ACC": float(np.mean(full_prediction == truth)),
            "AUC_original": hard_auc(truth, full_prediction),
            "AUC_probability": probability_auc(truth, full_probability),
            "train_seconds": float(audit.get("train_seconds", 0.0)),
            "peak_reserved_mib": audit.get("peak_reserved_mib"),
        })
    frame = pd.DataFrame(predictions).sort_values("sample_index").reset_index(drop=True)
    if len(frame) != 598 or frame["sample_index"].duplicated().any():
        raise RuntimeError(f"{group}: OOF coverage is not exactly once per patient")
    if frame["true_label"].value_counts().sort_index().tolist() != [209, 315, 74]:
        raise RuntimeError(f"{group}: OOF class coverage differs from locked cohort")
    return frame, folds


def group_summary(group, frame, folds):
    truth = frame["true_label"].to_numpy(dtype=np.int64)
    native = frame["native_prediction"].to_numpy(dtype=np.int64)
    full = frame["full_prediction"].to_numpy(dtype=np.int64)
    native_prob = frame[["native_prob_NC", "native_prob_MCI", "native_prob_AD"]].to_numpy()
    full_prob = frame[["full_prob_NC", "full_prob_MCI", "full_prob_AD"]].to_numpy()
    row = {"group": group, "folds": len(folds), "OOF_patients": len(frame)}
    for metric in METRICS:
        values = np.asarray([item[metric] for item in folds], dtype=np.float64)
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std_population"] = float(values.std(ddof=0))
    for name, prediction, probability in (
            ("native", native, native_prob), ("full", full, full_prob)):
        correct = prediction == truth
        row[f"{name}_OOF_correct"] = int(correct.sum())
        row[f"{name}_OOF_accuracy"] = float(correct.mean())
        row[f"{name}_AUC_original"] = hard_auc(truth, prediction)
        row[f"{name}_AUC_probability"] = probability_auc(truth, probability)
        matrix = confusion_matrix(truth, prediction, labels=[0, 1, 2])
        row[f"{name}_confusion_matrix"] = json.dumps(matrix.tolist(), separators=(",", ":"))
        for class_id, class_name in enumerate(("NC", "MCI", "AD")):
            row[f"{name}_correct_{class_name}"] = int(
                np.sum((truth == class_id) & (prediction == class_id)))
    row["native_to_full_corrected"] = int(np.sum((native != truth) & (full == truth)))
    row["native_to_full_damaged"] = int(np.sum((native == truth) & (full != truth)))
    row["native_to_full_net_correct"] = row["full_OOF_correct"] - row["native_OOF_correct"]
    return row


def compare_groups(left_group, right_group, frames, fold_rows, replicate_count):
    left, right = frames[left_group], frames[right_group]
    y = left["true_label"].to_numpy(dtype=np.int64)
    if not np.array_equal(y, right["true_label"].to_numpy()):
        raise RuntimeError(f"{left_group} vs {right_group}: OOF labels differ")
    if not np.array_equal(left["sample_index"], right["sample_index"]):
        raise RuntimeError(f"{left_group} vs {right_group}: OOF patient order differs")
    left_bundle = {
        "full_pred": left["full_prediction"].to_numpy(dtype=np.int64),
        "full_prob": left[["full_prob_NC", "full_prob_MCI", "full_prob_AD"]].to_numpy(),
    }
    right_bundle = {
        "full_pred": right["full_prediction"].to_numpy(dtype=np.int64),
        "full_prob": right[["full_prob_NC", "full_prob_MCI", "full_prob_AD"]].to_numpy(),
    }
    left_correct = left_bundle["full_pred"] == y
    right_correct = right_bundle["full_pred"] == y
    corrected = int(np.sum(~right_correct & left_correct))
    damaged = int(np.sum(right_correct & ~left_correct))
    mcnemar_p = float(binomtest(corrected, corrected + damaged, 0.5,
                                alternative="two-sided").pvalue) if corrected + damaged else 1.0
    intervals = stratified_paired_bootstrap(
        y, left_bundle, right_bundle, seed=BOOTSTRAP_SEED + sum(map(ord, left_group + right_group)),
        replicates=replicate_count)

    left_folds = {row["fold"]: row for row in fold_rows[left_group]}
    right_folds = {row["fold"]: row for row in fold_rows[right_group]}
    fold_differences = {
        metric: np.asarray([
            left_folds[fold][metric] - right_folds[fold][metric]
            for fold in range(1, 11)
        ], dtype=np.float64)
        for metric in METRICS
    }
    comparison = {
        "left_group": left_group, "right_group": right_group,
        "delta_direction": f"{left_group} minus {right_group}",
        "corrected_vs_right": corrected, "damaged_vs_right": damaged,
        "net_correct_vs_right": corrected - damaged,
        "McNemar_exact_p": mcnemar_p,
        "fold_ACC_wins_ties_losses": [
            int(np.sum(fold_differences["ACC"] > 0)),
            int(np.sum(fold_differences["ACC"] == 0)),
            int(np.sum(fold_differences["ACC"] < 0)),
        ],
        "fold_AUC_original_wins_ties_losses": [
            int(np.sum(fold_differences["AUC_original"] > 0)),
            int(np.sum(fold_differences["AUC_original"] == 0)),
            int(np.sum(fold_differences["AUC_original"] < 0)),
        ],
        "fold_AUC_probability_wins_ties_losses": [
            int(np.sum(fold_differences["AUC_probability"] > 0)),
            int(np.sum(fold_differences["AUC_probability"] == 0)),
            int(np.sum(fold_differences["AUC_probability"] < 0)),
        ],
        "bootstrap_replicates": replicate_count,
        "bootstrap_seed": BOOTSTRAP_SEED + sum(map(ord, left_group + right_group)),
    }
    for metric, values in fold_differences.items():
        comparison[f"delta_{metric}_fold_mean"] = float(values.mean())
        comparison[f"delta_{metric}_fold_population_std"] = float(values.std(ddof=0))
        comparison[f"paired_fold_t_p_{metric}"] = float(
            ttest_1samp(values, 0.0).pvalue) if np.any(values) else 1.0
        comparison[f"delta_{metric}_OOF"] = (
            float(np.mean(left_bundle["full_pred"] == y) -
                  np.mean(right_bundle["full_pred"] == y))
            if metric == "ACC" else
            hard_auc(y, left_bundle["full_pred"]) - hard_auc(y, right_bundle["full_pred"])
            if metric == "AUC_original" else
            probability_auc(y, left_bundle["full_prob"])
            - probability_auc(y, right_bundle["full_prob"])
        )
        comparison[f"delta_{metric}_paired_bootstrap_95ci_low"] = intervals[metric][0]
        comparison[f"delta_{metric}_paired_bootstrap_95ci_high"] = intervals[metric][1]
    return comparison


def main():
    replay_path = ROOT / "audits/checkpoint_replay.json"
    mechanism_path = ROOT / "audits/mechanism_summary.json"
    log_path = ROOT / "audits/log_format_audit.json"
    if not replay_path.is_file():
        raise RuntimeError(f"Required independent replay is missing: {replay_path}")
    replay = json.loads(replay_path.read_text())
    if (replay.get("best_latest_replay_status") != "PASS"
            or replay.get("best_latest_folds_passed") != 160
            or replay.get("folds_expected") != 160):
        raise RuntimeError("Best/Latest checkpoint replay must pass for all 160 folds")
    for path in (mechanism_path, log_path):
        if not path.is_file() or json.loads(path.read_text()).get("status") != "PASS":
            raise RuntimeError(f"Required independent audit is missing or failed: {path}")
    x, y, ids, raw_splits = runner.load_data()
    splits = {int(item["fold"]): item for item in raw_splits}
    all_frames, all_fold_rows, summaries = {}, [], []
    for group in runner.FORMAL_GROUPS:
        frame, folds = read_group(group, y, ids, splits)
        all_frames[group] = frame
        all_fold_rows.extend(folds)
        all_fold_rows_by_group = None
        summaries.append(group_summary(group, frame, folds))
    fold_by_group = {
        group: [row for row in all_fold_rows if row["group"] == group]
        for group in runner.FORMAL_GROUPS
    }
    oof_frame = pd.concat(
        [all_frames[group] for group in runner.FORMAL_GROUPS],
        ignore_index=True)
    oof_frame.to_csv(ROOT / "results/oof_predictions.csv", index=False)
    pd.DataFrame(all_fold_rows).sort_values(["group", "fold"]).to_csv(
        ROOT / "results/fold_metrics.csv", index=False)
    pd.DataFrame(summaries).sort_values("group").to_csv(
        ROOT / "results/group_summary.csv", index=False)

    primary_comparisons = [
        compare_groups(left, right, all_frames, fold_by_group, BOOTSTRAP_REPLICATES)
        for left, right in CONTRASTS
    ]
    diagnostic_comparisons = [
        compare_groups(left, right, all_frames, fold_by_group, BOOTSTRAP_REPLICATES)
        for left, right in DIAGNOSTIC_CONTRASTS
    ]
    comparisons = primary_comparisons + diagnostic_comparisons
    for family, family_rows, suffix in (
            ("primary", primary_comparisons, ""),
            ("diagnostic", diagnostic_comparisons, "_diagnostic")):
        for metric in METRICS:
            adjusted = holm_adjust([
                row[f"paired_fold_t_p_{metric}"] for row in family_rows])
            for row, p_value in zip(family_rows, adjusted):
                row[f"paired_fold_t_holm_p{suffix}_{metric}"] = p_value
        adjusted_mcnemar = holm_adjust([
            row["McNemar_exact_p"] for row in family_rows])
        for row, p_value in zip(family_rows, adjusted_mcnemar):
            row[f"McNemar_exact_holm_p{suffix}"] = p_value
    pd.DataFrame(comparisons).to_csv(ROOT / "results/paired_contrasts.csv", index=False)

    report = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
        "group_count": len(all_frames), "fold_count": len(all_fold_rows),
        "oof_rows": len(oof_frame), "patients_per_group": 598,
        "unique_patients_per_group": {
            group: int(frame["sample_index"].nunique()) for group, frame in all_frames.items()},
        "paired_contrasts": [f"{left} vs {right}" for left, right in CONTRASTS],
        "diagnostic_contrasts": [
            f"{left} vs {right}" for left, right in DIAGNOSTIC_CONTRASTS],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "holm_family_count": len(CONTRASTS),
        "diagnostic_holm_family_count": len(DIAGNOSTIC_CONTRASTS),
        "checkpoint_replay": {
            "best_latest_status": replay["best_latest_replay_status"],
            "latest_next_step_recovery_status": replay[
                "latest_next_step_recovery_status"],
            "latest_next_step_recovery_passed": replay[
                "latest_next_step_passed"],
            "latest_next_step_recovery_expected": replay[
                "latest_next_step_expected"],
        },
        "outputs": {
            "group_summary.csv": runner.sha_file(ROOT / "results/group_summary.csv"),
            "fold_metrics.csv": runner.sha_file(ROOT / "results/fold_metrics.csv"),
            "oof_predictions.csv": runner.sha_file(ROOT / "results/oof_predictions.csv"),
            "paired_contrasts.csv": runner.sha_file(ROOT / "results/paired_contrasts.csv"),
        },
    }
    (ROOT / "audits/aggregate_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
