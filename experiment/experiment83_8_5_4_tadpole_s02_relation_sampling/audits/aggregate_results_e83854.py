"""Aggregate six groups, OOF predictions, paired tests, and mechanisms."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, ttest_1samp
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83854 as runner
from relation_sampling import CLASS_NAMES, RELATION_CLASSES, RELATION_NAMES, _exposure_stats

GROUPS = runner.FORMAL_GROUPS
NEW_GROUPS = runner.TRAIN_GROUPS
METRICS = ("ACC", "AUC_original", "AUC_probability")
COMPARISONS = (
    ("T1", "H1"), ("T2", "T1"), ("T3", "T1"), ("T3", "T2"), ("T4", "T1"),
    ("T2", "H1"), ("T3", "H1"), ("T4", "H1"),
    ("T1", "H0"), ("T2", "H0"), ("T3", "H0"), ("T4", "H0"),
)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 83854
CLASS_NAMES = ("NC", "MCI", "AD")
RELATION_NAMES = ("NC-NC", "MCI-MCI", "AD-AD", "NC-MCI", "NC-AD", "MCI-AD")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability_auc(y, probability):
    return float(roc_auc_score(np.eye(3, dtype=np.float64)[y], probability,
                               average="macro", multi_class="ovr"))


def hard_auc(y, prediction):
    return float(roc_auc_score(np.eye(3, dtype=np.float64)[y],
                               np.eye(3, dtype=np.float64)[prediction],
                               average="macro", multi_class="ovr"))


def holm(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def load_groups(y, ids, splits):
    frames, fold_rows, summaries = {}, {}, []
    all_oof = []
    for group in GROUPS:
        predictions, rows = [], []
        folder = ROOT / "groups" / group
        for fold in range(1, 11):
            audit = json.loads((folder / "audits" / f"fold{fold:02d}.json").read_text())
            prediction = pd.read_csv(
                folder / "predictions" / f"fold{fold:02d}.csv").sort_values("sample_index")
            test = np.asarray(splits[fold]["test_idx"], dtype=np.int64)
            if audit.get("status") not in {"TRAINED", "HISTORICAL_REPLAY_VERIFIED"}:
                raise RuntimeError(f"{group} fold {fold}: audit is not a completed fold")
            if len(prediction) != len(test) or not np.array_equal(
                    prediction["sample_index"].to_numpy(), test):
                raise RuntimeError(f"{group} fold {fold}: OOF indices differ from locked split")
            if not np.array_equal(prediction["true_label"].to_numpy(), y[test]):
                raise RuntimeError(f"{group} fold {fold}: OOF labels differ from locked data")
            if not np.array_equal(prediction["sample_id"].to_numpy(),
                                  ids.iloc[test]["RID"].to_numpy()):
                raise RuntimeError(f"{group} fold {fold}: patient IDs differ from locked data")
            needed = {"native_pred_label", "native_prob_NC", "native_prob_MCI", "native_prob_AD"}
            if not needed.issubset(prediction.columns):
                raise RuntimeError(f"{group} fold {fold}: Native result columns are missing")
            truth = y[test]
            full_pred = prediction["pred_label"].to_numpy(dtype=np.int64)
            full_prob = prediction[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy()
            native_pred = prediction["native_pred_label"].to_numpy(dtype=np.int64)
            native_prob = prediction[["native_prob_NC", "native_prob_MCI", "native_prob_AD"]].to_numpy()
            rows.append({
                "group": group, "fold": fold,
                "train_count": len(splits[fold]["train_idx"]), "test_count": len(test),
                "best_epoch": int(audit["best_epoch"]), "latest_epoch": int(audit["latest_epoch"]),
                "ACC": float(np.mean(full_pred == truth)),
                "AUC_original": hard_auc(truth, full_pred),
                "AUC_probability": probability_auc(truth, full_prob),
                "native_ACC": float(np.mean(prediction["native_pred_label"].to_numpy() == truth)),
                "native_AUC_original": hard_auc(truth, native_pred),
                "native_AUC_probability": probability_auc(truth, native_prob),
                "native_full_corrected": int(audit.get("native_full_corrected", 0)),
                "native_full_damaged": int(audit.get("native_full_damaged", 0)),
                "train_seconds": float(audit.get("train_seconds", 0.0)),
                "peak_reserved_mib": audit.get("peak_reserved_mib"),
            })
            fold_oof = prediction.copy()
            fold_oof.insert(0, "group", group)
            predictions.append(fold_oof)
        frame = pd.concat(predictions, ignore_index=True).sort_values(
            "sample_index").reset_index(drop=True)
        if len(frame) != 598 or frame["sample_index"].duplicated().any():
            raise RuntimeError(f"{group}: OOF coverage is not exactly once per patient")
        if frame["true_label"].value_counts().sort_index().tolist() != [209, 315, 74]:
            raise RuntimeError(f"{group}: OOF class totals differ from locked cohort")
        frames[group], fold_rows[group] = frame, rows
        all_oof.append(frame)
        summaries.append(summarize(group, frame, rows))
    return frames, fold_rows, summaries, pd.concat(all_oof, ignore_index=True)


def summarize(group, frame, folds):
    truth = frame["true_label"].to_numpy(dtype=np.int64)
    summary = {"group": group, "folds": len(folds), "OOF_patients": len(frame)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in folds], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std_population"] = float(values.std(ddof=0))
    for metric in METRICS:
        values = np.asarray([row[f"native_{metric}"] for row in folds], dtype=np.float64)
        summary[f"native_{metric}_fold_mean"] = float(values.mean())
        summary[f"native_{metric}_fold_std_population"] = float(values.std(ddof=0))
    for name, pred_col, prob_cols in (
            ("native", "native_pred_label", ["native_prob_NC", "native_prob_MCI", "native_prob_AD"]),
            ("full", "pred_label", ["prob_NC", "prob_MCI", "prob_AD"])):
        pred = frame[pred_col].to_numpy(dtype=np.int64)
        probability = frame[prob_cols].to_numpy()
        correct = pred == truth
        matrix = confusion_matrix(truth, pred, labels=[0, 1, 2])
        summary[f"{name}_OOF_correct"] = int(correct.sum())
        summary[f"{name}_OOF_ACC"] = float(correct.mean())
        summary[f"{name}_OOF_AUC_original"] = hard_auc(truth, pred)
        summary[f"{name}_OOF_AUC_probability"] = probability_auc(truth, probability)
        summary[f"{name}_confusion_matrix"] = json.dumps(matrix.tolist(), separators=(",", ":"))
        for index, class_name in enumerate(CLASS_NAMES):
            summary[f"{name}_correct_{class_name}"] = int(
                np.sum((truth == index) & (pred == index)))
    summary["native_to_full_corrected"] = int(np.sum(
        (frame["native_pred_label"].to_numpy() != truth)
        & (frame["pred_label"].to_numpy() == truth)))
    summary["native_to_full_damaged"] = int(np.sum(
        (frame["native_pred_label"].to_numpy() == truth)
        & (frame["pred_label"].to_numpy() != truth)))
    summary["native_to_full_net_correct"] = summary["full_OOF_correct"] - summary["native_OOF_correct"]
    return summary


def stratified_bootstrap(y, left_pred, right_pred, left_prob, right_prob, seed):
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == index) for index in range(3)]
    deltas = {metric: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for metric in METRICS}
    for iteration in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in classes])
        truth = y[sampled]
        lpred, rpred = left_pred[sampled], right_pred[sampled]
        lprob, rprob = left_prob[sampled], right_prob[sampled]
        deltas["ACC"][iteration] = np.mean(lpred == truth) - np.mean(rpred == truth)
        deltas["AUC_original"][iteration] = hard_auc(truth, lpred) - hard_auc(truth, rpred)
        deltas["AUC_probability"][iteration] = probability_auc(truth, lprob) - probability_auc(truth, rprob)
    return {key: np.quantile(values, [0.025, 0.975]).tolist()
            for key, values in deltas.items()}


def compare(left_group, right_group, frames, folds_by_group, index):
    left, right = frames[left_group], frames[right_group]
    y = left["true_label"].to_numpy(dtype=np.int64)
    if not np.array_equal(y, right["true_label"].to_numpy()):
        raise RuntimeError(f"{left_group}/{right_group}: OOF truth differs")
    left_pred = left["pred_label"].to_numpy(dtype=np.int64)
    right_pred = right["pred_label"].to_numpy(dtype=np.int64)
    left_prob = left[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy()
    right_prob = right[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy()
    left_correct, right_correct = left_pred == y, right_pred == y
    corrected = int(np.sum(~right_correct & left_correct))
    damaged = int(np.sum(right_correct & ~left_correct))
    mcnemar = float(binomtest(corrected, corrected + damaged, 0.5).pvalue) if corrected + damaged else 1.0
    intervals = stratified_bootstrap(
        y, left_pred, right_pred, left_prob, right_prob,
        BOOTSTRAP_SEED + index * 31 + sum(map(ord, left_group + right_group)))
    left_rows = {row["fold"]: row for row in folds_by_group[left_group]}
    right_rows = {row["fold"]: row for row in folds_by_group[right_group]}
    result = {
        "left_group": left_group, "right_group": right_group,
        "delta_direction": f"{left_group} minus {right_group}",
        "corrected": corrected, "damaged": damaged, "net_correct": corrected - damaged,
        "McNemar_exact_p": mcnemar,
        "fold_ACC_wins_ties_losses": [0, 0, 0],
        "fold_AUC_original_wins_ties_losses": [0, 0, 0],
        "fold_AUC_probability_wins_ties_losses": [0, 0, 0],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    for metric in METRICS:
        diff = np.asarray([left_rows[f][metric] - right_rows[f][metric]
                           for f in range(1, 11)], dtype=np.float64)
        result[f"delta_{metric}_fold_mean"] = float(diff.mean())
        result[f"delta_{metric}_fold_population_std"] = float(diff.std(ddof=0))
        result[f"paired_fold_t_p_{metric}"] = float(ttest_1samp(diff, 0.0).pvalue) if np.any(diff) else 1.0
        result[f"fold_{metric}_wins_ties_losses"] = [
            int(np.sum(diff > 0)), int(np.sum(diff == 0)), int(np.sum(diff < 0))]
        if metric == "ACC":
            delta = float(np.mean(left_correct) - np.mean(right_correct))
        elif metric == "AUC_original":
            delta = hard_auc(y, left_pred) - hard_auc(y, right_pred)
        else:
            delta = probability_auc(y, left_prob) - probability_auc(y, right_prob)
        result[f"delta_{metric}_OOF"] = delta
        result[f"{metric}_paired_bootstrap_95ci"] = json.dumps(intervals[metric])
    return result


def flatten_sampling(group, fold, audit, labels):
    sample = audit.get("formal_sampling_audit") or {}
    exposure = sample.get("patient_exposure", {})
    checkpoint = ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"
    if not checkpoint.is_file():
        refs = json.loads((ROOT / "groups" / group / "checkpoint_refs" / f"fold{fold:02d}.json").read_text())
        checkpoint = Path(refs["best_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not exposure:
        pair_array = payload["relation_pairs"].numpy()
        train_ids = np.asarray(payload["train_idx"], dtype=np.int64)
        exposure = _exposure_stats(pair_array, train_ids, labels)
        if not sample.get("relation_counts"):
            sorted_labels = np.sort(labels[pair_array], axis=1)
            sample["relation_counts"] = {
                name: int(np.all(sorted_labels == np.asarray(classes), axis=1).sum())
                for name, classes in zip(RELATION_NAMES, RELATION_CLASSES)
            }
    row = {
        "group": group, "fold": fold, "sampler": sample.get("sampler", "historical"),
        "pair_count": sample.get("pair_count", audit.get("pair_count")),
        "same_pairs": sample.get("same_pairs", audit.get("same_pairs")),
        "different_pairs": sample.get("different_pairs", audit.get("different_pairs")),
        "target_support_count": sample.get("target_support_count"),
        "actual_support_count": sample.get("actual_support_count"),
        "actual_support_fraction": sample.get("actual_support_fraction"),
        "support_available": sample.get("support_available"),
        "relation_counts": json.dumps(sample.get("relation_counts", {}), sort_keys=True),
        "darg_gate_probabilities": json.dumps(audit.get("darg_gate_probabilities_best", [])),
        "darg_gate_scales": json.dumps(audit.get("darg_gate_scales", [])),
        "darg_gate_effective": json.dumps(audit.get("darg_effective_gate_best", [])),
        "relation_bce_best": None,
        "anchor_graph": json.dumps(audit.get("graph_mechanism", {}).get("edge_diagnostics", [])[4]
                                    if len(audit.get("graph_mechanism", {}).get("edge_diagnostics", [])) > 4 else {}),
    }
    for class_name in CLASS_NAMES:
        stats = exposure.get(class_name, {})
        for metric in ("mean", "std", "max", "p95", "gini"):
            row[f"{class_name}_{metric}"] = stats.get(metric)
    row["relation_bce_best"] = payload.get("history_at_best", {}).get("pair")
    return row


def write_report(group_rows, folds_by_group, paired, relation_summary,
                 relation_class_summary, intervention_summary, mechanism):
    lines = [
        "# Experiment 83.8.5.4 Results", "",
        "The protocol uses the historical transductive TADPOLE evaluation. Validation and test share the outer fold, and full-cohort RFE and scaling precede outer CV. These results are exploratory and are not an unbiased independent-generalization estimate.", "",
        "## Full-model results", "",
        "Fold means use population SD. AUC_original is hard-prediction macro OVR and remains the paper-comparison metric. AUC_probability and pooled OOF AUCs are reported separately.", "",
        "| Group | Model | ACC fold mean +/- SD | AUC_original fold mean +/- SD | AUC_probability fold mean +/- SD | OOF correct |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['group']} | Native | {100*row['native_ACC_fold_mean']:.4f} +/- {100*row['native_ACC_fold_std_population']:.4f}% | "
            f"{100*row['native_AUC_original_fold_mean']:.4f} +/- {100*row['native_AUC_original_fold_std_population']:.4f}% | "
            f"{100*row['native_AUC_probability_fold_mean']:.4f} +/- {100*row['native_AUC_probability_fold_std_population']:.4f}% | "
            f"{row['native_OOF_correct']}/598 |")
        lines.append(
            f"| {row['group']} | Full | {100*row['ACC_mean']:.4f} +/- {100*row['ACC_std_population']:.4f}% | "
            f"{100*row['AUC_original_mean']:.4f} +/- {100*row['AUC_original_std_population']:.4f}% | "
            f"{100*row['AUC_probability_mean']:.4f} +/- {100*row['AUC_probability_std_population']:.4f}% | "
            f"{row['full_OOF_correct']}/598 |")
    lines.extend(["", "## Pooled OOF results", "",
                  "Pooled OOF metrics aggregate all 598 one-time held-out predictions and are distinct from the unweighted mean of ten fold metrics.", "",
                  "| Group | Model | OOF correct | OOF ACC | OOF AUC_original | OOF AUC_probability |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in group_rows:
        for model_name in ("native", "full"):
            lines.append(f"| {row['group']} | {model_name.title()} | {row[f'{model_name}_OOF_correct']}/598 | "
                         f"{100*row[f'{model_name}_OOF_ACC']:.4f}% | "
                         f"{100*row[f'{model_name}_OOF_AUC_original']:.4f}% | "
                         f"{100*row[f'{model_name}_OOF_AUC_probability']:.4f}% |")
    lines.extend(["", "## Class results", "",
                  "Confusion-matrix rows are true NC/MCI/AD and columns are predicted NC/MCI/AD.", "",
                  "| Group | Model | Correct NC / MCI / AD | Confusion matrix |",
                  "|---|---|---:|---|"])
    for row in group_rows:
        for model_name in ("native", "full"):
            matrix = json.loads(row[f"{model_name}_confusion_matrix"])
            correct = "/".join(str(row[f"{model_name}_correct_{name}"]) for name in CLASS_NAMES)
            lines.append(f"| {row['group']} | {model_name.title()} | {correct} | `{json.dumps(matrix, separators=(',', ':'))}` |")
    lines.extend(["", "## Per-fold metrics", "",
                  "| Group | Fold | Native ACC | Native AUC_original | Native AUC_probability | Full ACC | Full AUC_original | Full AUC_probability |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for group in GROUPS:
        for fold in folds_by_group[group]:
            lines.append(f"| {group} | {fold['fold']} | {fold['native_ACC']:.6f} | "
                         f"{fold['native_AUC_original']:.6f} | {fold['native_AUC_probability']:.6f} | "
                         f"{fold['ACC']:.6f} | {fold['AUC_original']:.6f} | {fold['AUC_probability']:.6f} |")
    lines.extend(["", "## Paired comparisons", "",
                  "Bootstrap intervals are stratified by diagnosis with 2,000 resamples. Holm correction is across the 12 pre-registered contrasts for the paired-fold tests and exact McNemar tests.", "",
                  "| Contrast | corrected / damaged / net | McNemar p / Holm p | ACC delta OOF | ACC bootstrap 95% CI | AUC_original delta OOF | AUC_original bootstrap 95% CI | ACC fold W/T/L | AUC_original fold W/T/L | paired-fold Holm p ACC / AUC |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    def format_ci(value):
        interval = ast.literal_eval(value) if isinstance(value, str) else value
        return f"[{100*interval[0]:+.2f}, {100*interval[1]:+.2f}] pp"
    for row in paired:
        lines.append(
            f"| {row['left_group']} - {row['right_group']} | {row['corrected']} / {row['damaged']} / {row['net_correct']} | "
            f"{row['McNemar_exact_p']:.4g} / {row['McNemar_exact_holm_p']:.4g} | "
            f"{100*row['delta_ACC_OOF']:+.3f} pp | {format_ci(row['ACC_paired_bootstrap_95ci'])} | "
            f"{100*row['delta_AUC_original_OOF']:+.3f} pp | {format_ci(row['AUC_original_paired_bootstrap_95ci'])} | "
            f"{row['fold_ACC_wins_ties_losses']} | {row['fold_AUC_original_wins_ties_losses']} | "
            f"{row['paired_fold_t_holm_p_ACC']:.4g} / {row['paired_fold_t_holm_p_AUC_original']:.4g} |")
    lines.extend(["", "## Held-out DARG relation ranking", "",
                  "Relation ranking uses only patient-disjoint outer test-test pairs. The existing DARG RelationScorer is binary same-versus-different, so six relation OVR AUC uses its sigmoid score to rank each observed relation type against the other five. It is not a six-way classifier.", "",
                  "| Group | COGNITIVE_TEST SAME/DIFFERENT AUC | Mean six-relation OVR AUC | Cognitive support relation AUC | Non-support relation AUC |",
                  "|---|---:|---:|---:|---:|"])
    for row in relation_summary:
        support_auc = ("N/A" if row["support_relation_ovr_auc_mean"] is None
                       else f"{row['support_relation_ovr_auc_mean']:.4f}")
        lines.append(f"| {row['group']} | {row['same_different_auc_mean']:.4f} | {row['six_relation_ovr_auc_mean']:.4f} | "
                     f"{support_auc} | {row['non_support_relation_ovr_auc_mean']:.4f} |")
    lines.extend(["", "Six held-out disease-relation OVR AUCs for the COGNITIVE_TEST DARG scorer", "",
                  "| Group | NC-NC | MCI-MCI | AD-AD | NC-MCI | NC-AD | MCI-AD |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for group in GROUPS:
        values = {row["relation_name"]: row["relation_ovr_auc_mean"]
                  for row in relation_class_summary if row["group"] == group}
        lines.append("| " + group + " | " + " | ".join(
            f"{values[name]:.4f}" for name in RELATION_NAMES) + " |")
    lines.extend(["", "Cognitive-support held-out AUC is not estimable under the patient-disjoint rule because valid support edges are restricted to outer-training patients. The test-test support-pair count is zero. Non-support values use eligible held-out test-test pairs.", "",
                  "## Cognitive support coverage", "",
                  "Actual support is the count of selected formal relation pairs that overlap the frozen epoch-10 original COGNITIVE_TEST GraphLearn support. It is reported separately from held-out relation AUC because support pairs are train-only.", "",
                  "| Group | Target support pairs per fold | Actual mean +/- min-max | Target achieved | Fraction of all 4,096 pairs |", "|---|---:|---:|---:|---:|"])
    sampling = pd.read_csv(ROOT / "results/sampling_diagnostics.csv")
    for group in ("T1", "T2", "T3", "T4"):
        selected = sampling[sampling.group == group]
        target = float(selected["target_support_count"].mean())
        actual_mean = float(selected["actual_support_count"].mean())
        minimum = int(selected["actual_support_count"].min())
        maximum = int(selected["actual_support_count"].max())
        achieved = actual_mean / target if target else 0.0
        fraction = actual_mean / 4096.0
        lines.append(f"| {group} | {target:.1f} | {actual_mean:.1f} ({minimum}-{maximum}) | "
                     f"{100*achieved:.2f}% | {100*fraction:.2f}% |")
    lines.extend(["", "## Patient exposure", "",
                  "| Group | NC mean exposure / Gini | MCI mean exposure / Gini | AD mean exposure / Gini |",
                  "|---|---:|---:|---:|"])
    exposure_path = ROOT / "results/exposure_summary.csv"
    if exposure_path.is_file():
        for row in pd.read_csv(exposure_path).to_dict(orient="records"):
            lines.append(f"| {row['group']} | {row['NC_mean']:.2f} / {row['NC_gini']:.4f} | "
                         f"{row['MCI_mean']:.2f} / {row['MCI_gini']:.4f} | "
                         f"{row['AD_mean']:.2f} / {row['AD_gini']:.4f} |")
    lines.extend(["", "## DARG graph mechanism", "",
                  "| Group | Anchor effective gate | Positive / negative relation edges | Mean adjacency relative change | Mean GCN message relative norm |",
                  "|---|---:|---:|---:|---:|"])
    graph_path = ROOT / "results/anchor_graph_summary.csv"
    if graph_path.is_file():
        for row in pd.read_csv(graph_path).to_dict(orient="records"):
            lines.append(f"| {row['group']} | {row['effective_gate_mean']:.6f} | "
                         f"{row['positive_score_edges_mean']:.2f} / {row['negative_score_edges_mean']:.2f} | "
                         f"{row['adjacency_change_relative_mean']:.6f} | "
                         f"{row['gcn_message_raw_l2_ratio_mean']:.6f} |")
    lines.extend(["", "## DARG and DPSL inference effects", "",
                  "| Group | Native to Full corrected / damaged | DPSL Full-Native mean absolute logit delta | Direct-Off logit delta / changed | Disease-Neutral logit delta / changed | Cognitive graph off logit delta / changed |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in intervention_summary:
        group_row = next(item for item in group_rows if item["group"] == row["group"])
        lines.append(
            f"| {row['group']} | {group_row['native_to_full_corrected']} / {group_row['native_to_full_damaged']} | "
            f"{row['dpsl_full_native_logit_mean_abs']:.6f} | "
            f"{row['direct_off_logit_mean_abs']:.6f} / {row['direct_off_changed_mean']:.2f} | "
            f"{row['disease_neutral_logit_mean_abs']:.6f} / {row['disease_neutral_changed_mean']:.2f} | "
            f"{row['cognitive_graph_off_logit_mean_abs']:.6f} / {row['cognitive_graph_off_changed_mean']:.2f} |")
    lines.extend(["", "## Mechanism", "",
                  f"The signed graph, gate, message, and intervention aggregates are recorded in {mechanism['path']}. DARG training, direct graph propagation, and DPSL residual effects are reported separately there.", "",
                  "The paper reference is 94.81 +/- 2.95% ACC and 95.59 +/- 2.57% AUC. AUC_original remains the paper-comparison metric. Repeated tuning on TADPOLE makes all paired tests exploratory.", ""])
    (ROOT / "results/final_report.md").write_text("\n".join(lines))


def main():
    for needed in ("checkpoint_replay.json", "log_format_audit.json", "relation_generalization.json",
                   "intervention_replay.json"):
        path = ROOT / "audits" / needed
        if not path.is_file():
            raise RuntimeError(f"Required audit is missing: {path}")
    replay = json.loads((ROOT / "audits/checkpoint_replay.json").read_text())
    logs = json.loads((ROOT / "audits/log_format_audit.json").read_text())
    relation_audit = json.loads((ROOT / "audits/relation_generalization.json").read_text())
    intervention_audit = json.loads((ROOT / "audits/intervention_replay.json").read_text())
    if replay.get("status") != "PASS" or replay.get("folds_passed") != 60:
        raise RuntimeError("60/60 checkpoint replay is required")
    if replay.get("latest_next_step_passed") != 40:
        raise RuntimeError("40/40 deterministic Latest recovery is required")
    if (logs.get("status") != "PASS" or relation_audit.get("status") != "PASS"
            or intervention_audit.get("status") != "PASS"
            or intervention_audit.get("folds") != 60):
        raise RuntimeError("Log or held-out relation audit failed")
    _, y, ids, raw_splits = runner.load_data()
    splits = {int(row["fold"]): row for row in raw_splits}
    frames, folds_by_group, group_rows, oof = load_groups(y, ids, splits)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    oof.to_csv(results / "oof_predictions.csv", index=False)
    fold_frame = pd.DataFrame([row for group in GROUPS for row in folds_by_group[group]])
    fold_frame.to_csv(results / "fold_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(results / "group_summary.csv", index=False)

    paired_path = results / "paired_contrasts.csv"
    if paired_path.is_file():
        cached_paired = pd.read_csv(paired_path)
        observed = list(zip(cached_paired["left_group"], cached_paired["right_group"]))
        cache_ok = (observed == list(COMPARISONS)
                    and np.all(cached_paired["bootstrap_replicates"] == BOOTSTRAP_REPLICATES)
                    and "McNemar_exact_holm_p" in cached_paired.columns)
    else:
        cache_ok = False
    if cache_ok:
        paired = cached_paired.to_dict(orient="records")
    else:
        paired = [compare(left, right, frames, folds_by_group, index)
                  for index, (left, right) in enumerate(COMPARISONS)]
        for metric in METRICS:
            adjusted = holm([row[f"paired_fold_t_p_{metric}"] for row in paired])
            for row, pvalue in zip(paired, adjusted):
                row[f"paired_fold_t_holm_p_{metric}"] = pvalue
        adjusted = holm([row["McNemar_exact_p"] for row in paired])
        for row, pvalue in zip(paired, adjusted):
            row["McNemar_exact_holm_p"] = pvalue
        pd.DataFrame(paired).to_csv(paired_path, index=False)

    relation = pd.read_csv(results / "relation_generalization.csv")
    anchor = relation[relation["modality"] == "COGNITIVE_TEST"]
    auc_rows = anchor[anchor["same_different_auc"].notna()]
    ovr_rows = anchor[anchor["relation_ovr_auc"].notna()]
    support_ovr = anchor[anchor["support_relation_ovr_auc"].notna()]
    nonsupport_ovr = anchor[anchor["non_support_relation_ovr_auc"].notna()]
    relation_class_summary = []
    for group in GROUPS:
        for relation_name in RELATION_NAMES:
            selected = ovr_rows[(ovr_rows.group == group)
                                & (ovr_rows.relation_name == relation_name)]
            relation_class_summary.append({
                "group": group, "relation_name": relation_name,
                "relation_ovr_auc_mean": float(selected["relation_ovr_auc"].mean()),
                "relation_ovr_auc_std_population": float(selected["relation_ovr_auc"].std(ddof=0)),
                "held_out_pairs_mean": float(selected["test_test_pair_count"].mean()),
            })
    relation_summary = []
    for group in GROUPS:
        relation_summary.append({
            "group": group,
            "same_different_auc_mean": float(auc_rows.loc[auc_rows.group == group, "same_different_auc"].mean()),
            "six_relation_ovr_auc_mean": float(ovr_rows.loc[ovr_rows.group == group, "relation_ovr_auc"].mean()),
            "support_relation_ovr_auc_mean": (float(support_ovr.loc[support_ovr.group == group, "support_relation_ovr_auc"].mean())
                                               if not support_ovr.loc[support_ovr.group == group].empty else None),
            "non_support_relation_ovr_auc_mean": (float(nonsupport_ovr.loc[nonsupport_ovr.group == group, "non_support_relation_ovr_auc"].mean())
                                                  if not nonsupport_ovr.loc[nonsupport_ovr.group == group].empty else None),
            "held_out_support_pair_count": 0,
        })
    relation.to_csv(results / "relation_generalization.csv", index=False)
    pd.DataFrame(relation_summary).to_csv(results / "relation_summary.csv", index=False)
    pd.DataFrame(relation_class_summary).to_csv(results / "relation_class_summary.csv", index=False)

    intervention_frame = pd.read_csv(results / "intervention_diagnostics.csv")
    intervention_summary = []
    for group in GROUPS:
        selected = intervention_frame[intervention_frame.group == group]
        intervention_summary.append({
            "group": group,
            "dpsl_full_native_logit_mean_abs": float(selected["dpsl_full_native_logit_mean_abs"].mean()),
            "dpsl_full_native_logit_max_abs": float(selected["dpsl_full_native_logit_max_abs"].mean()),
            "direct_off_changed_mean": float(selected["full_vs_direct_off_prediction_changes"].map(
                lambda value: ast.literal_eval(value)["changed"]).mean()),
            "disease_neutral_changed_mean": float(selected["full_vs_disease_neutral_prediction_changes"].map(
                lambda value: ast.literal_eval(value)["changed"]).mean()),
            "cognitive_graph_off_changed_mean": float(selected["full_vs_cognitive_graph_off_prediction_changes"].map(
                lambda value: ast.literal_eval(value)["changed"]).mean()),
            "direct_off_logit_mean_abs": float(selected["full_vs_direct_off_logit_mean_abs"].mean()),
            "disease_neutral_logit_mean_abs": float(selected["full_vs_disease_neutral_logit_mean_abs"].mean()),
            "cognitive_graph_off_logit_mean_abs": float(selected["full_vs_cognitive_graph_off_logit_mean_abs"].mean()),
        })
    pd.DataFrame(intervention_summary).to_csv(results / "intervention_summary.csv", index=False)

    sampling_rows = []
    mechanism_rows = []
    modality_rows = []
    intervention_by_fold = {
        (row["group"], int(row["fold"])): row
        for row in intervention_frame.to_dict(orient="records")
    }
    for group in GROUPS:
        for fold in range(1, 11):
            audit_path = ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json"
            audit = json.loads(audit_path.read_text())
            intervention = intervention_by_fold[(group, fold)]
            sampling_row = flatten_sampling(group, fold, audit, y)
            sampling_row["darg_gate_probabilities"] = json.dumps(
                ast.literal_eval(intervention["gate_probabilities"]))
            sampling_row["darg_gate_scales"] = json.dumps(ast.literal_eval(intervention["gate_scales"]))
            sampling_row["darg_gate_effective"] = json.dumps(ast.literal_eval(intervention["effective_gates"]))
            sampling_rows.append(sampling_row)
            graph_info = json.loads(intervention["graph_mechanism"])
            edge_rows = graph_info.get("edge_diagnostics", [])
            for edge in edge_rows:
                modality_rows.append({"group": group, "fold": fold, **edge})
            mechanism_rows.append({
                "group": group, "fold": fold,
                "relation_bce_best": sampling_rows[-1]["relation_bce_best"],
                "effective_gates": json.dumps(ast.literal_eval(intervention["effective_gates"])),
                "gate_probabilities": json.dumps(ast.literal_eval(intervention["gate_probabilities"])),
                "gate_scales": json.dumps(ast.literal_eval(intervention["gate_scales"])),
                "signed_edge_diagnostics": json.dumps(edge_rows),
                "graph_message_relative_norms": json.dumps(audit.get("graph_message_relative_norm_mean_by_modality", [])),
                "direct_off": json.dumps(audit.get("direct_off", {})),
                "disease_neutral": json.dumps(audit.get("disease_neutral", {})),
                "cognitive_graph_off": json.dumps(audit.get("cognitive_graph_off", {})),
                "native_full_corrected": audit.get("native_full_corrected"),
                "native_full_damaged": audit.get("native_full_damaged"),
                "native_full_net": (audit.get("native_full_corrected", 0) or 0)
                                    - (audit.get("native_full_damaged", 0) or 0),
                "native_full_logit_max_abs": audit.get("native_full_logit_max_abs"),
                "dpsl_residual_max": audit.get("dpsl_residual_max"),
                "dpsl_shared_gradient_ratio_max": audit.get("dpsl_shared_gradient_ratio_max"),
                "dpsl_residual_off_exact_native": audit.get("dpsl_residual_off_exact_native"),
            })
    pd.DataFrame(sampling_rows).to_csv(results / "sampling_diagnostics.csv", index=False)
    pd.DataFrame(mechanism_rows).to_csv(results / "mechanism_diagnostics.csv", index=False)
    pd.DataFrame(modality_rows).to_csv(results / "graph_mechanism_by_modality.csv", index=False)
    sampling_frame = pd.DataFrame(sampling_rows)
    exposure_summary = []
    for group in GROUPS:
        selected = sampling_frame[sampling_frame.group == group]
        row = {"group": group}
        for class_name in CLASS_NAMES:
            for metric in ("mean", "std", "max", "p95", "gini"):
                row[f"{class_name}_{metric}"] = float(selected[f"{class_name}_{metric}"].dropna().mean())
        exposure_summary.append(row)
    pd.DataFrame(exposure_summary).to_csv(results / "exposure_summary.csv", index=False)

    anchor_graph_rows = [row for row in modality_rows if row.get("modality") == "COGNITIVE_TEST"]
    anchor_graph_summary = []
    for group in GROUPS:
        selected = [row for row in anchor_graph_rows if row["group"] == group]
        anchor_graph_summary.append({
            "group": group,
            "effective_gate_mean": float(np.mean([
                json.loads(next(row["effective_gates"] for row in mechanism_rows
                                if row["group"] == group and row["fold"] == edge["fold"]))[4]
                for edge in selected])),
            "positive_score_edges_mean": float(np.mean([row["positive_score_edges"] for row in selected])),
            "negative_score_edges_mean": float(np.mean([row["negative_score_edges"] for row in selected])),
            "adjacency_change_relative_mean": float(np.mean([row["adjacency_change_relative"] for row in selected])),
            "gcn_message_raw_l2_ratio_mean": float(np.mean([row["gcn_message_raw_l2_ratio"] for row in selected])),
        })
    pd.DataFrame(anchor_graph_summary).to_csv(results / "anchor_graph_summary.csv", index=False)

    mechanism = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "groups": list(GROUPS), "folds": 60,
        "relation_metrics": relation_summary,
        "six_relation_ovr_metrics": relation_class_summary,
        "inference_interventions": intervention_summary,
        "patient_exposure_summary": exposure_summary,
        "cognitive_anchor_graph_summary": anchor_graph_summary,
        "native_full": {row["group"]: {
            "native_correct": row["native_OOF_correct"], "full_correct": row["full_OOF_correct"],
            "corrected": row["native_to_full_corrected"], "damaged": row["native_to_full_damaged"],
            "net": row["native_to_full_net_correct"],
        } for row in group_rows},
        "DARG_and_DPSL_fold_diagnostics": "results/mechanism_diagnostics.csv",
        "signed_graph_by_modality": "results/graph_mechanism_by_modality.csv",
        "inference_interventions_by_fold": "results/intervention_diagnostics.csv",
        "sampling_and_exposure_diagnostics": "results/sampling_diagnostics.csv",
        "held_out_relation_scores": "results/relation_generalization.csv",
        "notes": [
            "Relation AUC uses only outer test-test patient pairs.",
            "RelationScorer is binary same-versus-different. Six relation OVR AUC uses its scalar score and does not imply a six-way output head.",
            "Support-stratified relation metrics use the shared epoch-10 frozen original cognitive graph support.",
            "Full-model classification changes are separated from DARG relation ranking and from direct inference interventions.",
        ],
        "path": "results/mechanism_diagnostics.csv",
    }
    (ROOT / "audits/mechanism_summary.json").write_text(json.dumps(mechanism, indent=2, sort_keys=True) + "\n")

    write_report(group_rows, folds_by_group, paired, relation_summary, relation_class_summary,
                 intervention_summary, mechanism)
    aggregate = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "groups": list(GROUPS), "folds": 60, "oof_rows": int(len(oof)),
        "patients_per_group": 598, "paired_comparisons": len(paired),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "holm_family_size": len(paired),
        "checkpoint_replay_status": replay["status"],
        "latest_next_step_status": replay["latest_next_step_recovery_status"],
        "log_audit_status": logs["status"], "relation_audit_status": relation_audit["status"],
        "intervention_audit_status": intervention_audit["status"],
        "outputs": {path.name: sha_file(path) for path in sorted(results.glob("*")) if path.is_file()},
    }
    (ROOT / "audits/aggregate_results.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
