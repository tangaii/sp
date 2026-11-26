"""Compute OOF metrics, paired contrasts and the final E83.8.3 report."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "common"))
import run_e8383 as runner
from metrics import acc_auc


CLASS_NAMES = {0: "NC", 1: "MCI", 2: "AD"}
HISTORICAL_ROOT = runner.N3_REFERENCE_ROOT
G4_ROOT = runner.N3_REFERENCE_ROOT
PAPER = {"ACC": .9481, "ACC_std": .0295, "AUC_original": .9559, "AUC_std": .0257}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def probability_auc(truth, probabilities):
    return float(roc_auc_score(np.eye(3, dtype=np.float64)[truth], probabilities,
                               average="macro", multi_class="ovr"))


def load_group_predictions(group, root=ROOT):
    frames = []
    fold_metrics = []
    for fold in range(1, 11):
        directory = root / "groups" / group
        audit = __import__("json").loads((directory / "audits" / f"fold{fold:02d}.json").read_text())
        if audit.get("status") != "TRAINED":
            raise RuntimeError(f"{group} fold {fold} is incomplete")
        train_idx = np.asarray(audit["train_idx"], dtype=np.int64)
        test_idx = np.asarray(audit["test_idx"], dtype=np.int64)
        if np.intersect1d(train_idx, test_idx).size:
            raise RuntimeError(f"{group} fold {fold} has train/test overlap")
        frame = pd.read_csv(directory / "predictions" / f"fold{fold:02d}.csv")
        if not np.array_equal(frame["sample_index"].to_numpy(dtype=np.int64), test_idx):
            raise RuntimeError(f"{group} fold {fold} prediction rows differ from test indices")
        truth = frame["true_label"].to_numpy(dtype=np.int64)
        prediction = frame["pred_label"].to_numpy(dtype=np.int64)
        probabilities = frame[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy(dtype=np.float64)
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1.0, atol=1e-6):
            raise RuntimeError(f"{group} fold {fold} probabilities are not valid distributions")
        metrics = acc_auc(truth, prediction, 3)
        metrics["AUC_probability"] = probability_auc(truth, probabilities)
        metrics["OOF_correct"] = int((truth == prediction).sum())
        metrics["fold"] = fold
        metrics["group"] = group
        metrics["test_count"] = len(frame)
        metrics["best_epoch"] = audit["best_epoch"]
        metrics["train_seconds"] = audit["train_seconds"]
        fold_metrics.append(metrics)
        frame.insert(0, "group", group)
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True).sort_values("sample_index").reset_index(drop=True)
    if len(oof) != 598 or oof["sample_index"].nunique() != 598:
        raise RuntimeError(f"{group} OOF predictions do not cover each of 598 patients once")
    return oof, fold_metrics


def historical_predictions(group):
    return load_group_predictions(group, HISTORICAL_ROOT)


def summarize_group(group, oof, fold_metrics):
    truth = oof["true_label"].to_numpy(dtype=np.int64)
    prediction = oof["pred_label"].to_numpy(dtype=np.int64)
    probabilities = oof[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy(dtype=np.float64)
    hard = acc_auc(truth, prediction, 3)
    result = {
        "group": group,
        "folds": len(fold_metrics),
        "ACC_mean": float(np.mean([row["ACC"] for row in fold_metrics])),
        "ACC_population_std": float(np.std([row["ACC"] for row in fold_metrics], ddof=0)),
        "AUC_original_mean": float(np.mean([row["AUC"] for row in fold_metrics])),
        "AUC_original_population_std": float(np.std([row["AUC"] for row in fold_metrics], ddof=0)),
        "AUC_probability_mean": float(np.mean([row["AUC_probability"] for row in fold_metrics])),
        "AUC_probability_population_std": float(np.std([row["AUC_probability"] for row in fold_metrics], ddof=0)),
        "OOF_ACC": hard["ACC"],
        "OOF_AUC_original": hard["AUC"],
        "OOF_AUC_probability": probability_auc(truth, probabilities),
        "OOF_correct": int((truth == prediction).sum()),
        "NC_correct": int(((truth == 0) & (prediction == 0)).sum()),
        "MCI_correct": int(((truth == 1) & (prediction == 1)).sum()),
        "AD_correct": int(((truth == 2) & (prediction == 2)).sum()),
        "NC_total": int((truth == 0).sum()),
        "MCI_total": int((truth == 1).sum()),
        "AD_total": int((truth == 2).sum()),
    }
    return result


def fold_contrasts(group, reference, left_metrics, right_metrics, rows, label):
    for metric in ("ACC", "AUC", "AUC_probability"):
        values = [(left[metric], right[metric]) for left, right in zip(left_metrics, right_metrics)]
        deltas = np.asarray([left - right for left, right in values], dtype=np.float64)
        rows.append({
            "comparison": label, "group": group, "reference": reference,
            "scope": "paired_folds", "fold": "mean",
            "metric": {"ACC": "ACC", "AUC": "AUC_original", "AUC_probability": "AUC_probability"}[metric],
            "group_value": float(np.mean([left for left, _ in values])),
            "reference_value": float(np.mean([right for _, right in values])),
            "difference_pp": float(deltas.mean() * 100),
            "difference_population_std_pp": float(deltas.std(ddof=0) * 100),
            "group_wins": int((deltas > 1e-12).sum()),
            "ties": int((np.abs(deltas) <= 1e-12).sum()),
            "group_losses": int((deltas < -1e-12).sum()),
            "mcnemar_corrections": "", "mcnemar_harms": "", "mcnemar_exact_p": "",
        })
        for fold_index, ((left, right), delta) in enumerate(zip(values, deltas)):
            rows.append({
                "comparison": label, "group": group, "reference": reference,
                "scope": "fold", "fold": left_metrics[fold_index]["fold"],
                "metric": {"ACC": "ACC", "AUC": "AUC_original", "AUC_probability": "AUC_probability"}[metric],
                "group_value": float(left), "reference_value": float(right),
                "difference_pp": float(delta * 100), "difference_population_std_pp": "",
                "group_wins": int(delta > 1e-12), "ties": int(abs(delta) <= 1e-12),
                "group_losses": int(delta < -1e-12),
                "mcnemar_corrections": "", "mcnemar_harms": "", "mcnemar_exact_p": "",
            })


def patient_contrast(group, reference, left_oof, right_oof, rows, label):
    left = left_oof.sort_values("sample_index").reset_index(drop=True)
    right = right_oof.sort_values("sample_index").reset_index(drop=True)
    check(np.array_equal(left["sample_index"], right["sample_index"]), f"{label} patient OOF rows differ")
    check(np.array_equal(left["true_label"], right["true_label"]), f"{label} patient labels differ")
    a = left["pred_label"].to_numpy(dtype=np.int64)
    b = right["pred_label"].to_numpy(dtype=np.int64)
    truth = left["true_label"].to_numpy(dtype=np.int64)
    corrected = int(((b != truth) & (a == truth)).sum())
    harmed = int(((b == truth) & (a != truth)).sum())
    p_value = float(binomtest(min(corrected, harmed), corrected + harmed, .5).pvalue) if corrected + harmed else 1.0
    delta = (a == truth).astype(np.int64) - (b == truth).astype(np.int64)
    rows.append({
        "comparison": label, "group": group, "reference": reference,
        "scope": "patient_oof", "fold": "all", "metric": "correctness",
        "class_name": "all",
        "group_value": float((a == truth).mean()), "reference_value": float((b == truth).mean()),
        "difference_pp": float(delta.mean() * 100), "difference_population_std_pp": "",
        "group_wins": corrected, "ties": int((delta == 0).sum()), "group_losses": harmed,
        "mcnemar_corrections": corrected, "mcnemar_harms": harmed, "mcnemar_exact_p": p_value,
    })
    for class_id, class_name in CLASS_NAMES.items():
        selected = truth == class_id
        class_corrected = int(((b != truth) & (a == truth) & selected).sum())
        class_harmed = int(((a != truth) & (b == truth) & selected).sum())
        class_delta = (a[selected] == truth[selected]).astype(np.int64) - (b[selected] == truth[selected]).astype(np.int64)
        class_p = (float(binomtest(min(class_corrected, class_harmed), class_corrected + class_harmed, .5).pvalue)
                   if class_corrected + class_harmed else 1.0)
        rows.append({
            "comparison": label, "group": group, "reference": reference,
            "scope": "patient_oof_by_class", "fold": "all", "metric": "correctness",
            "class_name": class_name,
            "group_value": float((a[selected] == truth[selected]).mean()),
            "reference_value": float((b[selected] == truth[selected]).mean()),
            "difference_pp": float(class_delta.mean() * 100), "difference_population_std_pp": "",
            "group_wins": class_corrected, "ties": int((class_delta == 0).sum()),
            "group_losses": class_harmed, "mcnemar_corrections": class_corrected,
            "mcnemar_harms": class_harmed, "mcnemar_exact_p": class_p,
        })


def paper_contrast(group, summary, rows):
    for metric, reference in (("ACC", PAPER["ACC"]), ("AUC_original", PAPER["AUC_original"])):
        value = summary[f"{metric}_mean"]
        rows.append({
            "comparison": f"{group} vs paper", "group": group, "reference": "paper",
            "scope": "published_mean", "fold": "all", "metric": metric,
            "group_value": value, "reference_value": reference,
            "difference_pp": (value - reference) * 100,
            "difference_population_std_pp": "", "group_wins": "", "ties": "", "group_losses": "",
            "mcnemar_corrections": "", "mcnemar_harms": "", "mcnemar_exact_p": "",
        })


def build_report(summaries, contrasts, mechanism=None):
    best_acc = max(summaries, key=lambda row: row["ACC_mean"])
    best_hard_auc = max(summaries, key=lambda row: row["AUC_original_mean"])
    best_prob_auc = max(summaries, key=lambda row: row["AUC_probability_mean"])
    baseline = next(row for row in summaries if row["group"] == "Y0")
    new_groups = [row for row in summaries if row["group"] != "Y0"]
    candidates = [row for row in new_groups
                  if row["ACC_mean"] > baseline["ACC_mean"]
                  and row["AUC_original_mean"] > baseline["AUC_original_mean"]
                  and row["AUC_probability_mean"] >= baseline["AUC_probability_mean"]]
    if candidates:
        status = "CANDIDATE"
        next_group = max(candidates, key=lambda row: (row["ACC_mean"], row["AUC_original_mean"]))["group"]
    else:
        any_change = any(row["ACC_mean"] != baseline["ACC_mean"]
                         or row["AUC_original_mean"] != baseline["AUC_original_mean"]
                         or row["AUC_probability_mean"] != baseline["AUC_probability_mean"]
                         for row in new_groups)
        status = "MIXED" if any_change else "NO_GO"
        next_group = "Y0"
    paper_reached = any(row["ACC_mean"] >= PAPER["ACC"]
                        and row["AUC_original_mean"] >= PAPER["AUC_original"] for row in summaries)
    lines = [
        "# Experiment 83.8.3 Results",
        "",
        f"Status: **{status}**. Next candidate to retain: **{next_group}**.",
        "",
        "## Fold Results",
        "",
        "| Group | ACC mean ± pop. SD | AUC_original mean ± pop. SD | AUC_probability mean ± pop. SD | OOF correct | NC | MCI | AD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['group']} | {row['ACC_mean']:.4%} ± {row['ACC_population_std']:.4%} "
            f"| {row['AUC_original_mean']:.4%} ± {row['AUC_original_population_std']:.4%} "
            f"| {row['AUC_probability_mean']:.4%} ± {row['AUC_probability_population_std']:.4%} "
            f"| {row['OOF_correct']}/598 | {row['NC_correct']}/209 | {row['MCI_correct']}/315 | {row['AD_correct']}/74 |"
        )
    lines += [
        "",
        "## Fixed References",
        "",
        "| Reference | ACC | AUC_original | AUC_probability |",
        "|---|---:|---:|---:|",
        "| E83.8.2 G4 / N0 | 92.9746% | 94.0967% | 96.2710% |",
        "| E83.8.2 N3 | 93.3079% | 94.1819% | 95.8360% |",
        "| Paper | 94.81% ± 2.95% | 95.59% ± 2.57% | not reported |",
        "",
        f"Highest ACC: **{best_acc['group']}** ({best_acc['ACC_mean']:.4%}). Highest AUC_original: "
        f"**{best_hard_auc['group']}** ({best_hard_auc['AUC_original_mean']:.4%}). Highest probability AUC: "
        f"**{best_prob_auc['group']}** ({best_prob_auc['AUC_probability_mean']:.4%}).",
        "",
        "## Paired Contrasts",
        "",
        "| Comparison | ΔACC pp | ΔAUC_original pp | ΔAUC_probability pp | OOF corrections / harms | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    comparison_names = list(dict.fromkeys(
        row["comparison"] for row in contrasts if row["scope"] in ("paired_folds", "patient_oof")))
    for label in comparison_names:
        selected = [row for row in contrasts if row["comparison"] == label]
        metrics = {row["metric"]: row for row in selected if row["scope"] == "paired_folds"}
        patient = next((row for row in selected if row["scope"] == "patient_oof"), None)
        if not metrics:
            continue
        accuracy = metrics["ACC"]["difference_pp"]
        hard_auc = metrics["AUC_original"]["difference_pp"]
        prob_auc = metrics["AUC_probability"]["difference_pp"]
        correction_harm = (f"{int(patient['mcnemar_corrections'])}/{int(patient['mcnemar_harms'])}"
                           if patient else "n/a")
        p_value = f"{float(patient['mcnemar_exact_p']):.5f}" if patient else "n/a"
        lines.append(f"| {label} | {accuracy:+.4f} | {hard_auc:+.4f} | {prob_auc:+.4f} "
                     f"| {correction_harm} | {p_value} |")
    lines += [
        "",
        "Each Δ column is the mean paired ten-fold difference in percentage points. `paired_contrasts.csv` contains "
        "all ten fold differences and patient OOF counts. Exact McNemar p-values use paired OOF predictions. "
        "The folds reuse overlapping training sets, so these patient comparisons do not establish independent-sample significance.",
        "",
        "## Mechanism",
        "",
    ]
    if mechanism:
        lines += [
            "| Group | Native ACC / AUC_original / probability AUC | Full ACC / AUC_original / probability AUC | Native→Full corrected / damaged | Protection-vs-unprotected corrected / damaged |",
            "|---|---:|---:|---:|---:|",
        ]
        for group in runner.FORMAL_GROUPS:
            item = mechanism["groups"][group]
            native = (f"{item['native_mean_ACC']:.4%} / {item['native_mean_AUC_original']:.4%} / "
                      f"{item['native_mean_AUC_probability']:.4%}")
            full = (f"{item['full_mean_ACC']:.4%} / {item['full_mean_AUC_original']:.4%} / "
                    f"{item['full_mean_AUC_probability']:.4%}")
            native_full = f"{item['native_to_full_corrected']} / {item['native_to_full_damaged']}"
            protection = (f"{item['dpsl_protected_vs_unprotected_corrected']} / "
                          f"{item['dpsl_protected_vs_unprotected_damaged']}")
            lines.append(f"| {group} | {native} | {full} | {native_full} | {protection} |")
        lines += [
            "",
            "The groupwise Native results measure changes caused by joint training. Native→Full counts measure direct DPSL inference changes. "
            "The protection comparison changes only the deterministic inference gate at the same Best checkpoint.",
            "",
            "Per-fold, per-class corrections and harms, six-modality edge/gate/message diagnostics, Direct-Off and Disease-Neutral "
            "interventions, relation gradients, confidence-gate distributions, residual ratios, and class-pair participation are in "
            "`audits/mechanism_summary.json` and `results/patient_mechanism.csv`.",
            "",
            "All six groups retain the N3 squared Frobenius ALoss, lambda 0.010 from epoch 11, with training-fold labels only. "
            "Per-modality graph gradients and M1 structural-zero status are included in the mechanism audit.",
            "",
        ]
    else:
        lines.append("Mechanism audit is not yet available.\n")
    lines += [
        "## Method Interpretation",
        "",
        "DARG gate effects are interpreted from six per-modality graph and signed-edge diagnostics, direct-message norms, "
        "and Direct-Off/Disease-Neutral interventions. Paired group-level differences retain the factorial comparison labels "
        "and are not attributed to an isolated module when another module also differs.",
        "",
        "The data set has already been used across multiple rounds of method selection. Small mean differences are not "
        "treated as stable gains. All groups retain full G4, the fixed TADPOLE protocol, N3 ALoss, the original prompts, "
        "and the N3 DPSL optimization limits.",
        "",
        f"Paper reference reached on both requested primary means: "
        f"**{paper_reached}**.",
        "",
    ]
    return "\n".join(lines), status, next_group


def main():
    runner.require_training_gate(list(runner.FORMAL_GROUPS))
    all_oof, all_folds, summaries = {}, {}, []
    for group in runner.FORMAL_GROUPS:
        oof, folds = load_group_predictions(group)
        all_oof[group], all_folds[group] = oof, folds
        summaries.append(summarize_group(group, oof, folds))
    contrasts = []
    comparisons = [(group, "Y0", f"{group} vs Y0") for group in ("Y1", "Y2", "Y3", "Y4", "Y5")]
    comparisons += [("Y2", "Y1", "Y2 vs Y1"), ("Y4", "Y1", "Y4 vs Y1"),
                    ("Y5", "Y2", "Y5 vs Y2"), ("Y4", "Y3", "Y4 vs Y3"),
                    ("Y5", "Y3", "Y5 vs Y3")]
    for group, reference, label in comparisons:
        fold_contrasts(group, reference, all_folds[group], all_folds[reference], contrasts, label)
        patient_contrast(group, reference, all_oof[group], all_oof[reference], contrasts, label)

    historical_n3, historical_n3_folds = historical_predictions("N3")
    historical_g4, historical_g4_folds = historical_predictions("N0")
    for group in runner.FORMAL_GROUPS:
        fold_contrasts(group, "E83.8.2-N3", all_folds[group], historical_n3_folds, contrasts,
                       f"{group} vs E83.8.2-N3")
        patient_contrast(group, "E83.8.2-N3", all_oof[group], historical_n3, contrasts,
                         f"{group} vs E83.8.2-N3")
        fold_contrasts(group, "E83.8.2-G4", all_folds[group], historical_g4_folds, contrasts,
                       f"{group} vs E83.8.2-G4")
        patient_contrast(group, "E83.8.2-G4", all_oof[group], historical_g4, contrasts,
                         f"{group} vs E83.8.2-G4")
        paper_contrast(group, next(row for row in summaries if row["group"] == group), contrasts)

    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / "group_summary.csv", index=False)
    fold_rows = [row for group in runner.FORMAL_GROUPS for row in all_folds[group]]
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    oof_all = pd.concat([all_oof[group] for group in runner.FORMAL_GROUPS], ignore_index=True)
    oof_all.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "paired_contrasts.csv", index=False)
    mechanism_path = ROOT / "audits/mechanism_summary.json"
    mechanism = __import__("json").loads(mechanism_path.read_text()) if mechanism_path.is_file() else None
    report, status, next_group = build_report(summaries, contrasts, mechanism)
    (output / "final_report.md").write_text(report, encoding="utf-8")
    reference_metrics = {
        "N3": {
            "ACC": float(np.mean([row["ACC"] for row in historical_n3_folds])),
            "AUC_original": float(np.mean([row["AUC"] for row in historical_n3_folds])),
            "AUC_probability": float(np.mean([row["AUC_probability"] for row in historical_n3_folds])),
        },
        "G4": {
            "ACC": float(np.mean([row["ACC"] for row in historical_g4_folds])),
            "AUC_original": float(np.mean([row["AUC"] for row in historical_g4_folds])),
            "AUC_probability": float(np.mean([row["AUC_probability"] for row in historical_g4_folds])),
        },
    }
    groups_both_primary_above_n3 = [
        row["group"] for row in summaries
        if row["group"] != "Y0"
        and row["ACC_mean"] > reference_metrics["N3"]["ACC"]
        and row["AUC_original_mean"] > reference_metrics["N3"]["AUC_original"]
    ]
    result = {
        "experiment": "Experiment 83.8.3", "status": status, "next_group": next_group,
        "group_count": len(summaries), "fold_count": len(fold_rows), "patient_predictions": len(oof_all),
        "highest_ACC_group": max(summaries, key=lambda row: row["ACC_mean"])["group"],
        "highest_AUC_original_group": max(summaries, key=lambda row: row["AUC_original_mean"])["group"],
        "highest_AUC_probability_group": max(summaries, key=lambda row: row["AUC_probability_mean"])["group"],
        "paper_primary_means_reached": any(row["ACC_mean"] >= PAPER["ACC"]
                                            and row["AUC_original_mean"] >= PAPER["AUC_original"]
                                            for row in summaries),
        "groups_both_primary_above_n3": groups_both_primary_above_n3,
        "summaries": summaries,
        "historical_references": {name: {"folds": 10, **values}
                                  for name, values in reference_metrics.items()},
    }
    (ROOT / "audits/analysis_summary.json").write_text(
        __import__("json").dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(__import__("json").dumps({key: result[key] for key in
                                    ("status", "next_group", "highest_ACC_group",
                                     "highest_AUC_original_group", "highest_AUC_probability_group",
                                     "paper_primary_means_reached")}, indent=2))


if __name__ == "__main__":
    main()
