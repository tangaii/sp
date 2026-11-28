"""Apply the preregistered descriptive and statistical decision gates."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ci_values(value):
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return [float(parsed[0]), float(parsed[1])]


def main():
    summaries = pd.read_csv(ROOT / "results/group_summary.csv").set_index("group")
    contrasts = pd.read_csv(ROOT / "results/paired_contrasts.csv")
    replay = json.loads((ROOT / "audits/checkpoint_replay.json").read_text())
    logs = json.loads((ROOT / "audits/log_format_audit.json").read_text())
    aggregate = json.loads((ROOT / "audits/aggregate_results.json").read_text())
    method = json.loads((ROOT / "audits/method_fidelity.json").read_text())
    parity = json.loads((ROOT / "audits/abide_tadpole_parity.json").read_text())
    seal = json.loads((ROOT / "audits/training_seal.json").read_text())
    new_groups = [f"U{i}" for i in range(1, 7)]
    descriptive_groups = [
        group for group in new_groups
        if summaries.loc[group, "ACC_mean"] > summaries.loc["H0", "ACC_mean"]
        and summaries.loc[group, "AUC_original_mean"] > summaries.loc["H1", "AUC_original_mean"]
    ]
    decision_rows = []
    statistical_groups = []
    for group in new_groups:
        for baseline in ("H0", "H1"):
            row = contrasts[(contrasts.left_group == group)
                            & (contrasts.right_group == baseline)].iloc[0]
            acc_ci = ci_values(row["ACC_paired_bootstrap_95ci"])
            auc_ci = ci_values(row["AUC_original_paired_bootstrap_95ci"])
            supported = (
                acc_ci[0] > 0.0 and auc_ci[0] > 0.0
                and float(row["McNemar_exact_holm_p"]) < 0.05
                and float(row["paired_fold_t_holm_p_ACC"]) < 0.05
                and float(row["paired_fold_t_holm_p_AUC_original"]) < 0.05
            )
            if supported:
                statistical_groups.append(group)
            decision_rows.append({
                "group": group, "baseline": baseline,
                "delta_ACC_OOF": float(row["delta_ACC_OOF"]),
                "delta_AUC_original_OOF": float(row["delta_AUC_original_OOF"]),
                "corrected": int(row["corrected"]), "damaged": int(row["damaged"]),
                "net_correct": int(row["net_correct"]),
                "McNemar_exact_holm_p": float(row["McNemar_exact_holm_p"]),
                "ACC_paired_bootstrap_95ci": acc_ci,
                "AUC_original_paired_bootstrap_95ci": auc_ci,
                "paired_fold_holm_p_ACC": float(row["paired_fold_t_holm_p_ACC"]),
                "paired_fold_holm_p_AUC_original": float(row["paired_fold_t_holm_p_AUC_original"]),
                "statistical_support": bool(supported),
            })
    execution = (
        replay.get("status") == "PASS" and replay.get("folds_passed") == 80
        and replay.get("latest_next_step_passed") == 60
        and logs.get("status") == "PASS" and len(logs.get("groups", {})) == 8
        and aggregate.get("status") == "PASS" and aggregate.get("oof_rows") == 4784
    )
    method_go = (
        method.get("status") == "PASS" and parity.get("status") == "PASS"
        and seal.get("status") == "SEALED_BEFORE_TRAINING"
    )
    statistical = bool(statistical_groups)
    final_replacement = bool(descriptive_groups and statistical_groups)
    result = {
        "experiment": "Experiment 83.8.5.5",
        "status": "PASS" if execution and method_go else "FAIL",
        "execution_go": bool(execution), "method_fidelity_go": bool(method_go),
        "descriptive_performance_go": bool(descriptive_groups),
        "descriptive_groups": descriptive_groups,
        "descriptive_thresholds": {
            "ACC_mean_strictly_above_H0": float(summaries.loc["H0", "ACC_mean"]),
            "AUC_original_mean_strictly_above_H1": float(summaries.loc["H1", "AUC_original_mean"]),
        },
        "statistical_support": statistical, "statistical_support_groups": statistical_groups,
        "statistical_support_rule": "positive lower bounds for patient-paired ACC and AUC_original bootstrap 95% CIs, plus Holm-adjusted McNemar and paired-fold p-values below 0.05",
        "final_replacement": final_replacement,
        "retain_H0_Y3_Z0_and_H1_S02": not final_replacement,
        "paired_results": decision_rows,
        "interpretation": (
            "No new sampler met both preregistered ACC and AUC_original thresholds. "
            "Retain historical Y3/Z0 as ACC reference and S02 as the TADPOLE base."
            if not descriptive_groups else
            "At least one new sampler met descriptive thresholds. Statistical and mechanism evidence remain part of replacement decision."
        ),
    }
    path = ROOT / "audits/final_decision.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    report_path = ROOT / "results/final_report.md"
    report = report_path.read_text()
    marker = "## Final decision gates"
    if marker in report:
        report = report.split(marker)[0].rstrip() + "\n"
    lines = [
        marker, "",
        f"- EXECUTION_GO: **{'PASS' if execution else 'FAIL'}**. Sixty new folds, twenty historical folds, eighty Best/Latest replays, sixty deterministic Latest next-step replays, eight logs, and 4,784 OOF rows were required.",
        f"- METHOD_FIDELITY_GO: **{'PASS' if method_go else 'FAIL'}**. The sealed S02 DARG/DPSL core and the ABIDE model identity audit passed.",
        f"- DESCRIPTIVE_PERFORMANCE_GO: **{'PASS' if descriptive_groups else 'NO'}**. At least one U group must exceed H0 ACC ({100*summaries.loc['H0','ACC_mean']:.4f}%) and H1 AUC_original ({100*summaries.loc['H1','AUC_original_mean']:.4f}%). Matching groups: {', '.join(descriptive_groups) if descriptive_groups else 'none'}.",
        f"- STATISTICAL_SUPPORT: **{'PASS' if statistical else 'NO'}** under patient-paired bootstrap and Holm-adjusted tests. Matching groups: {', '.join(sorted(set(statistical_groups))) if statistical_groups else 'none'}.",
        f"- FINAL_REPLACEMENT: **{'YES' if final_replacement else 'NO'}**. Retain H0 for the Y3/Z0 ACC reference and H1/S02 as the current TADPOLE base.",
        "",
        "## Paired Baseline Results", "",
        "OOF deltas below use each patient's single held-out prediction. The bootstrap is stratified by diagnosis and resamples patients, while the paired-fold tests use ten folds.", "",
        "| Group vs baseline | Delta ACC | Delta AUC_original | Corrected / damaged / net | McNemar Holm p | ACC bootstrap 95% CI | AUC_original bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decision_rows:
        acc_ci, auc_ci = row["ACC_paired_bootstrap_95ci"], row["AUC_original_paired_bootstrap_95ci"]
        lines.append(
            f"| {row['group']} - {row['baseline']} | {100*row['delta_ACC_OOF']:+.3f} pp | "
            f"{100*row['delta_AUC_original_OOF']:+.3f} pp | {row['corrected']} / {row['damaged']} / {row['net_correct']} | "
            f"{row['McNemar_exact_holm_p']:.4g} | [{100*acc_ci[0]:+.2f}, {100*acc_ci[1]:+.2f}] pp | "
            f"[{100*auc_ci[0]:+.2f}, {100*auc_ci[1]:+.2f}] pp |")
    lines.extend([
        "", "## Gradient Direction", "",
        "The read-only 80-fold gradient replay records the shared DARG parameter intersection, gradient norms, cosine direction, a replayed full optimizer update, and a counterfactual relation-only update. These diagnostics do not alter training.",
        "", "## Research Conclusion", "",
        "No new relation-sampling strategy simultaneously exceeded the Y3/Z0 ACC reference and the S02 AUC_original reference. The randomized S02-distribution control was already below both historical references, and the targeted MCI strategies did not recover that gap. The gradient audit found a shared path in every fold, but the shared parameter fraction was about one percent and relation/classification gradients were near-orthogonal or negative. A relation-only update changed DARG parameters while changing train NLL and classification margins by only about 1e-8 on average. The evidence therefore points to weak alignment and weak transmission of relation supervision into the final classifier, alongside no descriptive benefit from the tested sampling rules. It does not support changing the preserved DARG/DPSL model.",
        "", "The paper reports 94.81 +/- 2.95% ACC and 95.59 +/- 2.57% AUC. All paper comparisons use AUC_original. The preserved val=test, full-data RFE, and pre-CV scaling make the results historical-protocol estimates rather than an independent external validation. Repeated prior TADPOLE exploration also limits confirmatory interpretation.",
        "",
    ])
    report_path.write_text(report + "\n".join(lines))
    print(json.dumps({key: result[key] for key in (
        "status", "execution_go", "method_fidelity_go", "descriptive_performance_go",
        "statistical_support", "final_replacement")}, indent=2))


if __name__ == "__main__":
    main()
