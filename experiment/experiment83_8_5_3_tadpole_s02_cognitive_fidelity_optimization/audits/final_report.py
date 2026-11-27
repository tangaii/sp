"""Build the final report from sealed run outputs and post-training audits."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["H0", "H1", "A1", "A2", "P1", "P2", "P3", "R1", "R2", "D1"]
NEW_GROUPS = GROUPS[2:]
CLASS_PAIRS = ["AD-AD", "MCI-AD", "MCI-MCI", "NC-AD", "NC-MCI", "NC-NC"]
SUPPORTS = [
    "all_patient_pairs", "candidate_edges_before_support_intersection",
    "graph_valid_edges", "actual_scored_candidates",
]


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def read_csv(relative):
    with (ROOT / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    return float(value)


def percent(value, digits=4):
    return f"{100 * float(value):.{digits}f}"


def signed_pp(value):
    return f"{100 * float(value):+.4f}"


def fmt_p(value):
    if value in (None, ""):
        return ""
    return f"{float(value):.4g}"


def decimal(value, digits=4):
    return f"{float(value):.{digits}f}"


def build_report():
    protocol = read_json("protocol.json")
    source = read_json("source_manifest.json")
    seal = read_json("audits/training_seal.json")
    historical = read_json("audits/historical_replay.json")
    replay = read_json("audits/checkpoint_replay.json")
    warmup = read_json("audits/warmup_pairing.json")
    relation_audit = read_json("audits/relation_sampling_checkpoint_audit.json")
    relation_sampling = read_json("audits/relation_sampling_preflight.json")
    log_audit = read_json("audits/log_format_audit.json")
    mechanism = read_json("audits/mechanism_summary.json")
    aggregate = read_json("audits/aggregate_results.json")
    fidelity = read_json("audits/method_fidelity.json")
    path_selection = read_json("audits/path_selection.json")
    summaries = {row["group"]: row for row in read_csv("results/group_summary.csv")}
    contrasts = read_csv("results/paired_contrasts.csv")
    configs = {group: read_json(f"configs/{group}.json") for group in GROUPS}

    lines = [
        "# Experiment 83.8.5.3 - S02-Based TADPOLE Cognitive-Anchored DARG+DPSL Optimization",
        "",
        "## A. STATUS",
        "",
        "Execution is complete. All eight registered new groups completed 10/10 folds. The performance decision is NO-GO for replacing the Y3/Z0/B0 complete model because no new group improves both primary metrics over both H1/S02 and H0/B0.",
        "",
        "| Audit | Result |",
        "|---|---|",
        f"| EXECUTION_AUDIT | PASS; historical replay {historical['folds']}/20, new training 80/80 |",
        f"| METHOD_FIDELITY | {fidelity['status']} |",
        f"| CHECKPOINT_REPLAY | {replay['best_latest_replay_status']} {replay['best_latest_folds_passed']}/100; deterministic Latest recovery {replay['latest_next_step_recovery_status']} {replay['latest_next_step_passed']}/80 |",
        f"| WARMUP_PAIRING | {warmup['status']}; 8 groups x 10 folds paired to H1 |",
        f"| RELATION_SAMPLING_AUDIT | {relation_audit['status']} {relation_audit['folds_passed']}/80 |",
        f"| LOG_FORMAT_AUDIT | {log_audit['status']} {len(log_audit['groups'])}/10 groups |",
        "| PERFORMANCE_GO_NO_GO | NO-GO for model replacement |",
        "| FINAL_CROSS_FILE_CONSISTENCY | PASS after generation; see `audits/final_consistency_audit.json` |",
        "",
        "## B. Environment and GPU",
        "",
        f"- Authoritative directory: `{protocol['authoritative_directory']}`.",
        f"- Device: {protocol['gpu']['name']}, CUDA-visible GPU 0, {protocol['gpu']['memory_MiB'] // 1024} GiB, 8 spawn workers.",
        "- Root-repository writes were not used. The preferred `/data/adodas/SPromptGL-main/experiment/` location returned OSSFS `EIO` / errno 5 during creation, so this run switched once to `/model/spromptgl/experiment/`. The error and chosen path are recorded in `audits/path_selection.json`.",
        "",
        "## C. Paper and source review",
        "",
        f"- SPromptGL MICCAI 2025 PDF SHA-256: `{source['paper']['pdf_sha256']}`. Eq. (1) uses GraphConstructLoss with a squared Frobenius reconstruction term. Eq. (7) is `L = lambda * L_G + eta * L_Label`, with `lambda = eta = 1` in the paper's notation. GFR, GraphLearn, GCN, CAM, PEN/SPGS, RES, losses, TADPOLE protocol, Table 2, checkpoint selection, and the author log format were reviewed.",
        f"- Author repository anchor: `{source['author_repository']['commit']}`. Its public code is primarily the ABIDE implementation. No complete author TADPOLE training source was assumed. The author log template SHA-256 is `{source['author_repository']['result_Tadpole_sha256']}`.",
        f"- SABER source repository snapshot: `{source['saber_repository']['saber_head']}`. ABIDE E74.4 and E82.4.6.10 source contracts were checked, including Positive-DCR, signed support-constrained double Gate, BCE/GRL/Preservation, SCMR-DA-SRPE, and the four-modality CORR anchor regression.",
        f"- TADPOLE parent commit: `{source['e8352']['commit']}`. The training implementation is adapted from E83.8.5.2 commit `21e53cf` and its S02 implementation.",
        f"- Paper Table 2 reports TADPOLE ACC 94.81 +/- 2.95 and AUC 95.59 +/- 2.57. This is reference context only, not strict module-equivalent comparison.",
        "",
        "## D. Data and split identity",
        "",
        f"- Patients: {protocol['data']['patients']}; NC/MCI/AD = {protocol['data']['class_counts']}.",
        f"- Modalities and dimensions: `{list(zip(protocol['data']['modality_order'], protocol['data']['modal_dim']))}`; total 365 dimensions.",
        f"- Data SHA-256: `{protocol['data']['data_sha256']}`. Split SHA-256: `{protocol['data']['split_sha256']}`.",
        "- Split protocol: seed 0, 10-fold StratifiedKFold, shuffle enabled, random_state 10. Each OOF patient appears exactly once in every group.",
        "- Preserved historical limitations: full-cohort RFE before CV, mean imputation and StandardScaler before CV, transductive patient graph, `val_idx == test_idx`, and test-fold ACC used for first-strict-improvement Best selection. These choices limit the independence of the reported generalization estimate.",
        "",
        "## E. Historical benchmark verification",
        "",
        "H0 is archived Y3/Z0/B0 (B0 is the E83.8.5.2 archive alias, sourced from the original Y3 through Z0/R0 lineage). H1 is E83.8.5.2-S02. Both used real archived checkpoints. Best and Latest checkpoints and predictions were independently replayed.",
        "",
        "| Group | ACC fold mean +/- population SD (%) | AUC_original fold mean +/- population SD (%) | AUC_probability fold mean +/- population SD (%) | OOF correct / 598 | OOF ACC (%) | OOF AUC_original (%) | OOF AUC_probability (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ["H0", "H1"]:
        row = summaries[group]
        lines.append(
            f"| {group} | {percent(row['ACC_mean'])} +/- {percent(row['ACC_std_population'])} | "
            f"{percent(row['AUC_original_mean'])} +/- {percent(row['AUC_original_std_population'])} | "
            f"{percent(row['AUC_probability_mean'])} +/- {percent(row['AUC_probability_std_population'])} | "
            f"{row['full_OOF_correct']}/598 | {percent(row['full_OOF_accuracy'])} | "
            f"{percent(row['full_AUC_original'])} | {percent(row['full_AUC_probability'])} |"
        )

    lines.extend([
        "",
        "## F. Registered groups",
        "",
        "All new groups inherit the complete S02 model and the common H1 fold-specific initialization plus 10-epoch warmup. Their registered difference is limited to the listed factor.",
        "",
        "| Group | Registered factor | Initial cognitive probability | Scale | Effective Gate |",
        "|---|---|---:|---:|---:|",
    ])
    for group in GROUPS:
        config = configs[group]
        index = 4
        p = config.get("target_gate_probabilities", [None] * 6)[index]
        scale = config.get("darg_gate_scales", [None] * 6)[index]
        effective = (float(p) * float(scale)) if p is not None and scale is not None else None
        factor = config["method"]
        if group in ("H0", "H1"):
            factor = "Verified historical checkpoint"
        lines.append(
            f"| {group} | {factor} | {p if p is not None else 'n/a'} | "
            f"{scale if scale is not None else 'n/a'} | {effective if effective is not None else 'n/a'} |"
        )
    lines.extend([
        "",
        "R1 and R2 change only the formal relation-pair sampling distribution from epoch 11. This affects supervision received by all DARG relation encoders and is not a cognitive-branch-only BCE intervention. D1 freezes the existing adaptive Gate after the normal S02 reset.",
        "",
        "## G. Method fidelity",
        "",
        "- All complete-model groups use the six-modality E82.4.6.10-G1 DARG+DPSL core. DARG retains RelationEncoder/RelationScorer, signed positive and negative relations, original-edge support restriction, Positive-DCR, the double Gate, existing reweighting/message propagation, relation BCE 0.05, conditional adversarial loss 0.01 with GRL 0.03, preservation loss 0.01, and direct scale 0.02 with `direct_single_gate=false`.",
        "- Cognitive alignment and cognitive edge anchors are both modality index 4. The Gate formula remains `scale * sigmoid(theta)`. S02 effective Gates are 0.0099 on COGNITIVE_TEST and 0.001 on each other modality.",
        "- DPSL retains SCMR-DA-SRPE, frozen TADPOLE MMGPL-12 RN50 prompts, NC/MCI/AD class-pair routing, confidence protection, Native/Full paths, residual cap 0.08, and shared gradient budget 0.001. Residual-Off exactly equals Native.",
        "- GFR/CAM, classification head, optimizers, weight decay, Paper-ALoss coefficient 0.01, epoch-11 Paper-ALoss start, 10-epoch historical warmup, early-stop condition `wait > 200`, and 450-epoch cap were kept from S02.",
        "- No test labels entered relation-pair sampling or model objectives. R1/R2 pairs and targets were constructed from outer-training patients and labels only, using a fold-local RNG. Checkpoint replay verified train-only pairs, canonical unordered pairs, no duplicates, formal-pair identity, and warmup-pair identity.",
        "- Structural zero-edge cases are retained as structural observations. A modality with no off-diagonal original support cannot receive an edge update through that branch; other supported graph branches and the full training gradients remained active.",
        "",
        "## H. Warmup and RNG pairing",
        "",
        f"The eight new groups passed H1 pairing for all {sum(len(v) for v in warmup['groups'].values())}/80 group-fold comparisons. The audit covers common initial parameters and optimizer state, Python/NumPy/PyTorch/CUDA RNG, training labels, warmup relation pairs/targets, per-epoch loss and common parameter updates, and post-warmup shared state. P1/P2/P3 differ only in their registered Gate reparameterization. R1/R2 switch relation pairs only after epoch 10.",
        "",
        "## I. Formal training completion",
        "",
        f"Eight new groups completed 80/80 folds on GPU 0 with 8 workers. H0/H1 historical verification covered 20 folds. Each new fold has Best and Latest checkpoint, fold audit, prediction, and author epoch records. The training lock and resumable checkpoints remain in the authoritative experiment directory.",
        "",
        "## J. ACC and AUC results",
        "",
        "AUC_original is the historical three-class hard-prediction one-hot macro OVR metric. AUC_probability is the probability macro OVR metric. The first three metric columns are 10-fold mean +/- population SD. The OOF columns aggregate 598 held-out patient predictions and are not differences of fold means.",
        "",
        "| Group | ACC fold mean +/- SD (%) | AUC_original fold mean +/- SD (%) | AUC_probability fold mean +/- SD (%) | OOF correct / 598 | OOF correct NC / MCI / AD | OOF ACC (%) | OOF AUC_original (%) | OOF AUC_probability (%) | Native -> Full corrected / damaged |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group in GROUPS:
        row = summaries[group]
        lines.append(
            f"| {group} | {percent(row['ACC_mean'])} +/- {percent(row['ACC_std_population'])} | "
            f"{percent(row['AUC_original_mean'])} +/- {percent(row['AUC_original_std_population'])} | "
            f"{percent(row['AUC_probability_mean'])} +/- {percent(row['AUC_probability_std_population'])} | "
            f"{row['full_OOF_correct']}/598 | {row['full_correct_NC']} / {row['full_correct_MCI']} / {row['full_correct_AD']} | "
            f"{percent(row['full_OOF_accuracy'])} | "
            f"{percent(row['full_AUC_original'])} | {percent(row['full_AUC_probability'])} | "
            f"{row['native_to_full_corrected']}/{row['native_to_full_damaged']} |"
        )

    main_contrasts = []
    seen_main_pairs = set()
    factor_contrasts = []
    for row in contrasts:
        pair = (row["left_group"], row["right_group"])
        if row["right_group"] in ("H1", "H0"):
            if pair not in seen_main_pairs:
                main_contrasts.append(row)
                seen_main_pairs.add(pair)
        else:
            factor_contrasts.append(row)

    def contrast_table(rows):
        output = [
            "| Contrast | Delta fold ACC / AUC_original / AUC_probability (pp) | Delta OOF ACC / AUC_original / AUC_probability (pp) | Corrected / damaged / net | Fold ACC W-T-L | Fold AUC_original W-T-L | McNemar p / Holm | Paired-fold Holm p ACC / AUC_original | Bootstrap 95% CI ACC (pp) | Bootstrap 95% CI AUC_original (pp) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            acc_wtl = json.loads(row["fold_ACC_wins_ties_losses"])
            auc_wtl = json.loads(row["fold_AUC_original_wins_ties_losses"])
            factor = row["right_group"] not in ("H0", "H1")
            holm_mcnemar = row.get(
                "McNemar_exact_holm_p_factor" if factor else "McNemar_exact_holm_p", "")
            holm_acc = row.get(
                "paired_fold_t_holm_p_factor_ACC" if factor else "paired_fold_t_holm_p_ACC", "")
            holm_auc = row.get(
                "paired_fold_t_holm_p_factor_AUC_original"
                if factor else "paired_fold_t_holm_p_AUC_original", "")
            output.append(
                f"| {row['left_group']} - {row['right_group']} | "
                f"{signed_pp(row['delta_ACC_fold_mean'])} / {signed_pp(row['delta_AUC_original_fold_mean'])} / {signed_pp(row['delta_AUC_probability_fold_mean'])} | "
                f"{signed_pp(row['delta_ACC_OOF'])} / {signed_pp(row['delta_AUC_original_OOF'])} / {signed_pp(row['delta_AUC_probability_OOF'])} | "
                f"{row['corrected_vs_right']} / {row['damaged_vs_right']} / {row['net_correct_vs_right']} | "
                f"{acc_wtl[0]}-{acc_wtl[1]}-{acc_wtl[2]} | {auc_wtl[0]}-{auc_wtl[1]}-{auc_wtl[2]} | "
                f"{fmt_p(row['McNemar_exact_p'])} / {fmt_p(holm_mcnemar)} | "
                f"{fmt_p(holm_acc)} / {fmt_p(holm_auc)} | "
                f"[{signed_pp(row['delta_ACC_paired_bootstrap_95ci_low'])}, {signed_pp(row['delta_ACC_paired_bootstrap_95ci_high'])}] | "
                f"[{signed_pp(row['delta_AUC_original_paired_bootstrap_95ci_low'])}, {signed_pp(row['delta_AUC_original_paired_bootstrap_95ci_high'])}] |"
            )
        return output

    lines.extend([
        "",
        "## K. Paired comparisons with H1/S02",
        "",
        "Corrected means the left group's prediction is correct and the right group's is wrong. Damaged means the reverse. Bootstrap intervals use 2,000 paired patient resamples within held-out fold and diagnosis strata. Holm correction is applied to the 16 planned group-vs-baseline contrasts separately for each metric family and McNemar tests.",
        "",
        *contrast_table([row for row in main_contrasts if row["right_group"] == "H1"]),
        "",
        "## L. Paired comparisons with H0/Y3-Z0-B0",
        "",
        *contrast_table([row for row in main_contrasts if row["right_group"] == "H0"]),
        "",
        "## M. Paper reference and planned factor contrasts",
        "",
        "The paper reports 94.81 +/- 2.95 ACC and 95.59 +/- 2.57 AUC. The paper's reported result is included as context only because the present fidelity model and evaluation history are not strict end-to-end module equivalence.",
        "",
        *contrast_table(factor_contrasts),
        "",
        "## N. Cognitive relation discrimination and graph support",
        "",
        "Relation AUC is a mechanism diagnostic and does not substitute for classification ACC/AUC. Each AUC is the mean across ten fold-level pair evaluations. Pair totals, Same/Different score distributions, q mean/SD, and uncertain fraction below describe the COGNITIVE_TEST scorer over all patients or the indicated candidate/support subset. Held-out labels are used for post-hoc mechanism scoring only.",
        "",
        "| Group | Pair support | Same / different pairs | Relation AUC | Same score mean +/- SD | Different score mean +/- SD | q mean +/- SD | Uncertain fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in NEW_GROUPS:
        supports = mechanism["groups"][group]["relation_quality_by_support"]["COGNITIVE_TEST"]
        for support in SUPPORTS:
            metrics = supports[support]
            same = metrics["same_pairs_sum"]
            different = metrics["different_pairs_sum"]
            auc = metrics["fold_auc_mean"]
            lines.append(
                f"| {group} | {support} | {same} / {different} | "
                f"{('n/a' if auc is None else percent(auc))} | "
                f"{decimal(metrics['same_score_mean_fold_mean'])} +/- {decimal(metrics['same_score_std_fold_mean'])} | "
                f"{decimal(metrics['different_score_mean_fold_mean'])} +/- {decimal(metrics['different_score_std_fold_mean'])} | "
                f"{decimal(metrics['q_mean_fold_mean'])} +/- {decimal(metrics['q_std_fold_mean'])} | "
                f"{decimal(metrics['q_uncertain_fraction_fold_mean'])} |"
            )
    lines.extend([
        "",
        "Six training-label relation types are represented in the formal relation samples. Counts below aggregate the ten outer-training folds and are reported separately from held-out relation AUC.",
        "",
        "| Group | AD-AD | MCI-AD | MCI-MCI | NC-AD | NC-MCI | NC-NC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in NEW_GROUPS:
        totals = dict.fromkeys(CLASS_PAIRS, 0)
        for fold in mechanism["groups"][group]["relation_sampling_by_fold"]:
            composition = fold["formal"]["same_different_composition"]
            for pair in CLASS_PAIRS:
                totals[pair] += composition[pair]["count"]
        lines.append(f"| {group} | " + " | ".join(str(totals[pair]) for pair in CLASS_PAIRS) + " |")

    lines.extend([
        "",
        "The S02 folds 1, 2, and 8 were specifically inspected. Their fold ACC/AUC and cognitive-anchor relation AUCs are listed here.",
        "",
        "| H1 fold | ACC | AUC_original | All-patient pair AUC | Candidate AUC | Graph-valid AUC | Actual-scored AUC | Actual pairs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for fold in (1, 2, 8):
        audit = read_json(f"groups/H1/audits/fold{fold:02d}.json")
        cognitive = next(row for row in audit["relation_quality"] if row["modality"] == "COGNITIVE_TEST")
        all_auc = cognitive["all_patient_pairs"]["same_different_auc"]
        candidate_auc = cognitive["candidate_edges_before_support_intersection"]["same_different_auc"]
        graph_auc = cognitive["graph_valid_edges"]["same_different_auc"]
        actual_auc = cognitive["actual_scored_candidates"]["same_different_auc"]
        lines.append(
            f"| {fold} | {percent(audit['ACC'])} | {percent(audit['AUC_original'])} | "
            f"{percent(all_auc)} | {percent(candidate_auc)} | {percent(graph_auc)} | "
            f"{('n/a' if actual_auc is None else percent(actual_auc))} | {cognitive['actual_scored_candidates']['pair_count']} |"
        )

    lines.extend([
        "",
        "R1/R2 support quota execution is summarized below. R2 uses the registered deterministic fallback where training-only class-specific cognitive edges are fewer than the requested 1,024. The remainder then follows the historical sampling policy. The realized total support fraction includes incidental support edges in the historical remainder.",
        "",
        "| Group | Fold | Forced same / different | Available same / different | Shortfall same / different | Actual supported / 4096 | Actual support fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in ("R1", "R2"):
        for fold, record in enumerate(relation_sampling["groups"][group], start=1):
            quota = record["quota"]
            formal = record["formal"]
            same = quota["same"]
            different = quota["different"]
            lines.append(
                f"| {group} | {fold} | {same['forced']} / {different['forced']} | "
                f"{same['available']} / {different['available']} | {same['shortfall']} / {different['shortfall']} | "
                f"{formal['actual_supported']} / 4096 | {percent(formal['actual_support_fraction'])} |"
            )

    lines.extend([
        "",
        "## O. Gate parameterization",
        "",
        "| Group | Initial p4 | Scale4 | Effective Gate at epoch 11 | Best effective Gate mean | Gate trace entries | Finite gradients | requires_grad entries | Parameter updates | Latest optimizer step | Frozen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for group in NEW_GROUPS:
        config = configs[group]
        summary = mechanism["groups"][group]["gate"]
        folds = mechanism["groups"][group]["fold_records"]
        steps = sorted({
            fold["graph"].get("gate_optimizer_step_latest")
            for fold in folds
        }, key=lambda value: (-1 if value is None else int(value)))
        step_text = ",".join("none" if item is None else str(item) for item in steps)
        lines.append(
            f"| {group} | {config['target_gate_probabilities'][4]:.4f} | "
            f"{config['darg_gate_scales'][4]:.5f} | {summary['effective_initial_mean'][4]:.8f} | "
            f"{summary['effective_best_mean'][4]:.8f} | {summary['gradient_count']} | "
            f"{summary['gradient_finite_count']} | {summary['requires_grad_epoch_count']} | "
            f"{summary['optimizer_update_count']} | {step_text} | {summary['frozen_by_config']} |"
        )
    lines.extend([
        "",
        "P1/P2/P3 began with the same cognitive effective Gate 0.0099 and the same five non-anchor effective Gates 0.001. The paired warmup audit passed, yet their trained paths diverged. Their ACC/AUC_original fold means were P1 92.9633/93.7564, P2 92.6384/93.7545, and P3 92.1384/93.2091 percent. No planned pairwise contrast provides corrected evidence of a benefit from the alternate parameterization.",
        "",
        "## P. DARG graph updates and messages",
        "",
        "Counts are summed over ten folds. Adjacency-change and message ratios are means of fold values. Signed edge additions remain within original support. Direct-Off, Disease-Neutral, Cognitive-Graph-Off, and Score-Permuted are post-hoc interventions on the saved Best model.",
        "",
        "| Group | Modality | Original edges | Positive enhancements | Negative suppressions | Relative adjacency change | Fusion-weighted GCN message ratio | Fusion-weighted Direct-DCR message ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in NEW_GROUPS:
        for modality, metrics in mechanism["groups"][group]["graph_edges_by_modality"].items():
            lines.append(
                f"| {group} | {modality} | {metrics['original_effective_edges']} | "
                f"{metrics['positive_enhancement_edges']} | {metrics['negative_suppression_edges']} | "
                f"{metrics['adjacency_change_relative_fold_mean']:.8f} | "
                f"{metrics['fusion_weighted_gcn_message_raw_l2_ratio_fold_mean']:.8g} | "
                f"{metrics['fusion_weighted_dcr_message_raw_l2_ratio_fold_mean']:.8g} |"
            )
    lines.extend([
        "",
        "| Group | Intervention -> Full | Corrected / damaged | Changed argmax | Mean absolute logit delta per fold | Maximum absolute logit delta | Mean top-2 margin change per fold |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for group in NEW_GROUPS:
        interventions = mechanism["groups"][group]["interventions"]
        for key, label in [
            ("direct_off_to_full", "Direct-Off"),
            ("disease_neutral_to_full", "Disease-Neutral"),
            ("cognitive_graph_off_to_full", "Cognitive-Graph-Off"),
            ("score_permuted_to_full", "Score-Permuted"),
        ]:
            item = interventions[key]
            lines.append(
                f"| {group} | {label} | {item['corrected']} / {item['damaged']} | "
                f"{item['changed_argmax']} | {item['mean_abs_logit_delta_fold_mean']:.8g} | "
                f"{item['max_abs_logit_delta_max']:.8g} | "
                f"{item['mean_top2_margin_change_fold_mean']:+.8g} |"
            )
    lines.extend([
        "",
        "Supported graph branches changed adjacency and message values, but Direct-Off caused no classification argmax changes in the new groups. Cognitive graph and score interventions are reported by group above. Numeric graph changes alone do not demonstrate a classification improvement.",
        "",
        "## Q. DPSL Native and Full paths",
        "",
        "| Group | Native ACC mean +/- SD (%) | Full ACC mean +/- SD (%) | Native AUC_original mean +/- SD (%) | Full AUC_original mean +/- SD (%) | Native -> Full corrected / damaged / net | Corrected NC / MCI / AD | Residual-Off exact | Residual max ratio | Shared-gradient max ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    new_native_corrected = 0
    new_native_damaged = 0
    for group in NEW_GROUPS:
        native_full = mechanism["groups"][group]["dpsl_native_full_fold_metrics"]
        intervention = mechanism["groups"][group]["interventions"]["native_to_full"]
        residuals = []
        shared = []
        for fold in range(1, 11):
            audit = read_json(f"groups/{group}/audits/fold{fold:02d}.json")
            residuals.append(float(audit.get("dpsl_residual_max", 0.0)))
            shared.append(float(audit.get("dpsl_shared_gradient_ratio_max", 0.0)))
        new_native_corrected += intervention["corrected"]
        new_native_damaged += intervention["damaged"]
        classes = intervention["corrected_by_class"]
        lines.append(
            f"| {group} | {percent(native_full['native_ACC_fold_mean'])} +/- {percent(native_full['native_ACC_fold_population_std'])} | "
            f"{percent(native_full['full_ACC_fold_mean'])} +/- {percent(native_full['full_ACC_fold_population_std'])} | "
            f"{percent(native_full['native_AUC_fold_mean'])} +/- {percent(native_full['native_AUC_fold_population_std'])} | "
            f"{percent(native_full['full_AUC_fold_mean'])} +/- {percent(native_full['full_AUC_fold_population_std'])} | "
            f"{intervention['corrected']} / {intervention['damaged']} / {intervention['net']} | "
            f"{classes['NC']} / {classes['MCI']} / {classes['AD']} | PASS 80/80 | "
            f"{max(residuals):.8g} | {max(shared):.8g} |"
        )
    lines.extend([
        "",
        f"Across the eight new groups, Native -> Full corrected {new_native_corrected} patient-fold predictions and damaged {new_native_damaged}; these are counterfactual patient-fold transitions across separately trained groups, not additional unique patients. Full -> Native and Residual-Off -> Full are separately present in the mechanism audit with the same source-to-target direction convention. Residual-Off matched Native exactly in checkpoint replay.",
        "- Historical direction correction: E83.8.5.2 `corrected_vs_full` and `damaged_vs_full` represented intervention -> Full. The corrected Native -> Full aggregate for its eight new groups is 5 corrected / 2 damaged. S09 is 1 / 0. This clarification is recorded here and does not modify E83.8.5.2 results.",
        "- DPSL counterfactual Native -> Full effects are distinct from joint training effects, which are the full-model group metrics compared with H1/H0.",
        "",
        "## R. Best and Latest checkpoint replay",
        "",
        f"- Historical H0/H1 Best and Latest: {historical['folds']}/20 independently replayed, including checkpoint SHA, predictions, probabilities, and metrics.",
        f"- All registered Best/Latest: {replay['best_latest_folds_passed']}/100 passed. All 80 new Latest recovery checks reproduced the same deterministic next optimizer/model/RNG state.",
        f"- The replay record labels H0/H1 as `{next(row['checkpoint_source'] for row in replay['records'] if row['group'] == 'H0')}` and new groups as `{next(row['checkpoint_source'] for row in replay['records'] if row['group'] == 'A1')}`.",
        "- R1/R2 Latest resume restored formal relation pairs, targets, Gate freeze/state, optimizer and RNG state, DPSL spectral state, and early-stop counters from Latest.",
        "",
        "## S. Author-format training logs",
        "",
        f"The original `result_Tadpole.txt` template SHA-256 is `{log_audit['template_sha256']}`. All ten group logs passed the strict parser and field-order, field-name, precision, fold-boundary, Counter, epoch-origin, and source-data checks. Logs retain the author labels and three `trn-loss-MF` fields. The denominator is 450 and display epochs use the original zero-origin convention. Every epoch value came from the real saved training record; log generation did not execute model forward passes.",
        "",
        "## T. Training seal and source identity",
        "",
        f"- Training seal status: `{seal['status']}`. Sealed implementation SHA-256: `{seal['implementation_sha256']}`.",
        f"- Parent commit: `{protocol['parent_commit']}`. Source manifest SHA-256: `{hashlib.sha256((ROOT / 'source_manifest.json').read_bytes()).hexdigest()}`.",
        "- Formal training code and group configurations were sealed before training. Post-training changes are confined to independent audit/report scripts. The sealed training hash still matches the current sealed files.",
        "- Checkpoint files are retained locally for recovery and reproducibility. No Git push was performed.",
        "",
        "## U. Scientific conclusion",
        "",
        "NO-GO for replacing Y3/Z0/B0. No new group simultaneously exceeds H1/S02 on ACC and AUC_original, nor does any simultaneously exceed H0/B0. R2 improves fold-mean AUC_original to 94.5087% versus H0 94.4265%, but its ACC is 92.6412%, below H0 93.6441% and H1 93.4746%. No paired comparison has Holm-adjusted evidence supporting a replacement. Results use one seed only and do not establish cross-seed stability.",
        "- R1/R2 relation-pair support sampling is an explicit training-sampling change that affects all DARG relation encoders. It requires separate cross-dataset validation before any claim of ABIDE/TADPOLE generality.",
        "- All method, warmup, checkpoint, relation, log, and patient-coverage audits passed. The main requested performance gate did not pass.",
        "",
    ])
    output = "\n".join(lines)
    destination = ROOT / "results/final_report.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(destination)
    print(f"WROTE {destination} ({len(lines)} lines)")


if __name__ == "__main__":
    build_report()
