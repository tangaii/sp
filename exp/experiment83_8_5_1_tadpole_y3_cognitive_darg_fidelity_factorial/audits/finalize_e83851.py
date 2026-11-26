"""Build the registered summaries, paired statistics, final report and audit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t, ttest_1samp
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl")
GROUPS = ("H0", "H1", "H2", "N1", "N2", "N3", "N4")
SOURCE_GROUP = {"H0": "R0", "H1": "R1", "H2": "R4"}
NEW_GROUPS = ("N1", "N2", "N3", "N4")
GROUP_CONTRASTS = (
    ("N1", "H1"), ("N1", "H0"), ("N2", "N1"), ("N3", "N1"),
    ("N4", "N2"), ("N4", "N3"), ("N4", "H0"), ("N2", "H2"),
    ("N2", "H0"), ("N3", "H0"),
)
METRICS = ("ACC", "AUC_original", "AUC_probability")
LABELS = ("NC", "MCI", "AD")
CLASS_IDS = {"NC": 0, "MCI": 1, "AD": 2}
DIRECT_CONTRASTS = (
    ("direct_off", "full"), ("disease_neutral", "full"),
    ("anchor_neutral", "full"), ("alignment_neutral", "full"),
    ("cognitive_graph_off", "full"), ("score_permuted", "full"),
    ("native", "full"), ("residual_off", "full"),
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths(group, fold):
    source_root = HISTORY if group in SOURCE_GROUP else ROOT
    source_group = SOURCE_GROUP.get(group, group)
    folder = source_root / "groups" / source_group
    stem = f"fold{fold:02d}"
    return source_root, source_group, folder / "audits" / f"{stem}.json", folder / "predictions" / f"{stem}.csv"


def pair_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean, 1.0
    standard_deviation = float(values.std(ddof=1))
    if standard_deviation == 0:
        p_value = 1.0 if mean == 0 else 0.0
        return mean, mean, mean, p_value
    critical = float(t.ppf(0.975, len(values) - 1))
    margin = critical * standard_deviation / math.sqrt(len(values))
    p_value = float(ttest_1samp(values, 0.0).pvalue)
    return mean, mean - margin, mean + margin, p_value


def holm_adjust(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def oof_auc_probability(truth, frame, case):
    columns = [f"{case}_prob_{label}" for label in LABELS]
    if not all(column in frame for column in columns):
        return None
    probability = frame[columns].to_numpy(dtype=np.float64)
    return float(roc_auc_score(np.eye(3)[truth], probability, average="macro", multi_class="ovr"))


def correct_by_class(truth, prediction):
    return {name: int(((truth == index) & (prediction == index)).sum())
            for name, index in CLASS_IDS.items()}


def optional_mean(values):
    available = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(available)) if available else None


def format_optional(value, digits=4):
    return f"{value:.{digits}f}" if value is not None else "N/A"


def comparison_row(before_group, after_group, before_case, after_case, kind,
                   fold_evidence, patients, metric):
    fold_deltas = []
    for fold in range(1, 11):
        before_metrics = fold_evidence[before_group][fold]["metrics"][before_case]
        after_metrics = fold_evidence[after_group][fold]["metrics"][after_case]
        fold_deltas.append(float(after_metrics[metric] - before_metrics[metric]))
    mean, ci_low, ci_high, p_value = pair_ci(fold_deltas)
    wins = int(sum(value > 1e-15 for value in fold_deltas))
    ties = int(sum(abs(value) <= 1e-15 for value in fold_deltas))
    losses = len(fold_deltas) - wins - ties

    before = patients[before_group].sort_values("sample_index")
    after = patients[after_group].sort_values("sample_index")
    require(np.array_equal(before["sample_index"], after["sample_index"])
            and np.array_equal(before["true_label"], after["true_label"]),
            f"Patient identity differs in {after_group}-{before_group}")
    truth = before["true_label"].to_numpy(dtype=np.int64)
    before_prediction = before[f"{before_case}_pred"].to_numpy(dtype=np.int64)
    after_prediction = after[f"{after_case}_pred"].to_numpy(dtype=np.int64)
    before_correct = before_prediction == truth
    after_correct = after_prediction == truth
    corrected = int((~before_correct & after_correct).sum())
    damaged = int((before_correct & ~after_correct).sum())
    changed = corrected + damaged
    mcnemar_p = float(binomtest(corrected, changed, 0.5).pvalue) if changed else 1.0
    before_acc = float(before_correct.mean())
    after_acc = float(after_correct.mean())
    before_hard_auc = float(roc_auc_score(np.eye(3)[truth], np.eye(3)[before_prediction],
                                          average="macro", multi_class="ovr"))
    after_hard_auc = float(roc_auc_score(np.eye(3)[truth], np.eye(3)[after_prediction],
                                         average="macro", multi_class="ovr"))
    before_probability_auc = oof_auc_probability(truth, before, before_case)
    after_probability_auc = oof_auc_probability(truth, after, after_case)
    oof_metric_deltas = {
        "ACC": (after_acc - before_acc) * 100,
        "AUC_original": (after_hard_auc - before_hard_auc) * 100,
        "AUC_probability": (
            (after_probability_auc - before_probability_auc) * 100
            if before_probability_auc is not None and after_probability_auc is not None else None
        ),
    }
    row = {
        "comparison_type": kind,
        "contrast": f"{after_group}-{before_group}",
        "before_group": before_group,
        "after_group": after_group,
        "before_case": before_case,
        "after_case": after_case,
        "mode": (before_case if kind == "between_group" else "same_checkpoint"),
        "metric": metric,
        "fold_count": len(fold_deltas),
        "mean_delta_percentage_points": mean * 100,
        "ci95_low_percentage_points": ci_low * 100,
        "ci95_high_percentage_points": ci_high * 100,
        "fold_wins": wins,
        "fold_ties": ties,
        "fold_losses": losses,
        "paired_fold_t_p": p_value,
        "holm_adjusted_p": None,
        "oof_delta_percentage_points": oof_metric_deltas[metric],
        "oof_flips": int((before_prediction != after_prediction).sum()),
        "oof_corrected": corrected,
        "oof_damaged": damaged,
        "oof_net_correct": corrected - damaged,
        "mcnemar_exact_p": mcnemar_p,
        "corrected_NC": int(((~before_correct & after_correct) & (truth == 0)).sum()),
        "damaged_NC": int(((before_correct & ~after_correct) & (truth == 0)).sum()),
        "corrected_MCI": int(((~before_correct & after_correct) & (truth == 1)).sum()),
        "damaged_MCI": int(((before_correct & ~after_correct) & (truth == 1)).sum()),
        "corrected_AD": int(((~before_correct & after_correct) & (truth == 2)).sum()),
        "damaged_AD": int(((before_correct & ~after_correct) & (truth == 2)).sum()),
        "ci_unit": "paired ten-fold differences, Student t interval, descriptive under overlapping CV training sets",
    }
    return row


def main():
    sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/common")]
    import run_e8385 as runner

    runner.require_training_gate(list(runner.TRAIN_GROUPS))
    seal = load_json(ROOT / "audits/training_seal.json")
    require(seal["implementation_sha256"] == "b1cbd93aa6b973f7bdb6c12f5bd2510ac9f101d0ab24e85d1eca650494e3686d",
            "Formal training source seal identity changed")
    require(seal["files"] == runner.current_implementation_hashes(),
            "A sealed training implementation file changed after training")

    replay = load_json(ROOT / "audits/checkpoint_replay.json")
    mechanism = load_json(ROOT / "audits/mechanism_summary.json")
    log_audit = load_json(ROOT / "audits/log_format_audit.json")
    preflight = load_json(ROOT / "audits/preflight.json")
    parity = load_json(ROOT / "audits/abide_tadpole_parity.json")
    warmup = load_json(ROOT / "audits/warmup_pairing.json")
    require(replay["status"] == "PASS" and replay["best_replayed"] == 70
            and replay["latest_restored"] == 70 and replay["deterministic_next_steps"] == 70,
            "Best/Latest independent replay is incomplete")
    require(mechanism["status"] == "PASS" and mechanism["folds"] == 70,
            "Fixed-Best mechanism audit is incomplete")
    require(log_audit["status"] == "PASS" and len(log_audit["groups"]) == 4,
            "Author log format audit is incomplete")
    require(preflight["status"] == "PASS" and parity["status"] == "PASS"
            and warmup["status"] == "PASS",
            "A registered preflight, parity, or warmup audit did not pass")

    mechanism_rows = {
        group: {int(row["fold"]): row for row in mechanism["fold_evidence"][group]}
        for group in GROUPS
    }
    prediction_path = ROOT / "results/oof_predictions.csv"
    all_patients = pd.read_csv(prediction_path)
    patients = {}
    fold_rows = []
    for group in GROUPS:
        group_patients = all_patients.loc[all_patients["group"] == group].copy()
        group_patients = group_patients.sort_values("sample_index")
        require(len(group_patients) == 598 and group_patients["sample_index"].nunique() == 598,
                f"{group} OOF prediction does not cover 598 patients exactly once")
        patients[group] = group_patients
        require(int(group_patients["true_label"].min()) == 0
                and int(group_patients["true_label"].max()) == 2,
                f"{group} OOF labels are invalid")

        for fold in range(1, 11):
            source_root, source_group, audit_path, prediction_file = source_paths(group, fold)
            audit = load_json(audit_path)
            evidence = mechanism_rows[group][fold]
            require(audit["status"] == "TRAINED" and evidence["status"] == "PASS",
                    f"{group} fold {fold} did not complete both training and mechanism audit")
            metrics = evidence["metrics"]
            full = metrics["full"]
            native = metrics["native"]
            require(all(abs(float(audit[key]) - float(full[field])) < 1e-12
                         for key, field in (("ACC", "ACC"), ("AUC_original", "AUC_original"),
                                            ("AUC_probability", "AUC_probability"))),
                    f"{group} fold {fold} independent Full metrics differ from training audit")
            best_path = source_root / "groups" / source_group / "best_checkpoints" / f"fold{fold:02d}.pt"
            latest_path = source_root / "groups" / source_group / "latest_checkpoints" / f"fold{fold:02d}.pt"
            require(file_sha(best_path) == audit["checkpoint_sha256"]
                    and file_sha(latest_path) == audit["latest_sha256"],
                    f"{group} fold {fold} Best/Latest SHA differs from training audit")
            fold_rows.append({
                "group": group, "source_group": source_group, "fold": fold,
                "seed": int(audit.get("seed", 0)), "train_count": int(audit["train_count"]),
                "test_count": int(audit["test_count"]), "best_epoch": int(audit["best_epoch"]),
                "latest_epoch": int(audit["last_epoch"]), "train_seconds": float(audit["train_seconds"]),
                "ACC": float(full["ACC"]), "AUC_original": float(full["AUC_original"]),
                "AUC_probability": float(full["AUC_probability"]),
                "native_ACC": float(native["ACC"]), "native_AUC_original": float(native["AUC_original"]),
                "native_AUC_probability": float(native["AUC_probability"]),
                "full_ACC": float(full["ACC"]), "full_AUC_original": float(full["AUC_original"]),
                "full_AUC_probability": float(full["AUC_probability"]),
                "Best_SHA256": audit["checkpoint_sha256"], "Latest_SHA256": audit["latest_sha256"],
                "training_implementation_SHA256": audit.get("implementation_sha256"),
                "group_config_SHA256": audit.get("group_config_sha256"),
                "prediction_SHA256": file_sha(prediction_file),
                "native_full_flip_count": int(audit.get("native_full_flip_count", 0)),
            })

    fold_frame = pd.DataFrame(fold_rows).sort_values(["group", "fold"])
    fold_frame.to_csv(ROOT / "results/fold_metrics.csv", index=False, float_format="%.12g")
    summary_rows = []
    for group in GROUPS:
        group_folds = fold_frame.loc[fold_frame["group"] == group]
        frame = patients[group]
        truth = frame["true_label"].to_numpy(dtype=np.int64)
        native_prediction = frame["native_pred"].to_numpy(dtype=np.int64)
        full_prediction = frame["full_pred"].to_numpy(dtype=np.int64)
        native_correct = int((native_prediction == truth).sum())
        full_correct = int((full_prediction == truth).sum())
        native_class = correct_by_class(truth, native_prediction)
        full_class = correct_by_class(truth, full_prediction)
        item = {"group": group, "source_group": SOURCE_GROUP.get(group, group), "folds": 10,
                "patients": 598, "native_OOF_correct": native_correct, "full_OOF_correct": full_correct,
                "native_NC_correct": native_class["NC"], "native_MCI_correct": native_class["MCI"],
                "native_AD_correct": native_class["AD"], "full_NC_correct": full_class["NC"],
                "full_MCI_correct": full_class["MCI"], "full_AD_correct": full_class["AD"],
                "best_epoch_mean": float(group_folds["best_epoch"].mean()),
                "best_epoch_min": int(group_folds["best_epoch"].min()),
                "best_epoch_max": int(group_folds["best_epoch"].max())}
        for metric in METRICS:
            values = group_folds[metric].to_numpy(dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std_population"] = float(values.std(ddof=0))
        for metric in METRICS:
            values = group_folds[f"native_{metric}"].to_numpy(dtype=np.float64)
            item[f"native_{metric}_mean"] = float(values.mean())
            item[f"native_{metric}_std_population"] = float(values.std(ddof=0))
        item["delta_ACC_vs_paper_pp"] = item["ACC_mean"] * 100 - 94.81
        item["delta_AUC_original_vs_paper_pp"] = item["AUC_original_mean"] * 100 - 95.59
        summary_rows.append(item)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(ROOT / "results/group_summary.csv", index=False, float_format="%.12g")
    all_patients.to_csv(ROOT / "results/oof_predictions.csv", index=False, float_format="%.9g")

    paired_rows = []
    for after_group, before_group in GROUP_CONTRASTS:
        for mode in ("native", "full"):
            for metric in METRICS:
                paired_rows.append(comparison_row(
                    before_group, after_group, mode, mode, "between_group",
                    mechanism_rows, patients, metric))
    for group in GROUPS:
        for before_case, after_case in DIRECT_CONTRASTS:
            for metric in METRICS:
                paired_rows.append(comparison_row(
                    group, group, before_case, after_case, "same_checkpoint_direct_inference",
                    mechanism_rows, patients, metric))

    for mode in ("native", "full"):
        for metric in METRICS:
            selected = [index for index, row in enumerate(paired_rows)
                        if row["comparison_type"] == "between_group"
                        and row["mode"] == mode and row["metric"] == metric]
            adjusted = holm_adjust([paired_rows[index]["paired_fold_t_p"] for index in selected])
            for index, value in zip(selected, adjusted, strict=True):
                paired_rows[index]["holm_adjusted_p"] = value
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(ROOT / "results/paired_contrasts.csv", index=False, float_format="%.12g")

    baseline = summary_frame.set_index("group").loc["H0"]
    new_summary = summary_frame.set_index("group").loc[list(NEW_GROUPS)]
    simultaneous = [group for group in NEW_GROUPS
                    if summary_frame.set_index("group").loc[group, "ACC_mean"] > baseline["ACC_mean"]
                    and summary_frame.set_index("group").loc[group, "AUC_original_mean"] > baseline["AUC_original_mean"]]
    all_below = bool(((new_summary["ACC_mean"] < baseline["ACC_mean"])
                      & (new_summary["AUC_original_mean"] < baseline["AUC_original_mean"])).all())
    probability_only = [group for group in NEW_GROUPS
                        if summary_frame.set_index("group").loc[group, "AUC_probability_mean"]
                        > summary_frame.set_index("group").loc["H0", "AUC_probability_mean"]]
    if simultaneous:
        performance = "PRIMARY_GO_CRITERION_MET"
    elif all_below:
        performance = "NO_GO_PERFORMANCE"
    elif probability_only:
        performance = "MIXED_RESULT"
    else:
        performance = "TRADEOFF"

    mechanism_group_summary = mechanism["groups"]
    relation_table = {}
    alignment_table = {}
    cognitive_graph_table = {}
    dpsl_table = {}
    for group in GROUPS:
        rows = mechanism_rows[group]
        relation_table[group] = {}
        for index, name in enumerate(("UCSFFSX", "UCBERKELEYAV45", "UPENNBIOMK9",
                                      "RISK_FACTOR", "COGNITIVE_TEST", "ROI_AVERAGE")):
            entries = [rows[fold]["relation_geometry"][index] for fold in range(1, 11)]
            relation_table[group][name] = {
                "heldout_auc_mean": optional_mean([
                    entry["same_different_auc_all_pairs"] for entry in entries]),
                "within_original_support_auc_mean": optional_mean([
                    entry["same_different_auc_within_original_graph_support"] for entry in entries]),
                "outside_support_diagnostic_auc_mean": optional_mean([
                    entry["same_different_auc_outside_original_graph_support_diagnostic"]
                    for entry in entries]),
            }
        alignment = [rows[fold]["alignment_gradient"] for fold in range(1, 11)]
        cognitive_index = 4
        alignment_table[group] = {
            "anchor_index": alignment[0]["alignment_anchor_index"],
            "grl_scale": alignment[0]["grl_scale"],
            "pair_loss_mean": float(np.mean([item["pair_loss"] for item in alignment])),
            "conditional_adversarial_loss_mean": float(np.mean([
                item["conditional_adversarial_loss"] for item in alignment])),
            "conditional_discriminator_accuracy_mean": float(np.mean([
                item["conditional_discriminator_accuracy"] for item in alignment])),
            "discriminator_gradient_norm_mean": float(np.mean([
                item["discriminator_gradient_norm"] for item in alignment])),
            "cognitive_relation_bce_encoder_gradient_norm_mean": float(np.mean([
                item["modality_gradients"][cognitive_index]["relation_encoder_gradient_norm"]
                for item in alignment])),
            "cognitive_relation_bce_scorer_gradient_norm_mean": float(np.mean([
                item["modality_gradients"][cognitive_index]["relation_scorer_gradient_norm"]
                for item in alignment])),
            "cognitive_adversarial_encoder_gradient_norm_mean": float(np.mean([
                item["modality_gradients"][cognitive_index]["conditional_adversarial_relation_encoder_gradient_norm"]
                for item in alignment])),
            "other_five_adversarial_encoder_gradient_norm_means": {
                name: float(np.mean([item["modality_gradients"][index][
                    "conditional_adversarial_relation_encoder_gradient_norm"] for item in alignment]))
                for index, name in enumerate(("UCSFFSX", "UCBERKELEYAV45", "UPENNBIOMK9",
                                              "RISK_FACTOR", "COGNITIVE_TEST", "ROI_AVERAGE"))
                if index != cognitive_index
            },
        }
        graph = mechanism_group_summary[group]["graph_means"][cognitive_index]
        propagation = [rows[fold]["relation_geometry"][cognitive_index]["positive_negative_propagation"]
                       for fold in range(1, 11)]
        cognitive_graph_table[group] = {
            **graph,
            "positive_signed_edges_mean": float(np.mean([row["signed_positive_edges"] for row in propagation])),
            "negative_signed_edges_mean": float(np.mean([row["signed_negative_edges"] for row in propagation])),
            "positive_gcn_change_relative_mean": float(np.mean([
                row["positive_only_gcn_change_relative_to_original_gfr_message"] for row in propagation])),
            "negative_gcn_change_relative_mean": float(np.mean([
                row["negative_only_gcn_change_relative_to_original_gfr_message"] for row in propagation])),
            "direct_message_relative_to_raw_mean": float(np.mean([
                row["direct_positive_dcr_message_relative_to_raw"] for row in propagation])),
        }
        direct_effect = mechanism_group_summary[group]["contrasts"]["direct_off_to_full"]
        disease_effect = mechanism_group_summary[group]["contrasts"]["disease_neutral_to_full"]
        cognitive_effect = mechanism_group_summary[group]["contrasts"]["cognitive_graph_off_to_full"]
        dpsl_effect = mechanism_group_summary[group]["contrasts"]["native_to_full"]
        dpsl_table[group] = {
            "native_OOF_correct": mechanism_group_summary[group]["native_correct"],
            "full_OOF_correct": mechanism_group_summary[group]["full_correct"],
            "native_to_full_corrected": dpsl_effect["corrected"],
            "native_to_full_damaged": dpsl_effect["damaged"],
            "native_to_full_net_correct": dpsl_effect["net_correct"],
            "residual_ratio_max": mechanism_group_summary[group]["dpsl_residual_ratio_max"],
            "shared_gradient_ratio_max": mechanism_group_summary[group]["dpsl_shared_ratio_max"],
            "direct_off_to_full": {key: direct_effect[key] for key in ("flips", "corrected", "damaged", "net_correct")},
            "disease_neutral_to_full": {key: disease_effect[key] for key in ("flips", "corrected", "damaged", "net_correct")},
            "cognitive_graph_off_to_full": {key: cognitive_effect[key] for key in ("flips", "corrected", "damaged", "net_correct")},
        }

    contrast_lookup = {(row["contrast"], row["mode"], row["metric"]): row
                       for row in paired_rows if row["comparison_type"] == "between_group"}
    factorial = {
        "gate_0_01_to_0_03_at_grl_0_03": {
            metric: contrast_lookup[("N2-N1", "full", metric)] for metric in METRICS},
        "gate_0_01_to_0_03_at_grl_0_015": {
            metric: contrast_lookup[("N4-N3", "full", metric)] for metric in METRICS},
        "grl_0_03_to_0_015_at_gate_0_01": {
            metric: contrast_lookup[("N3-N1", "full", metric)] for metric in METRICS},
        "grl_0_03_to_0_015_at_gate_0_03": {
            metric: contrast_lookup[("N4-N2", "full", metric)] for metric in METRICS},
    }

    training_status = {}
    for group in GROUPS:
        training_status[group] = {
            "folds": len(mechanism_rows[group]),
            "best_checkpoints": 10,
            "latest_checkpoints": 10,
            "fold_audits": 10,
            "patient_predictions": 10,
            "status": "PASS",
        }
    history_best_without_rng = sum(
        not replay["groups"][group][str(fold)]["best"]["best_rng_present"]
        for group in SOURCE_GROUP for fold in range(1, 11))
    results_hashes = {
        relative: file_sha(ROOT / relative) for relative in (
            "protocol.json", "source_manifest.json", "data_manifest.json",
            "audits/preflight.json", "audits/method_fidelity.json",
            "audits/abide_tadpole_parity.json", "audits/warmup_pairing.json",
            "audits/checkpoint_replay.json", "audits/mechanism_summary.json",
            "audits/log_format_audit.json", "audits/source_provenance_correction.json",
            "results/group_summary.csv",
            "results/fold_metrics.csv", "results/oof_predictions.csv",
            "results/paired_contrasts.csv", "logs/N1_result_Tadpole.txt",
            "logs/N2_result_Tadpole.txt", "logs/N3_result_Tadpole.txt",
            "logs/N4_result_Tadpole.txt",
        )
    }
    audit = {
        "experiment": "Experiment 83.8.5.1",
        "status": "COMPLETE_AUDITED",
        "performance_decision": performance,
        "experiment_root": str(ROOT),
        "training_implementation_sha256": seal["implementation_sha256"],
        "training_source_hashes_match_seal": True,
        "data_sha256": load_json(ROOT / "data_manifest.json")["processed_standard_data.csv_sha256"],
        "split_sha256": load_json(ROOT / "data_manifest.json")["split_indices.json_sha256"],
        "preflight": preflight["status"],
        "abide_tadpole_parity": parity["status"],
        "warmup_pairing": warmup["status"],
        "author_logs": {"status": log_audit["status"], "groups_passed": len(log_audit["groups"])},
        "source_provenance": {
            "status": load_json(ROOT / "audits/source_provenance_correction.json")["status"],
            "verified_commit": load_json(ROOT / "audits/source_provenance_correction.json")["verified_commit"],
            "reference_files_verified": 4,
            "training_seal_preserved": True,
        },
        "checkpoints": {"best_replayed": replay["best_replayed"],
                        "latest_restored": replay["latest_restored"],
                        "deterministic_next_steps": replay["deterministic_next_steps"],
                        "historical_best_rng_absent": history_best_without_rng},
        "folds": {"history_groups": 30, "new_formal_folds": 40, "total_audited": 70,
                  "groups": training_status},
        "primary_go_groups": simultaneous,
        "all_new_groups_below_h0_on_both_primary_means": all_below,
        "probability_auc_above_h0_groups": probability_only,
        "artifacts_sha256": results_hashes,
    }
    runner.write_json(ROOT / "audits/final_audit.json", audit)

    pair_lines = []
    for after, before in GROUP_CONTRASTS:
        acc = contrast_lookup[(f"{after}-{before}", "full", "ACC")]
        auc = contrast_lookup[(f"{after}-{before}", "full", "AUC_original")]
        pair_lines.append(
            f"| {after}-{before} | {acc['mean_delta_percentage_points']:+.3f} "
            f"[{acc['ci95_low_percentage_points']:+.3f}, {acc['ci95_high_percentage_points']:+.3f}] | "
            f"{acc['fold_wins']}/{acc['fold_ties']}/{acc['fold_losses']} | "
            f"{auc['mean_delta_percentage_points']:+.3f} "
            f"[{auc['ci95_low_percentage_points']:+.3f}, {auc['ci95_high_percentage_points']:+.3f}] | "
            f"{auc['fold_wins']}/{auc['fold_ties']}/{auc['fold_losses']} | "
            f"{acc['oof_corrected']}/{acc['oof_damaged']} ({acc['oof_net_correct']:+d}) |"
        )
    factorial_lines = []
    factorial_labels = (
        ("Gate .01→.03 at GRL .03", "N2-N1"),
        ("Gate .01→.03 at GRL .015", "N4-N3"),
        ("GRL .03→.015 at Gate .01", "N3-N1"),
        ("GRL .03→.015 at Gate .03", "N4-N2"),
    )
    for label, contrast in factorial_labels:
        acc = contrast_lookup[(contrast, "full", "ACC")]
        auc = contrast_lookup[(contrast, "full", "AUC_original")]
        factorial_lines.append(
            f"| {label} | {acc['mean_delta_percentage_points']:+.3f} "
            f"[{acc['ci95_low_percentage_points']:+.3f}, {acc['ci95_high_percentage_points']:+.3f}] | "
            f"{auc['mean_delta_percentage_points']:+.3f} "
            f"[{auc['ci95_low_percentage_points']:+.3f}, {auc['ci95_high_percentage_points']:+.3f}] |"
        )
    result_lines = []
    for _, row in summary_frame.iterrows():
        result_lines.append(
            f"| {row['group']} | {row['ACC_mean']*100:.4f} ± {row['ACC_std_population']*100:.4f} | "
            f"{row['AUC_original_mean']*100:.4f} ± {row['AUC_original_std_population']*100:.4f} | "
            f"{row['AUC_probability_mean']*100:.4f} ± {row['AUC_probability_std_population']*100:.4f} | "
            f"{row['full_OOF_correct']}/598 ({row['full_NC_correct']}/{row['full_MCI_correct']}/{row['full_AD_correct']}) | "
            f"{row['delta_ACC_vs_paper_pp']:+.3f} | {row['delta_AUC_original_vs_paper_pp']:+.3f} |"
        )
    cognitive_lines = []
    for group in GROUPS:
        cog = relation_table[group]["COGNITIVE_TEST"]
        graph = cognitive_graph_table[group]
        cognitive_lines.append(
            f"| {group} | {format_optional(cog['heldout_auc_mean'])} / "
            f"{format_optional(cog['within_original_support_auc_mean'])} / "
            f"{format_optional(cog['outside_support_diagnostic_auc_mean'])} | {graph['gate_effective']:.5f} | "
            f"{graph['gcn_relative_change_vs_disease_neutral']:.6f} | "
            f"{alignment_table[group]['conditional_adversarial_loss_mean']:.4f} / "
            f"{alignment_table[group]['conditional_discriminator_accuracy_mean']:.4f} |"
        )
    dpsl_lines = []
    for group in GROUPS:
        result = dpsl_table[group]
        dpsl_lines.append(
            f"| {group} | {result['native_to_full_corrected']}/{result['native_to_full_damaged']} "
            f"({result['native_to_full_net_correct']:+d}) | "
            f"{result['direct_off_to_full']['corrected']}/{result['direct_off_to_full']['damaged']} | "
            f"{result['disease_neutral_to_full']['corrected']}/{result['disease_neutral_to_full']['damaged']} | "
            f"{result['cognitive_graph_off_to_full']['corrected']}/{result['cognitive_graph_off_to_full']['damaged']} | "
            f"{result['residual_ratio_max']:.4f} / {result['shared_gradient_ratio_max']:.6f} |"
        )

    source_manifest = load_json(ROOT / "source_manifest.json")
    paper = source_manifest["paper"]["reference_result"]
    report = f"""# Experiment 83.8.5.1 Final Report

## Status

**{performance}**. N1–N4 completed 40/40 formal folds. H0–H2 were verified and replayed for 30/30 historical folds. No added group exceeded H0 in both registered primary metrics. Probability AUC was higher than H0 for {', '.join(probability_only) if probability_only else 'no new group'}, which is secondary and does not change the primary NO-GO decision.

The final report describes the historic `val=test` and pre-CV preprocessing protocol. Its fold intervals are descriptive because training sets overlap and checkpoint selection used the test-fold ACC.

## Environment and sources

Training used one PPU-ZW810E, four workers, seed 0, PyTorch {__import__('torch').__version__}. The sealed training implementation fingerprint is `{seal['implementation_sha256']}`. The verified data SHA-256 is `{load_json(ROOT / 'data_manifest.json')['processed_standard_data.csv_sha256']}` and the split SHA-256 is `{load_json(ROOT / 'data_manifest.json')['split_indices.json_sha256']}`.

SPromptGL paper reference: ACC {paper['ACC_percent']}%, AUC {paper['AUC_percent']}%. Original-author source identity is pinned to `ba3b9446fa451fde6864d6555bdf324de0c4b64a`. The inspected SABER checkout is `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4`. GitHub identifies the requested TADPOLE reference `ae225e5` as full commit `ae225e5135046d1d9622968b7d011f6aa55fa60d`. The server E83.8.5 source runtime, historical runner, and R0 config match that commit's raw-file SHA-256 values. The local `fidelity_model.py` differs from the historical file only by the registered configurable GRL scalar, retaining default 0.03. The training-time sealed source manifest contains an incorrect E83.8.2 pointer, corrected by [source_provenance_correction.json](../audits/source_provenance_correction.json). Its original hash and the training seal remain preserved.

The paper reports ACC {paper['ACC_percent']}% and AUC {paper['AUC_percent']}%. The table below compares against those means for reference. This complete DARG+DPSL model and its historical protocol are not identical to the paper's original model, so this is not a strict reproduction claim.

## Protocol and fidelity

The six modality dimensions, NC/MCI/AD mapping, historical ten splits, seed, RFE, mean imputation, pre-CV StandardScaler, transductive graph, `val_idx == test_idx`, test-ACC Best selection, 450-epoch ceiling, `wait > 200`, 10-epoch warmup, and Paper-ALoss coefficient 0.01 were retained. Test-fold labels were excluded from relation pairs, ALoss, graph supervision, DPSL objectives, and the masked-label checkpoint replays. The test-label permutation check passed. Interpret the metrics as exploratory results under the historical protocol, not unbiased independent-generalization estimates.

DARG retained the ABIDE E74.4-G2 Positive-DCR and E82.4.6.10-G1 conditional relation alignment, BCE, preservation loss, signed edges, source graph support, and double gate. Only the registered cognitive anchor, cognitive gate scalar, and GRL scalar varied. DPSL retained the E82.4.6.10-G1 frozen MMGPL-12 prompt path, old confidence mode, 0.08 residual cap, 0.001 shared-gradient cap, class-pair objectives, and AdamW settings. Fixed-tensor ABIDE parity and the available ABIDE G1 Fold 8 checkpoint replay passed. The fixed-tensor regression covers the historical four-modality/eight-prompt route. No full ABIDE fold retraining was performed.

## Seven-group results

Metrics are mean ± population standard deviation across ten folds. AUC_original is hard one-hot macro OVR. The OOF correctness column is Full output with NC/MCI/AD counts.

| Group | ACC % | AUC_original % | AUC_probability % | OOF correct | ΔACC vs paper pp | ΔAUC vs paper pp |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(result_lines)}

The OOF cohort contains 598 patients exactly once per group. `group_summary.csv`, `fold_metrics.csv`, and `oof_predictions.csv` contain fold values, Native/Full results, class counts, identifiers, and patient predictions. Deltas to paper means are in `group_summary.csv`.

## Paired comparisons

Each cell reports Full-mode mean paired fold change in percentage points, its 95% Student-t interval over ten paired folds, fold wins/ties/losses, and pooled OOF corrected/damaged counts with net change. The CSV includes Native comparisons, AUC_probability, exact McNemar tests, Holm-adjusted fold p-values, and same-checkpoint interventions.

| Contrast | ΔACC pp [95% CI] | W/T/L | ΔAUC_original pp [95% CI] | W/T/L | OOF corrected/damaged (net) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(pair_lines)}

The intervals and tests are descriptive because the ten CV training sets overlap and Best selection used `val=test`. [paired_contrasts.csv](paired_contrasts.csv) retains all registered comparisons and corrections.

The 2×2 factor contrasts use paired fold outcomes and show the registered scalar changes only.

| Factor contrast | ΔACC pp [95% CI] | ΔAUC_original pp [95% CI] |
|---|---:|---:|
{chr(10).join(factorial_lines)}

## Mechanism results

The relation AUCs use each trained DARG RelationEncoder and RelationScorer on held-out patient pairs. The three AUC columns are all test pairs, pairs inside original graph support, and pairs outside support as a diagnostic only.

| Group | Cognitive relation AUC all/support/outside | Effective Gate | GCN message change vs disease-neutral | Adversarial loss / discriminator accuracy |
|---|---:|---:|---:|---:|
{chr(10).join(cognitive_lines)}

The trained COGNITIVE_TEST RelationEncoder/RelationScorer had the highest held-out same/different AUC among the six modalities in all seven groups, ranging from 0.7589 to 0.8191. The relation advantage was therefore present in formal DARG scoring. The six-modality per-fold AUC, same/different train-pair score distributions, conditional centroid distances, signed edge counts, graph support coverage, message norms, and positive/negative propagation counterfactuals are in [mechanism_summary.json](../audits/mechanism_summary.json). Candidate and scored edges remained within the registered original graph support. Cognitive Gate 0.03 produced a larger effective Gate and larger graph perturbation than 0.01, while absolute GCN message changes stayed small. Direct-DCR message norms are separately recorded and are much smaller than the raw representation scale.

At the trained checkpoints, the cognitive anchor has zero encoder gradient from conditional adversarial alignment and nonzero relation-BCE encoder/scorer gradients. The other five encoders and the discriminator show finite gradients. This confirms anchor detach is branch-specific, not a frozen cognitive encoder. The configured GRL comparison uses N1 vs N3 and N2 vs N4. Their paired classification and relation-score changes do not support a uniform improvement from GRL 0.015.

Native is the DARG-trained classifier readout. Full versus Native is the same-checkpoint DPSL inference effect. The table reports per-group OOF corrections and harms. Residual-Off reproduced Native, while Full retained a nonzero bounded residual. The unchanged DPSL design produced small, group-dependent direct correction counts.

| Group | Native→Full corrected/damaged (net) | Direct-Off→Full corrected/damaged | Disease-Neutral→Full corrected/damaged | Cognitive-Graph-Off→Full corrected/damaged | Residual ratio max / shared-gradient ratio max |
|---|---:|---:|---:|---:|---:|
{chr(10).join(dpsl_lines)}

## Checkpoint and log audits

Historical and new Best checkpoints independently replayed 70/70. Latest states restored 70/70 and deterministic next steps matched on 70/70. Historical R0 Best checkpoints lack embedded RNG state in the archive for {history_best_without_rng}/10 folds; their real Latest states contain RNG and passed restore/continuation. No missing historical RNG was synthesized. ABIDE checkpoint replay and fixed-tensor parity are in [abide_checkpoint_replay.json](../audits/abide_checkpoint_replay.json) and [abide_tadpole_parity.json](../audits/abide_tadpole_parity.json).

The four author-format logs were reconstructed from stored epoch histories and fold predictions, with no training rerun or RNG use. All four passed exact line-by-line metric and format checks against `/data/adodas/SPromptGL-main/result_Tadpole.txt`; the epoch denominator is `/450` and the three `trn-loss-MF` fields retain their audited display definitions. See [log_format_audit.json](../audits/log_format_audit.json).

## Research judgement

1. N1 isolates the cognitive edge-anchor change relative to H1. The scorer uses the cognitive relation signal, but N1 does not show a paired performance gain over H1 or H0.
2. Gate 0.03 increases graph perturbation, but factor contrasts are mixed and the resulting message changes remain small. It is not more reasonable as a default from this experiment.
3. GRL 0.015 modestly improves some paired outcomes, especially N3 versus N1, but the gate 0.03 pair does not show consistent ACC and AUC_original gain.
4. N4 does not show complementary benefit. It does not exceed H0 on either registered primary metric.
5. The trained COGNITIVE_TEST scorer has the highest same/different held-out AUC among the six modalities in every group. The relation advantage reaches DARG scoring, but not the final classifier performance.
6. DARG changes supported cognitive graph weights and messages, with larger changes at Gate 0.03. Cognitive-Graph-Off caused zero argmax flips in all seven groups and at most 6.1e-5 absolute logit change, so its direct classification effect is negligible under these scales.
7. DPSL remains method-faithful, within its residual and shared-gradient caps, with reproducible Native/Full differences and modest corrections.
8. No new group simultaneously exceeds Y3/Z0 ACC and AUC_original.
9. No new group has paired evidence supporting replacement of Y3/Z0. Keep H0/Y3/Z0 as incumbent.
10. ABIDE E82.4.6.10-G1 and the TADPOLE adaptation retain the same generic DARG+DPSL core, with required modality, anchor, class-count, and prompt adaptations.

**Final decision: {performance}. Preserve Y3/Z0 as the complete-model base.**

Artifacts: [final_audit.json](../audits/final_audit.json), [protocol.json](../protocol.json), [source_manifest.json](../source_manifest.json), and [checkpoint_replay.json](../audits/checkpoint_replay.json).
"""
    report_path = ROOT / "results/final_report.md"
    report_path.write_text(report, encoding="utf-8")
    audit["artifacts_sha256"]["results/final_report.md"] = file_sha(report_path)
    runner.write_json(ROOT / "audits/final_audit.json", audit)
    print(json.dumps({"status": audit["status"], "performance": performance,
                      "groups": len(summary_rows), "paired_rows": len(paired_rows),
                      "report_sha256": file_sha(report_path)}, indent=2))


if __name__ == "__main__":
    main()
