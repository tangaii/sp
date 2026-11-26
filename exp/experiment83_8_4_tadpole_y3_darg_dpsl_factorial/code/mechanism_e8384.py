"""Measure DARG and DPSL behavior from each sealed Best checkpoint."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "common"))
import run_e8384 as runner
from metrics import acc_auc
from scmr_runtime import confidence_gate
from source_runtime import E74_GROUP, TADPOLE_MODALITIES

CLASS_NAMES = {0: "NC", 1: "MCI", 2: "AD"}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def class_changes(truth, before, after):
    records = {}
    for class_id, name in ((0, "NC"), (1, "MCI"), (2, "AD")):
        selected = truth == class_id
        records[name] = {
            "corrected": int(((before != truth) & (after == truth) & selected).sum()),
            "damaged": int(((before == truth) & (after != truth) & selected).sum()),
        }
    return records


def relation_gradient_audit(model, x, labels, train_rows, pair, same, config, epoch):
    runner.zero_optimizers({name: optimizer for name, optimizer in
                            runner.make_optimizers(model, config, formal=True)[0].items()})
    model.train()
    _, aux = model(x, labels, train_rows, pair, same, dpsl_enabled=True)
    kind = runner.active_supervision_kind(config, epoch)
    coefficient = runner.configured_supervision_coefficient(config)
    _, total, _, _ = runner.historical_loss(aux, labels, train_rows, model, kind, coefficient)
    (total + aux["dpsl_loss"]).backward()
    relation = {}
    for name, parameter in model.darg.relation_named_parameters():
        grad = parameter.grad
        relation[name] = {"gradient_norm": float(grad.detach().norm()) if grad is not None else 0.0,
                          "finite": bool(grad is not None and torch.isfinite(grad).all().item())}
    dpsl = {}
    for name, parameter in model.dpsl.named_parameters():
        grad = parameter.grad
        dpsl[name] = {"gradient_norm": float(grad.detach().norm()) if grad is not None else 0.0,
                      "finite": bool(grad is not None and torch.isfinite(grad).all().item())}
    check(all(item["finite"] for item in relation.values()), "Non-finite relation parameter gradient")
    check(all(item["finite"] for item in dpsl.values()), "Non-finite DPSL parameter gradient")
    model.eval()
    return relation, dpsl


def analyze_fold(group, fold, config, checkpoint, audit, x_np, y_np, ids, split, patient_rows):
    device = torch.device("cuda:0")
    runner.historical.seed_everything(runner.SEED)
    model = runner.build_model(config, fold, device, spectral_stats=checkpoint["spectral_stats"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.darg.enter_positive_dcr()
    model.eval()
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    train_rows_np = np.asarray(split["train_idx"], dtype=np.int64)
    test_rows = np.asarray(split["test_idx"], dtype=np.int64)
    train_rows = torch.from_numpy(train_rows_np).long().to(device)
    pair = checkpoint["relation_pairs"].to(device)
    same = checkpoint["relation_targets"].to(device)
    parts, start = [], 0
    for dim in runner.MODAL_DIMS:
        parts.append(x[:, start:start + dim])
        start += dim

    with torch.no_grad():
        darg_fused, original_adjs, darg = model.darg.positive.rudag_forward(
            model.darg.core, parts, pair, E74_GROUP, fold=fold, collect_diagnostics=True)
        full_logp, full_aux = model(x, labels, train_rows, pair, same, dpsl_enabled=True)
        native_logp, native_aux = model(x, labels, train_rows, pair, same, dpsl_enabled=False)
        direct_off_logp, _ = model(x, labels, train_rows, pair, same, dpsl_enabled=True,
                                   darg_intervention="direct_off")
        neutral_logp, _ = model(x, labels, train_rows, pair, same, dpsl_enabled=True,
                                darg_intervention="disease_neutral")
        if config["dpsl_confidence_protection"]:
            model.dpsl._confidence_gate_override = 1.0
            unprotected_logp, unprotected_aux = model(
                x, labels, train_rows, pair, same, dpsl_enabled=True)
            model.dpsl._confidence_gate_override = None
        else:
            unprotected_logp, unprotected_aux = full_logp, full_aux

    check(tuple(model.darg.modalities) == TADPOLE_MODALITIES and len(darg["edge_diagnostics"]) == 6,
          f"{group} fold {fold} DARG diagnostics do not cover all modalities")
    check(darg["direct_scale"] == float(config.get("darg_direct_scale", .02))
          and model.darg.direct_single_gate is bool(config.get("darg_direct_single_gate", False))
          and model.darg.core.negative_scale == .25,
          f"{group} fold {fold} Positive-DCR scales differ")
    full_pred = full_logp.argmax(1).detach().cpu().numpy()[test_rows]
    native_pred = native_logp.argmax(1).detach().cpu().numpy()[test_rows]
    direct_off_pred = direct_off_logp.argmax(1).detach().cpu().numpy()[test_rows]
    neutral_pred = neutral_logp.argmax(1).detach().cpu().numpy()[test_rows]
    truth = y_np[test_rows]
    full_probability = full_logp.exp().detach().cpu().numpy()[test_rows]
    native_probability = native_logp.exp().detach().cpu().numpy()[test_rows]
    direct_off_probability = direct_off_logp.exp().detach().cpu().numpy()[test_rows]
    neutral_probability = neutral_logp.exp().detach().cpu().numpy()[test_rows]

    dpsl = full_aux["dpsl"]
    native_logits = full_aux["native_logits"].detach()
    native_prob_all = native_logits.softmax(-1)
    top = native_prob_all.topk(2, dim=-1).values
    gates = (dpsl["confidence_gate"] if "confidence_gate" in dpsl
             else confidence_gate(native_logits))
    margin_all = native_logits.topk(2, dim=-1).values
    margin = (margin_all[:, 0] - margin_all[:, 1]).detach().cpu().numpy()[test_rows]
    gate_values = gates.detach().cpu().numpy()[test_rows]
    after_ratio = dpsl.get("ratio", torch.zeros(len(x), device=device)).detach().cpu().numpy()[test_rows]
    before_delta = dpsl.get("delta_before_confidence", dpsl.get("delta"))
    if before_delta is None:
        before_ratio_all = after_ratio
    else:
        cam_norm = dpsl["cam_norm"].clamp_min(1e-12)
        before_ratio_all = (before_delta.norm(dim=-1) / cam_norm).detach().cpu().numpy()
    before_ratio = before_ratio_all[test_rows]
    dpsl_delta = dpsl.get("delta", torch.zeros_like(full_aux["cam"]))
    delta_norm = dpsl_delta.norm(dim=-1).detach().cpu().numpy()[test_rows]
    unprotected_pred = unprotected_logp.argmax(1).detach().cpu().numpy()[test_rows]
    low_margin_cut = float(np.quantile(margin, 0.25))
    high_margin_cut = float(np.quantile(margin, 0.75))
    low_confidence = margin <= low_margin_cut
    high_confidence = margin >= high_margin_cut

    class_pair_activity = []
    pair_ids = dpsl.get("class_pairs")
    pair_weights = dpsl.get("pair_weights")
    pair_outputs = dpsl.get("pair_outputs")
    if pair_ids is not None and pair_weights is not None and pair_outputs is not None:
        pair_ids = pair_ids.detach().cpu().numpy()
        pair_weights = pair_weights.detach().cpu().numpy()
        for pair_index, pair_output in enumerate(pair_outputs):
            mild, severe = sorted(int(value) for value in pair_ids[pair_index])
            weighted_raw = pair_output["raw_parallel"] + pair_output["raw_perp"]
            class_pair_activity.append({
                "pair": [mild, severe],
                "pair_names": [CLASS_NAMES[mild], CLASS_NAMES[severe]],
                "mean_patient_pair_weight": float(pair_weights[test_rows, pair_index].mean()),
                "mean_weighted_raw_residual_norm": float(
                    (weighted_raw[test_rows].norm(dim=-1)
                     * torch.from_numpy(pair_weights[test_rows, pair_index]).to(device)).mean()),
            })

    orig_edges, unified_edges = [], []
    modality_records = []
    raw_norm = darg["raw"].norm(dim=1).clamp_min(1e-12)
    for index, name in enumerate(TADPOLE_MODALITIES):
        original = darg["original_adjs"][index]
        unified = darg["unified_adjs"][index]
        diagnostic = darg["edge_diagnostics"][index]
        orig_edges.append(int(torch.triu((original > 0), diagonal=1).sum().item()))
        unified_edges.append(int(torch.triu((unified > 0), diagonal=1).sum().item()))
        message = darg["messages"][index]
        direct = darg["direct_messages"][index]
        weighted_direct = darg["weighted_direct_messages"][index]
        record = {
            **diagnostic,
            "modality": name,
            "original_graph_edges": orig_edges[-1],
            "unified_graph_edges": unified_edges[-1],
            "original_unified_edge_count_difference": unified_edges[-1] - orig_edges[-1],
            "unified_minus_original_frobenius": float((unified - original).norm()),
            "gcn_message_norm_ratio": float((message.norm() / darg["raw"].norm().clamp_min(1e-12)).item()),
            "direct_message_norm_ratio": float((direct.norm() / darg["raw"].norm().clamp_min(1e-12)).item()),
            "weighted_direct_message_norm_ratio": float((weighted_direct.norm() / darg["raw"].norm().clamp_min(1e-12)).item()),
        }
        modality_records.append(record)

    relation_gradients, dpsl_gradients = relation_gradient_audit(
        model, x, labels, train_rows, pair, same, config, int(checkpoint["epoch"]))
    history_best = checkpoint["history_at_best"]
    graph_gradient_audit = history_best.get("graph_gradient_audit", [])
    budget_ratios = {}
    for family in ("gfr", "darg", "cam", "head"):
        values = []
        for item in checkpoint.get("gradient_budget_history", []):
            entry = item.get("families", {}).get(family, {})
            if entry.get("historical", 0.0):
                values.append(float(entry.get("total_applied", 0.0)) / float(entry["historical"]))
        budget_ratios[family] = max(values, default=0.0)
    check(max(budget_ratios.values(), default=0.0) <= config["dpsl_shared_cap"] + 1e-8,
          f"{group} fold {fold} shared gradient budget exceeds 0.001")
    check(float(after_ratio.max()) <= .08 + 1e-7, f"{group} fold {fold} DPSL residual exceeds 0.08")

    native_metrics = acc_auc(truth, native_pred, 3)
    native_metrics["AUC_probability"] = runner.roc_auc_score(
        np.eye(3)[truth], native_probability, average="macro", multi_class="ovr")
    full_metrics = acc_auc(truth, full_pred, 3)
    full_metrics["AUC_probability"] = runner.roc_auc_score(
        np.eye(3)[truth], full_probability, average="macro", multi_class="ovr")
    directoff_metrics = acc_auc(truth, direct_off_pred, 3)
    neutral_metrics = acc_auc(truth, neutral_pred, 3)
    margin_spearman_raw = spearmanr(margin, gate_values).statistic if len(margin) > 1 else np.nan
    margin_spearman_defined = bool(np.isfinite(margin_spearman_raw))
    margin_spearman = float(margin_spearman_raw) if margin_spearman_defined else 0.0
    patient = pd.DataFrame({
        "group": group, "fold": fold, "sample_index": test_rows,
        "sample_id": ids.iloc[test_rows]["RID"].to_numpy(), "true_label": truth,
        "native_pred": native_pred, "full_pred": full_pred, "unprotected_full_pred": unprotected_pred,
        "direct_off_pred": direct_off_pred, "disease_neutral_pred": neutral_pred,
        "native_prob_NC": native_probability[:, 0], "native_prob_MCI": native_probability[:, 1],
        "native_prob_AD": native_probability[:, 2], "full_prob_NC": full_probability[:, 0],
        "full_prob_MCI": full_probability[:, 1], "full_prob_AD": full_probability[:, 2],
        "darg_direct_off_prob_NC": direct_off_probability[:, 0],
        "darg_direct_off_prob_MCI": direct_off_probability[:, 1],
        "darg_direct_off_prob_AD": direct_off_probability[:, 2],
        "disease_neutral_prob_NC": neutral_probability[:, 0],
        "disease_neutral_prob_MCI": neutral_probability[:, 1],
        "disease_neutral_prob_AD": neutral_probability[:, 2],
        "native_logit_margin": margin, "confidence_gate_g": gate_values,
        "dpsl_residual_ratio_unprotected": before_ratio,
        "dpsl_residual_ratio_actual": after_ratio, "dpsl_residual_norm_actual": delta_norm,
    })
    fold_summary = {
        "fold": fold, "best_epoch": int(checkpoint["epoch"]),
        "native": native_metrics, "full": full_metrics,
        "native_to_full_corrected": int(((native_pred != truth) & (full_pred == truth)).sum()),
        "native_to_full_damaged": int(((native_pred == truth) & (full_pred != truth)).sum()),
        "native_to_full_by_true_class": class_changes(truth, native_pred, full_pred),
        "dpsl_class_pair_activity": class_pair_activity,
        "protected_vs_unprotected_full_changed": int((full_pred != unprotected_pred).sum()),
        "dpsl_protected_vs_unprotected_corrected": int(((unprotected_pred != truth) & (full_pred == truth)).sum()),
        "dpsl_protected_vs_unprotected_damaged": int(((unprotected_pred == truth) & (full_pred != truth)).sum()),
        "direct_off": {"ACC": directoff_metrics["ACC"], "AUC_original": directoff_metrics["AUC"]},
        "disease_neutral": {"ACC": neutral_metrics["ACC"], "AUC_original": neutral_metrics["AUC"]},
        "direct_off_changed": int((full_pred != direct_off_pred).sum()),
        "direct_off_corrected": int(((direct_off_pred != truth) & (full_pred == truth)).sum()),
        "direct_off_damaged": int(((direct_off_pred == truth) & (full_pred != truth)).sum()),
        "direct_off_by_true_class": class_changes(truth, direct_off_pred, full_pred),
        "disease_neutral_changed": int((full_pred != neutral_pred).sum()),
        "disease_neutral_corrected": int(((neutral_pred != truth) & (full_pred == truth)).sum()),
        "disease_neutral_damaged": int(((neutral_pred == truth) & (full_pred != truth)).sum()),
        "disease_neutral_by_true_class": class_changes(truth, neutral_pred, full_pred),
        "confidence_gate": {"mean": float(gate_values.mean()), "std": float(gate_values.std()),
                            "min": float(gate_values.min()), "max": float(gate_values.max()),
                            "margin_spearman": margin_spearman,
                            "margin_spearman_defined": margin_spearman_defined},
        "residual_ratio": {"unprotected_mean": float(before_ratio.mean()),
                           "actual_mean": float(after_ratio.mean()),
                           "unprotected_max": float(before_ratio.max()),
                           "actual_max": float(after_ratio.max()),
                           "margin_quartiles": {"low_cut": low_margin_cut,
                                                "high_cut": high_margin_cut},
                           "low_confidence_unprotected_mean": float(before_ratio[low_confidence].mean()),
                           "low_confidence_actual_mean": float(after_ratio[low_confidence].mean()),
                           "high_confidence_unprotected_mean": float(before_ratio[high_confidence].mean()),
                           "high_confidence_actual_mean": float(after_ratio[high_confidence].mean()),
                           "shared_budget_ratio_max_by_family": budget_ratios},
        "darg_modalities": modality_records,
        "relation_parameter_gradients": relation_gradients,
        "dpsl_parameter_gradients": dpsl_gradients,
        "aloss": {
            "formula": audit["aloss_formula"],
            "lambda": float(audit["supervision_coefficient"]),
            "activation_epoch": 11,
            "best_epoch_graph": float(history_best["graph"]),
            "best_epoch_squared_aloss": float(history_best["graph_supervision"]),
            "coefficient_weighted_aloss_to_graph_ratio": float(
                float(audit["supervision_coefficient"]) * history_best["graph_supervision"]
                / max(history_best["graph"], 1e-30)),
            "gradient_audit_by_modality": graph_gradient_audit,
            "epoch11_counterfactual_update": next(
                (item for item in audit["update_vector_audits"] if item.get("epoch") == 11), None),
            "author_unsquared_log_observation": audit["aloss_observation_author_unsquared"],
        },
    }
    return fold_summary, patient


def mean_numeric(records, key):
    return float(np.mean([float(record[key]) for record in records]))


def main():
    runner.require_training_gate(list(runner.FORMAL_GROUPS))
    x, y, ids, splits = runner.load_data()
    split_map = {int(item["fold"]): item for item in splits}
    patient_frames = []
    all_groups = {}
    for group in runner.FORMAL_GROUPS:
        config = runner.group_config(group)
        group_dir = ROOT / "groups" / group
        fold_records = []
        for fold in range(1, 11):
            audit = json.loads((group_dir / "audits" / f"fold{fold:02d}.json").read_text())
            checkpoint = torch.load(group_dir / "best_checkpoints" / f"fold{fold:02d}.pt",
                                    map_location="cpu", weights_only=False)
            summary, patients = analyze_fold(group, fold, config, checkpoint, audit, x, y, ids,
                                             split_map[fold], None)
            fold_records.append(summary)
            patient_frames.append(patients)
            print(f"MECHANISM {group} fold {fold:02d}", flush=True)
            del checkpoint
            torch.cuda.empty_cache()

        modality_summary = {}
        for modality in TADPOLE_MODALITIES:
            records = [item for fold in fold_records for item in fold["darg_modalities"]
                       if item["modality"] == modality]
            fields = ("original_effective_edges", "candidate_edges_before_support_intersection",
                      "scored_edges", "positive_score_edges", "negative_score_edges",
                      "gate_effective", "adjacency_change_relative", "gcn_message_norm_ratio",
                      "direct_message_norm_ratio", "weighted_direct_message_norm_ratio",
                      "original_graph_edges", "unified_graph_edges",
                      "original_unified_edge_count_difference", "unified_minus_original_frobenius")
            modality_summary[modality] = {f"{field}_mean": mean_numeric(records, field) for field in fields}
        all_groups[group] = {
            "folds": fold_records,
            "native_mean_ACC": float(np.mean([item["native"]["ACC"] for item in fold_records])),
            "native_mean_AUC_original": float(np.mean([item["native"]["AUC"] for item in fold_records])),
            "native_mean_AUC_probability": float(np.mean([item["native"]["AUC_probability"] for item in fold_records])),
            "full_mean_ACC": float(np.mean([item["full"]["ACC"] for item in fold_records])),
            "full_mean_AUC_original": float(np.mean([item["full"]["AUC"] for item in fold_records])),
            "full_mean_AUC_probability": float(np.mean([item["full"]["AUC_probability"] for item in fold_records])),
            "native_to_full_corrected": sum(item["native_to_full_corrected"] for item in fold_records),
            "native_to_full_damaged": sum(item["native_to_full_damaged"] for item in fold_records),
            "direct_off_changed": sum(item["direct_off_changed"] for item in fold_records),
            "direct_off_corrected": sum(item["direct_off_corrected"] for item in fold_records),
            "direct_off_damaged": sum(item["direct_off_damaged"] for item in fold_records),
            "disease_neutral_changed": sum(item["disease_neutral_changed"] for item in fold_records),
            "disease_neutral_corrected": sum(item["disease_neutral_corrected"] for item in fold_records),
            "disease_neutral_damaged": sum(item["disease_neutral_damaged"] for item in fold_records),
            "dpsl_protected_vs_unprotected_corrected": sum(
                item["dpsl_protected_vs_unprotected_corrected"] for item in fold_records),
            "dpsl_protected_vs_unprotected_damaged": sum(
                item["dpsl_protected_vs_unprotected_damaged"] for item in fold_records),
            "darg_by_modality": modality_summary,
        }
    patient_mechanism = pd.concat(patient_frames, ignore_index=True)
    patient_mechanism.to_csv(ROOT / "results/patient_mechanism.csv", index=False)
    mechanism = {
        "experiment": "Experiment 83.8.4",
        "status": "PASS",
        "groups": all_groups,
        "patient_mechanism_rows": len(patient_mechanism),
        "dpsl_class_pairs": [[0, 1], [0, 2], [1, 2]],
        "dpsl_native_full_decomposition": "Within-group Native-to-Full is direct inference; cross-group Native is joint-training change.",
        "aloss_protocol": {"formula": "mean_m squared Frobenius norm over train-train normalized graphs",
                           "lambda": .01, "activation_epoch": 11,
                           "label_source": "training fold labels only"},
    }
    target = ROOT / "audits/mechanism_summary.json"
    target.write_text(json.dumps(mechanism, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "groups": list(all_groups),
                      "patient_rows": len(patient_mechanism),
                      "path": str(target)}, indent=2))


if __name__ == "__main__":
    main()
