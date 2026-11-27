"""Same-Best-checkpoint DARG/DPSL intervention and graph mechanism analysis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83853 as runner

GROUPS = runner.TRAIN_GROUPS
METHODS = (
    "full", "native", "residual_off", "direct_off", "disease_neutral",
    "cognitive_graph_off", "cognitive_score_permuted",
)


def top_margin(logits):
    values = logits.topk(2, dim=1).values
    return values[:, 0] - values[:, 1]


def summarize_transition(truth, source_pred, target_pred, source_logits, target_logits,
                         source_prob, target_prob, source_name, target_name):
    source_correct = source_pred == truth
    target_correct = target_pred == truth
    logit_delta = np.abs(target_logits - source_logits)
    prob_delta = np.abs(target_prob - source_prob)
    source_margin = top_margin(torch.as_tensor(source_logits)).numpy()
    target_margin = top_margin(torch.as_tensor(target_logits)).numpy()
    return {
        "source": source_name,
        "target": target_name,
        "test_patients": int(len(truth)),
        "changed_argmax": int(np.sum(target_pred != source_pred)),
        "corrected": int(np.sum(~source_correct & target_correct)),
        "damaged": int(np.sum(source_correct & ~target_correct)),
        "net": int(np.sum(target_correct) - np.sum(source_correct)),
        "corrected_by_class": {
            name: int(np.sum((truth == class_id) & ~source_correct & target_correct))
            for class_id, name in enumerate(("NC", "MCI", "AD"))},
        "damaged_by_class": {
            name: int(np.sum((truth == class_id) & source_correct & ~target_correct))
            for class_id, name in enumerate(("NC", "MCI", "AD"))},
        "mean_abs_logit_delta": float(logit_delta.mean()),
        "max_abs_logit_delta": float(logit_delta.max(initial=0.0)),
        "mean_abs_probability_delta": float(prob_delta.mean()),
        "max_abs_probability_delta": float(prob_delta.max(initial=0.0)),
        "mean_top2_margin_change": float((target_margin - source_margin).mean()),
    }


def forward_variant(model, x, y, train, pair, same, config, *, name):
    enabled = bool(config.get("dpsl_enabled", model.dpsl is not None))
    if name == "full":
        return model(x, y, train, pair, same, dpsl_enabled=enabled)
    if name == "native":
        return model(x, y, train, pair, same, dpsl_enabled=False)
    if name == "residual_off":
        if model.dpsl is None:
            return model(x, y, train, pair, same, dpsl_enabled=False)
        return model(x, y, train, pair, same, dpsl_enabled=True,
                     dpsl_intervention="residual_off")
    darg_mode = {
        "direct_off": "direct_off",
        "disease_neutral": "disease_neutral",
        "cognitive_graph_off": "cognitive_graph_off",
        "cognitive_score_permuted": "score_permuted",
    }[name]
    return model(x, y, train, pair, same, dpsl_enabled=enabled,
                 darg_intervention=darg_mode)


@torch.no_grad()
def relation_score_distributions(model, x, labels, parts, full_aux):
    n = len(labels)
    all_rows, all_cols = torch.triu_indices(n, n, offset=1, device=x.device)
    all_pairs = torch.stack((all_rows, all_cols), dim=1)
    _, _, all_logits = model.darg.core._encode_pairs(parts, all_pairs)
    all_keys = all_rows * n + all_cols

    def summarize(rows, cols, scores):
        row_np, col_np = rows.cpu().numpy(), cols.cpu().numpy()
        target = (labels[row_np] == labels[col_np])
        same_scores = scores[target]
        different_scores = scores[~target]
        auc = (float(roc_auc_score(target.astype(np.int8), scores))
               if len(np.unique(target)) == 2 else None)
        return {
            "pair_count": int(len(scores)), "same_pairs": int(target.sum()),
            "different_pairs": int((~target).sum()), "same_different_auc": auc,
            "same_score_mean": float(same_scores.mean()) if len(same_scores) else None,
            "same_score_std": float(same_scores.std()) if len(same_scores) else None,
            "different_score_mean": (float(different_scores.mean())
                                     if len(different_scores) else None),
            "different_score_std": (float(different_scores.std())
                                    if len(different_scores) else None),
            "q_mean": float(scores.mean()) if len(scores) else None,
            "q_std": float(scores.std()) if len(scores) else None,
            "q_uncertain_fraction": (float(np.mean((scores > 0.45) & (scores < 0.55)))
                                     if len(scores) else None),
        }

    records = []
    for modality, (part, logits) in enumerate(zip(parts, all_logits)):
        scores = torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy()
        original = model.darg.core.graphs[modality](part).detach()
        support = original > 0
        support.fill_diagonal_(False)
        support_rows, support_cols = torch.triu(support, diagonal=1).nonzero(as_tuple=True)
        candidate, candidate_rows, candidate_cols, _ = model.darg.core._candidate_support(
            modality, part, original)
        candidate_rows, candidate_cols = torch.triu(candidate, diagonal=1).nonzero(as_tuple=True)
        actual_rows = full_aux["edge_rows"][modality]
        actual_cols = full_aux["edge_cols"][modality]

        def score_subset(rows, cols):
            keys = rows * n + cols
            positions = torch.searchsorted(all_keys, keys)
            return scores[positions.detach().cpu().numpy()]

        records.append({
            "modality": runner.MODAL_NAMES[modality],
            "all_patient_pairs": summarize(all_rows, all_cols, scores),
            "graph_valid_edges": summarize(
                support_rows, support_cols, score_subset(support_rows, support_cols)),
            "candidate_edges_before_support_intersection": summarize(
                candidate_rows, candidate_cols, score_subset(candidate_rows, candidate_cols)),
            "actual_scored_candidates": summarize(
                actual_rows, actual_cols, score_subset(actual_rows, actual_cols)),
        })
    return records


def graph_record(group, fold, best, latest, fold_audit, config, relation_scores):
    gate_scale = np.asarray(config.get("darg_gate_scales", []), dtype=np.float64)
    gate_trace = fold_audit.get("gate_trace", [])
    initialized = fold_audit.get("gate_initialization") or best.get("warmup", {}).get(
        "gate_initialization", {})

    def get_gates(payload):
        state = payload["model"]
        logits = state.get("darg.core.adaptive_gate_logits")
        if logits is None:
            return []
        probabilities = torch.sigmoid(logits.detach().cpu()).numpy()
        return (gate_scale * probabilities).tolist()

    edge = fold_audit.get("graph_mechanism", {}).get("edge_diagnostics", [])
    relation = fold_audit.get("relation_quality", [])
    return {
        "group": group, "fold": fold,
        "gate_initial_effective": initialized.get("effective_gate_after"),
        "gate_initial_probabilities": initialized.get("probabilities_target"),
        "gate_best_effective": get_gates(best),
        "gate_latest_effective": get_gates(latest),
        "gate_scales": gate_scale.tolist(),
        "gate_frozen_config": bool(config.get("freeze_adaptive_gate", False)),
        "gate_requires_grad_epoch_count": int(sum(
            item.get("requires_grad") is True for item in gate_trace)),
        "gate_gradient_finite_count": int(sum(
            item.get("gradient_finite") is True for item in gate_trace)),
        "gate_gradient_count": len(gate_trace),
        "gate_gradient_norm_min": min(
            (float(item["gradient_norm"]) for item in gate_trace
             if item.get("gradient_norm") is not None), default=None),
        "gate_gradient_norm_max": max(
            (float(item["gradient_norm"]) for item in gate_trace
             if item.get("gradient_norm") is not None), default=None),
        "gate_update_count": int(sum(
            item.get("parameter_update_l2", 0.0) > 0 for item in gate_trace)),
        "gate_optimizer_step_latest": (
            gate_trace[-1].get("optimizer_step") if gate_trace else None),
        "edge_diagnostics": edge,
        "relation_quality_by_modality": relation,
        "relation_score_distributions": relation_scores,
        "relation_sampling": fold_audit.get("relation_sampling"),
    }


def analyze_fold(group, fold, x, y, ids, split, device):
    directory = ROOT / "groups" / group
    (directory / "counterfactuals").mkdir(parents=True, exist_ok=True)
    best_path = directory / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest_path = directory / "latest_checkpoints" / f"fold{fold:02d}.pt"
    prediction_path = directory / "predictions" / f"fold{fold:02d}.csv"
    audit_path = directory / "audits" / f"fold{fold:02d}.json"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    stored = pd.read_csv(prediction_path).sort_values("sample_index")
    fold_audit = json.loads(audit_path.read_text())
    config = runner.group_config(group)
    model = runner.build_model(config, fold, device, spectral_stats=best["spectral_stats"])
    model.load_state_dict(best["model"], strict=True)
    if model.use_darg:
        model.darg.enter_positive_dcr()
    model.eval()
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
    train = torch.as_tensor(split["train_idx"], dtype=torch.long, device=device)
    pair, same = best["relation_pairs"], best["relation_targets"]
    if pair is not None:
        pair, same = pair.to(device), same.to(device)

    outputs = {}
    full_aux = None
    with torch.no_grad():
        for name in METHODS:
            logp, aux = forward_variant(
                model, x_tensor, y_tensor, train, pair, same, config, name=name)
            if not torch.isfinite(logp).all():
                raise FloatingPointError(f"{group} fold {fold}: non-finite {name} logits")
            outputs[name] = {
                "logp": logp.detach(),
                "logits": aux["logits"].detach(),
                "prob": logp.exp().detach(),
                "prediction": logp.argmax(1).detach(),
            }
            if name == "full":
                full_aux = aux

    idx = np.asarray(split["test_idx"], dtype=np.int64)
    if not np.array_equal(stored["sample_index"].to_numpy(), idx):
        raise RuntimeError(f"{group} fold {fold}: stored prediction row order differs")
    full = outputs["full"]
    full_pred_all = full["prediction"].cpu().numpy()
    if not np.array_equal(full_pred_all[idx], stored["pred_label"].to_numpy()):
        raise RuntimeError(f"{group} fold {fold}: Full intervention replay prediction differs")
    full_prob_all = full["prob"].cpu().numpy()
    stored_prob = stored[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy()
    if np.max(np.abs(full_prob_all[idx] - stored_prob)) > 5e-6:
        raise RuntimeError(f"{group} fold {fold}: Full intervention replay probabilities differ")
    if not torch.equal(outputs["native"]["logp"], outputs["residual_off"]["logp"]):
        raise RuntimeError(f"{group} fold {fold}: Residual-Off is not exactly Native")

    truth = y[idx]
    result_rows = pd.DataFrame({
        "RID": ids.iloc[idx]["RID"].to_numpy(),
        "fold": fold,
        "sample_index": idx,
        "true_label": truth,
    })
    summaries = {}
    transitions = {
        "native_to_full": ("native", "full"),
        "full_to_native": ("full", "native"),
        "residual_off_to_full": ("residual_off", "full"),
        "direct_off_to_full": ("direct_off", "full"),
        "disease_neutral_to_full": ("disease_neutral", "full"),
        "cognitive_graph_off_to_full": ("cognitive_graph_off", "full"),
        "score_permuted_to_full": ("cognitive_score_permuted", "full"),
    }
    for transition, (source_name, target_name) in transitions.items():
        source, target = outputs[source_name], outputs[target_name]
        summaries[transition] = summarize_transition(
            truth, source["prediction"][idx].cpu().numpy(),
            target["prediction"][idx].cpu().numpy(), source["logits"][idx].cpu().numpy(),
            target["logits"][idx].cpu().numpy(), source["prob"][idx].cpu().numpy(),
            target["prob"][idx].cpu().numpy(), source_name, target_name)
    variant_metrics = {}
    for name in ("native", "full"):
        values = outputs[name]
        predicted = values["prediction"][idx].cpu().numpy()
        probability = values["prob"][idx].cpu().numpy()
        scores = runner.acc_auc(truth, predicted, 3)
        scores["AUC_probability"] = float(roc_auc_score(
            np.eye(3, dtype=np.float64)[truth], probability,
            average="macro", multi_class="ovr"))
        variant_metrics[name] = scores
    for name in METHODS:
        output = outputs[name]
        logits = output["logits"][idx].cpu().numpy()
        probability = output["prob"][idx].cpu().numpy()
        prediction = output["prediction"][idx].cpu().numpy()
        result_rows[f"{name}_pred"] = prediction
        for class_id, class_name in enumerate(("NC", "MCI", "AD")):
            result_rows[f"{name}_logit_{class_name}"] = logits[:, class_id]
            result_rows[f"{name}_prob_{class_name}"] = probability[:, class_id]
    result_rows.to_csv(directory / "counterfactuals" / f"fold{fold:02d}.csv", index=False)
    return {
        "group": group, "fold": fold, "status": "PASS",
        "counterfactual_sha256": runner.sha_file(
            directory / "counterfactuals" / f"fold{fold:02d}.csv"),
        "test_count": len(idx),
        "variant_metrics": variant_metrics,
        "interventions_vs_full": summaries,
        "graph": graph_record(group, fold, best, latest, fold_audit, config,
                               relation_score_distributions(
                                   model, x_tensor, y, [x_tensor[:, start:stop]
                                    for start, stop in zip(
                                        np.cumsum((0, *runner.MODAL_DIMS))[:-1],
                                        np.cumsum((0, *runner.MODAL_DIMS))[1:])],
                                   full_aux)),
    }


def aggregate_group(group, records):
    transition_names = (
        "native_to_full", "full_to_native", "residual_off_to_full", "direct_off_to_full",
        "disease_neutral_to_full", "cognitive_graph_off_to_full", "score_permuted_to_full",
    )
    aggregate = {}
    for name in transition_names:
        keys = ("test_patients", "changed_argmax", "corrected", "damaged", "net")
        aggregate[name] = {
            key: int(sum(item["interventions_vs_full"][name][key] for item in records))
            for key in keys
        }
        for key in (
            "mean_abs_logit_delta", "max_abs_logit_delta",
            "mean_abs_probability_delta", "max_abs_probability_delta",
            "mean_top2_margin_change",
        ):
            values = [item["interventions_vs_full"][name][key] for item in records]
            aggregate[name][key + "_fold_mean"] = float(np.mean(values))
            aggregate[name][key + "_max"] = float(np.max(values))
        aggregate[name]["corrected_by_class"] = {
            class_name: int(sum(item["interventions_vs_full"][name][
                "corrected_by_class"][class_name] for item in records))
            for class_name in ("NC", "MCI", "AD")}
        aggregate[name]["damaged_by_class"] = {
            class_name: int(sum(item["interventions_vs_full"][name][
                "damaged_by_class"][class_name] for item in records))
            for class_name in ("NC", "MCI", "AD")}
    variant_summary = {}
    for variant in ("native", "full"):
        for metric in ("ACC", "AUC", "AUC_probability"):
            values = [item["variant_metrics"][variant][metric] for item in records]
            variant_summary[f"{variant}_{metric}_fold_mean"] = float(np.mean(values))
            variant_summary[f"{variant}_{metric}_fold_population_std"] = float(np.std(values, ddof=0))
    gate_initial = [item["graph"]["gate_initial_effective"] for item in records]
    gate_best = np.asarray([item["graph"]["gate_best_effective"] for item in records], dtype=float)
    gate_latest = np.asarray([item["graph"]["gate_latest_effective"] for item in records], dtype=float)
    edge_rows = [edge for item in records for edge in item["graph"]["edge_diagnostics"]]
    modality_names = runner.MODAL_NAMES
    edge_summary = {}
    for modality in modality_names:
        rows = [row for row in edge_rows if row.get("modality") == modality]
        edge_summary[modality] = {}
        for key in (
            "original_effective_edges", "candidate_edges_before_support_intersection",
            "scored_edges", "modified_edges", "positive_enhancement_edges",
            "negative_suppression_edges", "positive_score_edges", "negative_score_edges",
            "uncertain_score_edges", "direct_positive_increment_edges",
        ):
            edge_summary[modality][key] = int(sum(int(row.get(key, 0)) for row in rows))
        for key in (
            "adjacency_change_relative", "fusion_weighted_gcn_message_raw_l2_ratio",
            "fusion_weighted_dcr_message_raw_l2_ratio", "direct_delta_relative_to_base",
            "q_mean", "q_std", "q_uncertain_fraction", "q_raw_mean", "q_raw_std",
            "q_raw_uncertain_fraction", "q_graph_mean", "q_graph_std",
            "q_graph_uncertain_fraction", "relation_logit_mean", "relation_logit_std",
        ):
            values = [float(row.get(key, 0.0)) for row in rows]
            edge_summary[modality][key + "_fold_mean"] = float(np.mean(values)) if values else None
            edge_summary[modality][key + "_fold_max"] = float(np.max(values)) if values else None
    relation_summary = {}
    for modality in modality_names:
        values = []
        for item in records:
            row = next((record for record in item["graph"]["relation_score_distributions"]
                        if record.get("modality") == modality), None)
            if row:
                values.append(row)
        relation_summary[modality] = {}
        for support_name in (
            "all_patient_pairs", "graph_valid_edges",
            "candidate_edges_before_support_intersection", "actual_scored_candidates",
        ):
            support_rows = [row.get(support_name, {}) for row in values]
            aucs = [float(row["same_different_auc"]) for row in support_rows
                    if row.get("same_different_auc") is not None]
            relation_summary[modality][support_name] = {
                "fold_auc_mean": float(np.mean(aucs)) if aucs else None,
                "pair_count_sum": int(sum(int(row.get("pair_count", 0)) for row in support_rows)),
                "auc_defined_folds": len(aucs),
                "same_pairs_sum": int(sum(int(row.get("same_pairs", 0)) for row in support_rows)),
                "different_pairs_sum": int(sum(int(row.get("different_pairs", 0)) for row in support_rows)),
                **{key + "_fold_mean": float(np.mean([float(row[key]) for row in support_rows
                                                          if row.get(key) is not None]))
                   if any(row.get(key) is not None for row in support_rows) else None
                   for key in ("same_score_mean", "same_score_std", "different_score_mean",
                               "different_score_std", "q_mean", "q_std",
                               "q_uncertain_fraction")},
            }
    relation_sampling = [item["graph"].get("relation_sampling") for item in records]
    return {
        "interventions": aggregate,
        "dpsl_native_full_fold_metrics": variant_summary,
        "gate": {
            "effective_initial_mean": np.mean(np.asarray(gate_initial, dtype=float), axis=0).tolist(),
            "effective_best_mean": gate_best.mean(axis=0).tolist(),
            "effective_best_population_std": gate_best.std(axis=0).tolist(),
            "effective_latest_mean": gate_latest.mean(axis=0).tolist(),
            "effective_latest_population_std": gate_latest.std(axis=0).tolist(),
            "gradient_finite_count": int(sum(
                item["graph"]["gate_gradient_finite_count"] for item in records)),
            "gradient_count": int(sum(item["graph"]["gate_gradient_count"] for item in records)),
            "optimizer_update_count": int(sum(item["graph"]["gate_update_count"] for item in records)),
            "requires_grad_epoch_count": int(sum(
                item["graph"]["gate_requires_grad_epoch_count"] for item in records)),
            "frozen_by_config": bool(records[0]["graph"]["gate_frozen_config"]),
        },
        "graph_edges_by_modality": edge_summary,
        "relation_quality_by_support": relation_summary,
        "relation_sampling_by_fold": relation_sampling,
    }


def main():
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Mechanism replay is pinned to the single assigned GPU")
    runner.configure_worker()
    x, y, ids, splits = runner.load_data()
    splits_by_id = {int(item["fold"]): item for item in splits}
    device = torch.device("cuda:0")
    groups = {}
    failures = []
    for group in GROUPS:
        records = []
        for fold in range(1, 11):
            try:
                record = analyze_fold(group, fold, x, y, ids,
                                     splits_by_id[fold], device)
            except Exception as exc:
                failures.append({"group": group, "fold": fold, "error": repr(exc)})
                print(f"FAIL {group} fold {fold:02d}: {exc!r}", flush=True)
            else:
                records.append(record)
                print(f"PASS mechanism {group} fold {fold:02d}", flush=True)
        groups[group] = {
            "status": "PASS" if len(records) == 10 else "INCOMPLETE",
            "folds": len(records),
            **(aggregate_group(group, records) if records else {}),
            "fold_records": records,
        }
    summary = {
        "experiment": runner.EXPERIMENT,
        "status": "PASS" if not failures and all(
            item["folds"] == 10 for item in groups.values()) else "FAIL",
        "groups": groups,
        "failures": failures,
        "script_sha256": runner.sha_file(Path(__file__)),
        "sealed_training_implementation_sha256": json.loads(
            (ROOT / "audits/training_seal.json").read_text())["implementation_sha256"],
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
        "device": torch.cuda.get_device_name(0),
        "labels_used_for_relation_quality": "post-training diagnostic only",
    }
    (ROOT / "audits/mechanism_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if summary["status"] != "PASS":
        raise SystemExit(1)
    print(json.dumps({"status": summary["status"],
                      "groups": {name: value["folds"]
                                 for name, value in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
