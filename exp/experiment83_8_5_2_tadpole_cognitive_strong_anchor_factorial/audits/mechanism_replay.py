"""Same-Best-checkpoint DARG/DPSL intervention and graph mechanism analysis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as runner

GROUPS = runner.TRAIN_GROUPS
METHODS = (
    "full", "native", "residual_off", "direct_off", "disease_neutral",
    "cognitive_graph_off", "cognitive_score_permuted",
)


def top_margin(logits):
    values = logits.topk(2, dim=1).values
    return values[:, 0] - values[:, 1]


def summarize_intervention(truth, full_pred, changed_pred, full_logits, changed_logits,
                           full_prob, changed_prob):
    full_correct = full_pred == truth
    changed_correct = changed_pred == truth
    logit_delta = np.abs(changed_logits - full_logits)
    prob_delta = np.abs(changed_prob - full_prob)
    full_margin = top_margin(torch.as_tensor(full_logits)).numpy()
    changed_margin = top_margin(torch.as_tensor(changed_logits)).numpy()
    return {
        "test_patients": int(len(truth)),
        "changed_argmax": int(np.sum(changed_pred != full_pred)),
        "corrected_vs_full": int(np.sum(~full_correct & changed_correct)),
        "damaged_vs_full": int(np.sum(full_correct & ~changed_correct)),
        "net_correct_vs_full": int(np.sum(changed_correct) - np.sum(full_correct)),
        "mean_abs_logit_delta": float(logit_delta.mean()),
        "max_abs_logit_delta": float(logit_delta.max(initial=0.0)),
        "mean_abs_probability_delta": float(prob_delta.mean()),
        "max_abs_probability_delta": float(prob_delta.max(initial=0.0)),
        "mean_top2_margin_change_vs_full": float((changed_margin - full_margin).mean()),
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


def graph_record(group, fold, best, latest, fold_audit, config):
    gate_scale = np.asarray(config.get("darg_gate_scales", []), dtype=np.float64)
    gate_trace = fold_audit.get("gate_trace", [])
    initialized = best.get("warmup", {}).get("gate_initialization", {})

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
        "gate_best_effective": get_gates(best),
        "gate_latest_effective": get_gates(latest),
        "gate_scales": gate_scale.tolist(),
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
    full_logits = full["logits"][idx].cpu().numpy()
    full_prob = full["prob"][idx].cpu().numpy()
    full_pred = full["prediction"][idx].cpu().numpy()
    for name in METHODS:
        output = outputs[name]
        logits = output["logits"][idx].cpu().numpy()
        probability = output["prob"][idx].cpu().numpy()
        prediction = output["prediction"][idx].cpu().numpy()
        result_rows[f"{name}_pred"] = prediction
        for class_id, class_name in enumerate(("NC", "MCI", "AD")):
            result_rows[f"{name}_logit_{class_name}"] = logits[:, class_id]
            result_rows[f"{name}_prob_{class_name}"] = probability[:, class_id]
        if name != "full":
            summaries[name] = summarize_intervention(
                truth, full_pred, prediction, full_logits, logits, full_prob, probability)
    result_rows.to_csv(directory / "counterfactuals" / f"fold{fold:02d}.csv", index=False)
    return {
        "group": group, "fold": fold, "status": "PASS",
        "counterfactual_sha256": runner.sha_file(
            directory / "counterfactuals" / f"fold{fold:02d}.csv"),
        "test_count": len(idx),
        "interventions_vs_full": summaries,
        "graph": graph_record(group, fold, best, latest, fold_audit, config),
    }


def aggregate_group(group, records):
    method_names = METHODS[1:]
    aggregate = {}
    for name in method_names:
        keys = (
            "test_patients", "changed_argmax", "corrected_vs_full",
            "damaged_vs_full", "net_correct_vs_full",
        )
        aggregate[name] = {
            key: int(sum(item["interventions_vs_full"][name][key] for item in records))
            for key in keys
        }
        for key in (
            "mean_abs_logit_delta", "max_abs_logit_delta",
            "mean_abs_probability_delta", "max_abs_probability_delta",
            "mean_top2_margin_change_vs_full",
        ):
            values = [item["interventions_vs_full"][name][key] for item in records]
            aggregate[name][key + "_fold_mean"] = float(np.mean(values))
            aggregate[name][key + "_max"] = float(np.max(values))
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
            "negative_suppression_edges",
        ):
            edge_summary[modality][key] = int(sum(int(row.get(key, 0)) for row in rows))
        for key in (
            "adjacency_change_relative", "fusion_weighted_gcn_message_raw_l2_ratio",
            "fusion_weighted_dcr_message_raw_l2_ratio", "direct_delta_relative_to_base",
        ):
            values = [float(row.get(key, 0.0)) for row in rows]
            edge_summary[modality][key + "_fold_mean"] = float(np.mean(values)) if values else None
            edge_summary[modality][key + "_fold_max"] = float(np.max(values)) if values else None
    relation_summary = {}
    for modality in modality_names:
        values = []
        for item in records:
            row = next((record for record in item["graph"]["relation_quality_by_modality"]
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
            }
    return {
        "interventions": aggregate,
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
        },
        "graph_edges_by_modality": edge_summary,
        "relation_quality_by_support": relation_summary,
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
