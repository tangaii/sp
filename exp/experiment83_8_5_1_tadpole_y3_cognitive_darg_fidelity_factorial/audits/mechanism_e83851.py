"""Run fixed-Best DARG and DPSL mechanism probes across all seven groups."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl")
GROUP_SOURCE = {"H0": "R0", "H1": "R1", "H2": "R4"}
PAIRED_GROUPS = (("N1", "H1"), ("N1", "H0"), ("N2", "N1"), ("N3", "N1"),
                 ("N4", "N2"), ("N4", "N3"), ("N4", "H0"), ("N2", "H2"))
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/common")]
import run_e8385 as runner
from source_runtime import darg_sources
RUNNER_GROUP_CONFIG = runner.group_config


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_historical_mechanism():
    sys.path.insert(0, str(HISTORY / "audits"))
    spec = importlib.util.spec_from_file_location(
        "e8385_mechanism_helpers", HISTORY / "audits/mechanism_e8385.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preflight = sys.modules["preflight_e8385"]
    module.ROOT = ROOT
    module.run = runner
    module.darg_sources = darg_sources
    module.require = require
    module.grade_graph.__globals__["run"] = runner
    preflight.run = runner
    return module, preflight


MECHANISM, PREFLIGHT = load_historical_mechanism()
MECHANISM.relation_geometry = None
ACTIVE_RELATION_CONTEXT = None
ALIGNMENT_EVIDENCE = None


def auc(same, scores):
    same = np.asarray(same, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(same) == 0 or len(np.unique(same)) != 2:
        return None
    return float(roc_auc_score(same, scores))


def relative_norm(value, reference):
    return float(value.norm() / reference.norm().clamp_min(1e-12))


def gradient_norm(gradients):
    squared = sum(float(gradient.detach().double().square().sum())
                  for gradient in gradients if gradient is not None)
    return float(np.sqrt(squared))


def alignment_diagnostics(model, aux, pair_same, config):
    modality_parameters = []
    relation_parameters = []
    for encoder, scorer in zip(model.gfr.encoders, model.gfr.scorers, strict=True):
        encoder_parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
        scorer_parameters = [parameter for parameter in scorer.parameters() if parameter.requires_grad]
        modality_parameters.append((encoder_parameters, scorer_parameters))
        relation_parameters.extend(encoder_parameters + scorer_parameters)

    bce_gradients = torch.autograd.grad(
        aux["pair_loss"], relation_parameters, retain_graph=True, allow_unused=True)
    encoders = [parameters[0] for parameters in modality_parameters]
    discriminator_parameters = [parameter for parameter in model.gfr.discriminator.parameters()
                                 if parameter.requires_grad]
    adversarial_parameters = [parameter for group in encoders for parameter in group]
    adversarial_parameters.extend(discriminator_parameters)
    adversarial_gradients = torch.autograd.grad(
        aux["adv_loss"], adversarial_parameters, allow_unused=True)

    bce_by_modality = []
    adversarial_by_modality = []
    offset = 0
    for encoder_parameters, scorer_parameters in modality_parameters:
        encoder_count = len(encoder_parameters)
        scorer_count = len(scorer_parameters)
        bce_encoder_norm = gradient_norm(bce_gradients[offset:offset + encoder_count])
        offset += encoder_count
        bce_scorer_norm = gradient_norm(bce_gradients[offset:offset + scorer_count])
        offset += scorer_count
        bce_by_modality.append({
            "relation_encoder_gradient_norm": bce_encoder_norm,
            "relation_scorer_gradient_norm": bce_scorer_norm,
        })

    offset = 0
    for encoder_parameters, _ in modality_parameters:
        count = len(encoder_parameters)
        adversarial_by_modality.append(gradient_norm(adversarial_gradients[offset:offset + count]))
        offset += count
    discriminator_gradient_norm = gradient_norm(adversarial_gradients[offset:])

    pair_same = pair_same.detach().bool()
    pair_scores = []
    for name, logits in zip(runner.MODAL_NAMES, aux["pair_logits"], strict=True):
        values = logits.detach()
        same_values = values[pair_same]
        different_values = values[~pair_same]
        same_numpy = pair_same.cpu().numpy().astype(np.int8)
        pair_scores.append({
            "modality": name,
            "same_pairs": int(same_values.numel()),
            "different_pairs": int(different_values.numel()),
            "same_logit_mean": float(same_values.mean()),
            "same_logit_std": float(same_values.std(unbiased=False)),
            "different_logit_mean": float(different_values.mean()),
            "different_logit_std": float(different_values.std(unbiased=False)),
            "train_pair_relation_auc": auc(same_numpy, values.cpu().numpy()),
        })

    anchor = config.get("alignment_anchor_index")
    if float(aux["pair_loss"].detach()) == 0.0 or discriminator_gradient_norm <= 0:
        raise RuntimeError("The trained relation BCE or conditional discriminator has no effective gradient")
    if anchor is not None:
        require(adversarial_by_modality[int(anchor)] == 0.0,
                "The configured adversarial anchor received a conditional-alignment encoder gradient")
        require(bce_by_modality[int(anchor)]["relation_encoder_gradient_norm"] > 0
                and bce_by_modality[int(anchor)]["relation_scorer_gradient_norm"] > 0,
                "The detached anchor no longer receives its ordinary relation BCE gradient")

    return {
        "alignment_anchor_index": anchor,
        "grl_scale": float(config.get("darg_grl_scale", 0.03)),
        "pair_loss": float(aux["pair_loss"].detach()),
        "preservation_loss": float(aux["preserve_loss"].detach()),
        "conditional_adversarial_loss": float(aux["adv_loss"].detach()),
        "conditional_discriminator_accuracy": float(aux["adv_info"]["accuracy"].detach()),
        "discriminator_gradient_norm": discriminator_gradient_norm,
        "pair_score_distributions": pair_scores,
        "modality_gradients": [{
            "modality": name,
            **bce_by_modality[index],
            "conditional_adversarial_relation_encoder_gradient_norm": adversarial_by_modality[index],
            "adversarial_encoder_gradient_status": (
                "DETACHED_ANCHOR" if anchor == index else
                "NONZERO" if adversarial_by_modality[index] > 0 else "ZERO"
            ),
        } for index, name in enumerate(runner.MODAL_NAMES)],
        "discriminator_gradient_finite": bool(torch.isfinite(
            torch.as_tensor(discriminator_gradient_norm)).item()),
    }


def relation_geometry(model, parts, test, labels):
    global ALIGNMENT_EVIDENCE
    require(ACTIVE_RELATION_CONTEXT is not None, "Missing fixed-checkpoint relation context")
    x, masked_labels, train, pair, same, config = ACTIVE_RELATION_CONTEXT
    state_before = runner.sha_object(model.state_dict())
    rng_before = runner.sha_object(runner.historical.capture_rng_state())
    with torch.enable_grad():
        _, full_aux = model(x, masked_labels, train, pair, same, dpsl_enabled=True)
        ALIGNMENT_EVIDENCE = alignment_diagnostics(model, full_aux, same, config)
    require(state_before == runner.sha_object(model.state_dict()),
            "Relation diagnostics modified fixed-checkpoint parameters")
    require(rng_before == runner.sha_object(runner.historical.capture_rng_state()),
            "Relation diagnostics consumed RNG state")
    row, col = torch.triu_indices(len(test), len(test), offset=1, device=test.device)
    heldout_pairs = torch.stack((test[row], test[col]), dim=1)
    _, relations, logits = model.gfr._encode_pairs(parts, heldout_pairs)
    same = labels[heldout_pairs[:, 0]] == labels[heldout_pairs[:, 1]]
    same_np = same.cpu().numpy().astype(np.int8)
    original_adjacencies = full_aux["adjacencies"]
    results, propagation = [], []
    raw = torch.cat(parts, dim=1)
    identity = torch.eye(len(raw), dtype=raw.dtype, device=raw.device)
    for index, (name, relation, logit, encoder, scorer, original) in enumerate(zip(
            runner.MODAL_NAMES, relations, logits, model.gfr.encoders, model.gfr.scorers,
            original_adjacencies, strict=True)):
        support = (original[heldout_pairs[:, 0], heldout_pairs[:, 1]] > 0).cpu().numpy()
        same_centroid = relation[same].mean(0)
        different = ~same
        different_centroid = relation[different].mean(0)
        cognitive_same = relations[4][same].mean(0)
        cognitive_different = relations[4][different].mean(0)
        results.append({
            "modality": name,
            "heldout_relation_pairs": int(len(same_np)),
            "same_different_auc_all_pairs": auc(same_np, logit.cpu().numpy()),
            "same_auc": auc(same_np, logit.cpu().numpy()),
            "same_different_auc_within_original_graph_support": auc(same_np[support], logit.cpu().numpy()[support]),
            "same_different_auc_outside_original_graph_support_diagnostic": auc(
                same_np[~support], logit.cpu().numpy()[~support]),
            "original_support_pair_count": int(support.sum()),
            "outside_support_pair_count": int((~support).sum()),
            "outside_support_is_diagnostic_only": True,
            "conditional_centroid_cosine_distance": {
                "same": float(1 - torch.nn.functional.cosine_similarity(
                    same_centroid, cognitive_same, dim=0)),
                "different": float(1 - torch.nn.functional.cosine_similarity(
                    different_centroid, cognitive_different, dim=0)),
            },
            "encoder_sha256": runner.sha_object(encoder.state_dict()),
            "scorer_sha256": runner.sha_object(scorer.state_dict()),
        })

        delta = full_aux["delta_adjacencies"][index]
        positive_delta = delta.clamp_min(0)
        negative_delta = delta.clamp_max(0)
        support = original != 0
        support.fill_diagonal_(False)
        require(not bool((delta[~support] != 0).any()),
                f"{name} signed propagation added unsupported graph edges")
        base_norm = model.gfr.utils.normalize_adj(original + identity)
        positive_norm = model.gfr.utils.normalize_adj(original + positive_delta + identity)
        negative_norm = model.gfr.utils.normalize_adj(original + negative_delta + identity)
        full_norm = model.gfr.utils.normalize_adj(original + delta + identity)
        base_message = model.gfr.gcn[index](raw, base_norm)
        positive_message = model.gfr.gcn[index](raw, positive_norm)
        negative_message = model.gfr.gcn[index](raw, negative_norm)
        full_message = model.gfr.gcn[index](raw, full_norm)
        positive_counterfactual = positive_message - base_message
        negative_counterfactual = negative_message - base_message
        combined_counterfactual = full_message - base_message
        direct = full_aux["direct_messages"][index]
        propagation_row = {
            "modality": name,
            "signed_positive_edges": int(torch.triu(delta > 0, diagonal=1).sum()),
            "signed_negative_edges": int(torch.triu(delta < 0, diagonal=1).sum()),
            "positive_only_gcn_change_relative_to_original_gfr_message": relative_norm(
                positive_counterfactual, base_message),
            "negative_only_gcn_change_relative_to_original_gfr_message": relative_norm(
                negative_counterfactual, base_message),
            "combined_gcn_change_relative_to_original_gfr_message": relative_norm(
                combined_counterfactual, base_message),
            "positive_only_gcn_change_l2": float(positive_counterfactual.norm()),
            "negative_only_gcn_change_l2": float(negative_counterfactual.norm()),
            "direct_positive_dcr_message_relative_to_raw": relative_norm(direct, raw),
            "direct_positive_dcr_message_l2": float(direct.norm()),
            "decomposition_definition": "Fixed-checkpoint counterfactual GCN differences on positive-only and negative-only signed adjacency deltas. These are diagnostics and are not assumed additive after graph normalization.",
        }
        propagation.append(propagation_row)
        results[-1]["positive_negative_propagation"] = propagation_row
    return results


def config_for(group):
    if group in GROUP_SOURCE:
        return json.loads((HISTORY / "configs" / f"{GROUP_SOURCE[group]}.json").read_text())
    if group in ("R0", "R1", "R4"):
        return json.loads((HISTORY / "configs" / f"{group}.json").read_text())
    return RUNNER_GROUP_CONFIG(group)


def analyze_registered_group(group, split, x, y, ids):
    global ACTIVE_RELATION_CONTEXT, ALIGNMENT_EVIDENCE
    source_group = GROUP_SOURCE.get(group, group)
    config = config_for(group)
    source_root = HISTORY if group in GROUP_SOURCE else ROOT
    best_path = source_root / "groups" / source_group / "best_checkpoints" / f"fold{int(split['fold']):02d}.pt"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    test = torch.as_tensor(split["test_idx"], dtype=torch.long, device=x.device)
    train = torch.as_tensor(split["train_idx"], dtype=torch.long, device=x.device)
    masked_labels = y.clone()
    masked_labels[test] = 0
    ACTIVE_RELATION_CONTEXT = (
        x, masked_labels, train, checkpoint["relation_pairs"].to(x.device),
        checkpoint["relation_targets"].to(x.device), config,
    )
    ALIGNMENT_EVIDENCE = None
    old_root = MECHANISM.ROOT
    old_group_config = runner.group_config
    runner.group_config = config_for
    MECHANISM.ROOT = HISTORY if group in GROUP_SOURCE else ROOT

    def grade_registered(_group, model, features, pair, initial=False):
        diagnostic_group = ("R4" if group.startswith("N") else source_group)
        return PREFLIGHT.grade_graph(diagnostic_group, model, features, pair, initial)

    old_grade_graph = MECHANISM.grade_graph
    MECHANISM.grade_graph = grade_registered
    old_relation_geometry = MECHANISM.relation_geometry
    MECHANISM.relation_geometry = relation_geometry
    try:
        row, patients = MECHANISM.analyze_fold(source_group, split, x, y, ids)
        row["alignment_gradient"] = ALIGNMENT_EVIDENCE
    finally:
        runner.group_config = old_group_config
        MECHANISM.ROOT = old_root
        MECHANISM.grade_graph = old_grade_graph
        MECHANISM.relation_geometry = old_relation_geometry
        ACTIVE_RELATION_CONTEXT = None
        ALIGNMENT_EVIDENCE = None
    row["group"] = group
    row["source_group"] = source_group
    row["training_gradient_audit"] = json.loads(
        (HISTORY if group in GROUP_SOURCE else ROOT)
        .joinpath("groups", source_group, "audits", f"fold{int(split['fold']):02d}.json")
        .read_text()
    ).get("graph_gradient_audit_at_best", [])
    patients["group"] = group
    return row, patients


def prediction_effect(before, after, group_before, group_after, mode="native"):
    left = before.sort_values("sample_index")
    right = after.sort_values("sample_index")
    require(np.array_equal(left["sample_index"], right["sample_index"])
            and np.array_equal(left["true_label"], right["true_label"]),
            f"OOF patient order differs for {group_before} and {group_after}")
    truth = left["true_label"].to_numpy(dtype=np.int64)
    prediction_before = left[f"{mode}_pred"].to_numpy(dtype=np.int64)
    prediction_after = right[f"{mode}_pred"].to_numpy(dtype=np.int64)
    result = MECHANISM.differences(truth, prediction_before, prediction_after)
    result.update({"mode": mode, "before": group_before, "after": group_after,
                   "ACC_before": float((prediction_before == truth).mean()),
                   "ACC_after": float((prediction_after == truth).mean()),
                   "AUC_original_before": runner.acc_auc(truth, prediction_before, 3)["AUC"],
                   "AUC_original_after": runner.acc_auc(truth, prediction_after, 3)["AUC"]})
    return result


def main():
    runner.require_training_gate(list(runner.TRAIN_GROUPS))
    runner.configure_worker()
    x_np, y_np, ids, splits = runner.load_data()
    x = torch.as_tensor(x_np, device="cuda:0")
    y = torch.as_tensor(y_np, device="cuda:0")
    groups, summaries, patient_frames = {}, {}, {}
    for group in runner.FORMAL_GROUPS:
        rows, frames = [], []
        for split in splits:
            record, patients = analyze_registered_group(group, split, x, y, ids)
            rows.append(record)
            frames.append(patients)
            print(f"PASS {group} fold {record['fold']:02d} fixed-Best DARG/DPSL audit", flush=True)
        patient_frame = pd.concat(frames, ignore_index=True)
        require(sorted(patient_frame["sample_index"].astype(int)) == list(range(598)),
                f"{group} mechanism output does not cover the 598 patients")
        groups[group] = rows
        patient_frames[group] = patient_frame
        summaries[group] = MECHANISM.group_summary(rows, patient_frame)

    contrasts = {}
    for after, before in PAIRED_GROUPS:
        contrasts[f"{after}-{before}"] = {
            "relation_scorer": MECHANISM.relation_contrast(groups, after, before),
            "native_training_effect": prediction_effect(
                patient_frames[before], patient_frames[after], before, after, "native"),
            "full_model_effect": prediction_effect(
                patient_frames[before], patient_frames[after], before, after, "full"),
        }
    combined_patients = pd.concat(patient_frames.values(), ignore_index=True)
    predictions_path = ROOT / "results/oof_predictions.csv"
    runner.atomic_write(predictions_path, lambda stream: stream.write(
        combined_patients.to_csv(index=False).encode()))
    report = {
        "experiment": runner.EXPERIMENT,
        "status": "PASS",
        "folds": 70,
        "historical_group_folds": 30,
        "new_group_folds": 40,
        "definition": "Interventions compare the same Best checkpoint. Held-out patient labels are used only in evaluation and relation-score diagnostics. Training relations and objectives remain outer-train only.",
        "relation_scorer_auc": "Actual trained DARG RelationEncoder and RelationScorer outputs. AUC is split into original effective graph support and outside-support diagnostic pairs.",
        "positive_negative_graph_message_definition": "Fixed-checkpoint GCN counterfactual differences after applying positive-only and negative-only signed adjacency deltas. Values are diagnostic and need not add because adjacency normalization changes with the edge set.",
        "groups": summaries,
        "fold_evidence": groups,
        "registered_factor_contrasts": contrasts,
        "native_vs_full": "Native is the learned training-effect readout. Full versus Native and fixed-weight ablations are direct inference effects.",
        "patient_output_sha256": runner.sha_file(predictions_path),
        "auditor_program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    runner.write_json(ROOT / "audits/mechanism_summary.json", report)
    print(json.dumps({"status": report["status"], "groups": list(groups),
                      "folds": report["folds"], "oof_prediction_sha256": report["patient_output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
