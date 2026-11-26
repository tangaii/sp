"""Evaluate fixed-checkpoint DARG and DPSL interventions across the locked folds."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as run
from preflight_e8385 import grade_graph, require
from scmr_runtime import confidence_gate, true_class_margin
from source_runtime import darg_sources
from metrics import acc_auc

LABELS = ("NC", "MCI", "AD")
CONTRASTS = (
    ("direct_off", "full"), ("disease_neutral", "full"),
    ("anchor_neutral", "full"), ("alignment_neutral", "full"),
    ("cognitive_graph_off", "full"), ("score_permuted", "full"),
    ("native", "full"), ("residual_off", "full"),
)
SHARED_FAMILIES = ("gfr", "darg", "cam", "head")


@contextmanager
def runtime_controls(model, config, *, alignment, edge, off=False):
    original = model.darg.cadr, model.darg.positive
    scales = tuple(0.0 if off and i == 4 else scale for i, scale in enumerate(config["darg_gate_scales"]))
    model.darg.cadr, model.darg.positive = darg_sources(
        run.MODAL_DIMS, run.MODAL_NAMES, scales, config["darg_direct_scale"],
        config["darg_direct_single_gate"], alignment, edge)
    try:
        yield
    finally:
        model.darg.cadr, model.darg.positive = original


def differences(truth, before, after):
    correct_before = before == truth
    correct_after = after == truth
    corrected = ~correct_before & correct_after
    damaged = correct_before & ~correct_after
    row = {"flips": int((before != after).sum()), "corrected": int(corrected.sum()),
           "damaged": int(damaged.sum()), "net_correct": int(corrected.sum() - damaged.sum())}
    row["classes"] = {name: {"corrected": int((corrected & (truth == index)).sum()),
                             "damaged": int((damaged & (truth == index)).sum())}
                      for index, name in enumerate(LABELS)}
    return row


def relative(first, reference):
    return float((first - reference).norm() / reference.norm().clamp_min(1e-12))


def projected_budget(latest, config):
    rows = latest["gradient_budget_history"]
    require(len(rows) == latest["epoch"] - run.PAIR_WARMUP_EPOCHS,
            "Incomplete historical SCMR gradient budget measurements")
    cap = config["dpsl_shared_cap"]
    max_ratio = dict.fromkeys(SHARED_FAMILIES, 0.0)
    nonzero_epochs = 0
    for row in rows:
        require(row["shared_cap"] == cap, "DPSL configured gradient cap changed")
        nonzero = False
        for family in SHARED_FAMILIES:
            value = row["families"][family]
            added = value["total_applied"]
            base = value["historical"]
            require(np.isfinite([added, base, value["applied_dot"], value["dot_bound"]]).all()
                    and 0 <= added <= cap * base + 1e-15
                    and abs(value["applied_dot"]) <= value["dot_bound"] + 1e-15,
                    f"DPSL {family} gradient budget violated at epoch {row['epoch']}")
            max_ratio[family] = max(max_ratio[family], added / base if base else 0.0)
            nonzero |= added > 0
        nonzero_epochs += nonzero
    require(nonzero_epochs > 0, "DPSL never applied a shared gradient")
    return {"epochs": len(rows), "epochs_nonzero": nonzero_epochs, "cap": cap,
            "max_applied_to_historical": max_ratio}


def relation_geometry(model, parts, test, labels):
    row, col = torch.triu_indices(len(test), len(test), offset=1, device=test.device)
    heldout_pairs = torch.stack((test[row], test[col]), dim=1)
    _, relations, logits = model.gfr._encode_pairs(parts, heldout_pairs)
    same = labels[heldout_pairs[:, 0]] == labels[heldout_pairs[:, 1]]
    same_cpu = same.cpu().numpy()
    results = []
    for name, relation, logit, encoder, scorer in zip(
            run.MODAL_NAMES, relations, logits, model.gfr.encoders, model.gfr.scorers, strict=True):
        require(bool(torch.isfinite(logit).all()) and bool(torch.isfinite(relation).all()),
                f"Nonfinite held-out relation outputs for {name}")
        distances = {}
        for relation_name, mask in (("same", same), ("different", ~same)):
            centroid = relation[mask].mean(0)
            cognitive_centroid = relations[4][mask].mean(0)
            distances[relation_name] = float(
                1 - F.cosine_similarity(centroid, cognitive_centroid, dim=0))
        results.append({"modality": name, "vv_pairs": int(len(same)),
                        "same_auc": float(roc_auc_score(same_cpu, logit.cpu().numpy())),
                        "conditional_centroid_cosine_distance": distances,
                        "encoder_sha256": run.sha_object(encoder.state_dict()),
                        "scorer_sha256": run.sha_object(scorer.state_dict())})
    return results


@torch.no_grad()
def analyze_fold(group, split, x, y, ids):
    fold = int(split["fold"])
    group_dir = ROOT / "groups" / group
    name = f"fold{fold:02d}"
    audit = json.loads((group_dir / "audits" / f"{name}.json").read_text())
    require(audit["status"] == "TRAINED", f"{group}:{fold} lacks formal execution")
    best_path = group_dir / "best_checkpoints" / f"{name}.pt"
    latest_path = group_dir / "latest_checkpoints" / f"{name}.pt"
    require(run.sha_file(best_path) == audit["checkpoint_sha256"]
            and run.sha_file(latest_path) == audit["latest_sha256"], "Mechanism checkpoint SHA differs")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    config = run.group_config(group)
    train = torch.as_tensor(split["train_idx"], device=x.device)
    test = torch.as_tensor(split["test_idx"], device=x.device)
    y_masked = y.clone()
    y_masked[test] = 0
    pair = best["relation_pairs"].to(x.device)
    same = best["relation_targets"].to(x.device)
    parts = list(x.split(run.MODAL_DIMS, dim=1))
    run.historical.seed_everything(run.SEED)
    model = run.build_model(config, fold, x.device, best["spectral_stats"])
    model.load_state_dict(best["model"], strict=True)
    model.darg.enter_positive_dcr()
    model.eval()
    initial_state = run.sha_object(model.state_dict())

    def evaluate(**kwargs):
        return model(x, y_masked, train, pair, same, **kwargs)

    alignment, edge = config["alignment_anchor_index"], config["edge_anchor_index"]
    original, graph = grade_graph(group, model, x, pair)
    cases = {
        "full": evaluate(),
        "direct_off": evaluate(darg_intervention="direct_off"),
        "disease_neutral": evaluate(darg_intervention="disease_neutral"),
        "native": evaluate(dpsl_enabled=False),
        "score_permuted": evaluate(darg_intervention="score_permuted"),
    }
    with runtime_controls(model, config, alignment=None, edge=edge):
        cases["alignment_neutral"] = evaluate()
    with runtime_controls(model, config, alignment=None, edge=None):
        cases["anchor_neutral"] = evaluate()
    with runtime_controls(model, config, alignment=alignment, edge=edge, off=True):
        cases["cognitive_graph_off"] = evaluate()
    model.dpsl._confidence_gate_override = 0.0
    try:
        cases["residual_off"] = evaluate()
    finally:
        model.dpsl._confidence_gate_override = None

    full_logp, full = cases["full"]
    neutral = cases["disease_neutral"][1]
    direct_off = cases["direct_off"][1]
    anchor_neutral = cases["anchor_neutral"][1]
    cognitive_off = cases["cognitive_graph_off"][1]
    require(torch.equal(cases["alignment_neutral"][0], full_logp), "Training-only anchor affected inference")
    require(torch.equal(cases["residual_off"][0], cases["native"][0]), "Residual-Off differs from Native")
    require(torch.equal(full["cam"], cases["native"][1]["cam"])
            and torch.equal(full["cam"], cases["residual_off"][1]["enhanced_cam"]),
            "DPSL isolation changed the upstream CAM")
    require(len(model.dpsl.spectrum.query_medical) == 12
            and not isinstance(model.dpsl.spectrum.query_medical, torch.nn.Parameter),
            "DPSL frozen MMGPL-12 identity differs")

    runtime_gate = full["dpsl"]["confidence_gate"]
    require(torch.equal(runtime_gate, confidence_gate(full["native_logits"], "old"))
            and not runtime_gate.requires_grad, "Historical DPSL confidence gate changed")
    residual = full["dpsl"]["residual"]
    ratio = full["dpsl"]["residual_ratio"]
    require(bool(torch.isfinite(residual).all()) and bool(torch.isfinite(ratio).all())
            and float(ratio.max()) <= .08 + 1e-7 and float(residual.norm()) > 0,
            "DPSL residual is absent, nonfinite, or exceeds the cap")
    require(run.sha_object(model.state_dict()) == initial_state, "Inference modified the Best model state")
    budget = projected_budget(latest, config)

    graph_rows = graph["edge_diagnostics"]
    for i, adjacency in enumerate(original):
        message = full["messages"][i]
        require(torch.equal(adjacency, neutral["unified_adjacencies"][i])
                and torch.equal(full["unified_adjacencies"][i], direct_off["unified_adjacencies"][i])
                and torch.equal(message, direct_off["messages"][i])
                and not bool(direct_off["direct_messages"][i].count_nonzero())
                and not bool(neutral["direct_messages"][i].count_nonzero()),
                "DARG intervention affects an unrelated propagation path")
        gate = graph["gates"][i]
        require(torch.equal(graph["direct_messages"][i], .02 * gate * graph["disease_residuals_bounded"][i])
                and not bool((graph["direct_delta_adjs"][i] < 0).any())
                and float(gate) <= config["darg_gate_scales"][i],
                "Positive-DCR double-gate or gate range differs")
        for key in ("unified_adjacencies", "messages", "direct_messages"):
            if i != 4:
                require(torch.equal(full[key][i], anchor_neutral[key][i])
                        and torch.equal(full[key][i], cognitive_off[key][i]),
                        "Cognitive intervention affected another modality at the same checkpoint")
        if i == 4:
            require(torch.equal(cognitive_off["unified_adjacencies"][i], adjacency)
                    and torch.equal(cognitive_off["messages"][i], neutral["messages"][i])
                    and not bool(cognitive_off["direct_messages"][i].count_nonzero()),
                    "Cognitive off did not preserve the original graph and GCN")
        diag = graph_rows[i]
        diag["learned_gate_probability"] = float(torch.sigmoid(model.gfr.adaptive_gate_logits[i]))
        diag["gcn_relative_change_vs_disease_neutral"] = relative(message, neutral["messages"][i])
        diag["gcn_relative_change_vs_anchor_neutral"] = relative(message, anchor_neutral["messages"][i])
        diag["candidate_to_all_patient_pairs"] = diag["candidate_edges_before_support_intersection"] / 178503
        diag["scored_to_candidate"] = diag["scored_edges"] / max(
            diag["candidate_edges_before_support_intersection"], 1)

    truth = y[test].cpu().numpy()
    patient = {"group": group, "fold": fold, "sample_index": test.cpu().numpy(),
               "sample_id": ids.iloc[split["test_idx"]]["RID"].to_numpy(), "true_label": truth,
               "dpsl_confidence_gate": runtime_gate[test].cpu().numpy(),
               "dpsl_residual_ratio": ratio[test].cpu().numpy()}
    pred, metrics, margins = {}, {}, {}
    for key, (logp, aux) in cases.items():
        require(bool(torch.isfinite(logp).all()), f"Nonfinite {key} log probabilities")
        pred[key] = logp[test].argmax(-1).cpu().numpy()
        prob = logp[test].exp().cpu().numpy()
        margins[key] = true_class_margin(aux["logits"][test], y[test])
        metric = acc_auc(truth, pred[key], 3)
        metrics[key] = {"ACC": metric["ACC"], "AUC_original": metric["AUC"],
                        "AUC_probability": float(roc_auc_score(np.eye(3)[truth], prob, average="macro"))}
        patient[f"{key}_pred"] = pred[key]
        patient[f"{key}_true_margin"] = margins[key].cpu().numpy()
        if key in ("full", "native"):
            for index, label in enumerate(LABELS):
                patient[f"{key}_prob_{label}"] = prob[:, index]
    saved = pd.read_csv(group_dir / "predictions" / f"{name}.csv")
    require(np.array_equal(pred["full"], saved["pred_label"].to_numpy())
            and np.array_equal(pred["native"], saved["native_pred_label"].to_numpy()),
            "Mechanism Full or Native differs from independent fold prediction")
    for key, prefix in (("full", ""), ("native", "native_")):
        require(all(np.array_equal(patient[f"{key}_prob_{label}"],
                                   saved[f"{prefix}prob_{label}"].to_numpy(dtype=np.float32))
                    for label in LABELS), "Mechanism probability differs from fold prediction")

    contrasts = {}
    for before, after in CONTRASTS:
        a_logp, a_aux = cases[before]
        b_logp, b_aux = cases[after]
        delta_margin = (margins[after] - margins[before]).cpu().numpy()
        contrast = differences(truth, pred[before], pred[after])
        contrast.update(
            max_abs_logit_delta=float((b_aux["logits"][test] - a_aux["logits"][test]).abs().max()),
            max_abs_prob_delta=float((b_logp[test].exp() - a_logp[test].exp()).abs().max()),
            cam_relative_delta=relative(b_aux["cam"][test], a_aux["cam"][test]),
            enhanced_cam_relative_delta=relative(b_aux["enhanced_cam"][test], a_aux["enhanced_cam"][test]),
            mean_true_margin_delta=float(delta_margin.mean()),
            mean_abs_true_margin_delta=float(np.abs(delta_margin).mean()),
        )
        for index, label in enumerate(LABELS):
            contrast["classes"][label]["mean_true_margin_delta"] = float(delta_margin[truth == index].mean())
        contrasts[f"{before}_to_{after}"] = contrast
    require(run.sha_file(best_path) == audit["checkpoint_sha256"]
            and run.sha_file(latest_path) == audit["latest_sha256"], "Mechanism audit changed checkpoint bytes")
    train_class_nll = {key: float(F.nll_loss(cases[key][0][train], y[train]))
                       for key in ("full", "score_permuted", "disease_neutral")}
    require(np.isfinite(list(train_class_nll.values())).all(), "Nonfinite training-set NLL in mechanism audit")
    return {"status": "PASS", "fold": fold, "best_sha256": audit["checkpoint_sha256"],
            "graph_modalities": graph_rows, "metrics": metrics, "contrasts": contrasts,
            "train_class_nll_eval_mode": train_class_nll,
            "relation_geometry": relation_geometry(model, parts, test, y),
            "dpsl": {"test_gate_mean": float(runtime_gate[test].mean()), "residual_max": float(ratio.max()),
                     "test_residual_mean": float(ratio[test].mean()), "shared_gradient_budget": budget},
            "parameter_state_unchanged": True}, pd.DataFrame(patient)


def group_summary(rows, patients):
    index = {(before, after): f"{before}_to_{after}" for before, after in CONTRASTS}
    contrasts = {}
    truth = patients["true_label"].to_numpy()
    for (before, after), key in index.items():
        result = differences(truth, patients[f"{before}_pred"].to_numpy(), patients[f"{after}_pred"].to_numpy())
        for name in ("max_abs_logit_delta", "max_abs_prob_delta"):
            result[name] = max(row["contrasts"][key][name] for row in rows)
        for name in ("cam_relative_delta", "enhanced_cam_relative_delta", "mean_abs_true_margin_delta"):
            result[f"{name}_mean"] = float(np.mean([row["contrasts"][key][name] for row in rows]))
        contrasts[key] = result
    nll_comparisons = {}
    for control in ("score_permuted", "disease_neutral"):
        deltas = np.array([row["train_class_nll_eval_mode"]["full"] -
                           row["train_class_nll_eval_mode"][control] for row in rows])
        nll_comparisons[control] = {"full_minus_control_mean": float(deltas.mean()),
                                    "full_lower_nll_folds": int((deltas < 0).sum())}
    return {"folds": len(rows), "graph_means": [{
        "modality": name, **{
            key: float(np.mean([row["graph_modalities"][i][key] for row in rows]))
            for key in ("original_effective_edges", "candidate_edges_before_support_intersection", "scored_edges",
                        "modified_edges", "positive_score_edges", "negative_score_edges", "q_mean", "q_std",
                        "gate_effective", "learned_gate_probability", "adjacency_change_relative",
                        "gcn_relative_change_vs_disease_neutral", "gcn_relative_change_vs_anchor_neutral",
                        "fusion_weighted_gcn_message_raw_l2_ratio", "fusion_weighted_dcr_message_raw_l2_ratio",
                        "effective_edge_coverage", "candidate_to_all_patient_pairs", "scored_to_candidate")}}
        for i, name in enumerate(run.MODAL_NAMES)],
        "contrasts": contrasts,
        "train_class_nll_eval_mode": nll_comparisons,
        "native_correct": int((patients["native_pred"].to_numpy() == truth).sum()),
        "full_correct": int((patients["full_pred"].to_numpy() == truth).sum()),
        "dpsl_residual_ratio_max": max(row["dpsl"]["residual_max"] for row in rows),
        "dpsl_shared_ratio_max": max(max(row["dpsl"]["shared_gradient_budget"]["max_applied_to_historical"].values())
                                     for row in rows)}


def relation_contrast(groups, after, before):
    summary = {}
    for index, name in enumerate(run.MODAL_NAMES):
        b = [row["relation_geometry"][index] for row in groups[before]]
        a = [row["relation_geometry"][index] for row in groups[after]]
        summary[name] = {
            "encoder_changed_folds": sum(x["encoder_sha256"] != y["encoder_sha256"] for x, y in zip(a, b)),
            "scorer_changed_folds": sum(x["scorer_sha256"] != y["scorer_sha256"] for x, y in zip(a, b)),
            "mean_same_auc_delta": float(np.mean([x["same_auc"] - y["same_auc"] for x, y in zip(a, b)])),
            "mean_conditioned_cosine_distance_delta": {
                label: float(np.mean([x["conditional_centroid_cosine_distance"][label]
                                      - y["conditional_centroid_cosine_distance"][label] for x, y in zip(a, b)]))
                for label in ("same", "different")},
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0-smoke", action="store_true")
    args = parser.parse_args()
    run.require_training_gate(run.TRAIN_GROUPS)
    run.configure_worker()
    x_np, y_np, ids, splits = run.load_data()
    x, y = (torch.as_tensor(array, device="cuda:0") for array in (x_np, y_np))
    if args.r0_smoke:
        row, _ = analyze_fold("R0", splits[0], x, y, ids)
        print(f"PASS R0 mechanism fold {row['fold']}")
        return
    groups, summaries, frames = {}, {}, []
    for group in run.FORMAL_GROUPS:
        group_frames = []
        groups[group] = []
        for split in splits:
            row, frame = analyze_fold(group, split, x, y, ids)
            groups[group].append(row)
            group_frames.append(frame)
            print(f"PASS {group} fold {split['fold']:02d} DARG/DPSL fixed interventions", flush=True)
        patients = pd.concat(group_frames, ignore_index=True)
        require(sorted(patients["sample_index"]) == list(range(598)), f"{group} invalid mechanism cohort")
        summaries[group] = group_summary(groups[group], patients)
        frames.append(patients)
    patient_path = ROOT / "results/oof_predictions.csv"
    run.atomic_write(patient_path, lambda stream: stream.write(pd.concat(frames, ignore_index=True).to_csv(index=False).encode()))
    run.write_json(ROOT / "audits/mechanism_summary.json", {
        "experiment": run.EXPERIMENT, "status": "PASS", "folds": 50,
        "GRAPH_ANCHOR_AUDIT": "PASS", "DARG_MECHANISM_AUDIT": "PASS", "DPSL_MECHANISM_AUDIT": "PASS",
        "definition": "Every contrast is before→after on the same checkpoint. Held-out labels enter metrics only.",
        "alignment_neutral": "Remove only the adversarial reference assignment at inference.",
        "anchor_neutral": "Remove the reference assignment and replace any cognitive special edge rule with ordinary mutual-topK.",
        "cognitive_graph_off": "Zero only the cognitive disease gate, retaining its original GraphLearn adjacency and GCN.",
        "score_permuted": "Permute q values on each modality's original scored support using the historical seeded intervention.",
        "train_class_nll_eval_mode": "Recompute classification NLL over outer training rows using the same Best in eval mode.",
        "groups": summaries, "fold_evidence": groups,
        "heldout_relation_geometry_contrasts": {
            "R1-R0": relation_contrast(groups, "R1", "R0"), "R3-R2": relation_contrast(groups, "R3", "R2")},
        "patient_audit_sha256": run.sha_file(patient_path), "program_sha256": run.sha_file(Path(__file__)),
    })


if __name__ == "__main__":
    main()
