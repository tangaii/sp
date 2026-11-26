"""Run the locked Experiment 83.8.4 pre-training checks."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "common"))
import run_e8384 as runner
from source_runtime import E74_GROUP, TADPOLE_MODALITIES
from scmr_runtime import confidence_gate

Y3_ROOT = ROOT.parent / "experiment83_8_3_tadpole_n3_darg_dpsl_optimization"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_splits(x, labels, splits):
    expected = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, labels))
    check(len(splits) == 10, "The locked split file must contain ten folds")
    for saved, (train, test) in zip(sorted(splits, key=lambda value: int(value["fold"])), expected):
        check(np.array_equal(np.asarray(saved["train_idx"], dtype=np.int64), train),
              f"Fold {saved['fold']} train indices differ from the locked CV rule")
        check(np.array_equal(np.asarray(saved["test_idx"], dtype=np.int64), test),
              f"Fold {saved['fold']} test indices differ from the locked CV rule")
        check(saved.get("val_equals_test") is True, f"Fold {saved['fold']} val and test differ")
        check(not np.intersect1d(train, test).size, f"Fold {saved['fold']} leaks train/test rows")


def check_prompts():
    prompts = torch.load(ROOT / "inputs/prompts/mmGPL12_class_rn50.pt", map_location="cpu", weights_only=False)
    scmr = torch.load(ROOT / "inputs/prompts/mmGPL12_class_scmr.pt", map_location="cpu", weights_only=False)
    check(prompts["medical"].shape == (12, 1024), "Frozen RN50 prompt vectors must be 12 x 1024")
    check(len(prompts["texts"]) == 12 and len(scmr["texts"]) == 12, "MMGPL-12 must contain twelve texts")
    check(scmr["q_medical"].shape == (12, 32), "DPSL prompt projection must retain twelve vectors")
    check(scmr["P_text"].shape == (1024, 32), "DPSL frozen text projection shape differs")
    check(list(prompts["groups"].keys()) == ["NC", "MCI", "AD"], "Prompt groups differ from NC/MCI/AD")
    check(all(len(value) == 4 for value in prompts["groups"].values()), "Each class must have four prompts")
    return {"rn50_shape": list(prompts["medical"].shape), "scmr_query_shape": list(scmr["q_medical"].shape),
            "texts": len(prompts["texts"]), "group_counts": {key: len(value) for key, value in prompts["groups"].items()},
            "rn50_sha256": sha256(ROOT / "inputs/prompts/mmGPL12_class_rn50.pt"),
            "scmr_sha256": sha256(ROOT / "inputs/prompts/mmGPL12_class_scmr.pt")}


def initial_group_contracts(device):
    identities = {}
    configs = {}
    models = {}
    optimizers_by_group = {}
    prompt_hashes = {}
    for group in runner.FORMAL_GROUPS:
        config = runner.group_config(group)
        scales = tuple(config["darg_gate_scales"])
        expected_scale = .01
        check(scales == (expected_scale,) * 6, f"{group} gate scales differ from the frozen design")
        check(config["supervision_kind"] == "paper_aloss"
              and config["supervision_coefficient"] == .01
              and config["graph_supervision_schedule"] == "epoch_11_onward",
              f"{group} does not preserve the N3 ALoss schedule")
        check(config["dpsl_shared_cap"] == .001 and config["dpsl_residual_cap"] == .08,
              f"{group} DPSL caps differ from N3")
        check(config["max_epochs"] == runner.MAX_EPOCHS == 450
              and config["early_stop_wait_gt"] == runner.PATIENCE == 200,
              f"{group} epoch protocol differs")
        check(config["base_reference"] == "H4" and
              config["training"] is (group != "Z0"), f"{group} training registration differs")
        check(config["dpsl_confidence_protection"] is True, f"{group} lost DPSL protection")
        expected_direct = {
            "Z0": (.02, False, "old"), "Z1": (.02, True, "old"),
            "Z2": (.10, True, "old"), "Z3": (.02, False, "multiclass"),
            "Z4": (.02, True, "multiclass"), "Z5": (.10, True, "multiclass"),
        }[group]
        check((float(config["darg_direct_scale"]), bool(config["darg_direct_single_gate"]),
               config["dpsl_confidence_mode"]) == expected_direct,
              f"{group} registered factorial fields differ")
        prompt_path = runner.prompt_path_for(config, runner.BASE_GROUPS[config["base_reference"]])
        prompt_hashes[group] = runner.sha_file(prompt_path)
        runner.historical.seed_everything(runner.SEED)
        model = runner.build_model(config, 1, device)
        optimizers, optimizer_config = runner.make_optimizers(model, config)
        check(model.use_darg and model.use_dpsl and model.nclass == 3, f"{group} lost DARG or DPSL")
        check(tuple(model.darg.modalities) == TADPOLE_MODALITIES, f"{group} modality identity differs")
        check(tuple(model.darg.gate_scales) == scales, f"{group} runtime gate binding differs")
        check(model.darg.core.direct_scale == expected_direct[0]
              and model.darg.direct_single_gate is expected_direct[1]
              and model.darg.core.negative_scale == .25,
              f"{group} DARG direct or negative scale differs")
        check(model.darg.core.edge_topk == 12, f"{group} relation candidate top-k differs")
        check(all(float(graph.th) == .9 for graph in model.gfr.graphs),
              f"{group} original GraphLearn threshold differs from 0.9")
        check(len(model.gfr.graphs) == 6, f"{group} must contain six GraphLearn modules")
        check(optimizer_config["gfr"] == [.012, .11]
              and optimizer_config["cam"] == [.012, .11]
              and optimizer_config["darg"] == [.001, .01],
              f"{group} historical optimizer settings differ")
        check(tuple(model.state_dict()) == tuple(runner.build_model(
            runner.group_config("Z0"), 1, device).state_dict()),
            f"{group} introduces or removes model state")
        identity = runner.initial_warmup_identity(model, optimizers)
        identities[group] = identity
        configs[group] = config
        models[group] = model
        optimizers_by_group[group] = optimizers
    reference = identities["Z0"]
    for group, identity in identities.items():
        check(identity == reference, f"{group} initial model, optimizer or RNG differs from Z0")
    check(len(set(prompt_hashes.values())) == 1,
          "The six group configs do not use the same original twelve DPSL prompt vectors")
    return configs, models, optimizers_by_group, identities, prompt_hashes


def darg_isolation_probe(configs, models, x_np, y_np, splits, device):
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    split = next(item for item in splits if int(item["fold"]) == 1)
    train_rows = np.asarray(split["train_idx"], dtype=np.int64)
    train_t = torch.from_numpy(train_rows).long().to(device)
    y0 = models["Z0"]
    supports = runner.historical.anchor_pair_support(y0, x)
    pair_np, same_np = runner.historical.make_pairs(train_rows, y_np, 1, supports)
    pair = torch.from_numpy(pair_np).long().to(device)
    same = torch.from_numpy(same_np).float().to(device)
    base_state = {key: value.detach().clone() for key, value in y0.state_dict().items()}
    parts = []
    start = 0
    for dim in runner.MODAL_DIMS:
        parts.append(x[:, start:start + dim])
        start += dim

    def evaluate(group):
        model = models[group]
        model.load_state_dict(base_state, strict=True)
        model.eval()
        with torch.no_grad():
            logp, aux = model(x, labels, train_t, pair, same, dpsl_enabled=False)
        return logp.detach().clone(), [item.detach().clone() for item in aux["unified_adjacencies"]], aux

    y0_first, graph_y0_first, y0_aux = evaluate("Z0")
    direct_z0 = [item.detach().clone() for item in y0_aux["direct_messages"]]
    evaluate("Z1")
    y0_after_y1, graph_y0_after_y1, _ = evaluate("Z0")
    check(torch.equal(y0_first, y0_after_y1), "Z0->Z1->Z0 changed Z0 output")
    check(all(torch.equal(a, b) for a, b in zip(graph_y0_first, graph_y0_after_y1)),
          "Z0->Z1->Z0 changed Z0 graph tensors")
    _, graph_z1, aux_z1 = evaluate("Z1")
    check(all(torch.equal(a, b) for a, b in zip(graph_y0_first, graph_z1)),
          "Single-gate DARG changed unified adjacency")
    check(any(not torch.equal(a, b) for a, b in zip(direct_z0, aux_z1["direct_messages"])),
          "Single-gate DARG did not remove the second direct gate")
    evaluate("Z2")
    y0_after_y2, graph_y0_after_y2, _ = evaluate("Z0")
    check(torch.equal(y0_first, y0_after_y2), "Z0->Z2->Z0 changed Z0 output")
    check(all(torch.equal(a, b) for a, b in zip(graph_y0_first, graph_y0_after_y2)),
          "Z0->Z2->Z0 changed Z0 graph tensors")

    audit_model = models["Z2"]
    audit_model.load_state_dict(base_state, strict=True)
    with torch.no_grad():
        _, original, details = audit_model.darg.positive.rudag_forward(
            audit_model.darg.core, parts, pair, E74_GROUP, fold=1, collect_diagnostics=True)
    names = list(audit_model.darg.modalities)
    runtime_globals = audit_model.darg.positive.rudag_forward.__globals__
    check(names == list(TADPOLE_MODALITIES) and "CORR" not in names,
          "TADPOLE modalities must not map RISK_FACTOR to CORR")
    check(runtime_globals.get("CORR_INDEX") is None,
          "A CORR-only branch is active for this six-modality TADPOLE binding")
    check(len(details["gates"]) == 6 and len(details["propagation_modes"]) == 6
          and len(details["edge_diagnostics"]) == 6,
          "DARG mechanism diagnostics do not include all six modalities")
    risk = details["edge_diagnostics"][3]
    check(risk["modality"] == "RISK_FACTOR" and risk["corr_effective_edge_coverage"] is None
          and risk["corr_graph_temperature"] == 1.0,
          "RISK_FACTOR is incorrectly receiving CORR-specific processing")
    selected, _, _ = audit_model.darg.positive._original_mutual_topk(
        audit_model.darg.core, 3, parts[3], original[3])
    expected_candidates = int(torch.triu(selected, diagonal=1).sum().item())
    check(risk["candidate_edges_before_support_intersection"] == expected_candidates,
          "RISK_FACTOR does not use the generic relation candidate rule")
    check(details["direct_scale"] == .10 and details["positive_only_direct"] is True,
          "Moderate single-gate Positive-DCR direct path differs")
    check(all(torch.isfinite(value).all() for value in details["unified_adjs"]),
          "DARG unified graph contains non-finite values")
    direct_z2 = details["direct_messages"]
    check(all(torch.isfinite(value).all() for value in direct_z2),
          "Moderate single-gate direct residual contains non-finite values")
    check(all(torch.allclose(a * 5.0, b, rtol=2e-5, atol=1e-10)
              for a, b in zip(aux_z1["direct_messages"], direct_z2)),
          "Z2 direct residual is not exactly the registered five-fold Z1 scale")
    cam_dimensions = int(y0_aux["cam"].shape[1])
    check(cam_dimensions == 240, "The locked CAM representation must contain 240 dimensions")
    return {
        "status": "PASS",
        "cam_dimensions": cam_dimensions,
        "modalities": names,
        "gate_scales_by_group": {group: list(model.darg.gate_scales) for group, model in models.items()},
        "isolation": {"Z0_Z1_Z0_exact": True, "Z0_Z2_Z0_exact": True,
                       "unified_adjacency_exact_Z0_Z1": True,
                       "second_gate_removed_Z1": True, "direct_scale_fivefold_Z2": True},
        "diagnostic_modalities": [item["modality"] for item in details["edge_diagnostics"]],
        "risk_factor": {key: risk[key] for key in ("original_effective_edges", "candidate_edges_before_support_intersection",
                                                     "scored_edges", "positive_score_edges", "negative_score_edges",
                                                     "gate_effective", "corr_graph_temperature")},
        "corr_index": runtime_globals.get("CORR_INDEX"),
    }, (pair, same, train_t, train_rows)


def dpsl_probe(config, x, y, split, checkpoint, pair, same, train_t, device):
    config_protected = config["Z3"]
    config_native = config["Z0"]
    stats = checkpoint["spectral_stats"]
    protected = runner.build_model(config_protected, 1, device, spectral_stats=stats)
    protected.load_state_dict(checkpoint["model"], strict=True)
    native = runner.build_model(config_native, 1, device, spectral_stats=stats)
    native.load_state_dict(checkpoint["model"], strict=True)
    protected.eval()
    native.eval()
    labels = torch.from_numpy(y).long().to(device)
    with torch.no_grad():
        _, native_aux = native(x, labels, train_t, pair, same, dpsl_enabled=False)
        _, full_aux = native(x, labels, train_t, pair, same, dpsl_enabled=True)
        _, guarded_aux = protected(x, labels, train_t, pair, same, dpsl_enabled=True)
    native_cam = native_aux["cam"]
    protected_cam = guarded_aux["enhanced_cam"]
    check(torch.equal(protected_cam, guarded_aux["cam"] + guarded_aux["dpsl"]["delta"]),
          "Protected DPSL CAM and reported residual differ")
    check(torch.equal(guarded_aux["logits"], protected.cam.raw_logits_from_cam(protected_cam)),
          "Protected Full logits do not use the gated CAM")
    check(float(guarded_aux["dpsl"]["confidence_gate"].min()) >= 0.0
          and float(guarded_aux["dpsl"]["confidence_gate"].max()) <= 1.0,
          "DPSL confidence gate is outside [0, 1]")
    check(float(guarded_aux["dpsl"]["ratio"].max()) <= .08 + 1e-7,
          "Protected DPSL CAM residual exceeds 0.08")
    uniform = torch.full((4, 3), 1.0 / 3.0, device=device)
    old_uniform = confidence_gate(uniform.log(), mode="old")
    new_uniform = confidence_gate(uniform.log(), mode="multiclass")
    check(torch.allclose(old_uniform, torch.full_like(old_uniform, 4.0 / 9.0), atol=1e-6),
          "Old multiclass gate does not reproduce 4/9 at uniform probabilities")
    check(torch.allclose(new_uniform, torch.ones_like(new_uniform), atol=1e-6),
          "New multiclass gate does not reproduce one at uniform probabilities")
    binary_logits = torch.tensor([[0.3, -0.4], [2.0, -1.0]], device=device)
    check(torch.equal(confidence_gate(binary_logits, mode="old"),
                      confidence_gate(binary_logits, mode="multiclass")),
          "Binary confidence compatibility branch changed")
    protected.dpsl._confidence_gate_override = 1.0
    with torch.no_grad():
        _, gate_one = protected(x, labels, train_t, pair, same, dpsl_enabled=True)
    check(torch.equal(gate_one["dpsl"]["delta"], gate_one["dpsl"]["delta_before_confidence"])
          and torch.equal(gate_one["enhanced_cam"], gate_one["cam"] + gate_one["dpsl"]["delta_before_confidence"]),
          "g=1 does not exactly recover the unprotected new-formula DPSL output")
    protected.dpsl._confidence_gate_override = 0.0
    with torch.no_grad():
        _, gate_zero = protected(x, labels, train_t, pair, same, dpsl_enabled=True)
    check(torch.equal(gate_zero["enhanced_cam"], gate_zero["cam"])
          and torch.equal(gate_zero["logits"], gate_zero["native_logits"]),
          "g=0 does not exactly recover Native output")
    protected.dpsl._confidence_gate_override = None
    permutation = torch.roll(labels, shifts=1)
    with torch.no_grad():
        permuted_logp, _ = protected(x, permutation, train_t, pair, same, dpsl_enabled=True)
        guarded_logp, _ = protected(x, labels, train_t, pair, same, dpsl_enabled=True)
    check(torch.equal(permuted_logp, guarded_logp), "Test-label permutation changed protected inference")

    protected.train()
    optimizers, _ = runner.make_optimizers(protected, config_protected, formal=True)
    runner.zero_optimizers(optimizers)
    logp, aux = protected(x, labels, train_t, pair, same, dpsl_enabled=True)
    _, total, _, _ = runner.historical_loss(
        aux, labels, train_t, protected, "paper_aloss", .01)
    check(torch.isfinite(total), "Protected DPSL training loss is non-finite")
    budget = runner.dpsl_optimizer_step(protected, optimizers, total, aux["scmr_objectives"], .001)
    check(all(torch.isfinite(parameter).all() for parameter in protected.parameters()),
          "Protected DPSL optimization produced non-finite model parameters")
    for optimizer in optimizers.values():
        for state in optimizer.state.values():
            check(all(torch.isfinite(value).all() for value in state.values() if torch.is_tensor(value)),
                  "Protected DPSL optimization produced non-finite optimizer state")
    shared_ratios = {
        family: float(budget["families"][family]["total_applied"])
        / max(float(budget["families"][family]["historical"]), 1e-30)
        for family in ("gfr", "darg", "cam", "head")
    }
    check(max(shared_ratios.values(), default=0.0) <= .001 + 1e-8,
          "Protected DPSL shared-family gradient budget exceeds 0.001")
    check(len(aux["scmr_objectives"]) > 0, "DPSL training objective was not computed")
    return {
        "status": "PASS", "g_min": float(guarded_aux["dpsl"]["confidence_gate"].min()),
        "g_max": float(guarded_aux["dpsl"]["confidence_gate"].max()),
        "g1_recovers_unprotected_new_formula": True, "g0_matches_native": True,
        "label_permutation_invariant": True,
        "residual_ratio_max": float(guarded_aux["dpsl"]["ratio"].max()),
        "shared_gradient_budget": budget,
        "shared_gradient_ratios": shared_ratios,
        "finite_loss": bool(torch.isfinite(total)), "finite_model_and_optimizer": True,
        "class_pairs": [[0, 1], [0, 2], [1, 2]],
        "uniform_probability_old_gate": 4.0 / 9.0,
        "uniform_probability_new_gate": 1.0,
        "binary_compatibility_exact": True,
        "native_cam_delta_max_abs": float((native_cam - guarded_aux["cam"]).abs().max()),
    }


def check_log_template():
    reference = Path("/data/adodas/SPromptGL-main/result_Tadpole.txt")
    copied = ROOT / "sources/author/result_Tadpole.txt"
    check(sha256(reference) == sha256(copied), "Local and source-snapshot author log templates differ")
    lines = reference.read_text(encoding="utf-8").splitlines()
    epoch = next(line for line in lines if re.match(r"^\s*\d+/\d+ trn-loss-MF:", line))
    check(epoch.count("trn-loss-MF:") == 3, "Author epoch template must retain three loss fields")
    check(" trn-auc: " in epoch and "  acc: " in epoch and " val-auc: " in epoch,
          "Author epoch field order differs")
    check(any(line.startswith("SEN_mean:") for line in lines)
          and any(line.startswith("SPE_mean:") for line in lines)
          and any(line.startswith("val=") for line in lines),
          "Author fold/final summary structure is incomplete")
    check("%3d/%d" in (ROOT / "sources/author/main.py").read_text(),
          "Author epoch field uses a different integer-width format")
    return {"status": "PASS", "sha256": sha256(reference), "line_count": len(lines),
            "epoch_line_sample": epoch, "loss_field_count": 3}


def check_z0_reuse(x, y, ids, splits):
    source = Y3_ROOT / "groups/Y3"
    local = ROOT / "groups/Z0"
    checks = []
    for fold in range(1, 11):
        for relative in (f"audits/fold{fold:02d}.json", f"predictions/fold{fold:02d}.csv",
                         f"best_checkpoints/fold{fold:02d}.pt", f"latest_checkpoints/fold{fold:02d}.pt"):
            left, right = local / relative, source / relative
            check(left.is_file() and right.is_file() and sha256(left) == sha256(right),
                  f"Z0 reuse differs from historical Y3 for {relative}")
        audit = json.loads((local / "audits" / f"fold{fold:02d}.json").read_text())
        frame = __import__("pandas").read_csv(local / "predictions" / f"fold{fold:02d}.csv")
        split = next(item for item in splits if int(item["fold"]) == fold)
        test = np.asarray(split["test_idx"], dtype=np.int64)
        check(audit["status"] == "TRAINED" and np.array_equal(frame.sample_index, test)
              and np.array_equal(frame.true_label, y[test]), f"Z0 fold {fold} identity differs")
        checks.append({"fold": fold, "best_sha256": sha256(local / "best_checkpoints" / f"fold{fold:02d}.pt"),
                       "latest_sha256": sha256(local / "latest_checkpoints" / f"fold{fold:02d}.pt"),
                       "prediction_sha256": sha256(local / "predictions" / f"fold{fold:02d}.csv")})
    historical_summary = __import__("pandas").read_csv(Y3_ROOT / "groups/Y3/fold_metrics.csv")
    local_summary = __import__("pandas").read_csv(local / "fold_metrics.csv")
    check(local_summary.to_csv(index=False) == historical_summary.to_csv(index=False),
          "Z0 fold metrics differ from the historical Y3 table")
    return {"status": "PASS", "folds": checks, "historical_group": "Y3",
            "metrics_exact": True, "predictions_exact": True, "best_latest_exact": True}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    check(device.type == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
          "One assigned PPU is required for the formal preflight")
    x_np, y_np, ids, splits = runner.load_data()
    check(len(ids) == 598 and x_np.shape == (598, 365), "TADPOLE input dimensions differ")
    check(np.bincount(y_np, minlength=3).tolist() == [209, 315, 74], "TADPOLE class encoding differs")
    check(tuple(runner.MODAL_DIMS) == (145, 148, 3, 36, 24, 9), "Six modality widths differ")
    check_splits(x_np, y_np, splits)
    check(sha256(ROOT / "inputs/processed_standard_data.csv") == runner.DATA_SHA,
          "Locked data SHA differs")
    check(sha256(ROOT / "inputs/split_indices.json") == runner.SPLIT_SHA,
          "Locked split SHA differs")
    prompt_report = check_prompts()
    configs, models, optimizers, identities, prompt_hashes = initial_group_contracts(device)
    z0_report = check_z0_reuse(x_np, y_np, ids, splits)
    darg_report, (pair, same, train_t, train_rows) = darg_isolation_probe(
        configs, models, x_np, y_np, splits, device)

    runner.historical.seed_everything(runner.SEED)
    warm_model = runner.build_model(configs["Z0"], 1, device)
    warm_optimizers, _ = runner.make_optimizers(warm_model, configs["Z0"])
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    supports = runner.historical.anchor_pair_support(warm_model, x)
    pair_np, same_np = runner.historical.make_pairs(train_rows, y_np, 1, supports)
    warm_pair = torch.from_numpy(pair_np).long().to(device)
    warm_same = torch.from_numpy(same_np).float().to(device)
    initial = runner.initial_warmup_identity(warm_model, warm_optimizers)
    warm_history, warm_rng, warm_updates, warm_logs = runner.run_warmup(
        warm_model, warm_optimizers, x, labels, train_t, warm_pair, warm_same,
        configs["Z0"], 1, "Z0", torch.as_tensor(splits[0]["test_idx"], device=device))
    warm_payload = runner.make_warmup_payload(
        1, warm_model, warm_optimizers, warm_history, warm_rng, warm_pair, warm_same,
        initial, configs["Z0"], train_rows, y_np, warm_updates, warm_logs)
    warmup_pairing = runner.compare_warmup(
        "Z0", 1, configs["Z0"], warm_payload, warm_pair, warm_same, train_rows, y_np)
    check(warmup_pairing["status"] == "PASS", "Z0 fold1 warmup does not reproduce historical N3")
    check(len(warm_logs) == 10 and [item["epoch"] for item in warm_logs] == list(range(1, 11)),
          "Warmup logging does not contain ten real epochs")
    warm_model.eval()
    with torch.no_grad():
        _, warm_aux = warm_model(x, labels, train_t, warm_pair, warm_same, dpsl_enabled=False)
    warm_spectral = runner.fit_spectral_statistics(
        warm_aux["cam"], labels, train_t, fold=1, warm_logits=warm_aux["native_logits"])
    warm_spectral["warmup_cam"] = warm_aux["cam"].detach().cpu()
    warm_spectral["warmup_logits"] = warm_aux["native_logits"].detach().cpu()
    warm_spectral, spectral_pairing = runner.compare_warmup_spectral_state(
        "Z0", 1, configs["Z0"], warm_spectral)

    historical_n3 = torch.load(runner.N3_REFERENCE_ROOT / "groups/N3/best_checkpoints/fold01.pt",
                               map_location="cpu", weights_only=False)
    dpsl_report = dpsl_probe(
        configs, x, y_np, splits[0], historical_n3, historical_n3["relation_pairs"].to(device),
        historical_n3["relation_targets"].to(device), train_t, device)
    log_template = check_log_template()
    report = {
        "experiment": "Experiment 83.8.4",
        "status": "PASS",
        "formal_training_allowed": True,
        "training_seal_sha256": None,
        "data_sha256": runner.DATA_SHA,
        "split_sha256": runner.SPLIT_SHA,
        "source_manifest_sha256": sha256(ROOT / "source_manifest.json"),
        "cohort": {"patients": 598, "class_counts": [209, 315, 74], "labels": [0, 1, 2],
                   "modality_dimensions": list(runner.MODAL_DIMS), "total_dimensions": 365,
                   "cam_dimensions": darg_report["cam_dimensions"]},
        "checks": [
            {"id": "LOCKED_DATA_AND_SPLIT", "status": "PASS"},
            {"id": "THREE_CLASS_SIX_MODALITY_PROTOCOL", "status": "PASS"},
            {"id": "MMGPL12_RN50_AND_SCMR_PROMPTS", "status": "PASS"},
            {"id": "Y3_REUSED_FORMAL_Z0_BEST_LATEST_AND_OOF", "status": "PASS"},
            {"id": "N3_ALOSS_AND_FULL_G4_CONFIG", "status": "PASS"},
            {"id": "SIX_MODALITY_DARG_BINDING_AND_CROSS_GROUP_ISOLATION", "status": "PASS"},
            {"id": "RISK_FACTOR_GENERIC_BRANCH_AND_SIX_MODALITY_DIAGNOSTICS", "status": "PASS"},
            {"id": "DPSL_CONFIDENCE_GATE_AND_GRADIENT_BUDGET", "status": "PASS"},
            {"id": "LOGGING_WARMUP_RNG_AND_AUTHOR_FORMAT", "status": "PASS"},
        ],
        "prompts": prompt_report,
        "prompt_vector_file_sha256_by_group": prompt_hashes,
        "initial_group_identities": identities,
        "z0_reuse": z0_report,
        "darg": darg_report,
        "dpsl": dpsl_report,
        "warmup_pairing_fold1": warmup_pairing,
        "warmup_spectral_pairing_fold1": spectral_pairing,
        "author_log_template": log_template,
        "groups": {group: {"status": "PASS", "formal_training_allowed": True,
                            "config_sha256": sha256(ROOT / "configs" / f"{group}.json"),
                            "model_contract": {"darg": True, "dpsl": True, "classes": 3},
                            "training_action": "REUSE_VERIFIED" if group == "Z0" else "TRAIN_FORMAL"}
                   for group in runner.FORMAL_GROUPS},
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    target = ROOT / "audits/preflight_start.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if (ROOT / "audits/training_seal.json").exists():
        raise RuntimeError("A sealed preflight cannot be overwritten")
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "groups": list(report["groups"]),
                      "warmup_pairing": warmup_pairing["status"],
                      "darg": darg_report["status"], "dpsl": dpsl_report["status"],
                      "path": str(target)}, indent=2))


if __name__ == "__main__":
    main()
