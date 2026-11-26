"""Run the locked-data, method, GRL, DPSL and paired-warmup preflight."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code/common"), str(ROOT / "code")]
import run_e8385 as run
from fidelity_model import HistoricalCAM, SignedDARG
from source_runtime import ABIDE_DIMS, ABIDE_MODALITIES, relation_pair_sources
from scmr_runtime import CLASS_PAIRS


HISTORY = run.HISTORICAL_ROOT
H0_PATHS = {str(fold): HISTORY / "groups/R0/best_checkpoints" / f"fold{fold:02d}.pt"
            for fold in range(1, 11)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def store(path, value):
    run.write_json(path, value)


def verify_locked_data():
    x, y, ids, splits = run.load_data()
    require(x.shape == (598, 365) and len(ids) == 598, "TADPOLE shape or patient order changed")
    require(np.bincount(y, minlength=3).tolist() == [209, 315, 74], "TADPOLE label counts changed")
    require(tuple(run.MODAL_DIMS) == (145, 148, 3, 36, 24, 9), "Modality dimensions changed")
    require(run.MODAL_NAMES[4] == "COGNITIVE_TEST", "Cognitive modality index is not four")
    require(len(splits) == 10, "Expected ten locked folds")
    regenerated = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    for (train, test), saved in zip(regenerated.split(x, y), splits, strict=True):
        require(int(saved["fold"]) >= 1, "Fold identity missing")
        require(np.array_equal(train, saved["train_idx"])
                and np.array_equal(test, saved["test_idx"]),
                f"Locked fold {saved['fold']} differs from StratifiedKFold")
        require(saved.get("val_idx", saved["test_idx"]) == saved["test_idx"],
                f"Fold {saved['fold']} validation and test indices differ")
    return x, y, ids, splits


def verify_factorial_configs():
    configs = {group: run.group_config(group) for group in run.TRAIN_GROUPS}
    common = None
    factors = {
        "N1": (0.01, 0.03), "N2": (0.03, 0.03),
        "N3": (0.01, 0.015), "N4": (0.03, 0.015),
    }
    for group, config in configs.items():
        require(config["alignment_anchor_index"] == 4 and config["edge_anchor_index"] == 4,
                f"{group} does not use both registered cognitive anchors")
        require(config["supervision_kind"] == "paper_aloss"
                and config["supervision_coefficient"] == 0.01
                and config["graph_supervision_schedule"] == "epoch_11_onward",
                f"{group} Y3/Z0 ALoss contract changed")
        require(config["darg_direct_scale"] == 0.02
                and config["darg_direct_single_gate"] is False,
                f"{group} Positive-DCR direct path changed")
        require(config["darg_initial_gate_probabilities"] == [0.1] * 6,
                f"{group} initial Gate probabilities changed")
        require(config["dpsl_residual_cap"] == 0.08
                and config["dpsl_shared_cap"] == 0.001
                and config["dpsl_confidence_mode"] == "old",
                f"{group} DPSL contract changed")
        gate_level, grl_level = factors[group]
        require(config["darg_gate_scales"][4] == gate_level
                and config["darg_grl_scale"] == grl_level,
                f"{group} differs from its registered factorial level")
        reduced = copy.deepcopy(config)
        for key in ("group", "method", "registered_factor_levels", "darg_gate_scales", "darg_grl_scale"):
            reduced.pop(key, None)
        if common is None:
            common = reduced
        else:
            require(reduced == common, f"{group} changes an unregistered config field")
    history_r1 = json.loads((HISTORY / "configs/R1.json").read_text())
    n1 = configs["N1"]
    shared_fields = (
        "base_reference", "darg_gate_scales", "darg_direct_scale", "darg_direct_single_gate",
        "alignment_anchor_index", "darg_initial_gate_probabilities", "supervision_kind",
        "supervision_coefficient", "graph_supervision_schedule", "dpsl_confidence_protection",
        "dpsl_confidence_mode", "dpsl_shared_cap", "dpsl_residual_cap", "data_sha256",
        "split_sha256", "training_seed", "max_epochs", "early_stop_wait_gt",
    )
    for field in shared_fields:
        require(history_r1[field] == n1[field], f"H1 to N1 unregistered difference in {field}")
    require(history_r1.get("darg_grl_scale", 0.03) == n1["darg_grl_scale"] == 0.03,
            "H1 to N1 changed the historical default GRL scale")
    require(history_r1.get("edge_anchor_index") is None and n1["edge_anchor_index"] == 4,
            "H1 to N1 must add only the cognitive edge anchor")
    require(history_r1.get("warmup_reference") == "R0" and n1["warmup_reference"] == "H0",
            "H1 and N1 do not share the same R0/H0 warmup source")
    require(history_r1.get("initial_base_weight_decay", 0.11)
            == n1.get("initial_base_weight_decay", 0.11) == 0.11
            and history_r1.get("intermediate_base_weight_decay")
            == n1.get("intermediate_base_weight_decay"),
            "H1 to N1 changed the inherited optimizer schedule")
    factors["H1_to_N1_registered_difference"] = {
        "status": "PASS", "shared_source_config_sha256": sha(HISTORY / "configs/R1.json"),
        "shared_fields": list(shared_fields), "only_method_difference": "edge_anchor_index: null -> 4",
        "effective_grl_scale_both": 0.03, "warmup_source": "R0/H0",
    }
    registered = {group: {"cognitive_gate_scale": levels[0], "grl_scale": levels[1]}
                  for group, levels in factors.items() if group in configs}
    registered["H1_to_N1_registered_difference"] = factors["H1_to_N1_registered_difference"]
    return registered


def abide_fixed_tensor_regression(device):
    torch.manual_seed(7331)
    darg = SignedDARG(
        ABIDE_DIMS, fold=8, device=device, modalities=ABIDE_MODALITIES,
        gate_scales=(0.01, 0.01, 0.01, 1.0), direct_scale=0.02,
        direct_single_gate=False, alignment_anchor_index=3, edge_anchor_index=3,
        initial_gate_probabilities=(0.1, 0.1, 0.1, 0.99), grl_scale=0.03,
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(7331)
    n = 64
    parts = [torch.randn((n, dim), generator=generator, device=device) for dim in ABIDE_DIMS]
    pair_index = torch.stack((torch.arange(32, device=device),
                              torch.arange(32, device=device) + 32), dim=1)
    pair_same = (torch.arange(32, device=device) % 2 == 0).float()
    darg.eval()
    fused, aux = darg(parts, pair_index, pair_same)
    for key in ("pair_loss", "adv_loss", "preserve_loss", "graph_loss"):
        require(bool(torch.isfinite(aux[key])), f"ABIDE {key} is non-finite")
    for original, disease_delta, direct_delta in zip(
            aux["adjacencies"], aux["disease_delta_adjs"], aux["direct_delta_adjs"], strict=True):
        support = (original > 0) & ~torch.eye(n, dtype=torch.bool, device=device)
        require(not bool((disease_delta[~support] != 0).any()), "DARG added an unsupported ABIDE patient edge")
        require(not bool((direct_delta[~support] != 0).any()), "Direct-DCR added an unsupported ABIDE patient edge")
        require(torch.allclose(disease_delta, disease_delta.T, atol=1e-7, rtol=0),
                "ABIDE signed disease edge update is not symmetric")
        require(torch.all(direct_delta >= -1e-8), "ABIDE direct path is not positive-only")
    prompt_path = ROOT / "code/sources/E82_4_6_G2_SCMR_DA_SRPE/inputs/prompts/prompts.pt"
    cam = HistoricalCAM(ABIDE_DIMS, hidden=10, heads=4, nlayer=2, nclass=2,
                        dropout=0.0, use_pen=True, prompt_path=str(prompt_path)).to(device)
    cam.eval()
    logp, cam_aux = cam(fused)
    require(tuple(cam.PEN.prompt_text_features.shape) == (8, 1024), "ABIDE binary path lost its eight prompts")
    require(tuple(logp.shape) == (n, 2) and bool(torch.isfinite(logp).all()),
            "ABIDE binary prompt path produced invalid logits")
    labels = torch.arange(n, device=device) % 2
    loss = F.nll_loss(logp, labels) + aux["graph_loss"] + .05 * aux["pair_loss"] \
        + .01 * aux["adv_loss"] + .01 * aux["preserve_loss"]
    loss.backward()
    encoder_grad = [float(torch.stack([p.grad.norm() for p in encoder.parameters()
                                       if p.grad is not None]).norm())
                    for encoder in darg.core.encoders]
    scorer_grad = [any(p.grad is not None and float(p.grad.norm()) > 0
                       for p in scorer.parameters()) for scorer in darg.core.scorers]
    require(all(value > 0 for value in encoder_grad), "ABIDE relation encoders did not receive training gradients")
    require(all(scorer_grad), "ABIDE relation scorer did not receive BCE gradients")
    require(any(p.grad is not None and float(p.grad.norm()) > 0 for p in cam.parameters()),
            "ABIDE CAM/PEN path did not receive classification gradients")
    return {
        "status": "PASS", "checkpoint_level": False,
        "fixed_test_tensor": {"seed": 7331, "patients": n, "dimensions": list(ABIDE_DIMS)},
        "modality_order": list(ABIDE_MODALITIES), "alignment_anchor_index": 3,
        "edge_anchor_index": 3, "grl_scale": darg.core.grl_scale,
        "direct_single_gate": darg.direct_single_gate, "query_count": 8,
        "binary_logits_shape": list(logp.shape), "relation_encoder_grad_norms": encoder_grad,
        "relation_scorer_gradients": scorer_grad,
        "edge_support_and_signed_double_gate_checked": True,
        "loss_coefficients": {"pair_bce": .05, "conditional_adversarial": .01,
                              "preservation": .01},
    }


def grl_and_dpsl_regression(x_np, y_np, splits, device):
    config = run.group_config("N1")
    fold = 1
    source_path = H0_PATHS[str(fold)]
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    run.historical.seed_everything(run.SEED)
    model = run.build_model(config, fold, device, source["spectral_stats"])
    model.load_state_dict(source["model"], strict=True)
    model.darg.enter_positive_dcr()
    model.eval()
    x = torch.as_tensor(x_np, device=device)
    labels = torch.as_tensor(y_np, device=device)
    train = torch.as_tensor(splits[0]["train_idx"], device=device)
    test = torch.as_tensor(splits[0]["test_idx"], device=device)
    y_train_only = labels.clone()
    y_train_only[test] = -7
    pair = source["relation_pairs"].to(device)
    same = source["relation_targets"].to(device)
    choose = torch.cat((torch.arange(0, 24, device=device), torch.arange(2048, 2072, device=device)))
    pair_small, same_small = pair[choose], same[choose]

    model.zero_grad(set_to_none=True)
    model.darg.core.grl_scale = .03
    pred_003, aux_003 = model(x, y_train_only, train, pair_small, same_small, dpsl_enabled=False)
    aux_003["adv_loss"].backward()
    encoder_003 = [None if not any(p.grad is not None for p in module.parameters()) else
                   torch.cat([p.grad.detach().flatten() for p in module.parameters() if p.grad is not None])
                   for module in model.darg.core.encoders]
    discriminator_003 = [p.grad.detach().clone() for p in model.darg.core.discriminator.parameters()]

    model.zero_grad(set_to_none=True)
    model.darg.core.grl_scale = .015
    pred_0015, aux_0015 = model(x, y_train_only, train, pair_small, same_small, dpsl_enabled=False)
    aux_0015["adv_loss"].backward()
    encoder_0015 = [None if not any(p.grad is not None for p in module.parameters()) else
                    torch.cat([p.grad.detach().flatten() for p in module.parameters() if p.grad is not None])
                    for module in model.darg.core.encoders]
    discriminator_0015 = [p.grad.detach().clone() for p in model.darg.core.discriminator.parameters()]
    require(torch.equal(pred_003, pred_0015), "Changing GRL changed the discriminator or classifier forward")
    require(torch.equal(aux_003["adv_info"]["logits"], aux_0015["adv_info"]["logits"]),
            "Changing GRL changed conditional discriminator logits")
    require(torch.equal(aux_003["adv_loss"], aux_0015["adv_loss"]), "Changing GRL changed its scalar loss")
    require(all(torch.equal(a, b) for a, b in zip(discriminator_003, discriminator_0015, strict=True)),
            "Changing GRL changed discriminator gradients")
    nonanchor_ratios = {}
    for index, (grad_003, grad_0015) in enumerate(zip(encoder_003, encoder_0015, strict=True)):
        if index == 4:
            require(grad_003 is None and grad_0015 is None,
                    "Cognitive alignment anchor was not detached on the adversarial branch")
            continue
        require(grad_003 is not None and grad_0015 is not None,
                f"Non-anchor modality {index} did not receive adversarial gradient")
        ratio = float(grad_0015.norm() / grad_003.norm().clamp_min(1e-30))
        require(abs(ratio - .5) <= 1e-6, f"GRL gradient ratio for modality {index} is {ratio}")
        nonanchor_ratios[run.MODAL_NAMES[index]] = ratio
    model.zero_grad(set_to_none=True)
    _, pair_aux = model(x, y_train_only, train, pair_small, same_small, dpsl_enabled=False)
    pair_aux["pair_loss"].backward()
    anchor_normal_grad = any(p.grad is not None and float(p.grad.norm()) > 0
                             for p in model.darg.core.encoders[4].parameters())
    require(anchor_normal_grad, "Cognitive alignment anchor was globally frozen from non-adversarial losses")

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        full, full_aux = model(x, y_train_only, train, pair, same, dpsl_enabled=True)
        native, _ = model(x, y_train_only, train, pair, same, dpsl_enabled=False)
        model.dpsl._confidence_gate_override = 0.0
        residual_off, residual_aux = model(x, y_train_only, train, pair, same, dpsl_enabled=True)
        model.dpsl._confidence_gate_override = None
    require(torch.equal(native, residual_off), "DPSL Residual-Off does not recover Native output")
    prompt = model.dpsl.spectrum.query_medical
    prompt_file = torch.load(ROOT / "inputs/prompts/mmGPL12_scmr.pt",
                             map_location="cpu", weights_only=False)
    require(tuple(prompt.shape) == (12, 32) and not prompt.requires_grad
            and not isinstance(prompt, torch.nn.Parameter)
            and tuple(prompt_file["medical"].shape) == (12, 1024)
            and tuple(prompt_file["q_medical"].shape) == (12, 32),
            "TADPOLE frozen MMGPL-12 prompt/query identity differs")
    expected_directed_pairs = tuple((severe, mild) for mild, severe in CLASS_PAIRS)
    require(len(full_aux["dpsl"]["pair_outputs"]) == 3
            and tuple(map(tuple, full_aux["dpsl"]["class_pairs"].tolist())) == expected_directed_pairs,
            "DPSL did not execute all three registered diagnosis pairs")
    residual_ratio = full_aux["dpsl"]["residual_ratio"]
    require(float(residual_ratio.max()) <= .08 + 1e-7,
            "DPSL residual exceeded the historical cap")
    require(float((full - native).abs().max()) > 0,
            "DPSL Full and Native did not produce a replayable difference")
    original = full_aux["adjacencies"]
    deltas = full_aux["disease_delta_adjs"]
    direct = full_aux["direct_delta_adjs"]
    support_stats = []
    for index, (adjacency, delta, direct_delta) in enumerate(zip(original, deltas, direct, strict=True)):
        support = (adjacency > 0) & ~torch.eye(len(adjacency), dtype=torch.bool, device=device)
        require(not bool((delta[~support] != 0).any()),
                f"TADPOLE modality {index} DARG changed an unsupported edge")
        require(not bool((direct_delta[~support] != 0).any()),
                f"TADPOLE modality {index} Direct-DCR changed an unsupported edge")
        require(torch.allclose(delta, delta.T, atol=1e-7, rtol=0),
                f"TADPOLE modality {index} signed edges are asymmetric")
        require(torch.all(direct_delta >= -1e-8), "TADPOLE Direct-DCR is not positive-only")
        support_stats.append({"modality": run.MODAL_NAMES[index],
                              "original_edges": int(torch.triu(support, diagonal=1).sum()),
                              "signed_positive": int((torch.triu(delta, diagonal=1) > 0).sum()),
                              "signed_negative": int((torch.triu(delta, diagonal=1) < 0).sum()),
                              "direct_positive": int((torch.triu(direct_delta, diagonal=1) > 0).sum())})
    relation_pairs = run.historical.make_pairs(
        np.asarray(splits[0]["train_idx"], dtype=np.int64), y_np, fold,
        run.historical.anchor_pair_support(model, x),
    )
    altered = y_np.copy()
    altered_test = altered[splits[0]["test_idx"]].copy()
    altered[splits[0]["test_idx"]] = altered_test[::-1]
    permuted_pairs = run.historical.make_pairs(
        np.asarray(splits[0]["train_idx"], dtype=np.int64), altered, fold,
        run.historical.anchor_pair_support(model, x),
    )
    require(np.array_equal(relation_pairs[0], permuted_pairs[0])
            and np.array_equal(relation_pairs[1], permuted_pairs[1]),
            "Test-label permutation changed relation training supervision")
    return {
        "status": "PASS", "reference_h0_best_sha256": sha(H0_PATHS["1"]),
        "grl": {"forward_identical": True, "loss_identical": True,
                "discriminator_gradients_identical": True,
                "anchor_index": 4, "anchor_detached_only_on_adversarial_branch": True,
                "anchor_receives_pair_loss_gradient": anchor_normal_grad,
                "nonanchor_gradient_ratios_0_015_over_0_03": nonanchor_ratios},
        "pair_supervision": {"count": int(len(relation_pairs[0])),
                             "same": int(relation_pairs[1].sum()),
                             "different": int((relation_pairs[1] == 0).sum()),
                             "test_label_permutation_invariant": True},
        "darg": {"two_gate_direct_path": True, "direct_scale": .02,
                 "direct_single_gate": False, "support_and_symmetry": support_stats},
        "dpsl": {"frozen_prompt_source_shape": list(prompt_file["medical"].shape),
                 "fixed_query_buffer_shape": list(prompt.shape), "frozen_prompt": True,
                 "class_pairs": [list(pair) for pair in expected_directed_pairs],
                 "full_native_max_abs": float((full - native).abs().max()),
                 "residual_off_native_max_abs": float((native - residual_off).abs().max()),
                 "residual_ratio_max": float(residual_ratio.max()),
                 "full_residual_off_difference_nonzero": bool((full != residual_off).any())},
    }


def warmup_pairing_preflight(x_np, y_np, splits, device):
    historical_best_sha = {str(fold): sha(path) for fold, path in H0_PATHS.items()}
    store(ROOT / "audits/preflight.json", {"historical_h0_best_sha256": historical_best_sha})
    x = torch.as_tensor(x_np, device=device)
    labels = torch.as_tensor(y_np, device=device)
    records = {group: {} for group in run.TRAIN_GROUPS}
    expected_states = {}
    for group in run.TRAIN_GROUPS:
        config = run.group_config(group)
        for split in splits:
            fold = int(split["fold"])
            run.historical.seed_everything(run.SEED)
            train = torch.as_tensor(split["train_idx"], device=device)
            result = run.prepare_warmup(group, fold, config, x, labels, train,
                                        np.asarray(split["test_idx"], dtype=np.int64), y_np)
            model, optimizers, _, pair, same, warmup, pairing, stats = result
            require(pairing["status"] == "PASS", f"{group} fold {fold} warmup did not pair")
            state = pairing["pre_epoch_11_state"]
            if str(fold) not in expected_states:
                expected_states[str(fold)] = state
            else:
                require(state == expected_states[str(fold)],
                        f"{group} fold {fold} post-warmup model/optimizer/RNG differs")
            records[group][str(fold)] = {
                "status": "PASS", "reference": "H0/R0",
                "historical_best_sha256": historical_best_sha[str(fold)],
                "initial_model_sha256": warmup["initial_model_sha256"],
                "initial_optimizer_sha256": warmup["initial_optimizer_sha256"],
                "initial_rng_sha256": warmup["initial_rng_sha256"],
                "relation_pairs_sha256": warmup["pair_sha256"],
                "warmup_history_sha256": run.sha_object(warmup["history"]),
                "post_warmup_model_sha256": warmup["base_state_sha256"],
                "post_warmup_optimizer_sha256": warmup["optimizer_state_sha256"],
                "post_warmup_rng_sha256": warmup["rng_state_sha256"],
                "prompt_sha256": pairing["prompt_vector_file_sha256"],
                "spectral_statistics_sha256": pairing["spectral_statistics_sha256"],
                "formal_grl_scale": config["darg_grl_scale"],
                "warmup_grl_scale": 0.03,
                "post_epoch_10_state": state,
            }
            del model, optimizers, pair, same, warmup, stats, result
            torch.cuda.empty_cache()
            print(f"PASS warmup pairing {group} fold {fold:02d}", flush=True)
    require(all(len(rows) == 10 for rows in records.values()), "Warmup preflight is incomplete")
    audit = {
        "experiment": run.EXPERIMENT, "status": "PASS", "reference_group": "H0/R0",
        "reference_folds": 10, "treatment_folds": 40,
        "all_groups_share_post_warmup_state_per_fold": True,
        "grl_0_015_does_not_affect_warmup": True,
        "DARG_graph_branch_active_from_epoch": 11,
        "group_folds": records, "pre_epoch_11_state_by_fold": expected_states,
    }
    store(ROOT / "audits/warmup_pairing.json", audit)
    return audit, historical_best_sha, expected_states


def main():
    run.configure_worker()
    device = torch.device("cuda:0")
    x_np, y_np, _, splits = verify_locked_data()
    factors = verify_factorial_configs()
    require(run.sha_file(ROOT / "code/sources/history_darg/cadr_gfr.py")
            == "f5032d96c9ac68afda47a253f794ddfe41373fabebbbbe27057f1425f988815d",
            "Archived CADR-GFR source identity changed")
    require(run.sha_file(ROOT / "code/sources/history_darg/soft_positive_dcr.py")
            == "3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718",
            "Archived Positive-DCR source identity changed")
    require(run.sha_file(ROOT / "code/sources/history_darg/e61_run.py")
            == "f5bd8236b6a5f66903f56a01832caab6c21177238f24676562f536df3af2b334",
            "Archived relation-pair sampler identity changed")
    pair_sources = relation_pair_sources()
    require(callable(pair_sources.make_outer_pairs), "Historical deterministic relation sampler missing")
    require(HistoricalCAM is not None, "Historical CAM/PEN path unavailable")
    abide = abide_fixed_tensor_regression(device)
    abide_checkpoint = json.loads((ROOT / "audits/abide_checkpoint_replay.json").read_text())
    require(abide_checkpoint.get("status") == "PASS"
            and abide_checkpoint.get("certificate", {}).get("best_reloaded") is True
            and abide_checkpoint.get("certificate", {}).get("latest_next_training_exact") is True,
            "ABIDE E82.4.6.10-G1 checkpoint-level replay is incomplete")
    method = grl_and_dpsl_regression(x_np, y_np, splits, device)
    method["historical_default_grl_exact"] = True
    method["grl_default_preserved_at_0_03"] = True
    store(ROOT / "audits/method_fidelity.json", {
        "experiment": run.EXPERIMENT, "status": "PASS",
        "source_sha256": {
            "cadr_gfr.py": sha(ROOT / "code/sources/history_darg/cadr_gfr.py"),
            "soft_positive_dcr.py": sha(ROOT / "code/sources/history_darg/soft_positive_dcr.py"),
            "e61_run.py": sha(ROOT / "code/sources/history_darg/e61_run.py"),
            "fidelity_model.py": sha(ROOT / "code/common/fidelity_model.py"),
            "source_runtime.py": sha(ROOT / "code/common/source_runtime.py"),
        },
        "darg_dpsl_preflight": method,
        "registered_factor_levels": factors,
        "registered_losses": {"relation_bce": .05, "conditional_adversarial": .01,
                               "relation_preservation": .01},
    })
    historical_best_sha = {str(fold): sha(path) for fold, path in H0_PATHS.items()}
    parity = {
        "experiment": run.EXPERIMENT, "status": "PASS",
        "ABIDE": {**abide, "historical_DARG_source": "E74.4-G2 Positive-DCR",
                  "historical_DPSL_source": "E82.4.6.10-G1 / SCMR-DA-SRPE"},
        "TADPOLE": {"status": "PASS", "patients": 598, "modalities": list(run.MODAL_NAMES),
                    "classes": 3, "cognitive_anchor_index": 4,
                    "MMGPL12_prompt_sha256": sha(ROOT / "inputs/prompts/mmGPL12_scmr.pt"),
                    "H0_fold01_best_sha256": historical_best_sha["1"],
                    "residual_off_native_equal": method["dpsl"]["residual_off_native_max_abs"] == 0.0,
                    "all_three_class_pairs": [list((severe, mild)) for mild, severe in CLASS_PAIRS]},
        "checkpoint_level_ABIDE": {
            "status": abide_checkpoint["status"],
            "group": abide_checkpoint["certificate"]["group"],
            "fold": abide_checkpoint["certificate"]["fold"],
            "best_epoch": abide_checkpoint["certificate"]["best_epoch"],
            "latest_epoch": abide_checkpoint["certificate"]["latest_epoch"],
            "best_reloaded": abide_checkpoint["certificate"]["best_reloaded"],
            "latest_next_training_exact": abide_checkpoint["certificate"]["latest_next_training_exact"],
            "checkpoint_sha256": abide_checkpoint["checkpoint_sha256"],
            "audit_sha256": sha(ROOT / "audits/abide_checkpoint_replay.json"),
        },
    }
    store(ROOT / "audits/abide_tadpole_parity.json", parity)
    warmup, historical_best_sha, expected_states = warmup_pairing_preflight(x_np, y_np, splits, device)
    preflight = {
        "experiment": run.EXPERIMENT, "status": "PASS", "formal_training_allowed": True,
        "preflight_program_sha256": sha(Path(__file__)),
        "abide_checkpoint_replay_sha256": sha(ROOT / "audits/abide_checkpoint_replay.json"),
        "data_sha256": run.DATA_SHA, "split_sha256": run.SPLIT_SHA,
        "groups": {group: {"status": "PASS", "formal_training_allowed": True,
                           "config_sha256": sha(ROOT / "configs" / f"{group}.json")}
                   for group in run.TRAIN_GROUPS},
        "historical_h0_best_sha256": historical_best_sha,
        "warmup": expected_states,
        "method_fidelity_sha256": sha(ROOT / "audits/method_fidelity.json"),
        "parity_sha256": sha(ROOT / "audits/abide_tadpole_parity.json"),
        "warmup_pairing_sha256": sha(ROOT / "audits/warmup_pairing.json"),
        "checks": {
            "data_and_split_identity": "PASS", "cognitive_index": "PASS",
            "factorial_config_control": "PASS", "double_gate_and_positive_dcr": "PASS",
            "signed_edge_support": "PASS", "relation_and_discriminator_gradients": "PASS",
            "anchor_adversarial_detach_only": "PASS", "test_label_permutation": "PASS",
            "dpsl_residual_off": "PASS", "dpsl_residual_budget": "PASS",
            "abide_8_prompt_binary": "PASS", "tadpole_12_prompt_three_class": "PASS",
            "historical_grl_default": "PASS", "reduced_grl_gradient_scale": "PASS",
            "warmup_pairing": "PASS", "source_identity": "PASS",
            "abide_checkpoint_replay": "PASS",
        },
        "warmup_pairing_status": warmup["status"],
    }
    store(ROOT / "audits/preflight.json", preflight)
    print("PASS Experiment 83.8.5.1 preflight; formal training can be sealed", flush=True)


if __name__ == "__main__":
    main()
