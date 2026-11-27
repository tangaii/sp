"""Preflight the ten shared warmups and all 80 registered branch transitions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83853 as run
from fidelity_model import HistoricalCAM, SignedDARG
from scmr_runtime import CLASS_PAIRS, source_config
from source_runtime import ABIDE_DIMS, ABIDE_MODALITIES
from state_digest import content_sha
from author_log_format import parse


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_locked_data():
    x, y, ids, splits = run.load_data()
    require(np.isfinite(x).all() and ids["RID"].is_unique, "Invalid feature, sample count or RID")
    manifest = json.loads((ROOT / "inputs/modality_manifest.json").read_text())
    order = list(run.MODAL_NAMES)
    require(manifest["modality_order"] == order, "Modality ordering differs")
    require([manifest["modality_dimensions"][m] for m in order] == list(run.MODAL_DIMS),
            "Modality widths differ")
    with (ROOT / "inputs/processed_standard_data.csv").open() as stream:
        columns = next(csv.reader(stream))
    expected = [col for m in order for col in manifest["features"][m]]
    require([c for c in columns if c != "LABEL"] == expected, "Features do not follow locked modality blocks")
    require(np.bincount(y, minlength=3).tolist() == [209, 315, 74], "Class counts differ")
    regenerated = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    for fold, ((train, test), saved) in enumerate(zip(regenerated.split(x, y), splits, strict=True), 1):
        require(saved["fold"] == fold and saved["val_equals_test"] is True, "Fold protocol differs")
        np.testing.assert_array_equal(train, saved["train_idx"])
        np.testing.assert_array_equal(test, saved["test_idx"])
        require(not np.intersect1d(train, test).size, "Train/test overlap")
    require(sorted(j for fold in splits for j in fold["test_idx"]) == list(range(598)),
            "OOF coverage differs")
    return {"status": "PASS", "shape": list(x.shape), "class_counts": [209, 315, 74],
            "modality_order": order, "modality_dimensions": list(run.MODAL_DIMS),
            "data_sha256": run.DATA_SHA, "split_sha256": run.SPLIT_SHA, "folds": 10, "oof_exact": True}


def verify_sources_and_configs():
    archive_files = json.loads((run.ARCHIVE / "audits/training_seal.json").read_text())["files"]
    parent = json.loads((ROOT / "audits/source_comparison.json").read_text())
    require(all(item["exact"] for item in parent["files"].values()), "Pinned GitHub source drift")
    preserved = {}
    for name in sorted((ROOT / "code/common").glob("*.py")) + sorted(
            (ROOT / "code/sources").rglob("*.py")):
        key = str(name.relative_to(ROOT))
        require(archive_files[key] == run.sha_file(name), f"Frozen arithmetic identity {key}")
        preserved[key] = archive_files[key]
    reference = json.loads((run.ARCHIVE / "configs/S02.json").read_text())
    factors = {"A1":(.99,.0075,0,False), "A2":(.99,.0125,0,False), "P1":(.9,.011,0,False),
               "P2":(.5,.0198,0,False), "P3":(.1,.099,0,False), "R1":(.99,.01,512,False),
               "R2":(.99,.01,1024,False), "D1":(.99,.01,0,True)}
    allowed = {"experiment","group","method","warmup_reference","registered_factor_levels",
               "darg_gate_scales","target_gate_probabilities","anchor_support_quota_per_target",
               "freeze_adaptive_gate"}
    for group, (p,s,q,f) in factors.items():
        config = run.group_config(group)
        require({key:v for key,v in config.items() if key not in allowed}
                == {key:v for key,v in reference.items() if key not in allowed}, f"{group}: unregistered change")
        require(config["darg_gate_scales"] == [.01,.01,.01,.01,s,.01]
                and config["target_gate_probabilities"] == [.1,.1,.1,.1,p,.1]
                and config["anchor_support_quota_per_target"] == q
                and config["freeze_adaptive_gate"] is f, f"{group}: registered factors differ")
    prompts = json.loads((ROOT / "prompt_manifest.json").read_text())
    for name, evidence in prompts["files"].items():
        path = ROOT / "inputs/prompts" / name
        require(run.sha_file(path) == evidence["sha256"], f"Prompt provenance {name}")
    prompt = torch.load(run.OLD_PROMPT_SCMR, map_location="cpu", weights_only=True)
    require(tuple(prompt["q_medical"].shape) == (12, 32), "Expected 12 locked queries")
    return {"status":"PASS", "frozen_source_files":preserved, "s02_keywise_verified":list(factors),
            "prompts":prompts, "class_pairs":[list(pair) for pair in CLASS_PAIRS]}


def grad_norm(parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    require(all(torch.isfinite(grad).all() for grad in gradients), "Non-finite training gradient")
    return sum(float(grad.square().sum()) for grad in gradients) ** .5


@torch.no_grad()
def graph_contract(model, x, labels, train, pair, same):
    model.eval()
    full, aux = model(x, labels, train, pair, same)
    native = model(x, labels, train, pair, same, dpsl_enabled=False)[0]
    residual_off = model(x, labels, train, pair, same, dpsl_intervention="residual_off")[0]
    require(torch.equal(residual_off, native), "Residual-Off is not exactly Native")
    require(aux["dpsl"]["residual_ratio"].max() <= 0.08000002, "Residual exceeds FP32 0.08 bound")
    require(aux["dpsl"]["class_pairs"].tolist() == [[1,0],[2,0],[2,1]], "DPSL class-pair set differs")
    require(tuple(model.dpsl.spectrum.query_medical.shape) == (12, 32)
            and not model.dpsl.spectrum.query_medical.requires_grad, "Medical Prompt buffer not frozen")
    require(tuple(model.dpsl.spectrum.query_anonymous.shape) == (12, 32)
            and not model.dpsl.spectrum.query_anonymous.requires_grad, "Anonymous Prompt buffer not frozen")
    require(all(graph.th == .9 for graph in model.gfr.graphs), "GraphLearn threshold differs")
    require(model.darg.cadr.alignment_anchor_index == 4
            and model.darg.positive.edge_anchor_index == 4, "Both anchor identities must be cognitive")
    require(model.darg.direct_scale == .02 and model.darg.direct_single_gate is False,
            "Direct double Gate contract differs")
    require(model.gfr.grl_scale == .03, "GRL scale differs")
    cognitive = aux["adjacencies"][4]
    same_pairs = model.darg.cadr.pair_features
    embeddings = model.gfr.encoders[4](x.split(run.MODAL_DIMS, dim=1)[4])
    require(torch.equal(same_pairs(embeddings[pair[:, 0]], embeddings[pair[:, 1]]),
                        same_pairs(embeddings[pair[:, 1]], embeddings[pair[:, 0]])),
            "Pair feature symmetry differs")
    deltas = aux["delta_adjacencies"]
    signed = 0
    for modality, (original, new, delta, direct) in enumerate(zip(
            aux["adjacencies"], aux["unified_adjacencies"], deltas, aux["direct_delta_adjacencies"])):
        require(torch.allclose(new, new.T, atol=2e-7, rtol=0), "Adjacency is asymmetric")
        require(torch.equal(new.diag(), original.diag()), "Graph diagonal changed")
        require(bool((delta[original <= 0] == 0).all()), "A graph edge was added")
        require(torch.equal(direct, delta.clamp_min(0)), "Positive-DCR support differs")
        expected = .02 * aux["gates"][modality] * aux["bounded_residuals"][modality]
        torch.testing.assert_close(aux["direct_messages"][modality], expected, atol=0, rtol=0)
        signed += int((delta > 0).any() and (delta < 0).any())
    return {"signed_modalities": signed, "signed_update_observation":
            "observed" if signed else "no_signed_modality_in_this_fold",
            "residual_off_exact": True,
            "cognitive_support_edges_unique": int(torch.triu(cognitive > 0, 1).sum()),
            "graph_delta": deltas[4].detach().clone(), "full_logp": full.detach()}


def formal_grad_check(model, opts, x, labels, train, pair, same, config, test, fold):
    model.train()
    rng = run.historical.capture_rng_state()
    poisoned = labels.clone()
    poisoned[test] = (poisoned[test] + 1) % 3
    _, control = model(x, poisoned, train, pair, same)
    cls_bad, base_bad, al_bad, target_bad = run.historical_loss(
        control, poisoned, train, model, "paper_aloss", .01)
    run.historical.restore_rng(rng)
    run.zero_optimizers(opts)
    _, aux = model(x, labels, train, pair, same)
    cls, base, aloss, target = run.historical_loss(aux, labels, train, model, "paper_aloss", .01)
    for original, alternate in zip((cls, base, aloss, aux["dpsl_loss"], target["target"]),
                                   (cls_bad, base_bad, al_bad, control["dpsl_loss"], target_bad["target"])):
        require(torch.equal(original, alternate), "Outer test labels changed the training objective")
    for key in aux["scmr_objectives"]:
        require(torch.equal(aux["scmr_objectives"][key], control["scmr_objectives"][key]),
                f"Outer test labels changed DPSL objective {key}")
    require(torch.equal(aux["logits"], control["logits"]), "Test labels affected train forward")
    del control, cls_bad, base_bad, al_bad, target_bad
    rel_logits = aux["pair_logits"]
    expected_bce = torch.stack([F.binary_cross_entropy_with_logits(j, same) for j in rel_logits]).mean()
    gaps = torch.stack([v[same.bool()].mean() - v[~same.bool()].mean() for v in rel_logits])
    require(torch.equal(expected_bce, aux["pair_loss"]), "Relation BCE differs")
    require(torch.equal(F.relu(.02 - gaps).mean(), aux["preserve_loss"]), "Preservation loss differs")
    require(torch.equal(base, cls + aux["graph_loss"] + .05 * aux["pair_loss"]
                        + .01 * aux["adv_loss"] + .01 * aux["preserve_loss"] + .01 * aloss),
            "Weighted S02 objective differs")
    pair_grads = torch.autograd.grad(aux["adv_loss"], aux["pair_reprs"],
                                     retain_graph=True, allow_unused=True)
    require(pair_grads[4] is None and all(g is not None and g.norm() > 0
                                         for i, g in enumerate(pair_grads) if i != 4),
            "Conditional GRL anchor-detach direction differs")
    probe = torch.tensor([1.0, 3.0], device=x.device, requires_grad=True)
    direction = torch.autograd.grad(model.darg.cadr.grad_reverse(probe, .03).sum(), probe)[0]
    require(torch.equal(direction, torch.full_like(probe, -.03)), "GRL sign differs")
    gate = model.gfr.adaptive_gate_logits
    initial = gate.detach().clone()
    budget = run.dpsl_optimizer_step(model, opts, base, aux["scmr_objectives"], .001)
    graph_grads = run.graph_gradient_audit(model, aux)
    run.validate_aloss_gradient_audit(graph_grads)
    measured = {
        "encoders": [grad_norm(module.parameters()) for module in model.gfr.encoders],
        "scorers": [grad_norm(module.parameters()) for module in model.gfr.scorers],
        "gfr": grad_norm(model.graph_parameters()), "cam": grad_norm(model.cam.parameters()),
        "dpsl": grad_norm(model.dpsl.parameters()),
    }
    require(all(z > 0 for z in (*measured["encoders"], *measured["scorers"],
                                measured["gfr"], measured["cam"], measured["dpsl"])),
            "Expected an active GFR, CAM, Relation Encoder/Scorer or DPSL gradient")
    require(budget["shared_cap"] == .001, "DPSL gradient cap differs")
    gate_change = float((gate.detach() - initial).norm())
    frozen = config.get("freeze_adaptive_gate", False)
    if frozen:
        require(gate.grad is None and gate not in opts["darg"].state and gate_change == 0,
                "D1 Gate updated or created an optimizer state")
    else:
        require(gate.grad is not None and gate.grad.norm() > 0 and gate in opts["darg"].state
                and gate_change > 0, "Adaptive Gate did not receive a gradient and update")
    after = run.author_epoch_metrics(eval_without_dpsl_ema(model, x, labels, train, pair, same), test, labels)
    if config["group"] == "H1":
        record = run.warmup_references()[fold]["first_formal_history"]
        require(float((base + aux["dpsl_loss"]).detach()) == record["loss"],
                "Recomputed S02 epoch-11 loss differs")
        require(after["ACC"] == record["test_acc"], "Recomputed S02 epoch-11 validation ACC differs")
    return {"gradient_norms": measured, "graphlearn_gradients": graph_grads,
            "gate_gradient_norm": float(gate.grad.norm()) if not frozen else None,
            "gate_update_l2": gate_change, "gate_optimizer_state": gate in opts["darg"].state,
            "train_label_boundary": "PASS", "BCE_GRL_preservation": "PASS", "shared_budget": budget}


@torch.no_grad()
def eval_without_dpsl_ema(model, x, labels, train, pair, same):
    model.eval()
    return model(x, labels, train, pair, same)[0]


def fold_check(fold, groups):
    run.configure_worker()
    x_np, y_np, _, splits = run.load_data()
    x = torch.from_numpy(x_np).cuda()
    y = torch.from_numpy(y_np).cuda()
    split = splits[fold - 1]
    test = np.asarray(split["test_idx"], dtype=np.int64)
    train = torch.as_tensor(split["train_idx"], device=x.device)
    test_t = torch.as_tensor(test, device=x.device)
    output, reference = {}, None
    for group in ("H1", *groups):
        run.historical.seed_everything(0)
        config = run.group_config(group)
        model, opts, opt_config, pair, same, warm, pairing, stats, relations = run.prepare_warmup(
            group, fold, config, x, y, train, test, y_np)
        require(opt_config == {"class":"Adam/AdamW","gfr":[.012,.11],"cam":[.012,.11],
                               "darg":[.001,.01],"dpsl":[.0005,.01]}, "Optimizer contract differs")
        graph = graph_contract(model, x, y, train, pair, same)
        delta = graph.pop("graph_delta")
        logp = graph.pop("full_logp")
        if reference is None:
            reference = (warm, pairing, delta, logp)
        ref_w, ref_p, ref_delta, ref_logp = reference
        require(warm["steps"] == ref_w["steps"], f"{group} per-epoch warmup differs")
        for key in ("pre_epoch_11_non_gate_sha256", "pre_epoch_11_optimizer_non_gate_sha256"):
            require(pairing[key] == ref_p[key], f"{group} paired epoch-11 {key} differs")
        if group in ("P1", "P2", "P3"):
            torch.testing.assert_close(delta, ref_delta, atol=2e-7, rtol=1e-6)
            torch.testing.assert_close(logp, ref_logp, atol=5e-6, rtol=1e-6)
        graph["cognitive_delta_max_abs_diff_vs_H1"] = float((delta - ref_delta).abs().max())
        graph["logp_max_abs_diff_vs_H1"] = float((logp - ref_logp).abs().max())
        gradients = formal_grad_check(model, opts, x, y, train, pair, same, config, test_t, fold)
        output[group] = {"status":"PASS", "fold":fold,
                         "gate":warm["gate_initialization"], "warmup":pairing,
                         "graph":graph, "gradients":gradients,
                         "relations":relations["relation_sampling_audit"]}
        print(f"Preflight {group} fold {fold:02d} PASS", flush=True)
    return fold, output, ref_w["steps"], int(torch.cuda.max_memory_reserved() / 2**20)


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



def verify_abide():
    fixed = abide_fixed_tensor_regression(torch.device("cuda:0"))
    archive = ROOT.parent / "experiment83_8_5_1_tadpole_y3_cognitive_darg_fidelity_factorial"
    certificate_file = archive / "audits/abide_checkpoint_replay.json"
    certificate = json.loads(certificate_file.read_text())
    require(certificate["status"] == "PASS" and certificate["certificate"]["best_reloaded"]
            and certificate["certificate"]["latest_next_training_exact"], "Archived ABIDE replay certificate")
    for key, path in certificate["checkpoint_paths"].items():
        require(run.sha_file(Path(path)) == certificate["checkpoint_sha256"][key],
                f"ABIDE checkpoint SHA differs for {key}")
    source = ROOT / "code/sources/history_darg/soft_positive_dcr.py"
    e74 = Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/soft_positive_dcr.py")
    require(run.sha_file(e74) == run.sha_file(source), "E74.4 Positive-DCR differs from ABIDE archive")
    scm_root = ROOT / "code/sources/E82_4_6_G2_SCMR_DA_SRPE/code/history_base"
    abide_root = ROOT.parent / "DPSL/code/history_base"
    common_exact = {}
    for name in ("common/spectral.py", "common/optimization.py", "config.json"):
        tadpole = scm_root / name
        abide = abide_root / name
        require(run.sha_file(tadpole) == run.sha_file(abide), f"ABIDE SCMR shared arithmetic differs at {name}")
        common_exact[name] = run.sha_file(tadpole)
    g2_text = (scm_root / "g2.py").read_text()
    require("attention.shape[1] == 8" in g2_text and "attention[:, 8:12]" in g2_text,
            "SCMR 8-query fallback and 12-query route are not both present")
    require("topk(2, dim=-1)" in g2_text and "torch.minimum" in g2_text,
            "SCMR 12-query class-pair routing is not source-anchored")
    evidence_text = (scm_root / "common/evidence.py").read_text()
    require("E8242 / \"G2_pdjc_da_srpe/checkpoints\"" in evidence_text,
            "TADPOLE Gaussian reference bank binding is not explicit")
    return {"status":"PASS", "four_modality_CORR_regression":fixed,
            "archived_checkpoint_certificate_sha256":run.sha_file(certificate_file),
            "positive_dcr_source":run.sha_file(source), "SCMR_numeric_source_match":True,
            "scmr_shared_exact_sha256": common_exact,
            "scmr_adaptation": {
                "g2_sha256": run.sha_file(scm_root / "g2.py"),
                "abide_g2_sha256": run.sha_file(abide_root / "g2.py"),
                "eight_query_fallback": True,
                "twelve_query_route": True,
                "adaptation_scope": "MMGPL-12 attention grouping and native Top-2 class-pair routing",
            },
            "evidence_binding_sha256": run.sha_file(scm_root / "common/evidence.py"),
            "scope":"Current fixed-tensor ABIDE regression, archived E82.4.6.10-G1 Fold 8 checkpoint SHA, and exact shared SCMR arithmetic with an explicit MMGPL-12 adapter."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-fold", type=int)
    args = parser.parse_args()
    run.configure_worker()
    if args.probe_fold:
        fold_check(args.probe_fold, run.TRAIN_GROUPS)
        return
    data = verify_locked_data()
    provenance = verify_sources_and_configs()
    abide = verify_abide()
    historic = json.loads((ROOT / "audits/historical_replay.json").read_text())
    require(historic["status"] == "PASS" and historic["folds"] == 20, "Historical replay missing")
    template = Path("/data/adodas/SPromptGL-main/result_Tadpole.txt")
    source = json.loads((ROOT / "source_manifest.json").read_text())
    require(run.sha_file(template) == source["author_repository"]["result_Tadpole_sha256"],
            "Original author log hash differs")
    parsed, _ = parse(template, 300)
    require(all(len(fold["epochs"]) == 300 for fold in parsed), "Original log epochs differ")
    start = time.monotonic()
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(fold_check, fold, run.TRAIN_GROUPS) for fold in range(1, 11)]
        records = [job.result() for job in jobs]
    groups = {group:[output[group] for _, output, _, _ in records] for group in ("H1", *run.TRAIN_GROUPS)}
    write = lambda name, value: run.write_json(ROOT / "audits" / name, value)
    write("warmup_pairing.json", {"status":"PASS", "reference":"H1/S02",
          "per_epoch_hashes_source":"S02 deterministic reexecution, verified against the official warmup",
          "reference_steps_by_fold":{str(fold): steps for fold, _, steps, _ in records},
          "groups":{g:[row["warmup"] for row in rows] for g, rows in groups.items()}})
    write("gate_initialization.json", {"status":"PASS", "formula":"scale * sigmoid(theta)",
          "groups":{g:[{"fold":row["fold"], **row["gate"], **row["graph"]} for row in rows]
                    for g, rows in groups.items()}})
    write("relation_sampling_preflight.json", {"status":"PASS",
          "groups":{g:[row["relations"] for row in rows] for g, rows in groups.items()}})
    write("abide_tadpole_parity.json", abide)
    write("method_fidelity.json", {"status":"PASS", "data":data, "sources":provenance, "ABIDE":abide,
          "formal_gradients":{g:[row["gradients"] for row in rows] for g, rows in groups.items()}})
    write("resource_probe.json", {
          "status":"PASS", "experiment":run.EXPERIMENT, "worker_count":8,
          "device":torch.cuda.get_device_name(0), "CUDA_VISIBLE_DEVICES":os.environ["CUDA_VISIBLE_DEVICES"],
          "device_memory_MiB": torch.cuda.get_device_properties(0).total_memory // 2**20,
          "pytorch":torch.__version__, "cuda":torch.version.cuda,
          "peak_reserved_mib_by_fold":[peak for _, _, _, peak in records],
          "worker_probe_elapsed_seconds":time.monotonic() - start,
          "smi":subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.total",
                                        "--format=csv,noheader"], text=True).strip()})
    write("preflight.json", {"status":"PASS", "groups":list(run.TRAIN_GROUPS),
          "formal_training_allowed":True, "implementation_sha256":run.implementation_fingerprint(),
          "historical_checkpoint_folds":20, "training_transitions":80,
          "author_template_folds":10, "data":data})


if __name__ == "__main__":
    main()
