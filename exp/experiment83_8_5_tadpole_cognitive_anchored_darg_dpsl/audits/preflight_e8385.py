"""Verify the registered controls, all warmups, and live PPU method behavior."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as run
from source_runtime import E74_GROUP

Z0 = ROOT.parent / "experiment83_8_4_tadpole_y3_darg_dpsl_factorial"
ANCHORS = [(None, None, .01), (4, None, .01), (None, None, .1),
           (4, None, .1), (4, 4, .1)]


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def snapshot(model):
    return {name: {"shape": list(param.shape), "trainable": param.requires_grad,
                   "elements": param.numel()}
            for name, param in model.named_parameters()}


def config_check():
    z0 = json.loads((Z0 / "configs/Z0.json").read_text())
    metadata = {"experiment", "group", "method", "kind", "training",
                "warmup_reference", "reuse_reference"}
    for group, (alignment, edges, cognitive_scale) in zip(run.FORMAL_GROUPS, ANCHORS):
        config = run.group_config(group)
        reference = {key: value for key, value in z0.items() if key not in metadata}
        expected = {**reference, "alignment_anchor_index": alignment, "edge_anchor_index": edges,
                    "darg_gate_scales": [.01, .01, .01, .01, cognitive_scale, .01],
                    "darg_initial_gate_probabilities": [.1] * 6}
        require({key: value for key, value in config.items() if key not in metadata} == expected,
                f"{group} has a change outside its preregistered control")
        require(config["warmup_reference"] == ("N3" if group == "R0" else "R0"),
                f"{group} warmup lineage differs")


def data_check(x, y, ids, splits):
    require(run.sha_file(ROOT / "inputs/sample_ids.csv")
            == "3c354e2623128209544b23d84fbac0e436f55b4c898c74af83d2e99e7f61d8e0",
            "Patient ID identities differ")
    require(len(ids.drop_duplicates()) == 598 and np.isfinite(x).all(), "Invalid locked data")
    regenerated = StratifiedKFold(10, shuffle=True, random_state=10).split(x, y)
    for split, (train, test) in zip(splits, regenerated, strict=True):
        require(train.tolist() == split["train_idx"] and test.tolist() == split["test_idx"]
                and split["val_equals_test"] is True, f"Split {split['fold']} changed")
    require(sorted(index for split in splits for index in split["test_idx"]) == list(range(598)),
            "Patient OOF coverage differs from the locked cohort")


def grade_graph(group, model, x, pair, initial=False):
    with torch.no_grad():
        fused, original, aux = model.darg.positive.rudag_forward(
            model.gfr, list(x.split(run.MODAL_DIMS, dim=1)), pair, E74_GROUP,
            collect_diagnostics=True)
    for index, (adj, adj0, q, row, col, diag, gate) in enumerate(zip(
            aux["unified_adjs"], original, aux["edge_q"], aux["edge_rows"], aux["edge_cols"],
            aux["edge_diagnostics"], aux["gates"], strict=True)):
        label = f"{group} {run.MODAL_NAMES[index]}"
        require(diag["modality"] == run.MODAL_NAMES[index], f"{label} modality identity differs")
        require(bool(torch.isfinite(q).all()) and bool(torch.isfinite(adj).all())
                and diag["normalized_graph_finite"], f"{label} has a nonfinite graph or score")
        require(torch.equal(adj.diag(), adj0.diag()) and torch.equal(adj.gt(0), adj0.gt(0))
                and adj.min().item() >= 0.0 and torch.equal(adj, adj.T),
                f"{label} changed support, diagonal, or symmetry")
        require(bool((adj0[row, col] > 0).all()) and bool((row < col).all())
                and len(q) == len(row) == diag["scored_edges"], f"{label} scored an invalid edge")
        relative = (adj[row, col] - adj0[row, col]).abs() / adj0[row, col]
        require(bool((relative <= .8 * gate + 2e-7).all()), f"{label} exceeded the reweighting bound")
        expected = diag["original_effective_edges"] if group == "R4" and index == 4 else None
        if expected is not None:
            require(diag["scored_edges"] == expected, "R4 COGNITIVE_TEST misses effective edges")
        if initial:
            require(abs(float(gate) - model.darg.gate_scales[index] * .1) < 1e-8,
                    f"{label} has incorrect initial effective gate")
        diag["max_abs_relative_edge_update"] = float(relative.max()) if len(relative) else 0.0
        diag["effective_edge_coverage"] = len(row) / max(diag["original_effective_edges"], 1)
    require(bool(torch.isfinite(fused).all()), f"{group} has nonfinite fused features")
    return original, aux


def compare_graphs(graphs):
    for reference, treatment, unchanged in (
            ("R0", "R1", range(6)), ("R0", "R2", (0, 1, 2, 3, 5)),
            ("R1", "R3", (0, 1, 2, 3, 5)), ("R3", "R4", (0, 1, 2, 3, 5))):
        orig_a, aux_a = graphs[reference]
        orig_b, aux_b = graphs[treatment]
        require(all(torch.equal(a, b) for a, b in zip(orig_a, orig_b)),
                f"{reference} vs {treatment} changed GraphLearn at the same state")
        for name in ("unified_adjs", "messages", "direct_messages", "edge_rows", "edge_cols", "edge_q"):
            require(all(torch.equal(aux_a[name][i], aux_b[name][i]) for i in unchanged),
                    f"{reference} vs {treatment} changed {name} in another modality")
        if treatment in ("R2", "R3"):
            for name in ("edge_rows", "edge_cols", "edge_q"):
                require(torch.equal(aux_a[name][4], aux_b[name][4]),
                        f"{treatment} gate unexpectedly changed candidate selection or scoring")
            require(abs(float(aux_b["gates"][4] / aux_a["gates"][4]) - 10) < 2e-6,
                    f"{treatment} cognitive effective gate did not increase tenfold")
    require(graphs["R3"][1]["edge_rows"][4].numel() < graphs["R4"][1]["edge_rows"][4].numel(),
            "R4 did not expand scoring")


def norm(parameters):
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    require(all(bool(torch.isfinite(grad).all()) for grad in gradients), "Nonfinite gradient")
    return float(torch.stack([grad.norm() for grad in gradients]).norm()) if gradients else 0.0


def budget_check(budget):
    require(budget["shared_cap"] == .001, "Historical shared gradient cap changed")
    for name in ("gfr", "darg", "cam", "head"):
        row = budget["families"][name]
        require(0 <= row["total_applied"] <= .001 * row["historical"]
                and abs(row["applied_dot"]) <= row["dot_bound"],
                f"DPSL exceeded the {name} shared gradient constraint")
    require(any(budget["families"][name]["total_applied"] > 0 for name in ("gfr", "darg", "cam", "head")),
            "No shared DPSL gradient applied")
    for name in ("prompt", "control"):
        require(budget["families"][name]["main"] > 0, f"Zero {name} classification gradient")


def live_method(group, model, config, x, y, train, test_np, pair, same, y_np, supports):
    model.train()
    gate = model.gfr.adaptive_gate_logits
    optimizers, _ = run.make_optimizers(model, config, formal=True)
    require(any(p is gate for pg in optimizers["darg"].param_groups for p in pg["params"]),
            f"{group} gate missing from DARG optimizer")
    require(model.darg.cadr.alignment_anchor_index == config["alignment_anchor_index"]
            and model.darg.positive.edge_anchor_index == config["edge_anchor_index"],
            f"{group} runtime differs from configuration")
    require(not model.gfr.edge_gate_logits.requires_grad, "Retired CADR edge gates should be frozen")
    label_alt = y_np.copy()
    label_alt[test_np] = (label_alt[test_np] + 1) % 3
    alt_pair, alt_same = run.historical.make_pairs(train.cpu().numpy(), label_alt, 1, supports)
    require(np.array_equal(alt_pair, pair.cpu().numpy()) and np.array_equal(alt_same, same.cpu().numpy())
            and np.isin(alt_pair, train.cpu().numpy()).all(), f"{group} consumes held-out relation targets")
    rng = run.historical.capture_rng_state()
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    alt = torch.as_tensor(label_alt, device=x.device)
    def collect(labels):
        logp, aux = model(x, labels, train, pair, same)
        classification, total, supervision, target = run.historical_loss(
            aux, labels, train, model, "paper_aloss", .01)
        outputs = [logp.detach(), classification.detach(), total.detach(), supervision.detach(),
                   aux["pair_loss"].detach(), aux["adv_loss"].detach(), aux["preserve_loss"].detach(),
                   *(v.detach() for v in aux["scmr_objectives"].values()), *[v.detach() for v in aux["pair_logits"]]]
        require(all(bool(torch.isfinite(value).all()) for value in outputs), f"{group} nonfinite train state")
        return aux, total, outputs, run.supervision_target_sha(target)
    _, _, changed, alt_target = collect(alt)
    run.historical.restore_rng(rng)
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    aux, total, outputs, target = collect(y)
    require(target == alt_target and all(torch.equal(a, b) for a, b in zip(outputs, changed, strict=True)),
            f"{group} training or supervision depends on held-out true labels")

    cond = F.one_hot(same.long(), num_classes=2).to(x.dtype)
    reps = aux["pair_reprs"]
    raw_logits = torch.cat([model.gfr.discriminator(rep, cond) for rep in reps])
    raw_loss = F.cross_entropy(raw_logits, aux["adv_info"]["targets"])
    reference = torch.autograd.grad(raw_loss, reps, retain_graph=True)
    actual = torch.autograd.grad(aux["adv_loss"], reps, retain_graph=True, allow_unused=True)
    adv = []
    for i, (observed, plain) in enumerate(zip(actual, reference, strict=True)):
        if i == config["alignment_anchor_index"]:
            require(observed is None, f"{group} does not detach the anchor adversarial input")
        else:
            require(observed is not None and float(plain.norm()) > 0
                    and torch.equal(observed, -model.gfr.grl_scale * plain),
                    f"{group} GRL differs at modality {i}")
        adv.append(0.0 if observed is None else float(observed.norm()))
    anchor_encoder = tuple(model.gfr.encoders[4].parameters())
    anchor_bce = torch.autograd.grad(aux["pair_loss"], anchor_encoder, retain_graph=True)
    require(any(float(value.norm()) > 0 for value in anchor_bce), "Cognitive relation encoder detached from BCE")
    budget = run.dpsl_optimizer_step(model, optimizers, total, aux["scmr_objectives"], .001)
    budget_check(budget)
    gradients = {"relation_encoders": [norm(module.parameters()) for module in model.gfr.encoders],
                 "relation_scorers": [norm(module.parameters()) for module in model.gfr.scorers],
                 "discriminator": norm(model.gfr.discriminator.parameters()),
                 "gate": gate.grad.detach().cpu().tolist(),
                 "gfr": norm(model.graph_parameters()), "cam": norm(model.cam.parameters()),
                 "dpsl": norm(model.dpsl.parameters())}
    require(all(value > 0 for name in ("relation_encoders", "relation_scorers", "discriminator", "gfr", "cam", "dpsl")
                for value in (gradients[name] if isinstance(gradients[name], list) else [gradients[name]]))
            and norm((gate,)) > 0 and gate in optimizers["darg"].state, f"{group} missing required training gradient")
    return {"status": "PASS", "adversarial_gradient_norms": adv,
            "cognitive_bce_encoder_gradient_norm": float(torch.stack([v.norm() for v in anchor_bce]).norm()),
            "gradients": gradients, "optimizer_families": budget, "test_label_perturbation": "PASS",
            "targets_sha256": target}


def main():
    config_check()
    r0_path = ROOT / "audits/r0_identity.json"
    r0 = json.loads(r0_path.read_text())
    require(r0["status"] == "PASS" and r0["best_replayed"] == 10 and r0["deterministic_next_steps"] == 10,
            "All ten historical folds must be independently replayed")
    x_np, y_np, ids, splits = run.load_data()
    data_check(x_np, y_np, ids, splits)
    run.configure_worker()
    device = torch.device("cuda:0")
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    pairing_by_group = {group: {} for group in run.FORMAL_GROUPS}
    pre_epoch11 = {}
    methods, warmup_graphs, best_graphs = {}, {}, {}
    parameter_reference = None
    for split in splits:
        fold = int(split["fold"])
        train = torch.as_tensor(split["train_idx"], device=device)
        test_np = np.asarray(split["test_idx"])
        for group in run.FORMAL_GROUPS:
            config = run.group_config(group)
            if fold == 1 and group == "R0":
                run.historical.seed_everything(run.SEED)
                support_model = run.build_model(config, fold, device)
                supports = run.historical.anchor_pair_support(support_model, x)
                del support_model
            run.historical.seed_everything(run.SEED)
            model, _, _, pair, same, _, pairing, _ = run.prepare_warmup(
                group, fold, {**config, "warmup_reference": "R0"}, x, y, train, test_np, y_np, verify_logs=True)
            pairing_by_group[group][str(fold)] = pairing
            start = pairing["pre_epoch_11_state"]
            if group == "R0":
                pre_epoch11[str(fold)] = start
            require(pre_epoch11[str(fold)] == start, f"{group} fold {fold} pre-epoch-11 state differs")
            if fold == 1:
                if parameter_reference is None:
                    parameter_reference = snapshot(model)
                require(parameter_reference == snapshot(model), f"{group} differs in named parameter identity")
                model.eval()
                warmup_graphs[group] = grade_graph(group, model, x, pair, initial=True)
                historical = torch.load(ROOT / "groups/R0/best_checkpoints/fold01.pt",
                                        map_location="cpu", weights_only=False)
                # Only the configured runtime differs across groups at this diagnostic checkpoint.
                model.load_state_dict(historical["model"], strict=True)
                best_graphs[group] = grade_graph(group, model, x, pair)
                methods[group] = live_method(group, model, config, x, y, train, test_np, pair, same, y_np, supports)
                methods[group]["initial_graph"] = warmup_graphs[group][1]["edge_diagnostics"]
                methods[group]["historical_best_shared_state_graph"] = best_graphs[group][1]["edge_diagnostics"]
            print(f"PASS {group} fold {fold:02d} warmup", flush=True)
        if fold == 1:
            compare_graphs(warmup_graphs)
            compare_graphs(best_graphs)
            warmup_graphs.clear()
            best_graphs.clear()
    run.write_json(ROOT / "audits/warmup_pairing_preflight.json",
                   {"status": "PASS", "reference_folds": 10, "treatment_folds": 40,
                    "warmup_author_metrics_independently_recomputed": True, "groups": pairing_by_group})
    run.write_json(ROOT / "audits/method_fidelity.json",
                   {"status": "PASS", "data_identity": "PASS", "parameter_identity": parameter_reference,
                    "config_isolation": "PASS", "same_state_graph_isolation": "PASS", "groups": methods})
    run.write_json(ROOT / "audits/preflight_start.json",
                   {"experiment": run.EXPERIMENT, "status": "PASS", "formal_training_allowed": True,
                    "r0_audit_sha256": run.sha_file(r0_path),
                    "r0_regression_sha256": run.sha_file(ROOT / "audits/r0_regression.json"),
                    "r0_best_sha256": {fold: row["best_sha256"] for fold, row in r0["groups"]["R0"].items()},
                    "method_sha256": run.sha_file(ROOT / "audits/method_fidelity.json"),
                    "warmup_audit_sha256": run.sha_file(ROOT / "audits/warmup_pairing_preflight.json"),
                    "groups": {g: {"status": "PASS", "formal_training_allowed": g != "R0",
                                   "config_sha256": run.sha_file(ROOT / f"configs/{g}.json")}
                               for g in run.FORMAL_GROUPS}, "warmup": pre_epoch11})


if __name__ == "__main__":
    main()
