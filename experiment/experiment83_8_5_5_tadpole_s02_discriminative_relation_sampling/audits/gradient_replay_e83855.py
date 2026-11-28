"""Read-only DARG relation/classification gradient-direction diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83855 as runner

HISTORICAL = {
    "H0": (Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl"), "R0"),
    "H1": (Path("/model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial"), "S02"),
}
GROUPS = runner.FORMAL_GROUPS


def source_checkpoint(group: str, fold: int) -> Path:
    if group in HISTORICAL:
        root, name = HISTORICAL[group]
        return root / "groups" / name / "best_checkpoints" / f"fold{fold:02d}.pt"
    return ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"


def diagnostic_rng_state(payload):
    if "rng" in payload:
        return payload["rng"], "checkpoint_rng"
    runner.historical.seed_everything(runner.SEED)
    return runner.historical.capture_rng_state(), "seed_0_replay_historical_rng_absent"


def flat_gradients(grads, names):
    values, used = [], []
    for name, grad in zip(names, grads):
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            continue
        if float(grad.detach().norm()) == 0.0:
            continue
        values.append(grad.detach().reshape(-1))
        used.append(name)
    return (torch.cat(values) if values else None), used


def cosine(left, right):
    if left is None or right is None or left.numel() == 0 or right.numel() == 0:
        return None
    left_norm, right_norm = left.norm(), right.norm()
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return None
    return float(torch.dot(left, right) / (left_norm * right_norm))


def true_margin(logp, labels, rows):
    selected = logp.index_select(0, rows).double()
    truth = labels.index_select(0, rows)
    true_logp = selected.gather(1, truth[:, None]).squeeze(1)
    masked = selected.clone()
    masked.scatter_(1, truth[:, None], float("-inf"))
    return float((true_logp - masked.max(dim=1).values).mean())


def nll(logp, labels, rows):
    return float(runner.historical.weighted_nll(logp.double(), labels, rows))


def margin_values(logp, labels, rows):
    selected = logp.index_select(0, rows).double()
    truth = labels.index_select(0, rows)
    true_logp = selected.gather(1, truth[:, None]).squeeze(1)
    masked = selected.clone()
    masked.scatter_(1, truth[:, None], float("-inf"))
    return (true_logp - masked.max(dim=1).values).detach().cpu().tolist()


def build_loaded(group: str, fold: int, payload, config, device):
    model = runner.build_model(config, fold, device, spectral_stats=payload["spectral_stats"])
    model.load_state_dict(payload["model"], strict=True)
    if model.use_darg:
        model.darg.enter_positive_dcr()
    optimizers, optimizer_config = runner.make_optimizers(model, config, formal=True)
    saved_optimizers = payload.get("optimizers", {})
    if set(saved_optimizers) != set(optimizers):
        raise RuntimeError(f"{group} fold {fold}: optimizer keys differ in diagnostic replay")
    for name, state in saved_optimizers.items():
        optimizers[name].load_state_dict(state)
    return model, optimizers, optimizer_config


def save_vectors(model, names):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()
            if name in names}


def vector_delta(before, after, names):
    values = [after[name].reshape(-1) - before[name].reshape(-1) for name in names
              if name in before and name in after]
    return torch.cat(values) if values else None


def relation_update_effect(group, fold, payload, config, x, labels, train, test, pair, same, device):
    model, optimizers, _ = build_loaded(group, fold, payload, config, device)
    model.eval()
    with torch.no_grad():
        before_logp, _ = model(x, labels, train, pair, same, dpsl_enabled=False)
    before = {
        "train_nll": nll(before_logp, labels, train),
        "train_margin": true_margin(before_logp, labels, train),
        "test_margin": true_margin(before_logp, labels, test),
        "test_margin_by_patient": margin_values(before_logp, labels, test),
    }
    model.train()
    before_parameters = {name: parameter.detach().clone()
                         for name, parameter in model.named_parameters()
                         if parameter.requires_grad}
    rng_state, rng_source = diagnostic_rng_state(payload)
    runner.historical.restore_rng(rng_state)
    runner.zero_optimizers(optimizers)
    _, aux = model(x, labels, train, pair, same, dpsl_enabled=False)
    relation_objective = 0.05 * aux["pair_loss"]
    relation_objective.backward()
    optimizers["darg"].step()
    relation_parameter_delta = [
        (parameter.detach() - before_parameters[name]).reshape(-1)
        for name, parameter in model.named_parameters()
        if name in before_parameters and parameter.requires_grad and name.startswith("darg.")]
    relation_update_norm = (float(torch.cat(relation_parameter_delta).norm())
                            if relation_parameter_delta else 0.0)
    model.eval()
    with torch.no_grad():
        after_logp, _ = model(x, labels, train, pair, same, dpsl_enabled=False)
    after = {
        "train_nll": nll(after_logp, labels, train),
        "train_margin": true_margin(after_logp, labels, train),
        "test_margin": true_margin(after_logp, labels, test),
        "test_margin_by_patient": margin_values(after_logp, labels, test),
        "logit_max_abs": float((after_logp - before_logp).abs().max()),
    }
    del model, optimizers
    return before, after, float(relation_objective.detach()), rng_source, relation_update_norm


def one_fold(group, fold, payload, config, x, labels, train, test, pair, same, device):
    model, _, _ = build_loaded(group, fold, payload, config, device)
    model.eval()
    with torch.enable_grad():
        _, aux = model(x, labels, train, pair, same, dpsl_enabled=False)
        native_logp = F.log_softmax(aux["native_logits"], dim=-1)
        classification_loss = runner.historical.weighted_nll(native_logp, labels, train)
        relation_loss = 0.05 * aux["pair_loss"]
        parameters = [(name, parameter) for name, parameter in model.named_parameters()
                      if parameter.requires_grad]
        names = [name for name, _ in parameters]
        tensors = [parameter for _, parameter in parameters]
        classification_grads = torch.autograd.grad(
            classification_loss, tensors, retain_graph=True, allow_unused=True)
        relation_grads = torch.autograd.grad(
            relation_loss, tensors, retain_graph=True, allow_unused=True)
        shared = []
        class_values, relation_values = [], []
        for name, class_grad, relation_grad in zip(names, classification_grads, relation_grads):
            if (class_grad is not None and relation_grad is not None
                    and torch.isfinite(class_grad).all() and torch.isfinite(relation_grad).all()
                    and float(class_grad.detach().norm()) > 0.0
                    and float(relation_grad.detach().norm()) > 0.0):
                shared.append(name)
                class_values.append(class_grad.detach().reshape(-1))
                relation_values.append(relation_grad.detach().reshape(-1))
        class_vector = torch.cat(class_values) if class_values else None
        relation_vector = torch.cat(relation_values) if relation_values else None
        all_class, class_names = flat_gradients(classification_grads, names)
        all_relation, relation_names = flat_gradients(relation_grads, names)
        class_grad_finite = all(grad is None or torch.isfinite(grad).all()
                                for grad in classification_grads)
        relation_grad_finite = all(grad is None or torch.isfinite(grad).all()
                                   for grad in relation_grads)
        relation_bce_value = float(aux["pair_loss"].detach())
        del model, aux

    # The next full optimizer step is replayed on a fresh model initialized
    # from the frozen Best checkpoint. It is never written back to disk.
    actual_model, actual_opts, _ = build_loaded(group, fold, payload, config, device)
    rng_state, rng_source = diagnostic_rng_state(payload)
    runner.historical.restore_rng(rng_state)
    actual_model.train()
    actual_before = save_vectors(actual_model, shared)
    runner.zero_optimizers(actual_opts)
    _, actual_aux = actual_model(x, labels, train, pair, same, dpsl_enabled=True)
    _, historical_total, _, _ = runner.historical_loss(
        actual_aux, labels, train, actual_model, "paper_aloss", 0.01)
    if actual_model.dpsl is not None:
        runner.dpsl_optimizer_step(actual_model, actual_opts, historical_total,
                                   actual_aux["scmr_objectives"], 0.001)
    else:
        historical_total.backward()
        runner.step_optimizers(actual_opts)
    actual_after = save_vectors(actual_model, shared)
    update_vector = vector_delta(actual_before, actual_after, shared)
    del actual_model, actual_opts, actual_aux

    before_effect, after_effect, relation_objective, relation_rng_source, relation_update_norm = relation_update_effect(
        group, fold, payload, config, x, labels, train, test, pair, same, device)

    row = {
        "group": group, "fold": fold, "status": "PASS",
        "classification_loss": float(classification_loss.detach()),
        "relation_bce_loss": relation_bce_value,
        "relation_supervision_coefficient": 0.05,
        "classification_gradient_norm_all": (float(all_class.norm()) if all_class is not None else 0.0),
        "relation_gradient_norm_all": (float(all_relation.norm()) if all_relation is not None else 0.0),
        "classification_gradient_finite": bool(class_grad_finite),
        "relation_gradient_finite": bool(relation_grad_finite),
        "shared_parameter_names": shared,
        "shared_parameter_count": len(shared),
        "shared_parameter_tensor_count": len(shared),
        "effective_shared_gradient_ratio": (
            sum(dict(parameters)[name].numel() for name in shared)
            / sum(parameter.numel() for _, parameter in parameters) if parameters else 0.0),
        "shared_classification_gradient_norm": (float(class_vector.norm()) if class_vector is not None else 0.0),
        "shared_relation_gradient_norm": (float(relation_vector.norm()) if relation_vector is not None else 0.0),
        "shared_relation_to_classification_norm_ratio": (
            float(relation_vector.norm() / class_vector.norm())
            if class_vector is not None and relation_vector is not None and float(class_vector.norm()) else None),
        "shared_gradient_cosine": cosine(class_vector, relation_vector),
        "all_classification_gradient_parameter_count": len(class_names),
        "all_relation_gradient_parameter_count": len(relation_names),
        "shared_path_status": "PASS" if shared else "NO_SHARED_GRAD_PATH",
        "actual_optimizer_update_norm": (float(update_vector.norm()) if update_vector is not None else 0.0),
        "actual_update_cosine_with_classification_gradient": cosine(update_vector, class_vector),
        "actual_update_cosine_with_relation_gradient": cosine(update_vector, relation_vector),
        "actual_optimizer_update_finite": bool(update_vector is not None and torch.isfinite(update_vector).all()),
        "relation_only_objective": relation_objective,
        "relation_only_parameter_update_norm": relation_update_norm,
        "relation_update_native_logit_max_abs": after_effect["logit_max_abs"],
        "relation_update_train_nll_before": before_effect["train_nll"],
        "relation_update_train_nll_after": after_effect["train_nll"],
        "relation_update_train_nll_delta": after_effect["train_nll"] - before_effect["train_nll"],
        "relation_update_train_margin_before": before_effect["train_margin"],
        "relation_update_train_margin_after": after_effect["train_margin"],
        "relation_update_train_margin_delta": after_effect["train_margin"] - before_effect["train_margin"],
        "relation_update_test_margin_before": before_effect["test_margin"],
        "relation_update_test_margin_after": after_effect["test_margin"],
        "relation_update_test_margin_delta": after_effect["test_margin"] - before_effect["test_margin"],
        "test_patient_margins": {
            "sample_indices": test.detach().cpu().tolist(),
            "before": before_effect["test_margin_by_patient"],
            "after": after_effect["test_margin_by_patient"],
            "delta": [right - left for left, right in zip(
                before_effect["test_margin_by_patient"], after_effect["test_margin_by_patient"])],
        },
        "diagnostic_rng_restored": True,
        "actual_update_rng_source": rng_source,
        "relation_update_rng_source": relation_rng_source,
        "relation_loss_definition": "0.05 * DARG RelationScorer BCE (counterfactual diagnostic only)",
        "actual_update_definition": "one frozen-Best next optimizer step with the sealed historical total and DPSL path",
    }
    return row


def main():
    runner.configure_worker()
    device = torch.device("cuda:0")
    x_np, y_np, _, raw_splits = runner.load_data()
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y_np, dtype=torch.long, device=device)
    splits = {int(row["fold"]): row for row in raw_splits}
    rows = []
    for group in GROUPS:
        config = runner.group_config(group)
        for fold in range(1, 11):
            outer_rng = runner.historical.capture_rng_state()
            payload = torch.load(source_checkpoint(group, fold), map_location="cpu", weights_only=False)
            train = torch.as_tensor(splits[fold]["train_idx"], dtype=torch.long, device=device)
            test = torch.as_tensor(splits[fold]["test_idx"], dtype=torch.long, device=device)
            pair = payload["relation_pairs"].to(device)
            same = payload["relation_targets"].to(device)
            try:
                rows.append(one_fold(group, fold, payload, config, x, labels,
                                     train, test, pair, same, device))
            finally:
                runner.historical.restore_rng(outer_rng)
                del payload, pair, same
                torch.cuda.empty_cache()
            print(f"GRADIENT {group} fold {fold:02d}", flush=True)

    passed = sum(row["status"] == "PASS" for row in rows)
    no_shared = sum(row["shared_path_status"] == "NO_SHARED_GRAD_PATH" for row in rows)
    report = {
        "experiment": runner.EXPERIMENT, "status": "PASS" if passed == len(rows) == 80 else "FAIL",
        "device": torch.cuda.get_device_name(0), "folds": len(rows), "folds_passed": passed,
        "groups": list(GROUPS), "no_shared_grad_path_folds": no_shared,
        "diagnostic_only": True, "optimizer_unchanged": True, "new_loss_added": False,
        "relation_loss_definition": "0.05 * existing DARG RelationScorer BCE",
        "classification_loss_definition": "existing weighted train-row Native classification NLL",
        "checks": {
            "shared_parameter_intersection_recorded": True,
            "nonfinite_gradients_rejected": True,
            "actual_optimizer_update_direction_recorded": True,
            "counterfactual_relation_update_nll_margin_recorded": True,
            "rng_restored_after_each_fold": True,
            "pcgrad_or_gradient_rewrite_used": False,
        },
        "fold_results": rows,
    }
    target = ROOT / "audits/gradient_replay.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "folds": len(rows),
                      "folds_passed": passed, "no_shared_grad_path_folds": no_shared,
                      "path": str(target)}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
