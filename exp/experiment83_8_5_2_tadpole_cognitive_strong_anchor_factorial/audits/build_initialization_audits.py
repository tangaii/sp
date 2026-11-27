"""Aggregate formal-fold Gate initialization, learning and warmup-pair evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as runner


def effective_gate(group, payload):
    state = payload["model"]
    logits = state["darg.core.adaptive_gate_logits"].detach().cpu()
    probability = torch.sigmoid(logits).numpy()
    scale = np.asarray(runner.group_config(group)["darg_gate_scales"], dtype=np.float64)
    return (scale * probability).tolist(), probability.tolist(), scale.tolist()


def optimizer_gate_state(group, fold, payload):
    config = runner.group_config(group)
    model = runner.build_model(config, fold, torch.device("cpu"),
                               spectral_stats=payload["spectral_stats"])
    optimizers, _ = runner.make_optimizers(model, config, formal=True)
    optimizer = optimizers["darg"]
    gate = model.darg.core.adaptive_gate_logits
    state_dict = optimizer.state_dict()
    parameter_id = None
    for actual_group, serialized_group in zip(optimizer.param_groups,
                                               state_dict["param_groups"]):
        for actual_parameter, serialized_id in zip(
                actual_group["params"], serialized_group["params"]):
            if actual_parameter is gate:
                parameter_id = serialized_id
                break
        if parameter_id is not None:
            break
    if parameter_id is None:
        raise RuntimeError(f"{group} fold {fold}: adaptive gate is absent from DARG optimizer")
    return payload["optimizers"]["darg"]["state"].get(parameter_id)


def main():
    seals = json.loads((ROOT / "audits/training_seal.json").read_text())
    if seals.get("status") != "SEALED_BEFORE_TRAINING":
        raise RuntimeError("Training seal is missing or invalid")
    group_records = {}
    warmup_records = {}
    all_gates = []
    for group in runner.TRAIN_GROUPS:
        fold_records = []
        warmups = []
        config = runner.group_config(group)
        target_probability = np.asarray(
            config["target_gate_probabilities"], dtype=np.float64)
        target_effective = np.asarray(config["darg_gate_scales"], dtype=np.float64) * target_probability
        for fold in range(1, 11):
            folder = ROOT / "groups" / group
            best_path = folder / "best_checkpoints" / f"fold{fold:02d}.pt"
            latest_path = folder / "latest_checkpoints" / f"fold{fold:02d}.pt"
            audit_path = folder / "audits" / f"fold{fold:02d}.json"
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            latest = torch.load(latest_path, map_location="cpu", weights_only=False)
            audit = json.loads(audit_path.read_text())
            init = best.get("warmup", {}).get("gate_initialization", {})
            status = init.get("status")
            expected_status = ("PASS_CONSTRUCTOR_INITIALIZATION"
                               if config.get("gate_initialization") == "constructor"
                               else "PASS")
            if status != expected_status:
                raise RuntimeError(f"{group} fold {fold}: Gate initialization status {status}")
            initialized = np.asarray(init["effective_gate_after"], dtype=np.float64)
            if not np.allclose(initialized, target_effective, atol=2e-7, rtol=0.0):
                raise RuntimeError(f"{group} fold {fold}: initialized effective Gate differs")
            best_gate, best_probability, scales = effective_gate(group, best)
            latest_gate, latest_probability, _ = effective_gate(group, latest)
            if not np.allclose(best_gate, np.asarray(scales) * best_probability,
                               atol=1e-10, rtol=0.0):
                raise RuntimeError(f"{group} fold {fold}: Best effective Gate formula differs")
            if not np.allclose(latest_gate, np.asarray(scales) * latest_probability,
                               atol=1e-10, rtol=0.0):
                raise RuntimeError(f"{group} fold {fold}: Latest effective Gate formula differs")
            trace = audit.get("gate_trace", [])
            if not trace or any(item.get("gradient_finite") is not True for item in trace):
                raise RuntimeError(f"{group} fold {fold}: missing or non-finite Gate gradient")
            if not any(float(item.get("parameter_update_l2", 0.0)) > 0 for item in trace):
                raise RuntimeError(f"{group} fold {fold}: Gate parameter never updated")
            if not np.isfinite(np.asarray(best_gate + latest_gate)).all():
                raise RuntimeError(f"{group} fold {fold}: Gate contains non-finite value")
            optimizer_state = optimizer_gate_state(group, fold, latest)
            if not optimizer_state or int(optimizer_state["step"].item()) <= 0:
                raise RuntimeError(f"{group} fold {fold}: Gate optimizer has no update state")
            if int(trace[-1]["optimizer_step"]) != int(optimizer_state["step"].item()):
                raise RuntimeError(f"{group} fold {fold}: Gate trace and optimizer step differ")

            pairing = audit["warmup_pair_audit"]
            allowed_status = {"PASS"} if group != "C03" else {
                "PASS_CONSTRUCTOR_GATE_DIFFERENCE"
            }
            if pairing.get("status") not in allowed_status:
                raise RuntimeError(
                    f"{group} fold {fold}: warmup pairing status {pairing.get('status')}")
            required_pair_checks = (
                "relation_pairs_exact", "history_exact", "shared_state_exact",
                "spectral_statistics_exact",
            )
            if any(pairing.get(key) is not True for key in required_pair_checks):
                if group != "C03" or pairing.get("spectral_statistics_exact") is not True:
                    raise RuntimeError(f"{group} fold {fold}: warmup pairing evidence failed")
            warmup_row = {
                "group": group, "fold": fold, "status": pairing["status"],
                "warmup_reference": pairing.get("reference"),
                "history_exact": pairing.get("history_exact"),
                "relation_pairs_exact": pairing.get("relation_pairs_exact"),
                "initial_rng_sha256": pairing.get("initial_rng_sha256"),
                "post_warmup_rng_sha256": pairing.get("post_warmup_rng_sha256"),
                "post_warmup_model_sha256": pairing.get("post_warmup_model_sha256"),
                "post_warmup_optimizer_sha256": pairing.get("post_warmup_optimizer_sha256"),
                "spectral_statistics_exact": pairing.get("spectral_statistics_exact"),
                "spectral_statistics_sha256": pairing.get("spectral_statistics_sha256"),
                "pre_epoch_11_state": pairing.get("pre_epoch_11_state"),
                "field_differences": pairing.get("field_differences", []),
            }
            warmups.append(warmup_row)
            gate_row = {
                "group": group, "fold": fold,
                "mode": init.get("mode"),
                "initialization_status": status,
                "probability_before": init.get("probabilities_before"),
                "probability_target": init.get("probabilities_target"),
                "probability_after": init.get("probabilities_after"),
                "gate_scales": scales,
                "effective_gate_initial": initialized.tolist(),
                "effective_gate_best": best_gate,
                "effective_gate_latest": latest_gate,
                "rng_unchanged": init.get("rng_unchanged"),
                "rng_sha256_before": init.get("rng_sha256_before"),
                "rng_sha256_after": init.get("rng_sha256_after"),
                "optimizer_state_removed": init.get("optimizer_state_removed"),
                "optimizer_non_gate_state_unchanged":
                    init.get("optimizer_non_gate_state_unchanged"),
                "parameter_identity_preserved": init.get("parameter_identity_preserved"),
                "gate_gradient_count": len(trace),
                "gate_gradient_all_finite": all(
                    item.get("gradient_finite") is True for item in trace),
                "gate_gradient_norm_min": min(
                    float(item["gradient_norm"]) for item in trace
                    if item.get("gradient_norm") is not None),
                "gate_gradient_norm_max": max(
                    float(item["gradient_norm"]) for item in trace
                    if item.get("gradient_norm") is not None),
                "gate_optimizer_step_latest": int(optimizer_state["step"].item()),
                "best_epoch": int(best["epoch"]),
                "latest_epoch": int(latest["epoch"]),
                "best_sha256": runner.sha_file(best_path),
                "latest_sha256": runner.sha_file(latest_path),
                "config_sha256": runner.sha_file(ROOT / "configs" / f"{group}.json"),
            }
            fold_records.append(gate_row)
            all_gates.append(gate_row)
        group_records[group] = {"folds": len(fold_records), "records": fold_records}
        warmup_records[group] = {"folds": len(warmups), "records": warmups}

    s_group_pairing = {}
    for fold in range(1, 11):
        rows = [row for group in runner.TRAIN_GROUPS if group.startswith("S")
                for row in warmup_records[group]["records"] if row["fold"] == fold]
        common_keys = (
            "initial_rng_sha256", "post_warmup_rng_sha256",
            "post_warmup_model_sha256", "post_warmup_optimizer_sha256",
            "spectral_statistics_sha256",
        )
        for key in common_keys:
            values = {row[key] for row in rows}
            if len(values) != 1:
                raise RuntimeError(f"S-group warmup differs at fold {fold}: {key}")
        best_states = {
            group: torch.load(ROOT / "groups" / group / "best_checkpoints" /
                              f"fold{fold:02d}.pt", map_location="cpu", weights_only=False)
            for group in runner.TRAIN_GROUPS if group.startswith("S")
        }
        pair_hashes = {runner.sha_object(
            (payload["relation_pairs"], payload["relation_targets"]))
            for payload in best_states.values()}
        if len(pair_hashes) != 1:
            raise RuntimeError(f"S-group relation pairs differ at fold {fold}")
        s_group_pairing[str(fold)] = {
            "status": "PASS", "shared_warmup_fields": common_keys,
            "shared_relation_pairs_sha256": next(iter(pair_hashes)),
            "groups": [group for group in runner.TRAIN_GROUPS if group.startswith("S")],
        }

    gate_report = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "groups": group_records, "total_folds": len(all_gates),
        "s_group_shared_warmup_pairing": s_group_pairing,
        "registered_formula": "g_m=s_m*sigmoid(theta_m)",
        "gate_learning_folds": sum(
            row["gate_gradient_all_finite"] and row["gate_optimizer_step_latest"] > 0
            for row in all_gates),
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
    }
    warmup_report = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "groups": warmup_records, "total_folds": 120,
        "s_group_pairing": s_group_pairing,
        "c03_constructor_gate_difference_is_registered": True,
        "c03_spectral_statistics_are_reported_separately": True,
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
    }
    (ROOT / "audits/gate_initialization.json").write_text(
        json.dumps(gate_report, indent=2, sort_keys=True) + "\n")
    (ROOT / "audits/warmup_pairing_actual.json").write_text(
        json.dumps(warmup_report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS", "gate_folds": gate_report["total_folds"],
        "gate_learning_folds": gate_report["gate_learning_folds"],
        "S_group_warmup_pairing": "PASS 10/10 folds",
    }, indent=2))


if __name__ == "__main__":
    main()
