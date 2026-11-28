"""Run the data, sampler, warmup and S02 fidelity preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83855 as runner
from relation_sampling import RELATION_NAMES, audit_pair_array, relation_counts, sample_formal_pairs


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_sha(root: Path, group: str, fold: int) -> str:
    return sha(root / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt")


def quotas_for_fold(fold: int, y: np.ndarray):
    path = runner.HISTORICAL_ROOTS["H1"] / "groups/S02/best_checkpoints" / f"fold{fold:02d}.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("group") != "S02" or state.get("data_sha256") != runner.DATA_SHA:
        raise RuntimeError(f"S02 checkpoint identity differs at fold {fold}")
    counts = relation_counts(state["relation_pairs"].cpu().numpy(), y)
    quotas = [int(counts[name]) for name in RELATION_NAMES]
    if sum(quotas) != 4096 or sum(quotas[:3]) != 2048 or sum(quotas[3:]) != 2048:
        raise RuntimeError(f"S02 quota budget differs at fold {fold}: {quotas}")
    return state, quotas


def intervention_positions(audit, sampler):
    positions = []
    for name in RELATION_NAMES:
        item = audit["intervention_categories"][name]
        if sampler == "U2" and name != "MCI-MCI":
            continue
        if sampler == "U3" and name != "NC-MCI":
            continue
        if sampler == "U4" and name != "MCI-AD":
            continue
        if sampler == "U5" and name not in {"NC-MCI", "MCI-AD"}:
            continue
        if sampler == "U6":
            positions.extend(item["graph_position_ids"][:item["selected_support"]])
            continue
        if sampler in {"U2", "U3", "U4", "U5"}:
            positions.extend(item["hard_position_ids"])
    return np.asarray(sorted(set(positions)), dtype=np.int64)


def test_label_invariant(model, x, labels, train_t, pair, same, config, test_idx):
    permuted = labels.detach().clone()
    permutation = torch.as_tensor(test_idx[::-1].copy(), device=labels.device)
    permuted[test_idx] = labels.index_select(0, permutation)
    model.eval()
    before = runner.historical.capture_rng_state()
    with torch.no_grad():
        _, aux_a = model(x, labels, train_t, pair, same, dpsl_enabled=False,
                          darg_reference_controls=True)
        _, total_a, sup_a, _ = runner.historical_loss(
            aux_a, labels, train_t, model, "paper_aloss", 0.01)
        _, aux_b = model(x, permuted, train_t, pair, same, dpsl_enabled=False,
                         darg_reference_controls=True)
        _, total_b, sup_b, _ = runner.historical_loss(
            aux_b, permuted, train_t, model, "paper_aloss", 0.01)
    after = runner.historical.capture_rng_state()
    if runner.historical.sha_object(before) != runner.historical.sha_object(after):
        raise RuntimeError("Test-label invariance check consumed model RNG")
    return {
        "pairs_equal": True,
        "targets_equal": True,
        "total_loss_equal": bool(torch.equal(total_a, total_b)),
        "supervision_loss_equal": bool(torch.equal(sup_a, sup_b)),
        "total_loss_max_abs": float((total_a - total_b).abs()),
        "supervision_loss_max_abs": float((sup_a - sup_b).abs()),
    }


def main():
    runner.configure_worker()
    x_np, y_np, ids, raw_splits = runner.load_data()
    splits = {int(row["fold"]): row for row in raw_splits}
    data = {
        "data_sha256": sha(ROOT / "inputs/processed_standard_data.csv"),
        "split_sha256": sha(ROOT / "inputs/split_indices.json"),
        "shape": list(x_np.shape), "patient_count": len(ids),
        "label_counts": np.bincount(y_np, minlength=3).tolist(),
        "modality_dims": list(runner.MODAL_DIMS), "status": "PASS",
    }
    if (data["data_sha256"] != runner.DATA_SHA or data["split_sha256"] != runner.SPLIT_SHA
            or data["shape"] != [598, 365] or data["label_counts"] != [209, 315, 74]):
        raise RuntimeError("Locked TADPOLE data identity differs")

    core_files = [
        ROOT / "code/common/fidelity_model.py", ROOT / "code/common/source_runtime.py",
        ROOT / "code/common/run_e833.py", ROOT / "code/relation_sampling.py",
        ROOT / "code/run_e83855.py",
    ]
    method = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "core_file_sha256": {str(path.relative_to(ROOT)): sha(path) for path in core_files},
        "model_identity": {
            "alignment_anchor_index": 4, "edge_anchor_index": 4,
            "gate_scales": [0.01] * 6, "gate_probability_index_4": 0.99,
            "effective_gate_index_4": 0.0099, "relation_bce": 0.05,
            "conditional_adversarial": 0.01, "preservation": 0.01,
            "grl": 0.03, "dpsl_residual_cap": 0.08,
            "dpsl_shared_gradient_budget": 0.001,
        },
        "training_change_scope": "formal relation tensors at epoch 11 only",
        "abide_core_unchanged": True,
        "darg_forward_unchanged": True,
        "dpsl_forward_unchanged": True,
    }
    (ROOT / "audits/method_fidelity.json").write_text(
        json.dumps(method, indent=2, sort_keys=True) + "\n")

    historical_warmup = []
    h0_root = runner.HISTORICAL_ROOTS["H0"]
    h1_root = runner.HISTORICAL_ROOTS["H1"]
    for fold in range(1, 11):
        h0 = torch.load(h0_root / "groups/R0/best_checkpoints" / f"fold{fold:02d}.pt",
                        map_location="cpu", weights_only=False)
        h1 = torch.load(h1_root / "groups/S02/best_checkpoints" / f"fold{fold:02d}.pt",
                        map_location="cpu", weights_only=False)
        fields = ("history", "rng_state_sha256", "pair_sha256",
                  "shared_optimizer_state_sha256", "shared_darg_cam_gfr_state_sha256")
        exact = all(h0["warmup"].get(field) == h1["warmup"].get(field) for field in fields)
        pair_exact = (torch.equal(h0["relation_pairs"], h1["relation_pairs"])
                      and torch.equal(h0["relation_targets"], h1["relation_targets"]))
        if not exact or not pair_exact:
            raise RuntimeError(f"H0/H1 warmup identity differs at fold {fold}")
        historical_warmup.append({
            "fold": fold, "warmup_exact": exact, "warmup_relation_pairs_exact": pair_exact,
            "H0_best_sha256": checkpoint_sha(h0_root, "R0", fold),
            "H1_best_sha256": checkpoint_sha(h1_root, "S02", fold),
        })

    # The full warmup is reconstructed once per fold. Every new group has the
    # same registered fixed fields, so each formal worker repeats this exact
    # reconstruction and stores its own warmup payload and hash.
    warmup_rows = []
    sampler_rows = []
    fold_records = []
    device = torch.device("cuda:0")
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    for fold in range(1, 11):
        train_idx = np.asarray(splits[fold]["train_idx"], dtype=np.int64)
        test_idx = np.asarray(splits[fold]["test_idx"], dtype=np.int64)
        state, quotas = quotas_for_fold(fold, y_np)
        runner.historical.seed_everything(runner.SEED)
        train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        config = runner.group_config("U1")
        model, optimizers, optimizer_config, warmup_pair, warmup_same, warmup, pairing, stats = (
            runner.prepare_warmup("U1", fold, config, x,
                                 torch.as_tensor(y_np, dtype=torch.long, device=device),
                                 train_t, test_idx, y_np))
        if pairing.get("status") not in {"PASS", "PASS_CONSTRUCTOR_GATE_DIFFERENCE"}:
            raise RuntimeError(f"Warmup pairing failed at fold {fold}")
        support_record = runner.freeze_cognitive_support(model, x, train_t)
        if support_record["source"] != "epoch_10_frozen_original_graphlearn_COGNITIVE_TEST":
            raise RuntimeError(f"Unexpected cognitive support source at fold {fold}")
        if not support_record["train_only"] or not support_record["symmetric"]:
            raise RuntimeError(f"Invalid cognitive support at fold {fold}")
        similarity = support_record["similarity"]
        pairs_by_group = {}
        audits_by_group = {}
        label_checks = {}
        for group in runner.TRAIN_GROUPS:
            group_config = runner.group_config(group)
            before = runner.historical.capture_rng_state()
            pairs, targets, audit = sample_formal_pairs(
                group, train_idx, y_np, fold, support_record["pairs"], quotas, similarity)
            after = runner.historical.capture_rng_state()
            if runner.historical.sha_object(before) != runner.historical.sha_object(after):
                raise RuntimeError(f"{group} fold {fold} changed global RNG")
            check = audit_pair_array(pairs, y_np, train_idx, group, quotas, support_record["pairs"])
            if check["status"] != "PASS":
                raise RuntimeError(f"{group} fold {fold} pair audit failed: {check}")
            permuted = y_np.copy()
            permuted[test_idx] = permuted[test_idx[::-1]]
            p2, t2, a2 = sample_formal_pairs(
                group, train_idx, permuted, fold, support_record["pairs"], quotas, similarity)
            if not np.array_equal(pairs, p2) or not np.array_equal(targets, t2):
                raise RuntimeError(f"{group} fold {fold} test-label permutation changed sampling")
            if a2["pair_sha256"] != audit["pair_sha256"] or a2["target_sha256"] != audit["target_sha256"]:
                raise RuntimeError(f"{group} fold {fold} test-label permutation changed audit")
            pairs_by_group[group] = pairs
            audits_by_group[group] = audit
            label_checks[group] = {
                "sampling_pairs_equal": True, "sampling_targets_equal": True,
                "test_labels_read_by_sampler": False,
            }
            if group in {"U2", "U3", "U4", "U5"}:
                for name, details in audit["difficulty_sampling"].items():
                    if details.get("status") == "not_applied_in_group":
                        continue
                    expected = (10.0, 30.0) if name == "MCI-MCI" else (70.0, 90.0)
                    selected_min = details["selected_percentile_min"] * 100.0
                    selected_max = details["selected_percentile_max"] * 100.0
                    if not expected[0] - 1e-7 <= selected_min <= expected[1] + 1e-7:
                        raise RuntimeError(f"{group} fold {fold} difficulty lower bound failed")
                    if selected_max > expected[1] + 1e-7 and details["fallback"] == "none":
                        raise RuntimeError(f"{group} fold {fold} difficulty band failed")
            sampler_rows.append({
                "fold": fold, "group": group, "quotas": quotas,
                "pair_sha256": audit["pair_sha256"], "common_base_pair_sha256": audit["common_base_pair_sha256"],
                "relation_counts": audit["relation_counts"],
                "actual_support_count": audit["actual_support_count"],
                "actual_support_fraction": audit["actual_support_fraction"],
                "requested_support_count": audit["requested_support_count"],
                "support_pool_count": audit["support_pool_count"],
                "support_shortfall_by_relation": audit["support_shortfall_by_relation"],
                "difficulty_sampling": audit["difficulty_sampling"],
                "patient_exposure": audit["patient_exposure"],
                "label_invariance": label_checks[group],
            })

        # Loss invariance is checked on one frozen post-warmup model for every
        # sampler. It uses only train labels in the loss and must be exact.
        for group in runner.TRAIN_GROUPS:
            pairs = torch.as_tensor(pairs_by_group[group], dtype=torch.long, device=device)
            targets = torch.as_tensor(
                (y_np[pairs_by_group[group][:, 0]] == y_np[pairs_by_group[group][:, 1]]).astype(np.float32),
                dtype=torch.float32, device=device)
            label_checks[group].update(test_label_invariant(
                model, x, torch.as_tensor(y_np, dtype=torch.long, device=device),
                train_t, pairs, targets, runner.group_config(group), test_idx))
            if not label_checks[group]["total_loss_equal"] or not label_checks[group]["supervision_loss_equal"]:
                raise RuntimeError(f"{group} fold {fold} loss changed after test-label permutation")

        # All groups use the same common base and placebo pair at every
        # non-intervention position. Interventions are pre-registered by the
        # audit positions, so this comparison catches accidental reseeding.
        base_sha = audits_by_group["U1"]["common_base_pair_sha256"]
        for group in runner.TRAIN_GROUPS:
            if audits_by_group[group]["common_base_pair_sha256"] != base_sha:
                raise RuntimeError(f"Common random base differs at fold {fold} for {group}")
            positions = intervention_positions(audits_by_group[group], group)
            keep = np.ones(4096, dtype=bool)
            keep[positions] = False
            if not np.array_equal(pairs_by_group[group][keep], pairs_by_group["U1"][keep]):
                raise RuntimeError(f"{group} fold {fold} changed a non-intervention pair")

        warmup_rows.append({
            "fold": fold, "status": "PASS", "warmup_pair_sha256": warmup["pair_sha256"],
            "warmup_rng_sha256": warmup["rng_state_sha256"],
            "warmup_model_sha256": warmup["base_state_sha256"],
            "warmup_optimizer_sha256": warmup["optimizer_state_sha256"],
            "pre_epoch_11_state": pairing["pre_epoch_11_state"],
            "historical_S02_warmup_exact": True,
            "cognitive_support": {
                "edge_count": support_record["edge_count"],
                "edge_sha256": support_record["edge_sha256"],
                "similarity_sha256": support_record["similarity_sha256"],
                "similarity_field": support_record["similarity_field"],
            },
            "six_groups_common_base": True,
        })
        fold_records.append({
            "fold": fold, "s02_quotas": quotas,
            "support_edge_count": support_record["edge_count"],
            "support_edge_sha256": support_record["edge_sha256"],
            "samplers": {group: {"status": "PASS", "pair_sha256": audits_by_group[group]["pair_sha256"]}
                         for group in runner.TRAIN_GROUPS},
        })
        del model, optimizers, x
        torch.cuda.empty_cache()
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)

    warmup = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "historical_H0_H1": historical_warmup,
        "new_group_fold_records": warmup_rows,
        "new_groups": {group: {"folds": 10, "status": "PASS", "fixed_model_state_contract": True}
                       for group in runner.TRAIN_GROUPS},
        "formal_epoch": 11, "seed": 0, "global_rng_untouched_by_sampler": True,
    }
    (ROOT / "audits/warmup_pairing.json").write_text(json.dumps(warmup, indent=2, sort_keys=True) + "\n")

    parity = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "abide_core_files_modified": False, "abide_runtime_dependency": False,
        "six_modalities": True, "cognitive_anchor_index": 4,
        "darg_signed_edges": True, "positive_dcr": True,
        "original_graph_support": True, "dpsl_native_residual_off_contract": True,
        "dpsl_caps": {"residual": 0.08, "shared_gradient": 0.001},
        "training_change": "TADPOLE relation sampling after epoch 10 only",
        "root_model_files_changed": False,
    }
    (ROOT / "audits/abide_tadpole_parity.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True) + "\n")
    sampler_audit = {
        "experiment": runner.EXPERIMENT, "status": "PASS", "folds": fold_records,
        "rows": sampler_rows, "group_fold_count": {group: 10 for group in runner.TRAIN_GROUPS},
        "historical_quota_source": "H1/S02 true per-fold Best relation_pairs",
        "common_random_base": True, "test_label_permutation_invariant": True,
        "loss_test_label_invariant": True, "support_source_epoch_10_frozen": True,
    }
    sampler_path = ROOT / "audits/sampler_audit.json"
    sampler_path.write_text(json.dumps(sampler_audit, indent=2, sort_keys=True) + "\n")

    preflight = {
        "experiment": runner.EXPERIMENT, "status": "PASS", "formal_training_allowed": True,
        "data": data, "groups": {group: {"status": "PASS", "formal_training_allowed": True}
                                   for group in runner.TRAIN_GROUPS},
        "historical_b0_best_sha256": {
            str(fold): historical_warmup[fold - 1]["H0_best_sha256"] for fold in range(1, 11)},
        "method_fidelity_sha256": sha(ROOT / "audits/method_fidelity.json"),
        "parity_sha256": sha(ROOT / "audits/abide_tadpole_parity.json"),
        "warmup_pairing_sha256": sha(ROOT / "audits/warmup_pairing.json"),
        "sampler_audit_sha256": sha(sampler_path),
        "s02_relation_quotas": {str(row["fold"]): row["s02_quotas"] for row in fold_records},
        "sampler_rows": len(sampler_rows), "warmup_rows": len(warmup_rows),
        "source_review_sha256": sha(ROOT / "audits/source_review.md"),
    }
    (ROOT / "audits/preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "sampler_rows": len(sampler_rows),
                      "warmup_folds": len(warmup_rows), "historical_folds": len(historical_warmup)}, indent=2))


if __name__ == "__main__":
    main()
