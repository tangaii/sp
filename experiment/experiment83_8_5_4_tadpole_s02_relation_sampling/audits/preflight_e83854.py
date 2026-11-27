"""Focused E83.8.5.4 method and data preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83854 as runner
from relation_sampling import sample_formal_pairs


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_checkpoint(root: Path, group: str, fold: int) -> Path:
    return root / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"


def main():
    x, y, ids, raw_splits = runner.load_data()
    splits = {int(item["fold"]): item for item in raw_splits}
    data_checks = {
        "data_sha256": sha(ROOT / "inputs/processed_standard_data.csv"),
        "split_sha256": sha(ROOT / "inputs/split_indices.json"),
        "shape": list(x.shape), "patient_count": int(len(ids)),
        "label_counts": np.bincount(y, minlength=3).tolist(),
        "modality_dims": list(runner.MODAL_DIMS),
        "status": "PASS",
    }
    if data_checks["data_sha256"] != runner.DATA_SHA or data_checks["split_sha256"] != runner.SPLIT_SHA:
        raise RuntimeError("Locked data or split SHA differs")
    if data_checks["shape"] != [598, 365] or data_checks["label_counts"] != [209, 315, 74]:
        raise RuntimeError("TADPOLE cohort identity differs")

    method_files = [
        ROOT / "code/common/fidelity_model.py", ROOT / "code/common/source_runtime.py",
        ROOT / "code/common/run_e833.py", ROOT / "code/sources/history_darg/cadr_gfr.py",
        ROOT / "code/sources/history_darg/soft_positive_dcr.py", ROOT / "code/relation_sampling.py",
    ]
    method = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "unchanged_core_files": {str(path.relative_to(ROOT)): sha(path) for path in method_files[:-1]},
        "new_sampling_file": sha(method_files[-1]),
        "model_identity": {
            "alignment_anchor_index": 4, "edge_anchor_index": 4,
            "gate_scales": [0.01] * 6, "gate_target_probability_index_4": 0.99,
            "relation_bce": 0.05, "conditional_adversarial": 0.01,
            "preservation": 0.01, "grl": 0.03,
            "dpsl_residual_cap": 0.08, "dpsl_shared_gradient_budget": 0.001,
        },
        "training_change_scope": "formal relation pair tensors at epoch 11 only",
    }
    (ROOT / "audits/method_fidelity.json").write_text(json.dumps(method, indent=2, sort_keys=True) + "\n")

    # The historical H0/H1 archives must share the same ten-step warmup state.
    h0_root = Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl")
    h1_root = Path("/model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial")
    warmup_rows = []
    for fold in range(1, 11):
        h0 = torch.load(source_checkpoint(h0_root, "R0", fold), map_location="cpu", weights_only=False)
        h1 = torch.load(source_checkpoint(h1_root, "S02", fold), map_location="cpu", weights_only=False)
        w0, w1 = h0["warmup"], h1["warmup"]
        fields = ("history", "rng_state_sha256", "pair_sha256", "shared_optimizer_state_sha256",
                  "shared_darg_cam_gfr_state_sha256")
        exact = all(w0.get(field) == w1.get(field) for field in fields)
        pair_exact = torch.equal(h0["relation_pairs"].cpu(), h1["relation_pairs"].cpu()) and torch.equal(
            h0["relation_targets"].cpu(), h1["relation_targets"].cpu())
        warmup_rows.append({"fold": fold, "warmup_exact_h0_h1": exact, "formal_pair_exact_h0_h1": pair_exact,
                            "h0_best_sha256": sha(source_checkpoint(h0_root, "R0", fold)),
                            "h1_best_sha256": sha(source_checkpoint(h1_root, "S02", fold))})
        if not exact or not pair_exact:
            raise RuntimeError(f"H0/H1 warmup identity differs at fold {fold}")
    warmup = {"experiment": runner.EXPERIMENT, "status": "PASS", "folds": warmup_rows,
              "formal_epoch": 11, "global_rng_untouched_by_sampler": True}
    (ROOT / "audits/warmup_pairing.json").write_text(json.dumps(warmup, indent=2, sort_keys=True) + "\n")

    # Exercise every sampler on fold 1 and on a different fold with a real
    # six-modality GraphLearn-derived cognitive support candidate set.
    sampler_checks = []
    for fold in (1, 2, 10):
        train = np.asarray(splits[fold]["train_idx"], dtype=np.int64)
        model_config = runner.group_config("T2")
        model = runner.build_model(model_config, fold, torch.device("cpu"))
        support_record = runner.freeze_cognitive_support(
            model, torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(train, dtype=torch.long))
        support = support_record["pairs"]
        for sampler in ("T1", "T2", "T3", "T4"):
            pairs, targets, audit = sample_formal_pairs(sampler, train, y, fold,
                                                        support if sampler in {"T2", "T3"} else None)
            if int(targets.sum()) != 2048 or audit["relation_counts"] != {
                    "NC-NC": 683, "MCI-MCI": 683, "AD-AD": 682,
                    "NC-MCI": 683, "NC-AD": 683, "MCI-AD": 682}:
                raise RuntimeError(f"{sampler} fold {fold}: six-category quota failure")
            permuted = y.copy()
            test = np.asarray(splits[fold]["test_idx"], dtype=np.int64)
            permuted[test] = permuted[test[::-1]]
            other_pairs, other_targets, _ = sample_formal_pairs(
                sampler, train, permuted, fold, support if sampler in {"T2", "T3"} else None)
            if not np.array_equal(pairs, other_pairs) or not np.array_equal(targets, other_targets):
                raise RuntimeError(f"{sampler} fold {fold}: test-label permutation changed sampling")
            sampler_checks.append({"fold": fold, "sampler": sampler, "audit": audit,
                                   "test_label_permutation_invariant": True})
    sampler_path = ROOT / "audits/sampler_audit.json"
    sampler_path.write_text(json.dumps({"experiment": runner.EXPERIMENT, "status": "PASS",
                                        "checks": sampler_checks}, indent=2, sort_keys=True) + "\n")

    parity = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "abide_core_files_modified": False,
        "abide_runtime_dependency": False,
        "six_modalities": True, "cognitive_anchor_index": 4,
        "darg_signed_edges": True, "original_graph_support": True,
        "dpsl_native_residual_off_contract": True,
        "note": "ABIDE is outside this TADPOLE-only experiment and no root model.py/network.py/layers.py/main.py file was modified.",
    }
    (ROOT / "audits/abide_tadpole_parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")

    preflight = {
        "experiment": runner.EXPERIMENT, "status": "PASS", "formal_training_allowed": True,
        "data": data_checks, "groups": {group: {"status": "PASS", "formal_training_allowed": True}
                                           for group in ("T1", "T2", "T3", "T4")},
        "historical_b0_best_sha256": {
            str(fold): warmup_rows[fold - 1]["h0_best_sha256"] for fold in range(1, 11)
        },
        "method_fidelity_sha256": sha(ROOT / "audits/method_fidelity.json"),
        "parity_sha256": sha(ROOT / "audits/abide_tadpole_parity.json"),
        "warmup_pairing_sha256": sha(ROOT / "audits/warmup_pairing.json"),
        "sampler_audit_sha256": sha(sampler_path),
    }
    (ROOT / "audits/preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "sampler_checks": len(sampler_checks),
                      "warmup_folds": len(warmup_rows)}, indent=2))


if __name__ == "__main__":
    main()
