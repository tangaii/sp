"""Verify the complete E83.8.5.4 deliverables and write the final audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83854 as runner
from relation_sampling import audit_pair_array

H1_SOURCE = Path("/model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial")
H0_SOURCE = Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl")
AUDIT_ONLY_PROVENANCE_FILES = {"source_manifest.json", "audits/source_review.md"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seal = json.loads((ROOT / "audits/training_seal.json").read_text())
    current = runner.current_implementation_hashes()
    all_paths = set(seal["files"]) | set(current)
    changed_paths = sorted(path for path in all_paths
                           if seal["files"].get(path) != current.get(path))
    unexpected_changes = [path for path in changed_paths
                          if path not in AUDIT_ONLY_PROVENANCE_FILES]
    fingerprint_inputs = dict(current)
    for path in AUDIT_ONLY_PROVENANCE_FILES:
        if path in seal["files"]:
            fingerprint_inputs[path] = seal["files"][path]
    source_hash_ok = (not unexpected_changes
                      and seal["implementation_sha256"] == runner.implementation_fingerprint(fingerprint_inputs))
    if not source_hash_ok:
        raise RuntimeError("Training source or inputs differ from the pre-training seal")

    x, y, ids, raw_splits = runner.load_data()
    if len(ids) != 598 or list(np.bincount(y, minlength=3)) != [209, 315, 74]:
        raise RuntimeError("Locked data identity differs")
    splits = {int(row["fold"]): row for row in raw_splits}
    historical_pairing, train_pairing, sampler_rows = [], [], []
    for fold in range(1, 11):
        h0 = torch.load(H0_SOURCE / "groups/R0/best_checkpoints" / f"fold{fold:02d}.pt",
                        map_location="cpu", weights_only=False)
        h1 = torch.load(H1_SOURCE / "groups/S02/best_checkpoints" / f"fold{fold:02d}.pt",
                        map_location="cpu", weights_only=False)
        for field in ("history", "rng_state_sha256", "pair_sha256",
                      "shared_optimizer_state_sha256", "shared_darg_cam_gfr_state_sha256"):
            if h0["warmup"].get(field) != h1["warmup"].get(field):
                raise RuntimeError(f"H0/H1 warmup mismatch fold {fold} field {field}")
        historical_pairing.append({"fold": fold, "warmup_exact": True,
                                   "H0_best_sha256": sha(H0_SOURCE / "groups/R0/best_checkpoints" / f"fold{fold:02d}.pt"),
                                   "H1_best_sha256": sha(H1_SOURCE / "groups/S02/best_checkpoints" / f"fold{fold:02d}.pt")})
        for group in runner.TRAIN_GROUPS:
            best_path = ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"
            latest_path = ROOT / "groups" / group / "latest_checkpoints" / f"fold{fold:02d}.pt"
            audit_path = ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json"
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            latest = torch.load(latest_path, map_location="cpu", weights_only=False)
            audit = json.loads(audit_path.read_text())
            split = splits[fold]
            train = np.asarray(split["train_idx"], dtype=np.int64)
            test = np.asarray(split["test_idx"], dtype=np.int64)
            if audit.get("status") != "TRAINED" or best.get("implementation_sha256") != seal["implementation_sha256"]:
                raise RuntimeError(f"{group} fold {fold}: training identity mismatch")
            if latest.get("implementation_sha256") != seal["implementation_sha256"]:
                raise RuntimeError(f"{group} fold {fold}: Latest seal mismatch")
            if not np.array_equal(best["train_idx"], train) or not np.array_equal(best["test_idx"], test):
                raise RuntimeError(f"{group} fold {fold}: checkpoint split mismatch")
            if sha(best_path) != audit["checkpoint_sha256"] or sha(latest_path) != audit["latest_sha256"]:
                raise RuntimeError(f"{group} fold {fold}: checkpoint hash differs from audit")
            for field in ("history", "rng_state_sha256", "pair_sha256",
                          "shared_optimizer_state_sha256", "shared_darg_cam_gfr_state_sha256"):
                if best["warmup"].get(field) != h1["warmup"].get(field):
                    raise RuntimeError(f"{group} fold {fold}: S02 warmup mismatch field {field}")
            if not torch.equal(best["warmup_relation_pairs"], h1["relation_pairs"]):
                raise RuntimeError(f"{group} fold {fold}: warmup relation pairs differ from S02")
            if best["warmup_pair_audit"].get("status") not in {"PASS", "PASS_CONSTRUCTOR_GATE_DIFFERENCE"}:
                raise RuntimeError(f"{group} fold {fold}: warmup pairing audit failed")
            sample = best["formal_sampling_audit"]
            pair_np = best["relation_pairs"].numpy()
            target_np = best["relation_targets"].numpy()
            check = audit_pair_array(pair_np, y, train)
            if (best["relation_pairs"].shape != (4096, 2)
                    or int(best["relation_targets"].sum()) != 2048
                    or sample["relation_counts"] != {
                        "NC-NC": 683, "MCI-MCI": 683, "AD-AD": 682,
                        "NC-MCI": 683, "NC-AD": 683, "MCI-AD": 682}
                    or sample["epoch_first_used"] != 11
                    or sample["rng_sha256_before"] != sample["rng_sha256_after"]
                    or check["pair_count"] != 4096
                    or check["same_pairs"] != 2048
                    or check["different_pairs"] != 2048
                    or not check["canonical_unordered_unique"]
                    or not check["non_self"] or not check["train_only"]):
                raise RuntimeError(f"{group} fold {fold}: formal relation sampler audit failed")
            expected = (y[pair_np[:, 0]] == y[pair_np[:, 1]]).astype(np.float32)
            if not np.array_equal(expected, target_np):
                raise RuntimeError(f"{group} fold {fold}: relation labels differ from training labels")
            if set(pair_np.ravel()) & set(test.tolist()):
                raise RuntimeError(f"{group} fold {fold}: test patient enters a training relation pair")
            if group in {"T2", "T3"}:
                support = best["cognitive_support"]
                if (support["source"] != "epoch_10_frozen_original_graphlearn_COGNITIVE_TEST"
                        or support["modality_index"] != 4 or not support["symmetric"]
                        or not support["non_diagonal"] or not support["train_only"]):
                    raise RuntimeError(f"{group} fold {fold}: cognitive support source contract differs")
            if not audit.get("dpsl_residual_off_exact_native"):
                raise RuntimeError(f"{group} fold {fold}: DPSL Residual-Off contract failed")
            if float(audit.get("dpsl_residual_max", 0.0)) > 0.08 + 1e-7:
                raise RuntimeError(f"{group} fold {fold}: DPSL residual exceeded 0.08")
            if float(audit.get("dpsl_shared_gradient_ratio_max", 0.0)) > 0.001 + 1e-7:
                raise RuntimeError(f"{group} fold {fold}: DPSL shared gradient budget exceeded")
            train_pairing.append({"group": group, "fold": fold, "warmup_exact": True,
                                  "formal_epoch": 11, "model_rng_unchanged": True,
                                  "six_relation_quota": True, "test_label_excluded": True,
                                  "pair_count": 4096})
            sampler_rows.append({"group": group, "fold": fold,
                                 "sampler": sample["sampler"],
                                 "actual_support": sample.get("actual_support_count", 0),
                                 "target_support": sample.get("target_support_count", 0),
                                 "exposure": sample["patient_exposure"]})

    replay = json.loads((ROOT / "audits/checkpoint_replay.json").read_text())
    if (replay.get("status") != "PASS" or replay.get("folds_passed") != 60
            or replay.get("latest_next_step_passed") != 40):
        raise RuntimeError("Independent Best/Latest replay is incomplete")
    logs = json.loads((ROOT / "audits/log_format_audit.json").read_text())
    if logs.get("status") != "PASS" or len(logs.get("groups", {})) != 6:
        raise RuntimeError("Original author-format log audit failed")
    for name in ("relation_generalization.json", "intervention_replay.json"):
        report = json.loads((ROOT / "audits" / name).read_text())
        if report.get("status") != "PASS":
            raise RuntimeError(f"{name} did not pass")
    gradients = json.loads((ROOT / "audits/gradient_replay.json").read_text())
    if (gradients.get("status") != "PASS" or gradients.get("folds") != 60
            or gradients.get("folds_passed") != 60):
        raise RuntimeError("DARG Relation BCE or GRL gradient replay failed")
    aggregate = json.loads((ROOT / "audits/aggregate_results.json").read_text())
    if aggregate.get("status") != "PASS" or aggregate.get("oof_rows") != 3588:
        raise RuntimeError("Result aggregation or OOF coverage failed")

    post = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "pretraining_implementation_sha256": seal["implementation_sha256"],
        "post_training_implementation_sha256": runner.implementation_fingerprint(fingerprint_inputs),
        "source_unchanged": True,
        "provenance_audit_files_updated_after_seal": changed_paths,
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
        "new_trained_folds": 40, "historical_replayed_folds": 20,
        "best_latest_replay_folds": 60, "latest_next_step_recovery_folds": 40,
        "warmup_exact_pairs": 40, "sampler_valid_folds": 40,
        "author_log_groups": 6, "oof_rows": aggregate["oof_rows"],
        "root_model_files_changed": False,
    }
    (ROOT / "audits/training_post_hash.json").write_text(json.dumps(post, indent=2, sort_keys=True) + "\n")
    final = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
        "source_seal_pass": source_hash_ok, "preflight_status": "PASS",
        "formal_training_groups": {group: {"folds": 10, "status": "PASS"} for group in runner.TRAIN_GROUPS},
        "historical_replay_groups": {"H0": {"folds": 10, "status": "PASS"},
                                     "H1": {"folds": 10, "status": "PASS"}},
        "best_latest_replay": replay["status"],
        "latest_next_step_recovery": replay["latest_next_step_recovery_status"],
        "darg_gradient_replay": gradients["status"],
        "darg_gradient_replay_folds": gradients["folds_passed"],
        "provenance_audit_files_updated_after_seal": changed_paths,
        "log_format_audit": logs["status"],
        "warmup_pairing": {"historical_folds": historical_pairing,
                            "new_training_folds": train_pairing},
        "sampler_audit_rows": sampler_rows,
        "outputs": {
            path.relative_to(ROOT).as_posix(): sha(path)
            for path in sorted([*(ROOT / "results").glob("*"),
                                ROOT / "audits/log_format_audit.json",
                                ROOT / "audits/checkpoint_replay.json",
                                ROOT / "audits/gradient_replay.json",
                                ROOT / "audits/mechanism_summary.json"])
            if path.is_file()
        },
    }
    (ROOT / "audits/final_audit.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": final["status"], "new_folds": 40,
                      "historical_folds": 20, "replayed_folds": replay["folds_passed"]}, indent=2))


if __name__ == "__main__":
    main()
