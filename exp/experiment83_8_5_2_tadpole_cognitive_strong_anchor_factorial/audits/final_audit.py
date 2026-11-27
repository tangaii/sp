"""Final integrity checks for all registered folds, checkpoints and deliverables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as runner

AUDIT_FILES = (
    "abide_4modal_regression.json", "gate_initialization.json",
    "warmup_pairing_actual.json", "checkpoint_replay.json",
    "mechanism_summary.json", "log_format_audit.json",
    "aggregate_results.json",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    seal = read_json(ROOT / "audits/training_seal.json")
    if seal.get("status") != "SEALED_BEFORE_TRAINING":
        raise RuntimeError("Training source was not sealed before formal training")
    current_hashes = runner.current_implementation_hashes()
    if current_hashes != seal.get("files"):
        raise RuntimeError("A sealed training source, configuration or input changed")
    if runner.implementation_fingerprint(current_hashes) != seal.get(
            "implementation_sha256"):
        raise RuntimeError("Sealed training implementation fingerprint differs")
    if runner.sha_file(ROOT / "inputs/processed_standard_data.csv") != runner.DATA_SHA:
        raise RuntimeError("Processed data SHA differs from protocol")
    if runner.sha_file(ROOT / "inputs/split_indices.json") != runner.SPLIT_SHA:
        raise RuntimeError("Split SHA differs from protocol")
    for filename in AUDIT_FILES:
        report = read_json(ROOT / "audits" / filename)
        if filename == "checkpoint_replay.json":
            if report.get("best_latest_replay_status") != "PASS":
                raise RuntimeError("Independent Best/Latest checkpoint replay failed")
            continue
        if report.get("status") != "PASS":
            raise RuntimeError(f"Prerequisite audit failed: {filename}")

    preflight = read_json(ROOT / "audits/preflight.json")
    replay = read_json(ROOT / "audits/checkpoint_replay.json")
    logs = read_json(ROOT / "audits/log_format_audit.json")
    mechanism = read_json(ROOT / "audits/mechanism_summary.json")
    gate = read_json(ROOT / "audits/gate_initialization.json")
    warmup = read_json(ROOT / "audits/warmup_pairing_actual.json")
    abide = read_json(ROOT / "audits/abide_4modal_regression.json")
    aggregate = read_json(ROOT / "audits/aggregate_results.json")

    x, y, ids, raw_splits = runner.load_data()
    splits = {int(item["fold"]): item for item in raw_splits}
    fold_count = 0
    best_count = latest_count = prediction_count = 0
    group_fold_summary = {}
    for group in runner.FORMAL_GROUPS:
        folder = ROOT / "groups" / group
        rows = []
        seen = []
        for fold in range(1, 11):
            audit_path = folder / "audits" / f"fold{fold:02d}.json"
            best_path = folder / "best_checkpoints" / f"fold{fold:02d}.pt"
            latest_path = folder / "latest_checkpoints" / f"fold{fold:02d}.pt"
            prediction_path = folder / "predictions" / f"fold{fold:02d}.csv"
            audit = read_json(audit_path)
            if group.startswith("B"):
                expected_status = "HISTORICAL_REPLAY_VERIFIED"
                if group == "B2" and audit.get("source_group") != "N1":
                    raise RuntimeError("B2 historical source is not N1")
            else:
                expected_status = "TRAINED"
                if audit.get("implementation_sha256") != seal["implementation_sha256"]:
                    raise RuntimeError(f"{group} fold {fold}: training implementation changed")
            if audit.get("status") != expected_status:
                raise RuntimeError(f"{group} fold {fold}: fold audit status mismatch")
            if runner.sha_file(best_path) != audit.get("checkpoint_sha256"):
                raise RuntimeError(f"{group} fold {fold}: Best checkpoint hash mismatch")
            latest_sha = audit.get("latest_sha256", audit.get("latest_checkpoint_sha256"))
            if runner.sha_file(latest_path) != latest_sha:
                raise RuntimeError(f"{group} fold {fold}: Latest checkpoint hash mismatch")
            prediction = pd.read_csv(prediction_path)
            expected_test = np.asarray(splits[fold]["test_idx"], dtype=np.int64)
            if (len(prediction) != len(expected_test)
                    or not np.array_equal(prediction["sample_index"].to_numpy(), expected_test)
                    or prediction["sample_index"].duplicated().any()):
                raise RuntimeError(f"{group} fold {fold}: prediction fold coverage mismatch")
            if (not np.array_equal(
                    prediction["true_label"].to_numpy(), y[expected_test])
                    or not np.array_equal(
                        prediction["sample_id"].astype(str).to_numpy(),
                        ids.iloc[expected_test]["RID"].astype(str).to_numpy())):
                raise RuntimeError(f"{group} fold {fold}: patient identity or label mismatch")
            seen.extend(prediction["sample_index"].astype(int).tolist())
            best_count += 1
            latest_count += 1
            prediction_count += 1
            fold_count += 1
            rows.append({
                "fold": fold,
                "status": expected_status,
                "best_epoch": audit.get("best_epoch"),
                "latest_epoch": audit.get("latest_epoch"),
                "ACC": audit.get("ACC"),
                "AUC_original": audit.get("AUC_original"),
                "AUC_probability": audit.get("AUC_probability"),
                "best_sha256": runner.sha_file(best_path),
                "latest_sha256": runner.sha_file(latest_path),
                "prediction_sha256": runner.sha_file(prediction_path),
            })
        if len(seen) != 598 or len(set(seen)) != 598:
            raise RuntimeError(f"{group}: OOF patients are not covered exactly once")
        group_fold_summary[group] = {"folds": len(rows), "records": rows}

    if (fold_count != 160 or best_count != 160
            or latest_count != 160 or prediction_count != 160):
        raise RuntimeError("The registered 160 historical/new fold set is incomplete")
    if (replay.get("best_latest_folds_passed") != 160
            or replay.get("folds_expected") != 160):
        raise RuntimeError("Independent Best/Latest replay did not pass 160/160 folds")
    if gate.get("total_folds") != 120 or gate.get("gate_learning_folds") != 120:
        raise RuntimeError("Formal Gate initialization/learning audit is incomplete")
    if warmup.get("total_folds") != 120:
        raise RuntimeError("Formal warmup pairing audit is incomplete")
    if not abide.get("status") == "PASS" or abide.get("folds_verified") != 10:
        raise RuntimeError("ABIDE four-modality historical regression is incomplete")
    if len(logs.get("groups", {})) != 16 or len(logs.get("verification", {})) != 16:
        raise RuntimeError("Author log-format audit does not cover all 16 groups")
    if aggregate.get("oof_rows") != 16 * 598:
        raise RuntimeError("OOF aggregate does not contain 598 rows per group")
    group_summary = pd.read_csv(ROOT / "results/group_summary.csv")
    baseline = group_summary[group_summary["group"] == "B0"].iloc[0]
    if (int(baseline["full_OOF_correct"]) != 560
            or [int(baseline[f"full_correct_{name}"]) for name in ("NC", "MCI", "AD")]
            != [205, 291, 64]):
        raise RuntimeError("B0 OOF baseline differs from the registered history")

    blockers = []
    if (replay.get("latest_next_step_passed") != 120
            or replay.get("latest_next_step_expected") != 120
            or replay.get("latest_next_step_recovery_status") != "PASS"):
        blockers.append({
            "gate": "latest_next_step_deterministic_recovery",
            "status": replay.get("latest_next_step_recovery_status"),
            "passed": replay.get("latest_next_step_passed"),
            "expected": replay.get("latest_next_step_expected"),
            "reason": "Identical Latest+RNG replay produced different model and optimizer state hashes.",
        })
    report_path = ROOT / "results/final_report.md"
    report_sha = runner.sha_file(report_path) if report_path.is_file() else None
    result = {
        "experiment": runner.EXPERIMENT,
        "status": "BLOCKED" if blockers else "PASS",
        "completed": not blockers, "registered_group_count": 16,
        "historical_folds": 40, "new_training_folds": 120,
        "best_checkpoint_replay": "PASS 160/160",
        "latest_checkpoint_replay": "PASS 160/160",
        "latest_next_step_recovery": (
            f"{replay.get('latest_next_step_recovery_status')} "
            f"{replay.get('latest_next_step_passed')}/"
            f"{replay.get('latest_next_step_expected')}"),
        "blockers": blockers,
        "historical_checkpoint_source": "server archives",
        "training_seal_status": seal["status"],
        "training_implementation_sha256": seal["implementation_sha256"],
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
        "group_oof_patient_coverage": "PASS 598/598 each",
        "oof_total_rows": aggregate["oof_rows"],
        "baseline_B0_oof_correct": int(baseline["full_OOF_correct"]),
        "baseline_B0_correct_NC_MCI_AD": [
            int(baseline[f"full_correct_{name}"]) for name in ("NC", "MCI", "AD")],
        "ABIDE_E74_1_four_modality_regression": "PASS",
        "warmup_gate_pairing": "PASS",
        "gate_initialization_and_learning": "PASS 120/120",
        "author_log_format": "PASS 16/16",
        "method_fidelity_preflight": preflight["status"],
        "mechanism_groups": len(mechanism["groups"]),
        "final_report_sha256": report_sha,
        "folds": group_fold_summary,
        "prerequisite_audit_sha256": {
            name: runner.sha_file(ROOT / "audits" / name) for name in AUDIT_FILES},
        "script_sha256": runner.sha_file(Path(__file__)),
    }
    (ROOT / "audits/final_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "best_replay": result["best_checkpoint_replay"],
        "latest_replay": result["latest_checkpoint_replay"],
        "latest_next_step": result["latest_next_step_recovery"],
        "oof_rows": result["oof_total_rows"],
        "final_report_sha256": result["final_report_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
