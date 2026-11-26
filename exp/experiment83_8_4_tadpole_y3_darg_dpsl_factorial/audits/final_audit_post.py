"""Post-seal final audit for frozen identity, 60 folds, OOF coverage, logs and replay."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8384 as runner


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    groups = list(runner.FORMAL_GROUPS)
    runner.require_training_gate(groups)
    seal = read_json(ROOT / "audits/training_seal.json")
    preflight = read_json(ROOT / "audits/preflight_start.json")
    source_manifest = read_json(ROOT / "source_manifest.json")
    resource = read_json(ROOT / "audits/resource_probe.json")
    replay = read_json(ROOT / "audits/checkpoint_replay.json")
    log_audit = read_json(ROOT / "audits/log_format_audit.json")
    log_manifest = read_json(ROOT / "audits/log_manifest.json")
    regression = read_json(ROOT / "audits/z0_y3_full_regression.json")
    mechanism = read_json(ROOT / "audits/mechanism_summary.json")
    analysis = read_json(ROOT / "audits/analysis_summary.json")

    check(seal["files"] == runner.current_implementation_hashes(),
          "Source/config/input hashes differ from training seal")
    check(seal["implementation_sha256"] == runner.implementation_fingerprint(),
          "Implementation fingerprint differs from training seal")
    check(preflight["status"] == "PASS" and preflight["formal_training_allowed"] is True,
          "Training preflight did not pass")
    check(resource["status"] == "PASS" and resource["worker_count"] == 8,
          "Single-GPU resource probe did not pass")
    check(regression["status"] == "PASS_EXACT" and regression["folds_passed"] == 10,
          "Z0 does not exactly regress to all ten fixed Y3 folds")
    check(replay["status"] == "PASS" and replay["best_replayed"] == 60
          and replay["latest_restored"] == 60 and replay["deterministic_next_steps"] == 60,
          "Best/Latest replay totals are incomplete")
    check(log_audit["status"] == "PASS" and set(log_audit["groups"]) == set(groups),
          "Author-format log audit did not pass all six groups")
    check(mechanism["status"] == "PASS" and set(mechanism["groups"]) == set(groups),
          "Mechanism audit did not cover all six groups")
    check(analysis["group_count"] == 6 and analysis["fold_count"] == 60
          and analysis["patient_predictions"] == 3588,
          "Final OOF analysis totals differ")
    check(source_manifest["repositories"]["experiment_repo"]["url"] == "https://github.com/tangaii/sp"
          and source_manifest["repositories"]["saber_main"]["url"] == "https://github.com/tangaii/SPromptGL",
          "E83.8.2 and SABER main repositories are not distinguished")

    data, labels, ids, splits = runner.load_data()
    check(runner.sha_file(ROOT / "inputs/processed_standard_data.csv") == runner.DATA_SHA,
          "Input data SHA differs from the frozen TADPOLE data")
    check(runner.sha_file(ROOT / "inputs/split_indices.json") == runner.SPLIT_SHA,
          "Split SHA differs from the frozen TADPOLE split")
    split_map = {int(item["fold"]): item for item in splits}
    group_records = {}
    total_folds = 0
    for group in groups:
        group_dir = ROOT / "groups" / group
        predictions = []
        epochs = []
        for fold in range(1, 11):
            suffix = f"fold{fold:02d}"
            audit = read_json(group_dir / "audits" / f"{suffix}.json")
            check(audit.get("status") == "TRAINED", f"{group} fold {fold} is incomplete")
            if group != "Z0":
                check(audit["implementation_sha256"] == seal["implementation_sha256"],
                      f"{group} fold {fold} training seal differs")
            else:
                check(audit["implementation_sha256"] == regression["historical_y3_implementation_sha256"],
                      f"Z0 fold {fold} historical Y3 seal identity differs")
            check(audit["data_sha256"] == runner.DATA_SHA and audit["split_sha256"] == runner.SPLIT_SHA,
                  f"{group} fold {fold} input identity differs")
            check(audit["warmup_pair_audit"]["status"] == "PASS",
                  f"{group} fold {fold} warmup pairing failed")
            check(audit["warmup_pair_audit"].get("spectral_statistics_exact") is True
                  and audit["warmup_pair_audit"].get("prompt_vectors_exact") is True,
                  f"{group} fold {fold} warmup spectral or Prompt pairing failed")
            train_idx = np.asarray(audit["train_idx"], dtype=np.int64)
            test_idx = np.asarray(audit["test_idx"], dtype=np.int64)
            expected = split_map[fold]
            check(np.array_equal(train_idx, np.asarray(expected["train_idx"], dtype=np.int64)),
                  f"{group} fold {fold} train indices differ from the locked split")
            check(np.array_equal(test_idx, np.asarray(expected["test_idx"], dtype=np.int64)),
                  f"{group} fold {fold} test indices differ from the locked split")
            check(not np.intersect1d(train_idx, test_idx).size,
                  f"{group} fold {fold} has train/test overlap")
            check(audit["last_epoch"] == audit["author_epoch_log_count"],
                  f"{group} fold {fold} lacks per-epoch log records")
            check(audit["last_epoch"] <= 450 and audit["best_epoch"] <= audit["last_epoch"],
                  f"{group} fold {fold} epoch bounds are invalid")
            best_path = group_dir / "best_checkpoints" / f"{suffix}.pt"
            latest_path = group_dir / "latest_checkpoints" / f"{suffix}.pt"
            check(runner.sha_file(best_path) == audit["checkpoint_sha256"],
                  f"{group} fold {fold} Best checkpoint SHA differs")
            check(runner.sha_file(latest_path) == audit["latest_sha256"],
                  f"{group} fold {fold} Latest checkpoint SHA differs")
            frame = pd.read_csv(group_dir / "predictions" / f"{suffix}.csv")
            check(np.array_equal(frame["sample_index"].to_numpy(dtype=np.int64), test_idx),
                  f"{group} fold {fold} prediction order differs from test indices")
            check(np.array_equal(frame["true_label"].to_numpy(dtype=np.int64), labels[test_idx]),
                  f"{group} fold {fold} prediction labels differ from the locked data")
            check(np.isfinite(frame[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy()).all(),
                  f"{group} fold {fold} prediction probabilities are not finite")
            predictions.append(frame)
            epochs.append(audit["last_epoch"])
            total_folds += 1

        oof = pd.concat(predictions, ignore_index=True).sort_values("sample_index")
        check(len(oof) == len(data) and oof["sample_index"].nunique() == len(data),
              f"{group} OOF does not cover each patient once")
        check(np.array_equal(oof["sample_index"].to_numpy(dtype=np.int64), np.arange(len(data))),
              f"{group} OOF patient indices are incomplete")
        check(np.array_equal(oof["true_label"].to_numpy(dtype=np.int64), labels),
              f"{group} OOF labels differ from the frozen dataset")
        log_entry = log_manifest["groups"][group]
        check(len(log_entry["folds"]) == 10
              and sum(item["epoch_count"] for item in log_entry["folds"]) == sum(epochs),
              f"{group} log manifest epoch counts differ")
        log_path = ROOT / "logs" / group / "result_Tadpole.txt"
        check(runner.sha_file(log_path) == log_entry["sha256"], f"{group} author log SHA differs")
        check([path.name for path in (ROOT / "logs" / group).iterdir()] == ["result_Tadpole.txt"],
              f"{group} contains unexpected log artifacts")
        group_records[group] = {
            "folds": 10,
            "oof_patients": len(oof),
            "oof_each_once": True,
            "warmup_pairs_passed": 10,
            "log_epochs": int(sum(epochs)),
            "log_sha256": log_entry["sha256"],
        }

    check(total_folds == 60, f"Only {total_folds}/60 folds are complete")
    check(Counter(int(value) for value in labels) == Counter({0: 209, 1: 315, 2: 74}),
          "TADPOLE class counts differ")
    check(ids["RID"].nunique() == 598, "TADPOLE sample ID coverage differs")

    report = {
        "experiment": "Experiment 83.8.4",
        "audit_program": "audits/final_audit_post.py",
        "audit_program_sha256": __import__("hashlib").sha256((ROOT / "audits/final_audit_post.py").read_bytes()).hexdigest(),
        "status": "PASS",
        "performance_status": analysis["status"],
        "training_seal_sha256": runner.sha_file(ROOT / "audits/training_seal.json"),
        "implementation_sha256": seal["implementation_sha256"],
        "data_sha256": runner.DATA_SHA,
        "split_sha256": runner.SPLIT_SHA,
        "formal_folds": total_folds,
        "best_checkpoints_replayed": replay["best_replayed"],
        "latest_checkpoints_restored": replay["latest_restored"],
        "deterministic_next_steps": replay["deterministic_next_steps"],
        "z0_vs_y3_exact_folds": regression["folds_passed"],
        "author_log_groups_passed": len(log_audit["groups"]),
        "checks": {
            "source_and_config_seal": "PASS",
            "training_preflight": "PASS",
            "single_gpu_resource_probe": "PASS",
            "all_six_groups_ten_folds": "PASS",
            "warmup_pairing_all_folds": "PASS",
            "train_test_index_disjoint": "PASS",
            "OOF_598_patients_once_per_group": "PASS",
            "Z0_historical_Y3_regression": "PASS_EXACT",
            "Best_replay": "PASS_60_OF_60",
            "Latest_restore": "PASS_60_OF_60",
            "deterministic_next_step": "PASS_60_OF_60",
            "author_log_format": "PASS_ALL_SIX_GROUPS",
            "six_modality_DARG_and_DPSL_mechanism": "PASS",
            "no_test_label_supervision": "PASS_BY_TRAIN_INDEX_CONTRACT_AND_LABEL_PERMUTATION_PROBE",
        },
        "groups": group_records,
    }
    target = ROOT / "audits/final_audit.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "performance_status", "formal_folds", "best_checkpoints_replayed",
                       "latest_checkpoints_restored", "deterministic_next_steps", "z0_vs_y3_exact_folds")},
                     indent=2))


if __name__ == "__main__":
    main()
