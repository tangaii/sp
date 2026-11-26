"""Compare all ten fresh Y0 folds with the fixed E83.8.2-N3 records."""
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
import run_e8383 as runner


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact(left, right, path="root"):
    if torch.is_tensor(left):
        check(torch.is_tensor(right) and torch.equal(left, right), f"Tensor differs at {path}")
    elif isinstance(left, np.ndarray):
        check(isinstance(right, np.ndarray) and np.array_equal(left, right), f"Array differs at {path}")
    elif isinstance(left, dict):
        check(isinstance(right, dict) and left.keys() == right.keys(), f"Mapping keys differ at {path}")
        for key in left:
            exact(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        check(isinstance(right, type(left)) and len(left) == len(right), f"Sequence differs at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            exact(a, b, f"{path}[{index}]")
    else:
        check(type(left) is type(right) and left == right,
              f"Value differs at {path}: {left!r} != {right!r}")


def compare_subset(left, right, keys, label):
    for key in keys:
        check(key in left and key in right, f"{label} is missing {key}")
        exact(left[key], right[key], f"{label}.{key}")


def main():
    runner.require_training_gate(["Y0"])
    new_dir = ROOT / "groups/Y0"
    old_dir = runner.N3_REFERENCE_ROOT / "groups/N3"
    prediction_columns = [
        "sample_index", "sample_id", "true_label", "pred_label", "prob_NC", "prob_MCI", "prob_AD",
        "native_pred_label", "native_prob_NC", "native_prob_MCI", "native_prob_AD",
    ]
    best_keys = (
        "epoch", "best_val_acc", "model", "optimizers", "optimizer_config", "history_at_best",
        "spectral_stats", "relation_pairs", "relation_targets", "dpsl_ema_updates", "gradient_budget",
        "gradient_budget_history", "update_vector_audits", "lambda_fold", "metrics_at_best",
        "prediction_sha256",
    )
    latest_keys = (
        "epoch", "model", "optimizers", "optimizer_config", "rng", "best_epoch", "best_acc", "wait",
        "history", "spectral_stats", "gradient_budget_history", "update_vector_audits", "lambda_fold",
        "aloss_calibration", "relation_pairs", "relation_targets", "dpsl_ema_updates",
    )
    warmup_keys = (
        "epochs", "history", "initial_model_sha256", "initial_optimizer_sha256", "initial_rng_sha256",
        "base_state_sha256", "optimizer_state_sha256", "shared_darg_cam_gfr_state_sha256",
        "shared_optimizer_state_sha256", "rng_state_sha256", "pair_sha256", "train_index_sha256",
        "train_labels_sha256",
    )
    folds = {}
    for fold in range(1, 11):
        suffix = f"fold{fold:02d}"
        new_audit = json.loads((new_dir / "audits" / f"{suffix}.json").read_text())
        old_audit = json.loads((old_dir / "audits" / f"{suffix}.json").read_text())
        check(new_audit.get("status") == old_audit.get("status") == "TRAINED",
              f"Y0 or N3 fold {fold} is incomplete")
        for key in ("best_epoch", "last_epoch", "stop_wait", "ACC", "AUC_original", "AUC_probability"):
            check(new_audit[key] == old_audit[key], f"Y0 fold {fold} differs in {key}")
        check(new_audit["warmup_pair_audit"]["status"] == "PASS",
              f"Y0 fold {fold} failed its historical warmup pairing")

        new_best = torch.load(new_dir / "best_checkpoints" / f"{suffix}.pt",
                              map_location="cpu", weights_only=False)
        old_best = torch.load(old_dir / "best_checkpoints" / f"{suffix}.pt",
                              map_location="cpu", weights_only=False)
        new_latest = torch.load(new_dir / "latest_checkpoints" / f"{suffix}.pt",
                                map_location="cpu", weights_only=False)
        old_latest = torch.load(old_dir / "latest_checkpoints" / f"{suffix}.pt",
                                map_location="cpu", weights_only=False)
        compare_subset(new_best, old_best, best_keys, f"{suffix}.Best")
        compare_subset(new_latest, old_latest, latest_keys, f"{suffix}.Latest")
        compare_subset(new_best["warmup"], old_best["warmup"], warmup_keys, f"{suffix}.warmup")

        new_predictions = pd.read_csv(new_dir / "predictions" / f"{suffix}.csv")
        old_predictions = pd.read_csv(old_dir / "predictions" / f"{suffix}.csv")
        check(list(new_predictions.columns) == prediction_columns,
              f"Y0 fold {fold} output columns differ from the historical contract")
        check(list(old_predictions.columns) == prediction_columns,
              f"N3 fold {fold} output columns differ from the historical contract")
        check(len(new_predictions) == len(old_predictions), f"Y0 fold {fold} prediction count differs")
        for column in prediction_columns:
            check(np.array_equal(new_predictions[column].to_numpy(), old_predictions[column].to_numpy()),
                  f"Y0 fold {fold} predictions differ in {column}")
        folds[str(fold)] = {
            "status": "PASS_EXACT",
            "best_epoch": new_audit["best_epoch"],
            "latest_epoch": new_audit["last_epoch"],
            "best_model_optimizer_exact": True,
            "latest_model_optimizer_rng_history_exact": True,
            "warmup_and_relation_pair_exact": True,
            "predictions_probabilities_and_metrics_exact": True,
            "prediction_sha256": hashlib.sha256(
                new_predictions["pred_label"].to_numpy(dtype=np.int64).tobytes()).hexdigest(),
            "best_checkpoint_sha256": runner.sha_file(new_dir / "best_checkpoints" / f"{suffix}.pt"),
            "latest_checkpoint_sha256": runner.sha_file(new_dir / "latest_checkpoints" / f"{suffix}.pt"),
        }
        print(f"PASS Y0 vs N3 fold {fold:02d}", flush=True)

    report = {
        "experiment": "Experiment 83.8.3",
        "status": "PASS_EXACT",
        "reference": "E83.8.2-N3 fixed historical checkpoints",
        "folds_passed": len(folds),
        "checks": [
            "Best epochs, predictions and probabilities",
            "Best model and optimizer state",
            "Latest model, optimizer, RNG and full epoch history",
            "Warmup model, optimizer, RNG, relation pairs and train labels",
            "ACC, AUC_original and AUC_probability",
        ],
        "folds": folds,
    }
    target = ROOT / "audits/y0_n3_full_regression.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "folds_passed": report["folds_passed"],
                      "path": str(target)}, indent=2))


if __name__ == "__main__":
    main()
