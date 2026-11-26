"""Hold later groups until the fresh Y0 fold1 exactly matches historical N3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "common"))
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
        check(type(left) is type(right) and left == right, f"Value differs at {path}: {left!r} != {right!r}")


def compare_checkpoint_subset(new, old, keys, label):
    for key in keys:
        check(key in new and key in old, f"{label} is missing {key}")
        exact(new[key], old[key], f"{label}.{key}")


def main():
    runner.require_training_gate(["Y0"])
    new_group = ROOT / "groups/Y0"
    old_group = runner.N3_REFERENCE_ROOT / "groups/N3"
    new_audit = json.loads((new_group / "audits/fold01.json").read_text())
    old_audit = json.loads((old_group / "audits/fold01.json").read_text())
    check(new_audit.get("status") == "TRAINED", "Y0 fold1 is not complete")
    new_best = torch.load(new_group / "best_checkpoints/fold01.pt", map_location="cpu", weights_only=False)
    old_best = torch.load(old_group / "best_checkpoints/fold01.pt", map_location="cpu", weights_only=False)
    new_latest = torch.load(new_group / "latest_checkpoints/fold01.pt", map_location="cpu", weights_only=False)
    old_latest = torch.load(old_group / "latest_checkpoints/fold01.pt", map_location="cpu", weights_only=False)

    check(new_audit["best_epoch"] == old_audit["best_epoch"], "Y0 Best epoch differs from historical N3")
    check(new_audit["last_epoch"] == old_audit["last_epoch"], "Y0 stopping epoch differs from historical N3")
    check(new_audit["ACC"] == old_audit["ACC"] and new_audit["AUC_original"] == old_audit["AUC_original"],
          "Y0 hard-prediction metrics differ from historical N3")
    check(new_audit["AUC_probability"] == old_audit["AUC_probability"],
          "Y0 probability AUC differs from historical N3")
    compare_checkpoint_subset(
        new_best, old_best,
        ("epoch", "best_val_acc", "model", "optimizers", "optimizer_config", "history_at_best",
         "spectral_stats", "relation_pairs", "relation_targets", "dpsl_ema_updates", "gradient_budget",
         "gradient_budget_history", "update_vector_audits", "lambda_fold", "metrics_at_best",
         "prediction_sha256"), "Best")
    compare_checkpoint_subset(
        new_latest, old_latest,
        ("epoch", "model", "optimizers", "optimizer_config", "rng", "best_epoch", "best_acc", "wait",
         "history", "spectral_stats", "gradient_budget_history", "update_vector_audits", "lambda_fold",
         "aloss_calibration", "relation_pairs", "relation_targets", "dpsl_ema_updates"), "Latest")
    old_warmup, new_warmup = old_best["warmup"], new_best["warmup"]
    compare_checkpoint_subset(
        new_warmup, old_warmup,
        ("epochs", "history", "initial_model_sha256", "initial_optimizer_sha256", "initial_rng_sha256",
         "base_state_sha256", "optimizer_state_sha256", "shared_darg_cam_gfr_state_sha256",
         "shared_optimizer_state_sha256", "rng_state_sha256", "pair_sha256", "train_index_sha256",
         "train_labels_sha256"), "Warmup")
    check(new_audit["warmup_pair_audit"]["status"] == "PASS",
          "Y0 fold1 did not pass its historical warmup check")

    new_predictions = pd.read_csv(new_group / "predictions/fold01.csv")
    old_predictions = pd.read_csv(old_group / "predictions/fold01.csv")
    prediction_columns = ["sample_index", "sample_id", "true_label", "pred_label", "prob_NC", "prob_MCI", "prob_AD",
                          "native_pred_label", "native_prob_NC", "native_prob_MCI", "native_prob_AD"]
    check(list(new_predictions.columns) == prediction_columns,
          "Y0 fold1 output columns differ from the fixed N3 prediction contract")
    check(list(old_predictions.columns) == prediction_columns,
          "Historical N3 fold1 output columns differ from the fixed prediction contract")
    for column in prediction_columns:
        if column in {"prob_NC", "prob_MCI", "prob_AD", "native_prob_NC", "native_prob_MCI", "native_prob_AD"}:
            check(np.array_equal(new_predictions[column].to_numpy(), old_predictions[column].to_numpy()),
                  f"Y0 fold1 probabilities differ in {column}")
        else:
            check(np.array_equal(new_predictions[column].to_numpy(), old_predictions[column].to_numpy()),
                  f"Y0 fold1 predictions differ in {column}")

    result = {
        "experiment": "Experiment 83.8.3",
        "status": "PASS",
        "group": "Y0",
        "fold": 1,
        "reference": "E83.8.2-N3",
        "checks": {
            "warmup": "PASS_EXACT",
            "best_epoch": "PASS_EXACT",
            "latest_epoch_and_early_stop": "PASS_EXACT",
            "best_model_and_optimizer": "PASS_EXACT",
            "latest_model_optimizer_rng_and_history": "PASS_EXACT",
            "predictions_and_probabilities": "PASS_EXACT",
            "acc_auc_and_probability_auc": "PASS_EXACT",
            "logging_state_invariance": "PASS_EXACT",
        },
        "Y0": {"best_epoch": new_audit["best_epoch"], "latest_epoch": new_audit["last_epoch"],
               "ACC": new_audit["ACC"], "AUC_original": new_audit["AUC_original"],
               "AUC_probability": new_audit["AUC_probability"],
               "checkpoint_sha256": new_audit["checkpoint_sha256"]},
        "N3": {"best_epoch": old_audit["best_epoch"], "latest_epoch": old_audit["last_epoch"],
               "ACC": old_audit["ACC"], "AUC_original": old_audit["AUC_original"],
               "AUC_probability": old_audit["AUC_probability"],
               "checkpoint_sha256": old_audit["checkpoint_sha256"]},
        "prediction_sha256": hashlib.sha256(
            new_predictions["pred_label"].to_numpy(dtype=np.int64).tobytes()).hexdigest(),
    }
    target = ROOT / "audits/y0_n3_fold01_regression.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
