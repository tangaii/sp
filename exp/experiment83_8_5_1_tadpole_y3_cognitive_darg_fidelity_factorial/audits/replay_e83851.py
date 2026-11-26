"""Independently replay all registered Best and Latest checkpoints."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl")
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/common")]
import run_e8385 as runner
from metrics import acc_auc


SPEC = importlib.util.spec_from_file_location("e8385_replay_helpers", HISTORY / "audits/replay_e8385.py")
replay_helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay_helpers)
replay_helpers.ROOT = ROOT
replay_helpers.runner = runner

GROUP_SOURCE = {"H0": "R0", "H1": "R1", "H2": "R4"}
GROUPS = (*GROUP_SOURCE, "N1", "N2", "N3", "N4")


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_paths(group: str, fold: int):
    source_group = GROUP_SOURCE.get(group, group)
    source_root = HISTORY if group in GROUP_SOURCE else ROOT
    folder = source_root / "groups" / source_group
    stem = f"fold{fold:02d}"
    return {
        "root": source_root,
        "source_group": source_group,
        "best": folder / "best_checkpoints" / f"{stem}.pt",
        "latest": folder / "latest_checkpoints" / f"{stem}.pt",
        "audit": folder / "audits" / f"{stem}.json",
        "prediction": folder / "predictions" / f"{stem}.csv",
    }


def config_for(group: str):
    if group in GROUP_SOURCE:
        source_group = GROUP_SOURCE[group]
        config = json.loads((HISTORY / "configs" / f"{source_group}.json").read_text())
    else:
        config = runner.group_config(group)
    return config


def verify_optimizer(model, config, checkpoint, group, fold, device):
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.darg is not None:
        model.darg.enter_positive_dcr()
        expected_grl = float(config.get("darg_grl_scale", 0.03))
        check(model.darg.core.grl_scale == expected_grl,
              f"{group} fold {fold} GRL scalar did not restore from registered config")
    optimizers, optimizer_config = runner.make_optimizers(model, config, formal=True)
    check(optimizer_config == checkpoint["optimizer_config"],
          f"{group} fold {fold} optimizer configuration differs")
    check(optimizers.keys() == checkpoint["optimizers"].keys(),
          f"{group} fold {fold} optimizer parameter groups differ")
    for name, state in checkpoint["optimizers"].items():
        optimizers[name].load_state_dict(copy.deepcopy(state))
    for optimizer in optimizers.values():
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    check(torch.isfinite(value).all().item(),
                          f"{group} fold {fold} optimizer has non-finite state")
    return optimizers


def optimizer_model_state(group, fold, config, checkpoint):
    device = torch.device("cuda:0")
    runner.historical.seed_everything(runner.SEED)
    model = runner.build_model(config, fold, device, checkpoint["spectral_stats"])
    optimizers = verify_optimizer(model, config, checkpoint, group, fold, device)
    return model, optimizers


def verify_pairs(checkpoint, split, labels, group, fold):
    train = np.asarray(split["train_idx"], dtype=np.int64)
    test = np.asarray(split["test_idx"], dtype=np.int64)
    check(checkpoint["train_idx"] == train.tolist() and checkpoint["test_idx"] == test.tolist(),
          f"{group} fold {fold} checkpoint partition differs")
    pairs = checkpoint["relation_pairs"].detach().cpu().numpy()
    same = checkpoint["relation_targets"].detach().cpu().numpy().astype(np.int64)
    check(pairs.shape == (4096, 2) and np.isin(pairs, train).all()
          and np.array_equal(same, labels[pairs[:, 0]] == labels[pairs[:, 1]])
          and np.bincount(same, minlength=2).tolist() == [2048, 2048],
          f"{group} fold {fold} relation pair scope or balance differs")
    check(not np.intersect1d(train, test).size and len(train) + len(test) == len(labels),
          f"{group} fold {fold} locked partition is invalid")


def replay_best(group, fold, config, paths, audit, best, frame, x, labels, split):
    model, optimizers = optimizer_model_state(group, fold, config, best)
    pair = best["relation_pairs"].to("cuda:0")
    same = best["relation_targets"].to("cuda:0")
    train = torch.as_tensor(split["train_idx"], dtype=torch.long, device="cuda:0")
    test = np.asarray(split["test_idx"], dtype=np.int64)
    test_tensor = torch.as_tensor(test, dtype=torch.long, device="cuda:0")
    x_tensor = torch.as_tensor(x, device="cuda:0")
    y_tensor = torch.as_tensor(labels, dtype=torch.long, device="cuda:0")
    y_eval = y_tensor.clone()
    y_eval[test_tensor] = 0
    model.eval()
    with torch.no_grad():
        logp, aux = model(x_tensor, y_eval, train, pair, same, dpsl_enabled=True)
        native_logp = model(x_tensor, y_eval, train, pair, same, dpsl_enabled=False)[0]
    prediction = logp.argmax(1).detach().cpu().numpy()[test]
    probabilities = logp.exp().detach().cpu().numpy()[test]
    native_prediction = native_logp.argmax(1).detach().cpu().numpy()[test]
    native_probabilities = native_logp.exp().detach().cpu().numpy()[test]
    check(np.array_equal(prediction, frame["pred_label"].to_numpy(dtype=np.int64)),
          f"{group} fold {fold} Best predictions differ from the patient output")
    saved_probability = frame[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy(dtype=np.float32)
    check(np.array_equal(probabilities, saved_probability),
          f"{group} fold {fold} Best probabilities differ from the patient output")
    check(np.array_equal(native_prediction, frame["native_pred_label"].to_numpy(dtype=np.int64)),
          f"{group} fold {fold} Native predictions differ from the patient output")
    saved_native = frame[["native_prob_NC", "native_prob_MCI", "native_prob_AD"]].to_numpy(dtype=np.float32)
    check(np.array_equal(native_probabilities, saved_native),
          f"{group} fold {fold} Native probabilities differ from the patient output")
    truth = labels[test]
    metrics = acc_auc(truth, prediction, 3)
    from sklearn.metrics import roc_auc_score
    metrics["AUC_probability"] = float(roc_auc_score(
        np.eye(3, dtype=np.float64)[truth], probabilities, average="macro", multi_class="ovr"))
    expected = {"ACC": audit["ACC"], "AUC": audit["AUC_original"],
                "AUC_probability": audit["AUC_probability"]}
    check(metrics == best["metrics_at_best"] == expected,
          f"{group} fold {fold} checkpoint metrics differ")
    check(int(best["epoch"]) == int(audit["best_epoch"]),
          f"{group} fold {fold} Best epoch differs")
    confidence = aux.get("dpsl", {}).get("confidence_gate")
    if config.get("dpsl_confidence_protection", False):
        check(confidence is not None and float(aux["dpsl"]["ratio"].max()) <= .08 + 1e-7,
              f"{group} fold {fold} DPSL confidence or residual cap differs")
    return {
        "status": "PASS", "prediction_exact": True, "probability_exact_float32": True,
        "native_exact": True, "metrics": metrics, "best_epoch": int(best["epoch"]),
        "model_state_sha256": runner.sha_object(best["model"]),
        "optimizer_state_sha256": runner.sha_object(best["optimizers"]),
        "best_rng_present": "rng" in best,
        "best_rng_sha256": runner.sha_object(best["rng"]) if "rng" in best else None,
        "confidence_gate_range": ([float(confidence.min()), float(confidence.max())]
                                   if confidence is not None else None),
    }


def latest_next_step(group, fold, config, latest, x, labels, split):
    model, optimizers = optimizer_model_state(group, fold, config, latest)
    pair = latest["relation_pairs"].to("cuda:0")
    same = latest["relation_targets"].to("cuda:0")
    train = torch.as_tensor(split["train_idx"], dtype=torch.long, device="cuda:0")
    x_tensor = torch.as_tensor(x, device="cuda:0")
    y_tensor = torch.as_tensor(labels, dtype=torch.long, device="cuda:0")
    y_train_only = y_tensor.clone()
    y_train_only[torch.as_tensor(split["test_idx"], dtype=torch.long, device="cuda:0")] = 0
    runner.historical.restore_rng(latest["rng"])
    check(runner.sha_object(runner.historical.capture_rng_state())
          == runner.sha_object(latest["rng"]), f"{group} fold {fold} Latest RNG restore differs")
    model.train()
    runner.zero_optimizers(optimizers)
    next_epoch = int(latest["epoch"]) + 1
    logp, aux = model(x_tensor, y_train_only, train, pair, same, dpsl_enabled=True)
    graph_kind = runner.active_supervision_kind(config, next_epoch)
    coefficient = runner.configured_supervision_coefficient(config, latest.get("lambda_fold"))
    _, historical_total, _, _ = runner.historical_loss(
        aux, y_train_only, train, model, graph_kind, coefficient)
    total = historical_total + aux["dpsl_loss"]
    check(torch.isfinite(total).all().item(), f"{group} fold {fold} Latest next-step loss is non-finite")
    budget = runner.dpsl_optimizer_step(
        model, optimizers, historical_total, aux["scmr_objectives"],
        float(config.get("dpsl_shared_cap", .001)))
    check(all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
          f"{group} fold {fold} Latest next-step parameters are non-finite")
    model.eval()
    with torch.no_grad():
        _, eval_aux = model(x_tensor, y_train_only, train, pair, same, dpsl_enabled=True)
    model.train()
    model.dpsl.advance_ema(eval_aux["cam"])
    return {
        "loss": float(total.detach()), "model_sha256": runner.sha_object(model.state_dict()),
        "optimizer_sha256": runner.sha_object({key: value.state_dict() for key, value in optimizers.items()}),
        "rng_sha256": runner.sha_object(runner.historical.capture_rng_state()),
        "prediction_sha256": runner.sha_object(logp.detach()),
        "shared_gradient_ratio_max": max((float(item.get("actual_ratio", 0.0))
                                            for item in budget.get("families", {}).values()), default=0.0),
    }


def replay_fold(group, fold, x, labels, ids, split, history_audit):
    paths = artifact_paths(group, fold)
    config = config_for(group)
    audit = json.loads(paths["audit"].read_text())
    check(audit.get("status") == "TRAINED", f"{group} fold {fold} training audit is missing")
    best_sha, latest_sha = sha(paths["best"]), sha(paths["latest"])
    check(best_sha == audit["checkpoint_sha256"] and latest_sha == audit["latest_sha256"],
          f"{group} fold {fold} checkpoint file identity differs")
    if group in GROUP_SOURCE:
        historic_record = history_audit["groups"][paths["source_group"]][str(fold)]
        check(best_sha == historic_record["best_sha256"]
              and latest_sha == historic_record["latest_sha256"],
              f"{group} fold {fold} source checkpoint SHA differs from the archived replay manifest")
    best = torch.load(paths["best"], map_location="cpu", weights_only=False)
    latest = torch.load(paths["latest"], map_location="cpu", weights_only=False)
    check(latest.get("best_sha256") == best_sha and latest.get("best_epoch") == best["epoch"]
          and latest.get("epoch") == audit["last_epoch"],
          f"{group} fold {fold} Latest-to-Best linkage differs")
    check([item["epoch"] for item in latest["history"]]
          == list(range(11, latest["epoch"] + 1)),
          f"{group} fold {fold} Latest epoch history is incomplete")
    first_best = max(latest["history"], key=lambda item: item["test_acc"])
    check(first_best["epoch"] == best["epoch"]
          and latest["wait"] == latest["epoch"] - best["epoch"]
          and (latest["epoch"] == runner.MAX_EPOCHS or latest["wait"] == runner.PATIENCE + 1),
          f"{group} fold {fold} Best selection or early stopping differs")
    check(set(latest["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
          and len(latest["rng"]["torch_cuda"]) == 1,
          f"{group} fold {fold} Latest RNG state is incomplete")
    verify_pairs(best, split, labels, group, fold)
    verify_pairs(latest, split, labels, group, fold)
    if group in GROUP_SOURCE:
        frame = pd.read_csv(paths["prediction"], float_precision="round_trip")
    else:
        frame = pd.read_csv(paths["prediction"], float_precision="round_trip")
    test = np.asarray(split["test_idx"], dtype=np.int64)
    check(np.array_equal(frame["sample_index"], test)
          and np.array_equal(frame["true_label"], labels[test])
          and np.array_equal(frame["sample_id"], ids.iloc[test]["RID"])
          and np.all(frame["fold"] == fold) and np.all(frame["seed"] == runner.SEED),
          f"{group} fold {fold} output patient identity differs")
    best_result = replay_best(group, fold, config, paths, audit, best, frame, x, labels, split)
    first = latest_next_step(group, fold, config, latest, x, labels, split)
    second = latest_next_step(group, fold, config, latest, x, labels, split)
    check(first == second, f"{group} fold {fold} Latest deterministic continuation differs")
    if group in GROUP_SOURCE:
        historic = history_audit["groups"][paths["source_group"]][str(fold)]["latest"]
        check(first["model_sha256"] == historic["next_step_model_sha256"]
              and first["optimizer_sha256"] == historic["next_step_optimizer_sha256"]
              and first["rng_sha256"] == historic["next_step_rng_sha256"]
              and first["loss"] == historic["next_step_loss"],
              f"{group} fold {fold} Latest continuation differs from archived independent replay")
    return {
        "best": best_result,
        "latest": {"restore_status": "PASS", "next_step_deterministic": True,
                   "latest_epoch": int(latest["epoch"]), "model_state_sha256": runner.sha_object(latest["model"]),
                   "optimizer_state_sha256": runner.sha_object(latest["optimizers"]),
                   "rng_sha256": runner.sha_object(latest["rng"]),
                   "relation_pairs_sha256": runner.sha_object(latest["relation_pairs"]),
                   "spectral_statistics_sha256": runner.sha_object(latest["spectral_stats"]),
                   "next_step": first},
        "best_sha256": best_sha, "latest_sha256": latest_sha,
        "source_group": paths["source_group"],
        "configuration": {"grl_scale": float(config.get("darg_grl_scale", .03)),
                          "gate_scales": config["darg_gate_scales"],
                          "alignment_anchor_index": config["alignment_anchor_index"],
                          "edge_anchor_index": config["edge_anchor_index"]},
        "prediction_file_sha256": sha(paths["prediction"]),
    }


def main():
    runner.configure_worker()
    x, labels, ids, splits = runner.load_data()
    history_audit = json.loads((HISTORY / "audits/checkpoint_replay.json").read_text())
    check(history_audit["status"] == "PASS" and history_audit["best_replayed"] == 50
          and history_audit["latest_restored"] == 50,
          "Historical E83.8.5 replay manifest is not complete")
    groups = {}
    for group in GROUPS:
        folds = {}
        for split in sorted(splits, key=lambda row: int(row["fold"])):
            fold = int(split["fold"])
            folds[str(fold)] = replay_fold(group, fold, x, labels, ids, split, history_audit)
            print(f"PASS {group} fold {fold:02d} Best/Latest", flush=True)
        groups[group] = folds
        test_indices = np.concatenate([
            np.asarray(splits[index]["test_idx"], dtype=np.int64) for index in range(10)
        ])
        check(np.array_equal(np.sort(test_indices), np.arange(598)),
              f"{group} does not cover all 598 OOF patients")
    report = {
        "experiment": "Experiment 83.8.5.1",
        "status": "PASS",
        "groups": groups,
        "best_replayed": sum(row["best"]["status"] == "PASS" for folds in groups.values()
                              for row in folds.values()),
        "latest_restored": sum(row["latest"]["restore_status"] == "PASS" for folds in groups.values()
                                for row in folds.values()),
        "deterministic_next_steps": sum(row["latest"]["next_step_deterministic"] for folds in groups.values()
                                        for row in folds.values()),
        "historical_group_folds": 30,
        "new_group_folds": 40,
        "auditor_source_sha256": sha(Path(__file__)),
    }
    target = ROOT / "audits/checkpoint_replay.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "best_replayed", "latest_restored", "deterministic_next_steps")}, indent=2))


if __name__ == "__main__":
    main()
