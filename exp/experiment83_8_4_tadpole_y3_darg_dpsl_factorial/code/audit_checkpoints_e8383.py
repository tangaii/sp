"""Replay each Best and Latest checkpoint and verify deterministic continuation."""
from __future__ import annotations

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


def finite_optimizer(optimizers):
    for optimizer in optimizers.values():
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    check(torch.isfinite(value).all().item(), "Checkpoint optimizer contains non-finite state")


def build_loaded(group, fold, checkpoint, config, x, y, split):
    device = torch.device("cuda:0")
    runner.historical.seed_everything(runner.SEED)
    model = runner.build_model(config, fold, device, spectral_stats=checkpoint["spectral_stats"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.darg.enter_positive_dcr()
    optimizers, optimizer_config = runner.make_optimizers(model, config, formal=True)
    check(optimizer_config == checkpoint["optimizer_config"], f"{group} fold {fold} optimizer config differs")
    for name, state in checkpoint["optimizers"].items():
        optimizers[name].load_state_dict(state)
    pair = checkpoint["relation_pairs"].to(device)
    same = checkpoint["relation_targets"].to(device)
    train_rows = torch.as_tensor(split["train_idx"], dtype=torch.long, device=device)
    x_tensor = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(y).long().to(device)
    return model, optimizers, pair, same, train_rows, x_tensor, labels


def best_replay(group, fold, config, audit, best, frame, x, y, split):
    model, optimizers, pair, same, train_rows, x_tensor, labels = build_loaded(
        group, fold, best, config, x, y, split)
    model.eval()
    with torch.no_grad():
        logp, aux = model(x_tensor, labels, train_rows, pair, same, dpsl_enabled=True)
    test_rows = np.asarray(split["test_idx"], dtype=np.int64)
    probabilities = logp.exp().detach().cpu().numpy()[test_rows]
    prediction = logp.argmax(1).detach().cpu().numpy()[test_rows]
    check(np.array_equal(prediction, frame["pred_label"].to_numpy(dtype=np.int64)),
          f"{group} fold {fold} Best replay predictions differ")
    saved_probabilities = frame[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy(dtype=np.float64)
    max_probability_difference = float(np.max(np.abs(probabilities - saved_probabilities)))
    check(max_probability_difference <= 1e-12, f"{group} fold {fold} Best replay probabilities differ")
    check(int(best["epoch"]) == int(audit["best_epoch"]), f"{group} fold {fold} Best epoch differs")
    check(int(best["epoch"]) == int(best["metrics_at_best"].get("epoch", best["epoch"])),
          f"{group} fold {fold} Best metric epoch differs")
    finite_optimizer(optimizers)
    confidence = aux.get("dpsl", {}).get("confidence_gate")
    if config["dpsl_confidence_protection"]:
        check(confidence is not None, f"{group} fold {fold} confidence gate was not replayed")
        check(float(aux["dpsl"]["ratio"].max()) <= .08 + 1e-7,
              f"{group} fold {fold} replay residual exceeds the cap")
    return {"status": "PASS", "prediction_exact": True,
            "probability_max_abs_difference": max_probability_difference,
            "best_epoch": int(best["epoch"]),
            "confidence_gate_range": ([float(confidence.min()), float(confidence.max())]
                                       if confidence is not None else None)}


def next_step_signature(group, fold, config, latest, x, y, split):
    model, optimizers, pair, same, train_rows, x_tensor, labels = build_loaded(
        group, fold, latest, config, x, y, split)
    runner.historical.restore_rng(latest["rng"])
    model.train()
    runner.zero_optimizers(optimizers)
    next_epoch = int(latest["epoch"]) + 1
    logp, aux = model(x_tensor, labels, train_rows, pair, same, dpsl_enabled=True)
    graph_kind = runner.active_supervision_kind(config, next_epoch)
    coefficient = runner.configured_supervision_coefficient(config, latest.get("lambda_fold"))
    _, historical_total, _, _ = runner.historical_loss(
        aux, labels, train_rows, model, graph_kind, coefficient)
    total = historical_total + aux["dpsl_loss"]
    check(torch.isfinite(total).all().item(), f"{group} fold {fold} next-step loss is non-finite")
    if model.dpsl is not None:
        budget = runner.dpsl_optimizer_step(
            model, optimizers, historical_total, aux["scmr_objectives"],
            float(config["dpsl_shared_cap"]))
    else:
        historical_total.backward()
        runner.step_optimizers(optimizers)
        budget = {"families": {}, "shared_cap": 0.0}
    check(all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
          f"{group} fold {fold} next-step parameters are non-finite")
    finite_optimizer(optimizers)
    model.eval()
    with torch.no_grad():
        _, eval_aux = model(x_tensor, labels, train_rows, pair, same, dpsl_enabled=True)
    model.train()
    if model.dpsl is not None:
        model.dpsl.advance_ema(eval_aux["cam"])
    return {
        "model_sha256": runner.sha_object(model.state_dict()),
        "optimizer_sha256": runner.sha_object({key: value.state_dict() for key, value in optimizers.items()}),
        "rng_sha256": runner.sha_object(runner.historical.capture_rng_state()),
        "loss": float(total.detach()),
        "prediction_sha256": runner.sha_object(logp.detach()),
        "budget": budget,
    }


def main():
    runner.require_training_gate(list(runner.FORMAL_GROUPS))
    x, y, ids, splits = runner.load_data()
    split_by_fold = {int(item["fold"]): item for item in splits}
    results = {}
    for group in runner.FORMAL_GROUPS:
        group_dir = ROOT / "groups" / group
        config = runner.group_config(group)
        folds = {}
        for fold in range(1, 11):
            audit = json.loads((group_dir / "audits" / f"fold{fold:02d}.json").read_text())
            if audit.get("status") != "TRAINED":
                raise RuntimeError(f"{group} fold {fold} is incomplete")
            best_path = group_dir / "best_checkpoints" / f"fold{fold:02d}.pt"
            latest_path = group_dir / "latest_checkpoints" / f"fold{fold:02d}.pt"
            check(runner.sha_file(best_path) == audit["checkpoint_sha256"],
                  f"{group} fold {fold} Best SHA differs")
            check(runner.sha_file(latest_path) == audit["latest_sha256"],
                  f"{group} fold {fold} Latest SHA differs")
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            latest = torch.load(latest_path, map_location="cpu", weights_only=False)
            frame = pd.read_csv(group_dir / "predictions" / f"fold{fold:02d}.csv")
            best_result = best_replay(group, fold, config, audit, best, frame, x, y, split_by_fold[fold])
            first = next_step_signature(group, fold, config, latest, x, y, split_by_fold[fold])
            second = next_step_signature(group, fold, config, latest, x, y, split_by_fold[fold])
            check(first == second, f"{group} fold {fold} Latest next-step continuation is not deterministic")
            folds[str(fold)] = {
                "best": best_result,
                "latest": {"restore_status": "PASS", "next_step_deterministic": True,
                           "latest_epoch": int(latest["epoch"]),
                           "next_step_model_sha256": first["model_sha256"],
                           "next_step_optimizer_sha256": first["optimizer_sha256"],
                           "next_step_rng_sha256": first["rng_sha256"],
                           "next_step_loss": first["loss"]},
            }
            print(f"PASS {group} fold {fold:02d} Best/Latest", flush=True)
            del best, latest
            torch.cuda.empty_cache()
        results[group] = folds
    report = {
        "experiment": "Experiment 83.8.3",
        "status": "PASS",
        "best_replayed": sum(item["best"]["status"] == "PASS" for group in results.values()
                              for item in group.values()),
        "latest_restored": sum(item["latest"]["restore_status"] == "PASS" for group in results.values()
                               for item in group.values()),
        "deterministic_next_steps": sum(item["latest"]["next_step_deterministic"] for group in results.values()
                                        for item in group.values()),
        "groups": results,
    }
    target = ROOT / "audits/checkpoint_replay.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "best_replayed", "latest_restored", "deterministic_next_steps")}, indent=2))


if __name__ == "__main__":
    main()
