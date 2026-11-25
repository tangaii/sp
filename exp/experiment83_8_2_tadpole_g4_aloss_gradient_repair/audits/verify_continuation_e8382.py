"""Verify checkpoint continuation against the sealed E83.8.2 epoch ordering."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import copy
import hashlib
import json
import multiprocessing

import numpy as np
import torch

from audit_finalize_e8382 import AUDITS, GROUPS, ROOT, load_checkpoint_model, runner, sha_file


def initialize_worker():
    runner.configure_worker()


def best_snapshot_check(group, best, rng, x_np, y_np, device):
    _, x, labels, train, pair, same, model = load_checkpoint_model(group, best, x_np, y_np, device)
    runner.historical.restore_rng(rng)
    before = runner.sha_object(runner.historical.capture_rng_state())
    model.eval()
    with torch.no_grad():
        logp, aux = model(x, labels, train, pair, same, dpsl_enabled=True)
    predictions = logp[best["test_idx"]].argmax(1).cpu().numpy().astype(np.int64)
    if hashlib.sha256(predictions.tobytes()).hexdigest() != best["prediction_sha256"]:
        raise RuntimeError("Best inference differs from the selected predictions")
    if model.dpsl is None:
        return best, False

    # Best records the selection forward. Training then advances the EMA using
    # that forward's CAM before the next training epoch, as Latest already does.
    ema_updates = int(model.dpsl.ema_updates)
    model.dpsl.advance_ema(aux["cam"])
    if int(model.dpsl.ema_updates) != ema_updates + 1:
        raise RuntimeError("Expected one DPSL EMA advancement after Best selection")
    if runner.sha_object(runner.historical.capture_rng_state()) != before:
        raise RuntimeError("Best inference or EMA advancement changed RNG")
    continuation_model = copy.deepcopy(model.state_dict())
    return {**best, "model": continuation_model}, True


def next_step(group, checkpoint, rng, x_np, y_np, device):
    config, x, labels, train, pair, same, model = load_checkpoint_model(group, checkpoint, x_np, y_np, device)
    optimizers, optimizer_config = runner.make_optimizers(model, config, formal=True)
    if optimizer_config != checkpoint["optimizer_config"] or optimizers.keys() != checkpoint["optimizers"].keys():
        raise RuntimeError("Checkpoint optimizer configuration differs from the sealed configuration")
    for name, state in checkpoint["optimizers"].items():
        optimizers[name].load_state_dict(copy.deepcopy(state))
    runner.historical.restore_rng(rng)
    epoch = int(checkpoint["epoch"]) + 1
    kind = runner.active_supervision_kind(config, epoch)
    coefficient = runner.configured_supervision_coefficient(config, checkpoint.get("lambda_fold"))
    model.train()
    runner.zero_optimizers(optimizers)
    _, aux = model(x, labels, train, pair, same, dpsl_enabled=True)
    classification, historical_total, supervision, _ = runner.historical_loss(
        aux, labels, train, model, kind, coefficient)
    history = {
        "epoch": epoch, "loss": float((historical_total + aux["dpsl_loss"]).detach()),
        "classification": float(classification.detach()), "graph": float(aux["graph_loss"].detach()),
        "graph_supervision": float(supervision.detach()), "pair": float(aux["pair_loss"].detach()),
        "adv": float(aux["adv_loss"].detach()), "preserve": float(aux["preserve_loss"].detach()),
        "dpsl": float(aux["dpsl_loss"].detach()), "supervision_kind": kind,
        "supervision_coefficient": coefficient or 0.0,
    }
    if model.dpsl is None:
        historical_total.backward()
        runner.step_optimizers(optimizers)
    else:
        runner.dpsl_optimizer_step(model, optimizers, historical_total, aux["scmr_objectives"],
                                   float(config["dpsl_shared_cap"]))
    model.eval()
    test = checkpoint["test_idx"]
    with torch.no_grad():
        logp, eval_aux = model(x, labels, train, pair, same, dpsl_enabled=True)
    history["test_acc"] = float((logp.argmax(1)[test] == labels[test]).sum()) / len(test)
    if model.dpsl is not None:
        model.dpsl.advance_ema(eval_aux["cam"])
    return {
        "history": history,
        "model_sha256": runner.sha_object(model.state_dict()),
        "optimizer_sha256": runner.sha_object({name: opt.state_dict() for name, opt in optimizers.items()}),
        "rng_sha256": runner.sha_object(runner.historical.capture_rng_state()),
    }


def verify_case(payload):
    group, fold, kind, rng_record, x_np, y_np = payload
    device = torch.device("cuda:0")
    group_path = ROOT / "groups" / group
    fold_audit = json.loads((group_path / "audits" / f"fold{fold:02d}.json").read_text())
    latest_path = group_path / "latest_checkpoints" / f"fold{fold:02d}.pt"
    latest_sha = sha_file(latest_path)
    if fold_audit["status"] != "TRAINED" or latest_sha != fold_audit["latest_sha256"]:
        raise RuntimeError("Latest identity differs from the formal fold audit")
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    checkpoint_path, checkpoint_sha, checkpoint = latest_path, latest_sha, latest
    rng, reference, ema_advanced = latest["rng"], None, False
    if kind == "Best":
        checkpoint_path = group_path / "best_checkpoints" / f"fold{fold:02d}.pt"
        checkpoint_sha = sha_file(checkpoint_path)
        best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        rng = rng_record["rng"]
        if (rng_record["best_epoch"] != int(best["epoch"])
                or rng_record["best_sha256"] != latest["best_sha256"]
                or rng_record["best_sha256"] != checkpoint_sha
                or checkpoint_sha != fold_audit["checkpoint_sha256"]
                or rng_record["rng_sha256"] != runner.sha_object(rng)):
            raise RuntimeError("Best recovery RNG does not identify the selected checkpoint")
        checkpoint, ema_advanced = best_snapshot_check(group, best, rng, x_np, y_np, device)
        reference = next(row for row in latest["history"] if row["epoch"] == best["epoch"] + 1)
    first = next_step(group, checkpoint, rng, x_np, y_np, device)
    second = next_step(group, checkpoint, rng, x_np, y_np, device)
    if first != second:
        raise RuntimeError(f"Next-step determinism differs for {group}/{fold}/{kind}")
    if reference is not None:
        diffs = {key: [expected, reference[key]] for key, expected in first["history"].items()
                 if expected != reference[key]}
        if diffs:
            raise RuntimeError(f"Next step differs from the formal trajectory for {group}/{fold}: {diffs}")
    return {
        "group": group, "fold": fold, "checkpoint": kind, "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha, "next_epoch": first["history"]["epoch"],
        "prepared_best_ema": ema_advanced, "recorded_next_epoch_exact": reference is not None,
        "twin_step_exact": True, "status": "PASS", **first,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 11), default=list(range(1, 11)))
    parser.add_argument("--checkpoint", choices=("Best", "Latest", "Both"), default="Both")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Continuation verification requires the assigned single GPU")
    seal_path = AUDITS / "training_seal.json"
    seal = json.loads(seal_path.read_text())
    if seal["files"] != runner.current_implementation_hashes():
        raise RuntimeError("Training seal does not match the current implementation")
    recovery = json.loads((AUDITS / "checkpoint_recovery.json").read_text())
    rng_path = AUDITS / "best_recovery_rng.pt"
    if sha_file(rng_path) != recovery["best_recovery_rng_sidecar_sha256"]:
        raise RuntimeError("Best RNG recovery sidecar SHA differs")
    rng_records = torch.load(rng_path, map_location="cpu", weights_only=False)["states"]
    x_np, y_np, _, _ = runner.load_data()
    kinds = ("Best", "Latest") if args.checkpoint == "Both" else (args.checkpoint,)
    tasks = [(g, f, k, rng_records[f"{g}/fold{f:02d}"], x_np, y_np)
             for g in args.groups for f in args.folds for k in kinds]
    with runner.lock(AUDITS / "training.lock"):
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker) as pool:
            results = list(pool.map(verify_case, tasks))
    result = {
        "experiment": "Experiment 83.8.2", "status": "PASS",
        "training_seal_sha256": sha_file(seal_path), "audit_script_sha256": sha_file(Path(__file__)),
        "best_rng_sidecar_sha256": sha_file(rng_path),
        "source": "Best selection forward, post-selection DPSL EMA advancement and actual next epoch from Latest history",
        "best_replayed_and_next_epoch_matched": sum(r["recorded_next_epoch_exact"] for r in results),
        "best_dpsl_ema_advances": sum(r["prepared_best_ema"] for r in results),
        "latest_deterministic_steps": sum(r["checkpoint"] == "Latest" for r in results),
        "latest_step_scope": "Shadow step from the stored post-epoch state, without extending formal training",
        "cases": results,
    }
    value = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(value)
        print({key: count for key, count in result.items() if key != "cases"}, flush=True)
    else:
        print(value, end="", flush=True)


if __name__ == "__main__":
    main()
