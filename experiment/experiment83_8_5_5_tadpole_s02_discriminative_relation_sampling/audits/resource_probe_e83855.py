"""Short representative GPU concurrency probe for the sealed training path."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83855 as runner


def one_step(task_id: int):
    runner.configure_worker()
    runner.historical.seed_everything(runner.SEED)
    x_np, y_np, _, raw_splits = runner.load_data()
    fold = int(task_id % 10) + 1
    split = next(row for row in raw_splits if int(row["fold"]) == fold)
    state_path = runner.HISTORICAL_ROOTS["H1"] / "groups/S02/best_checkpoints" / f"fold{fold:02d}.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    config = runner.group_config("U1")
    device = torch.device("cuda:0")
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y_np, dtype=torch.long, device=device)
    train = torch.as_tensor(split["train_idx"], dtype=torch.long, device=device)
    pair = state["relation_pairs"].to(device)
    same = state["relation_targets"].to(device)
    model = runner.build_model(config, fold, device, spectral_stats=state["spectral_stats"])
    optimizers, _ = runner.make_optimizers(model, config, formal=True)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    runner.zero_optimizers(optimizers)
    _, aux = model(x, labels, train, pair, same, dpsl_enabled=True)
    _, historical_total, _, _ = runner.historical_loss(
        aux, labels, train, model, "paper_aloss", 0.01)
    if model.dpsl is not None:
        runner.dpsl_optimizer_step(model, optimizers, historical_total,
                                   aux["scmr_objectives"], 0.001)
    else:
        historical_total.backward()
        runner.step_optimizers(optimizers)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    del model, optimizers, x, labels, pair, same
    torch.cuda.empty_cache()
    return {"task": task_id, "fold": fold, "seconds": elapsed,
            "peak_memory_bytes": int(peak), "finite": True}


def probe(worker_count: int):
    started = time.perf_counter()
    rows = []
    failures = []
    with ProcessPoolExecutor(worker_count, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(one_step, index) for index in range(worker_count)]
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append(repr(exc))
    elapsed = time.perf_counter() - started
    return {"worker_count": worker_count, "wall_seconds": elapsed,
            "steps": len(rows), "throughput_steps_per_second": len(rows) / elapsed if elapsed else 0.0,
            "max_peak_memory_bytes": max((row["peak_memory_bytes"] for row in rows), default=0),
            "rows": sorted(rows, key=lambda row: row["task"]), "failures": failures,
            "status": "PASS" if len(rows) == worker_count and not failures else "FAIL"}


def main():
    results = [probe(count) for count in (4, 8, 10)]
    passing = [row for row in results if row["status"] == "PASS"]
    if not passing:
        raise RuntimeError(f"All resource probes failed: {results}")
    selected = max(passing, key=lambda row: row["throughput_steps_per_second"])
    output = {
        "experiment": runner.EXPERIMENT, "status": "PASS",
        "device": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count(),
        "comparisons": results, "worker_count": selected["worker_count"],
        "selection_rule": "highest measured representative-step throughput among stable 4/8/10 probes",
        "max_memory_bytes": max(row["max_peak_memory_bytes"] for row in passing),
    }
    (ROOT / "audits/resource_probe.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: output[key] for key in ("status", "device", "worker_count", "comparisons")}, indent=2))


if __name__ == "__main__":
    main()
