"""Measure model-update throughput and device utilization for the sealed workload."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as run


def probe_worker(worker, steps, barrier, output):
    try:
        run.configure_worker()
        fold = worker + 1
        group = "R4" if worker % 2 == 0 else "R3"
        config = run.group_config(group)
        x_np, y_np, _, splits = run.load_data()
        x = torch.as_tensor(x_np, device="cuda")
        labels = torch.as_tensor(y_np, device=x.device)
        split = splits[worker]
        train = torch.as_tensor(split["train_idx"], device=x.device)
        best = torch.load(ROOT / f"groups/R0/best_checkpoints/fold{fold:02d}.pt",
                          map_location="cpu", weights_only=False)
        run.historical.seed_everything(run.SEED)
        model = run.build_model(config, fold, x.device, best["spectral_stats"])
        model.darg.enter_positive_dcr()
        model.load_state_dict(best["model"], strict=True)
        pair = best["relation_pairs"].to(x.device)
        same = best["relation_targets"].to(x.device)
        optimizers, _ = run.make_optimizers(model, config, formal=True)

        def step():
            model.train()
            run.zero_optimizers(optimizers)
            _, aux = model(x, labels, train, pair, same)
            _, total, _, _ = run.historical_loss(aux, labels, train, model, "paper_aloss", .01)
            if not torch.isfinite(total + aux["dpsl_loss"]):
                raise FloatingPointError(f"Nonfinite probe update for {group}:{fold}")
            run.dpsl_optimizer_step(model, optimizers, total, aux["scmr_objectives"], .001)
            model.eval()
            with torch.no_grad():
                _, evaluation = model(x, labels, train, pair, same)
            model.dpsl.advance_ema(evaluation["cam"])

        step()
        torch.cuda.synchronize()
        barrier.wait(timeout=120)
        started = time.monotonic()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        ended = time.monotonic()
        output.put({"status": "PASS", "group": group, "fold": fold, "started": started,
                    "ended": ended, "steps": steps, "peak_allocated": torch.cuda.max_memory_allocated(),
                    "peak_reserved": torch.cuda.max_memory_reserved()})
    except Exception:
        output.put({"status": "FAIL", "fold": worker + 1, "trace": traceback.format_exc()})
        barrier.abort()


def sample_device(stop, samples):
    args = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits", "-i", "0"]
    while not stop.is_set():
        now = time.monotonic()
        result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=True)
        util, used, total = (int(value) for value in result.stdout.strip().split(","))
        samples.append((now, util, used, total))
        stop.wait(.25)


def trial(workers, steps):
    ctx = multiprocessing.get_context("spawn")
    barrier, output = ctx.Barrier(workers), ctx.Queue()
    processes = [ctx.Process(target=probe_worker, args=(i, steps, barrier, output)) for i in range(workers)]
    stop, samples = threading.Event(), []
    monitor = threading.Thread(target=sample_device, args=(stop, samples))
    monitor.start()
    for process in processes:
        process.start()
    try:
        measurements = [output.get(timeout=125) for _ in processes]
    except queue.Empty:
        measurements = [{"status": "FAIL", "trace": "Probe workers did not complete within 125 seconds"}]
    finally:
        for process in processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join()
        stop.set()
        monitor.join()
    if len(measurements) != workers or any(row["status"] != "PASS" for row in measurements):
        return {"workers": workers, "status": "FAIL", "measurements": measurements}
    start = min(row["started"] for row in measurements)
    end = max(row["ended"] for row in measurements)
    active = [row for row in samples if start <= row[0] <= end]
    if not active:
        return {"workers": workers, "status": "FAIL", "measurements": measurements,
                "reason": "No device sample overlapped the timed update window"}
    peak = max(row[2] for row in samples)
    total = active[0][3]
    return {"workers": workers, "status": "PASS" if peak < .9 * total else "FAIL",
            "seconds": end - start, "steps_per_second": workers * steps / (end - start),
            "median_seconds_per_worker_step": statistics.median(
                (row["ended"] - row["started"]) / steps for row in measurements),
            "gpu_utilization_percent_median": statistics.median(row[1] for row in active),
            "gpu_utilization_percent_max": max(row[1] for row in active),
            "gpu_device_peak_mib": peak, "gpu_device_total_mib": total,
            "active_device_samples": len(active), "measurements": sorted(measurements, key=lambda row: row["fold"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    preflight = json.loads((ROOT / "audits/preflight_start.json").read_text())
    if not preflight["formal_training_allowed"]:
        raise RuntimeError("The resource probe requires the method preflight")
    results = []
    for workers in (1, 2, 4, 8):
        row = trial(workers, args.steps)
        results.append(row)
        print(f"{row['status']} workers={workers} throughput={row.get('steps_per_second')} "
              f"peak_mib={row.get('gpu_device_peak_mib')}", flush=True)
    eligible = [row for row in results if row["status"] == "PASS"]
    best = max(eligible, key=lambda row: row["steps_per_second"]) if eligible else None
    record = {"experiment": run.EXPERIMENT, "status": "PASS" if best else "FAIL",
              "utc": datetime.now(timezone.utc).isoformat(),
              "worker_count": best["workers"] if best else 0, "gpu": "PPU-ZW810E", "gpu_index": 0,
              "cpu_threads_per_worker": 1, "global_worker_limit": 8,
              "timing": "Full forward, SCMR gradient budgets, optimizer update, evaluation forward, EMA and synchronization; no checkpoint writes",
              "selection": "Highest measured aggregate throughput among successful trials below 90% GPU memory",
              "trials": results}
    run.write_json(ROOT / "audits/resource_probe.json", record)
    return 0 if best else 1


if __name__ == "__main__":
    raise SystemExit(main())
