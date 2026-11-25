"""Probe eight independent GPU workers before formal execution."""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "common"))
import run_e8382 as runner


def worker(group):
    runner.configure_worker()
    runner.historical.seed_everything(runner.SEED)
    x_np, y_np, _, splits = runner.load_data()
    config = runner.group_config(group)
    device = torch.device("cuda:0")
    before = torch.cuda.memory_reserved(0)
    model = runner.build_model(config, 1, device)
    opts, _ = runner.make_optimizers(model, config)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    train = torch.as_tensor(splits[0]["train_idx"], dtype=torch.long, device=device)
    supports = runner.historical.anchor_pair_support(model, x)
    pairs, same = runner.historical.make_pairs(splits[0]["train_idx"], y_np, 1, supports)
    pair = torch.from_numpy(pairs).long().to(device)
    same = torch.from_numpy(same).float().to(device)
    model.train()
    runner.zero_optimizers(opts)
    started = time.monotonic()
    _, aux = model(x, labels, train, pair, same, dpsl_enabled=False, darg_warmup=True)
    loss, _ = runner.warmup_loss(aux, labels, train, model, "none", None)
    loss.backward()
    runner.step_optimizers(opts)
    torch.cuda.synchronize()
    return {"status": "PASS", "group": group, "loss": float(loss.detach()),
            "reserved_mib": torch.cuda.memory_reserved(0) / 2**20,
            "allocated_mib": torch.cuda.memory_allocated(0) / 2**20,
            "delta_reserved_mib": (torch.cuda.memory_reserved(0) - before) / 2**20,
            "seconds": time.monotonic() - started,
            "device": torch.cuda.get_device_name(0)}


def main():
    if torch.cuda.device_count() != 1:
        raise RuntimeError("resource probe requires exactly one GPU")
    groups = [f"N{index}" for index in range(7)]
    results = {}
    failures = []
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
        tasks = {pool.submit(worker, groups[index % len(groups)]): index for index in range(8)}
        for future in as_completed(tasks):
            index = tasks[future]
            try:
                result = future.result()
                results[str(index)] = result
            except Exception as exc:
                failures.append({"worker": index, "error": repr(exc)})
    if failures:
        raise RuntimeError(json.dumps(failures))
    payload = {
        "experiment": "Experiment 83.8.2", "status": "PASS", "worker_count": 8,
        "device": torch.cuda.get_device_name(0), "groups": {
            group: {"status": "PASS", "probe_runs": [item for item in results.values()
                                                       if item["group"] == group]}
            for group in groups
        },
        "workers": results,
        "training_seal_sha256": None,
    }
    (ROOT / "audits/resource_probe.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
