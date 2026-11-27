"""Short 1/2/4/8 worker probe using the registered TADPOLE DARG path."""
from __future__ import annotations
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"

def worker(worker_id, steps, queue):
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        sys.path.insert(0, str(CODE))
        import torch
        import run_e8385 as runner
        runner.configure_worker()
        x_np, y_np, ids, splits = runner.load_data()
        split = splits[0]
        runner.historical.seed_everything(runner.SEED + worker_id)
        device = torch.device("cuda:0")
        x = torch.from_numpy(x_np).to(device)
        labels = torch.from_numpy(y_np).long().to(device)
        train = torch.tensor(split["train_idx"], dtype=torch.long, device=device)
        config = runner.group_config("S09")
        model = runner.build_model(config, 1, device)
        supports = runner.historical.anchor_pair_support(model, x)
        pair_np, same_np = runner.historical.make_pairs(train.cpu().numpy(), y_np, 1, supports)
        pair = torch.from_numpy(pair_np).long().to(device)
        same = torch.from_numpy(same_np).float().to(device)
        model.darg.enter_positive_dcr()
        optimizers, _ = runner.make_optimizers(model, config)
        for _ in range(1):
            runner.zero_optimizers(optimizers)
            _, aux = model(x, labels, train, pair, same, dpsl_enabled=False)
            _, total, _, _ = runner.historical_loss(aux, labels, train, model, "none")
            total.backward()
            runner.step_optimizers(optimizers)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            runner.zero_optimizers(optimizers)
            _, aux = model(x, labels, train, pair, same, dpsl_enabled=False)
            _, total, _, _ = runner.historical_loss(aux, labels, train, model, "none")
            total.backward()
            runner.step_optimizers(optimizers)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        queue.put({"worker": worker_id, "status": "PASS", "steps": steps,
                   "elapsed_seconds": elapsed, "steps_per_second": steps / max(elapsed, 1e-9),
                   "peak_memory_mib": int(torch.cuda.max_memory_reserved() / 2**20)})
    except Exception as exc:
        queue.put({"worker": worker_id, "status": "FAIL", "error": repr(exc)})

def run_trial(workers, steps=2):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=worker, args=(i, steps, queue)) for i in range(workers)]
    for process in processes:
        process.start()
    records = [queue.get(timeout=300) for _ in processes]
    for process in processes:
        process.join(timeout=60)
    if any(record["status"] != "PASS" for record in records):
        raise RuntimeError(f"worker probe failed for {workers}: {records}")
    return {"workers": workers, "status": "PASS",
            "aggregate_steps_per_second": sum(item["steps_per_second"] for item in records),
            "worker_records": sorted(records, key=lambda item: item["worker"])}

def main():
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("resource probe requires exactly one physical GPU")
    trials = [run_trial(workers) for workers in (1, 2, 4, 8)]
    selected = max(trials, key=lambda item: item["aggregate_steps_per_second"])
    result = {
        "experiment": "Experiment 83.8.5.2",
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "gpu_index": 0,
        "gpu_memory_mib": 98304,
        "cpu_threads_per_worker": 1,
        "global_worker_limit": 8,
        "probe": "registered S09 TADPOLE six-modality DARG forward, graph loss, backward and optimizer step",
        "steps_per_worker": 2,
        "trials": trials,
        "worker_count": selected["workers"],
        "selection_basis": "highest aggregate DARG steps per second",
        "selected_aggregate_steps_per_second": selected["aggregate_steps_per_second"],
        "historical_compatible_probe": {
            "source": "/model/spromptgl/experiment/experiment83_8_5_1_tadpole_y3_cognitive_darg_fidelity_factorial/audits/resource_probe.json",
            "historical_best_workers": 4
        }
    }
    (ROOT / "audits/resource_probe.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

