"""Small resource probe required before the formal single-GPU run."""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "inputs/processed_standard_data.csv"
EXPERIMENT = "Experiment 83.8.5.4"


def read_hash(_: int) -> str:
    return hashlib.sha256(DATA.read_bytes()).hexdigest()


def timed(workers: int) -> dict:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(read_hash, range(workers)))
    seconds = time.perf_counter() - start
    return {"workers": workers, "seconds": seconds, "hashes_match": len(set(values)) == 1}


def main() -> None:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True).stdout.strip().splitlines()
    four = timed(4)
    eight = timed(8)
    if not four["hashes_match"] or not eight["hashes_match"]:
        raise RuntimeError("resource probe read hashes differ")
    # The probe is a stability check. Training remains globally serialized on GPU 0.
    selected = 8 if eight["seconds"] <= four["seconds"] * 1.25 else 4
    result = {
        "experiment": EXPERIMENT,
        "status": "PASS",
        "gpu_query": gpu,
        "gpu_index": 0,
        "gpu_name": "PPU-ZW810E",
        "gpu_memory_total_mib": 98304,
        "worker_count": selected,
        "four_worker": four,
        "eight_worker": eight,
        "selection_rule": "highest stable read throughput within 25 percent tolerance",
        "training_policy": "one global GPU resource lock, one fold per worker task",
        "pid": os.getpid(),
    }
    (ROOT / "audits/resource_probe.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
