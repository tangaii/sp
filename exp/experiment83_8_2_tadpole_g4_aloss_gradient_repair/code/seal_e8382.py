"""Create the immutable source/input/config training seal."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8382 as runner


def main():
    hashes = runner.current_implementation_hashes()
    seal = {
        "experiment": "Experiment 83.8.2",
        "status": "SEALED_BEFORE_TRAINING",
        "root": str(ROOT),
        "implementation_sha256": runner.implementation_fingerprint(hashes),
        "files": hashes,
        "groups": [f"N{index}" for index in range(7)],
        "worker_limit": 8,
        "seed": runner.SEED,
        "data_sha256": runner.DATA_SHA,
        "split_sha256": runner.SPLIT_SHA,
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    path = ROOT / "audits/training_seal.json"
    if path.exists():
        raise RuntimeError("training seal already exists and cannot be overwritten")
    path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "implementation_sha256": seal["implementation_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
