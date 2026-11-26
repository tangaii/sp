"""Freeze the E83.8.4 code, inputs and configs before formal training."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8384 as runner


def main():
    audit_dir = ROOT / "audits"
    preflight_path = audit_dir / "preflight_start.json"
    seal_path = audit_dir / "training_seal.json"
    if seal_path.exists():
        raise RuntimeError("The E83.8.4 training seal already exists")
    if not preflight_path.is_file():
        raise RuntimeError("Preflight must be complete before training is sealed")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "PASS" or preflight.get("formal_training_allowed") is not True:
        raise RuntimeError("Only a passing preflight can be sealed")
    hashes = runner.current_implementation_hashes()
    seal = {
        "experiment": "Experiment 83.8.4",
        "status": "SEALED_BEFORE_TRAINING",
        "root": str(ROOT),
        "implementation_sha256": runner.implementation_fingerprint(hashes),
        "files": hashes,
        "groups": list(runner.FORMAL_GROUPS),
        "worker_limit": 8,
        "seed": runner.SEED,
        "data_sha256": runner.DATA_SHA,
        "split_sha256": runner.SPLIT_SHA,
        "preflight_sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": runner.sha_file(ROOT / "source_manifest.json"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(seal_path), "sha256": runner.sha_file(seal_path),
                      "implementation_sha256": seal["implementation_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
