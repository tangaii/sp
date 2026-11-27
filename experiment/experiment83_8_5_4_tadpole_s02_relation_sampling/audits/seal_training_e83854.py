"""Write the immutable pre-training source and audit seal."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83854 as runner


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    preflight = ROOT / "audits/preflight.json"
    resource = ROOT / "audits/resource_probe.json"
    if not preflight.is_file() or not resource.is_file():
        raise RuntimeError("preflight and resource probe are required")
    preflight_data = json.loads(preflight.read_text())
    resource_data = json.loads(resource.read_text())
    if preflight_data.get("status") != "PASS" or not preflight_data.get("formal_training_allowed"):
        raise RuntimeError("preflight does not allow formal training")
    if resource_data.get("status") != "PASS":
        raise RuntimeError("resource probe did not pass")
    files = runner.current_implementation_hashes()
    seal = {
        "experiment": runner.EXPERIMENT,
        "status": "SEALED_BEFORE_TRAINING",
        "files": files,
        "implementation_sha256": runner.implementation_fingerprint(files),
        "preflight_sha256": sha(preflight),
        "resource_probe_sha256": sha(resource),
        "data_sha256": runner.DATA_SHA,
        "split_sha256": runner.SPLIT_SHA,
    }
    (ROOT / "audits/training_seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": seal["status"], "implementation_sha256": seal["implementation_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
