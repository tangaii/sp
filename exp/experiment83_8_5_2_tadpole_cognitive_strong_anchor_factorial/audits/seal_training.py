"""Create the immutable pre-training source seal."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as runner

def main():
    preflight = ROOT / "audits/preflight.json"
    resource = ROOT / "audits/resource_probe.json"
    if not preflight.is_file() or not resource.is_file():
        raise RuntimeError("preflight and resource probe are required")
    preflight_payload = json.loads(preflight.read_text())
    resource_payload = json.loads(resource.read_text())
    if preflight_payload.get("status") != "PASS" or not preflight_payload.get("formal_training_allowed"):
        raise RuntimeError("preflight is not a formal PASS")
    if resource_payload.get("status") != "PASS" or not 1 <= int(resource_payload.get("worker_count", 0)) <= 8:
        raise RuntimeError("resource probe is not a usable PASS")
    files = runner.current_implementation_hashes()
    seal = {
        "experiment": runner.EXPERIMENT,
        "status": "SEALED_BEFORE_TRAINING",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_directory": str(ROOT),
        "implementation_sha256": runner.implementation_fingerprint(files),
        "files": files,
        "preflight_sha256": runner.sha_file(preflight),
        "resource_probe_sha256": runner.sha_file(resource),
        "method_fidelity_sha256": runner.sha_file(ROOT / "audits/method_fidelity.json"),
        "parity_sha256": runner.sha_file(ROOT / "audits/abide_tadpole_parity.json"),
        "warmup_pairing_sha256": runner.sha_file(ROOT / "audits/warmup_pairing.json"),
        "registered_groups": list(runner.FORMAL_GROUPS),
        "registered_training_groups": list(runner.TRAIN_GROUPS),
        "data_sha256": runner.DATA_SHA,
        "split_sha256": runner.SPLIT_SHA,
        "source_edit_policy": "No training implementation or registered configuration edits after this seal",
    }
    (ROOT / "audits/training_seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": seal["status"], "implementation_sha256": seal["implementation_sha256"],
                      "worker_count": resource_payload["worker_count"]}, indent=2))
if __name__ == "__main__":
    main()

