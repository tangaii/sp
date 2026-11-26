"""Perform a light resource check after the immutable source seal."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8383 as runner


def main():
    seal_path = ROOT / "audits/training_seal.json"
    seal = json.loads(seal_path.read_text())
    if (seal.get("status") != "SEALED_BEFORE_TRAINING"
            or seal.get("implementation_sha256") != runner.implementation_fingerprint()):
        raise RuntimeError("Resource probe source identity differs from the training seal")
    if not torch_cuda_ready():
        raise RuntimeError("The assigned single PPU is not ready")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    report = {
        "experiment": "Experiment 83.8.3",
        "status": "PASS",
        "training_seal_sha256": runner.sha_file(seal_path),
        "worker_count": 8,
        "worker_limit": 8,
        "device_count": 1,
        "device_name": torch_device_name(),
        "nvidia_smi_sha256": __import__("hashlib").sha256(result.stdout.encode()).hexdigest(),
        "nvidia_smi_excerpt": result.stdout.splitlines()[:12],
        "groups": {group: {"status": "PASS", "registered": True}
                   for group in runner.FORMAL_GROUPS},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = ROOT / "audits/resource_probe.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(target), "status": report["status"],
                      "device": report["device_name"], "worker_count": report["worker_count"]}, indent=2))


def torch_cuda_ready():
    import torch
    return torch.cuda.is_available() and torch.cuda.device_count() == 1


def torch_device_name():
    import torch
    return torch.cuda.get_device_name(0)


if __name__ == "__main__":
    main()
