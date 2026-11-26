"""Read-only E82.4.6.10-G1 checkpoint audit captured in this experiment."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABIDE_ROOT = Path("/model/spromptgl/experiment/experiment82_4_6_10_fold8_dual_asd")
manifest = json.loads((ABIDE_ROOT / "manifest.json").read_text())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = str(manifest["config"]["gpu"].get("threads", 1))
sys.path.insert(0, "/model/spromptgl/experiment/experiment82_3_6_ps_pgr/runtime/python")
sys.path.insert(0, str(ABIDE_ROOT))

from execute import checkpoint_audit
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.init()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    certificate = checkpoint_audit("G1", manifest)
    if certificate.get("status") not in {
        "COMPLETE_AUDITED", "AUDITED_WITH_RELATION_COVERAGE_LIMITATION"
    }:
        raise RuntimeError(f"ABIDE checkpoint audit did not complete: {certificate.get('status')}")
    result = {
        "experiment": "Experiment 83.8.5.1",
        "status": "PASS",
        "scope": "Read-only independent replay of E82.4.6.10-G1 Fold 8",
        "device_mapping": {"visible_device": 0, "physical_uuid": manifest["config"]["gpu"]["uuid"],
                           "reason": "Current PPU driver accepts index mapping and rejects the historical UUID string"},
        "source_experiment_status": manifest["status"],
        "source_manifest_sha256": sha(ABIDE_ROOT / "manifest.json"),
        "source_implementation_sha256": manifest["implementation_sha"],
        "checkpoint_paths": {
            "best": str(ABIDE_ROOT / "checkpoints/G1_f08.best.pt"),
            "latest": str(ABIDE_ROOT / "checkpoints/G1_f08.latest.pt"),
        },
        "checkpoint_sha256": {
            "best": sha(ABIDE_ROOT / "checkpoints/G1_f08.best.pt"),
            "latest": sha(ABIDE_ROOT / "checkpoints/G1_f08.latest.pt"),
        },
        "certificate": certificate,
        "audit_program_sha256": sha(Path(__file__)),
    }
    target = ROOT / "audits/abide_checkpoint_replay.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "group": certificate["group"],
        "fold": certificate["fold"],
        "best_epoch": certificate["best_epoch"],
        "latest_epoch": certificate["latest_epoch"],
        "best_reloaded": certificate["best_reloaded"],
        "latest_next_training_exact": certificate["latest_next_training_exact"],
        "output": str(target),
    }, indent=2))


if __name__ == "__main__":
    main()
