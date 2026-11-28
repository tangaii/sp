"""Seal the preflighted implementation before formal training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83855 as runner


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preflight_path = ROOT / "audits/preflight.json"
    resource_path = ROOT / "audits/resource_probe.json"
    preflight = json.loads(preflight_path.read_text())
    resource = json.loads(resource_path.read_text())
    if (preflight.get("experiment") != runner.EXPERIMENT
            or preflight.get("status") != "PASS"
            or preflight.get("formal_training_allowed") is not True
            or resource.get("experiment") != runner.EXPERIMENT
            or resource.get("status") != "PASS"):
        raise RuntimeError("Preflight or resource probe is not a formal-training PASS")
    hashes = runner.current_implementation_hashes()
    root_commit = subprocess.run(
        ["git", "-C", "/data/adodas/SPromptGL-main", "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    root_files = {
        name: sha(Path("/data/adodas/SPromptGL-main") / name)
        for name in ("main.py", "model.py", "network.py", "layers.py")
    }
    audit_scripts = {
        path.relative_to(ROOT).as_posix(): sha(path)
        for path in sorted((ROOT / "audits").glob("*.py"))
    }
    seal = {
        "experiment": runner.EXPERIMENT,
        "status": "SEALED_BEFORE_TRAINING",
        "implementation_sha256": runner.implementation_fingerprint(hashes),
        "files": hashes,
        "preflight_sha256": sha(preflight_path),
        "resource_probe_sha256": sha(resource_path),
        "selected_worker_count": int(resource["worker_count"]),
        "selected_resource_probe": "highest stable measured throughput among 4/8/10 workers",
        "root_git_commit": root_commit,
        "root_model_file_sha256": root_files,
        "root_model_files_modified": False,
        "audit_scripts_sha256": audit_scripts,
        "data_sha256": runner.DATA_SHA,
        "split_sha256": runner.SPLIT_SHA,
        "source_review_sha256": sha(ROOT / "audits/source_review.md"),
        "method_fidelity_sha256": sha(ROOT / "audits/method_fidelity.json"),
        "parity_sha256": sha(ROOT / "audits/abide_tadpole_parity.json"),
        "warmup_pairing_sha256": sha(ROOT / "audits/warmup_pairing.json"),
    }
    (ROOT / "audits/training_seal.json").write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: seal[key] for key in (
        "status", "implementation_sha256", "selected_worker_count", "root_git_commit")}, indent=2))


if __name__ == "__main__":
    main()
