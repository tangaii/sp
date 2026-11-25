"""Attach the immutable training seal hash to preflight and resource audits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    seal_path = ROOT / "audits/training_seal.json"
    seal_sha = hashlib.sha256(seal_path.read_bytes()).hexdigest()
    for name in ("preflight_start.json", "resource_probe.json"):
        path = ROOT / "audits" / name
        payload = json.loads(path.read_text())
        if payload.get("training_seal_sha256") not in (None, seal_sha):
            raise RuntimeError(f"{name} already references a different training seal")
        payload["training_seal_sha256"] = seal_sha
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(seal_sha)


if __name__ == "__main__":
    main()
