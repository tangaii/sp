"""Create the repository and file-hash manifest for E83.8.3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(paths):
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def main():
    source_paths = [path for path in (ROOT / "sources").rglob("*") if path.is_file()]
    author_paths = [ROOT / "sources/author" / name for name in
                    ("main.py", "model.py", "network.py", "layers.py", "utils.py", "result_Tadpole.txt")]
    e8382_root = ROOT / "sources/upstream_e8382/exp/experiment83_8_2_tadpole_g4_aloss_gradient_repair"
    e8382_paths = [path for path in e8382_root.rglob("*") if path.is_file()]
    selected_history = [
        "sources/e83_5_reference_4649c5d/protocol.json",
        "sources/e83_5_reference_4649c5d/common/run_e833.py",
        "sources/e83_5_reference_4649c5d/common/fidelity_model.py",
        "sources/e83_5_reference_4649c5d/results/group_summary.csv",
        "sources/e83_6_review/run_e836.py",
        "sources/e83_6_review/fidelity_model.py",
        "sources/e83_6_review/final_report.md",
        "sources/e83_7_review/run_e837.py",
        "sources/e83_7_review/fidelity_model.py",
        "sources/e83_7_final_report.md",
        "sources/saber_e83_8_protocol.json",
        "sources/saber_e83_8_report.md",
        "sources/saber_e83_8_group_summary.csv",
        "sources/saber_e83_8_1_protocol.json",
        "sources/saber_e83_8_1_report.md",
        "sources/saber_e83_8_1_group_summary.csv",
        "sources/saber_e83_8_1_final_audit.json",
    ]
    history_paths = [ROOT / name for name in selected_history]
    manifest = {
        "experiment": "Experiment 83.8.3",
        "repositories": {
            "author": {
                "url": "https://github.com/wanxixi11/SPromptGL",
                "commit": "ba3b9446fa451fde6864d6555bdf324de0c4b64a",
                "snapshot_sha256": hashes(author_paths),
                "author_log_template_sha256": sha256(author_paths[-1]),
            },
            "saber_main": {
                "url": "https://github.com/tangaii/SPromptGL",
                "commit": "5ad91878764863b04089a94c497707cc7459168b",
                "historical_source_snapshot_sha256": hashes(history_paths),
                "historical_assets": ["E83.5", "E83.6", "E83.7", "E83.8", "E83.8.1"],
            },
            "e8382": {
                "url": "https://github.com/tangaii/sp",
                "commit": "252fa96b5ff2ac89250cda378c777fc36a99d316",
                "snapshot_sha256": hashes(e8382_paths),
                "snapshot_root": str(e8382_root.relative_to(ROOT)),
                "distinct_repository_from_saber_main": True,
            },
            "paper": {
                "url": "https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf",
                "sha256": sha256(ROOT / "sources/sp/2024 MIA (MMGPL)Multimodal Medical Data Analysis with Graph Prompt Learning.pdf"),
            },
        },
        "training_inputs": {
            "data_sha256": sha256(ROOT / "inputs/processed_standard_data.csv"),
            "split_sha256": sha256(ROOT / "inputs/split_indices.json"),
            "prompt_manifest_sha256": sha256(ROOT / "prompt_manifest.json"),
            "prompt_files_sha256": hashes((ROOT / "inputs/prompts").glob("*.pt")),
            "group_config_sha256": hashes((ROOT / "configs").glob("*.json")),
        },
        "saber_historical_source_commit": "5ad91878764863b04089a94c497707cc7459168b",
        "fixed_e8382_source_commit": "252fa96b5ff2ac89250cda378c777fc36a99d316",
        "e8382_n3_training_configuration": json.loads((e8382_root / "configs/N3.json").read_text()),
        "e8383_training_configuration": {
            group: json.loads((ROOT / "configs" / f"{group}.json").read_text())
            for group in ("Y0", "Y1", "Y2", "Y3", "Y4", "Y5")
        },
    }
    target = ROOT / "source_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(target), "sha256": sha256(target),
                      "e8382_files": len(e8382_paths), "history_files": len(history_paths)}, indent=2))


if __name__ == "__main__":
    main()
