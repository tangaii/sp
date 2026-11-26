"""Record immutable protocol, source identity, and prerequisites before formal training."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as run

SABER_ROOT = Path("/data/adodas/SPromptGL-main")
AUTHOR_LOG_SHA = "5c80d16204e0882e5e264e42952c714397c2d31f07835b9e60d6dafd9488e3b7"


def command(*parts):
    return subprocess.check_output(parts, text=True).strip()


def main():
    seal_path = ROOT / "audits/training_seal.json"
    if seal_path.exists():
        raise RuntimeError("A training seal already exists")
    if any(list((ROOT / f"groups/{group}").glob("latest_checkpoints/*.pt")) for group in run.TRAIN_GROUPS):
        raise RuntimeError("A new formal checkpoint already exists before source sealing")
    preflight = ROOT / "audits/preflight_start.json"
    probe = ROOT / "audits/resource_probe.json"
    if any(json.loads(path.read_text()).get("status") != "PASS" for path in (preflight, probe)):
        raise RuntimeError("Formal preflight or resource probe did not pass")
    template = ROOT / "sources/author/result_Tadpole.txt"
    if run.sha_file(template) != AUTHOR_LOG_SHA or run.sha_file(SABER_ROOT / template.name) != AUTHOR_LOG_SHA:
        raise RuntimeError("The author log template differs")

    refs = ROOT / "sources/reference"
    upstream_files = sorted(path for path in refs.rglob("*") if path.is_file())
    if not upstream_files:
        raise RuntimeError("The frozen source reference snapshots are missing")
    groups = {group: run.group_config(group) for group in run.FORMAL_GROUPS}
    protocol = {
        "experiment": run.EXPERIMENT, "title": "TADPOLE COGNITIVE_TEST-Anchored DARG + DPSL",
        "base": "E83.8.3-Y3 / E83.8.4-Z0 full G4 of E82.4.6.10-G1",
        "patients": 598, "label_encoding": {"NC": 0, "MCI": 1, "AD": 2},
        "label_counts": {"NC": 209, "MCI": 315, "AD": 74},
        "modality_names": list(run.MODAL_NAMES), "modality_dimensions": list(run.MODAL_DIMS),
        "data_sha256": run.DATA_SHA, "split_sha256": run.SPLIT_SHA, "seed": 0,
        "split": "StratifiedKFold(n_splits=10, shuffle=True, random_state=10); val equals test",
        "preprocessing": "Historical full-cohort RFE, mean imputation and StandardScaler before CV",
        "graph": "Transductive features, threshold .9, historical second-modality overwrite, existing graph support only",
        "warmup_epochs": 10, "epoch_cap": 450, "early_stop": "wait > 200",
        "checkpoint_selection": "First strict validation ACC improvement after warmup",
        "latest_persistence": "Every tenth epoch, every strict improvement and terminal epoch",
        "relation_supervision": {"train_only": True, "pairs_per_fold": 4096, "same": 2048, "different": 2048,
                                 "bce": .05, "conditional_adversarial": .01, "preservation": .01,
                                 "grl": .03, "non_anchor_mutual_topk": 12, "thresholds": [.45, .55]},
        "dpsl": {"prompts": 12, "frozen_rn50": True, "residual_cap": .08, "shared_gradient_budget": .001},
        "aloss": {"kind": "Y3 squared Frobenius", "first_epoch": 11, "coefficient": .01},
        "darg_direct_double_gate": True,
        "official_folds": 50, "new_folds": 40, "groups": groups,
        "workers": {"physical_gpus": 1, "gpu_index": 0, "worker_limit": 8,
                    "schedule": "Highest stable measured aggregate throughput"},
        "metric_primary": {"ACC": "fraction of correct hard predictions",
                           "AUC_original": "hard one-hot predictions, macro one-versus-rest"},
        "metric_auxiliary": {"AUC_probability": "soft class scores, macro one-versus-rest"},
        "aggregate": "Mean and population standard deviation over 10 held-out folds; pooled OOF separately",
        "comparisons": ["R1-R0", "R2-R0", "R3-R0", "R3-R1", "R3-R2", "R4-R3", "R4-R0"],
        "decision": "Strictly higher ACC and AUC_original means than R0",
        "paper_reference": {"ACC": .9481, "ACC_std": .0295, "AUC": .9559, "AUC_std": .0257},
        "log_template_sha256": AUTHOR_LOG_SHA, "log_reported_epoch_cap": 450,
    }
    run.write_json(ROOT / "protocol.json", protocol)
    source = {
        "experiment": run.EXPERIMENT,
        "paper": {"url": "https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf",
                  "direct_download_sha256": "219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89"},
        "repositories": {
            "author": {"url": "https://github.com/wanxixi11/SPromptGL",
                       "commit": "ba3b9446fa451fde6864d6555bdf324de0c4b64a",
                       "result_Tadpole_git_blob": "0446670764286afbeb86573dadfdbb50ae6af15e"},
            "SABER": {"url": "https://github.com/tangaii/SPromptGL",
                      "local_checkout_commit": command("git", "-C", str(SABER_ROOT), "rev-parse", "HEAD"),
                      "prior_frozen_provenance_commit": "5ad91878764863b04089a94c497707cc7459168b"},
            "TADPOLE": {"url": "https://github.com/tangaii/sp",
                        "verified_N3_commit": "252fa96b5ff2ac89250cda378c777fc36a99d316",
                        "N3_config_git_blob": "73d91d16f3510df1338fafafa31e2fd284527011"},
            "Y3_short_ref": {"requested": "cbaaf1c", "resolved": False,
                             "identity": "Local archived source seal, config and checkpoint replay"},
        },
        "dependencies": {"python": platform.python_version(), "python_executable": sys.executable,
                         **{name: metadata.version(name) for name in
                            ("torch", "numpy", "pandas", "scipy", "scikit-learn")},
                         "torch_cuda_abi": torch.version.cuda,
                         "gpu_driver": command("nvidia-smi", "--query-gpu=name,driver_version",
                                               "--format=csv,noheader"),
                         "deterministic_algorithms": True, "streams": {"threads_per_worker": 1}},
        "source_review": "audits/source_review.md",
        "reference_files": {str(path.relative_to(ROOT)): run.sha_file(path) for path in upstream_files},
        "formal_and_reference_files": run.current_implementation_hashes(),
    }
    run.write_json(ROOT / "source_manifest.json", source)
    files = run.current_implementation_hashes()
    run.write_json(seal_path, {
        "experiment": run.EXPERIMENT, "status": "SEALED_BEFORE_TRAINING",
        "utc": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": run.implementation_fingerprint(files), "files": files,
        "preflight_sha256": run.sha_file(preflight), "resource_probe_sha256": run.sha_file(probe),
    })
    run.require_training_gate(run.TRAIN_GROUPS)
    print(run.implementation_fingerprint(files), flush=True)


if __name__ == "__main__":
    main()
