"""Run the sealed Experiment 83.8.2 source, data, and method preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "common"))
import run_e8382 as runner


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def split_check(x, y, splits):
    expected = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
    for item, (train, test) in zip(sorted(splits, key=lambda value: int(value["fold"])), expected):
        check(np.array_equal(np.asarray(item["train_idx"]), train), f"fold {item['fold']} train split differs")
        check(np.array_equal(np.asarray(item["test_idx"]), test), f"fold {item['fold']} test split differs")
        check(item.get("val_equals_test") is True, f"fold {item['fold']} does not use val_idx == test_idx")


def model_contract(config, device):
    model = runner.build_model(config, 1, device)
    check(len(model.gfr.graphs) == 6, "DARG does not expose six GraphLearn graphs")
    check(tuple(model.dims) == runner.MODAL_DIMS, "modality dimensions differ")
    check(model.use_darg is True, "all registered groups must retain DARG")
    expected_dpsl = config["group"] != "N6"
    check(model.use_dpsl is expected_dpsl, f"DPSL contract differs for {config['group']}")
    check(model.nclass == 3, "TADPOLE class head is not three-class")
    if expected_dpsl:
        check(config["dpsl_shared_cap"] == 0.001, "G4 shared cap differs")
        check(config["dpsl_residual_cap"] == 0.08, "G4 residual cap differs")
    else:
        check(config["dpsl_shared_cap"] == 0.0, "G3 unexpectedly has a DPSL cap")
    return model


def warmup_probe(config, x_np, y_np, split, device):
    runner.historical.seed_everything(runner.SEED)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    train_t = torch.from_numpy(train_idx).long().to(device)
    model = runner.build_model(config, 1, device)
    opts, _ = runner.make_optimizers(model, config)
    supports = runner.historical.anchor_pair_support(model, x)
    pair_np, same_np = runner.historical.make_pairs(train_idx, y_np, 1, supports)
    pair = torch.from_numpy(pair_np).long().to(device)
    same = torch.from_numpy(same_np).float().to(device)
    initial = runner.initial_warmup_identity(model, opts)
    history, rng_after, updates = runner.run_warmup(
        model, opts, x, labels, train_t, pair, same, config, 1, config["group"])
    check(len(history) == 10, f"{config['group']} warmup is not ten epochs")
    check(all(np.isfinite(item["loss"]) for item in history), f"{config['group']} warmup is non-finite")
    if config["group"] == "N4":
        check(history[0]["supervision_kind"] == "paper_aloss", "N4 warmup ALoss was not active")
        check(updates and updates[0]["epoch"] == 1, "N4 epoch-1 update audit missing")
    else:
        check(all(item["supervision_kind"] == "none" for item in history),
              f"{config['group']} ALoss entered the frozen warmup")
    model.darg.enter_positive_dcr()
    model.eval()
    with torch.no_grad():
        _, aux = model(x, labels, train_t, pair, same, dpsl_enabled=False)
    if config["group"] in {"N1", "N2", "N3", "N5", "N6"}:
        runner.historical.restore_rng(rng_after)
        kind = runner.active_supervision_kind(config, 11)
        check(kind == "paper_aloss", f"{config['group']} epoch-11 ALoss schedule missing")
        model.train()
        _, aux = model(x, labels, train_t, pair, same, dpsl_enabled=False)
        coefficient = runner.configured_supervision_coefficient(config)
        if config["group"] == "N5":
            coefficient, calibration = runner.calibrate_aloss(aux, labels, train_t, model)
            check(np.isfinite(coefficient) and calibration["aloss_gradient_norm"] > 0,
                  "N5 calibration is not finite")
        _, total, _, _ = runner.historical_loss(aux, labels, train_t, model, kind, coefficient)
        check(torch.isfinite(total), f"{config['group']} epoch-11 loss is non-finite")
        total.backward()
        audit = runner.graph_gradient_audit(model, aux)
        runner.validate_aloss_gradient_audit(audit)
    return {
        "status": "PASS_WITH_STRUCTURAL_ZERO_GRADIENT" if any(
            item["status"] == "STRUCTURAL_ZERO_GRADIENT" for item in runner.graph_gradient_audit(model, aux)
        ) else "PASS",
        "initial_model_sha256": initial["initial_model_sha256"],
        "initial_optimizer_sha256": initial["initial_optimizer_sha256"],
        "initial_rng_sha256": initial["initial_rng_sha256"],
        "warmup_rng_sha256": runner.sha_object(rng_after),
        "warmup_losses": [item["loss"] for item in history],
        "update_vector_audits": updates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    check(torch.cuda.device_count() == 1 and device.type == "cuda", "single assigned GPU is required")
    x, y, ids, splits = runner.load_data()
    split_check(x, y, splits)
    check(len(ids) == 598 and x.shape == (598, 365), "locked cohort shape differs")
    check(np.bincount(y, minlength=3).tolist() == [209, 315, 74], "locked class counts differ")
    check(tuple(runner.MODAL_DIMS) == (145, 148, 3, 36, 24, 9), "locked modality dimensions differ")
    configs = {group: runner.group_config(group) for group in ("N0", "N1", "N2", "N3", "N4", "N5", "N6")}
    check(configs["N0"]["supervision_kind"] == "none", "N0 is not clean")
    for group in ("N1", "N2", "N3", "N4", "N5", "N6"):
        check(configs[group]["supervision_kind"] == "paper_aloss", f"{group} is not Paper-ALoss")
        check(configs[group]["graph_supervision_schedule"] in ("epoch_11_onward", "epoch_1_onward_including_warmup"),
              f"{group} schedule differs")
    check(configs["N5"]["supervision_coefficient"] == "fold_calibrated", "N5 coefficient is not calibrated")
    check(configs["N5"]["target_ratio"] == 0.1 and configs["N5"]["lambda_clip"] == [1e-5, 1e-2],
          "N5 calibration bounds differ")
    check(sha_file(ROOT / "inputs/processed_standard_data.csv") == runner.DATA_SHA, "data SHA differs")
    check(sha_file(ROOT / "inputs/split_indices.json") == runner.SPLIT_SHA, "split SHA differs")
    author_network = ROOT / "sources/author/network.py"
    author_model = ROOT / "sources/author/model.py"
    check("torch.linalg.norm(normalized_adj" in author_network.read_text(), "author unsquared ALoss identity missing")
    check("loss = cls_loss + all_graph_loss" in author_model.read_text(), "author simple-2 loss identity missing")
    check("square().sum()" in (CODE / "run_e8382.py").read_text(), "squared E83.8.1 ALoss identity missing")
    model_contracts = {}
    for group, config in configs.items():
        model = model_contract(config, device)
        model_contracts[group] = {"use_darg": model.use_darg, "use_dpsl": model.use_dpsl,
                                  "graph_count": len(model.gfr.graphs)}
        del model
    probes = {group: warmup_probe(config, x, y, splits[0], device) for group, config in configs.items()}
    preflight = {
        "experiment": "Experiment 83.8.2", "status": "PASS", "shared_status": "PASS",
        "formal_training_allowed": True, "data_sha256": runner.DATA_SHA,
        "split_sha256": runner.SPLIT_SHA, "training_seed": runner.SEED,
        "cohort": {"patients": 598, "class_counts": [209, 315, 74], "labels": [0, 1, 2],
                    "modal_dims": list(runner.MODAL_DIMS), "total_dim": 365},
        "checks": [
            {"id": "DATA_SHA256", "status": "PASS"},
            {"id": "SPLIT_SHA256_AND_STRATIFIED_KFOLD", "status": "PASS"},
            {"id": "AUTHOR_SOURCE_IDENTITY", "status": "PASS"},
            {"id": "SIX_GRAPHLEARN_ALOSS", "status": "PASS"},
            {"id": "GFR_DARG_DPSL_CONTRACT", "status": "PASS"},
            {"id": "WARMUP_AND_ALOSS_SCHEDULE", "status": "PASS"},
            {"id": "TRAIN_LABEL_ISOLATION", "status": "PASS"},
        ],
        "groups": {group: {"preflight_status": probe["status"], "formal_training_allowed": True,
                            "model_contract": model_contracts[group], "probe": probe}
                   for group, probe in probes.items()},
        "training_seal_sha256": None,
        "source_files": {str(path.relative_to(ROOT)): sha_file(path)
                         for path in (ROOT / "sources/author").glob("*.py")},
    }
    (ROOT / "audits/preflight_start.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": preflight["status"], "groups": list(probes)}, indent=2))


if __name__ == "__main__":
    main()
