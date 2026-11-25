"""Record per-fold, per-modality ALoss gradients at the registered first update."""

from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path

import torch

from audit_finalize_e8382 import (
    AUDITS, ROOT, build_after_warmup, model_inputs, per_modality_gradient_probe,
    runner, sha_file, summarize_gradients,
)


GROUPS = ("N1", "N2", "N3", "N4", "N5", "N6")


def check_summary(previous, calculated, location="aloss_gradients"):
    if isinstance(calculated, dict):
        for key, value in calculated.items():
            check_summary(previous[key], value, f"{location}.{key}")
    elif isinstance(calculated, list):
        for index, (before, after) in enumerate(zip(previous, calculated, strict=True)):
            check_summary(before, after, f"{location}[{index}]")
    elif isinstance(calculated, float):
        if not math.isclose(previous, calculated, rel_tol=1e-12, abs_tol=1e-20):
            raise RuntimeError(f"Gradient summary mismatch at {location}: {previous} vs {calculated}")
    elif previous != calculated:
        raise RuntimeError(f"Gradient summary mismatch at {location}: {previous} vs {calculated}")


def probe_fold(payload):
    group, fold, epoch, x_np, y_np, split = payload
    device = torch.device("cuda:0")
    directory = ROOT / "groups" / group
    path = directory / "latest_checkpoints" / f"fold{fold:02d}.pt"
    audit = json.loads((directory / "audits" / f"fold{fold:02d}.json").read_text())
    latest_sha = sha_file(path)
    if latest_sha != audit["latest_sha256"] or audit["status"] != "TRAINED":
        raise RuntimeError(f"Formal Latest identity mismatch for {group}/{fold}")
    latest = torch.load(path, map_location="cpu", weights_only=False)
    is_warmup = epoch == 1
    if is_warmup:
        config, x, labels, train, pair, same, model, optimizers, _ = model_inputs(
            group, fold, x_np, y_np, split, device)
        state = runner.initial_warmup_identity(model, optimizers)
        reference = latest["warmup"]["history"][0]
    else:
        config, x, labels, train, pair, same, model, _, _, rng, state, _ = build_after_warmup(
            group, fold, x_np, y_np, split, device)
        runner.historical.restore_rng(rng)
        reference = latest["history"][0]
    state_keys = (
        "initial_model_sha256", "initial_optimizer_sha256", "initial_rng_sha256",
    ) if is_warmup else (
        "shared_darg_cam_gfr_state_sha256", "shared_optimizer_state_sha256",
        "rng_state_sha256", "pair_sha256",
    )
    if any(state[key] != latest["warmup"][key] for key in state_keys):
        raise RuntimeError(f"Paired state identity mismatch for {group}/{fold}")

    model.train()
    _, aux = model(x, labels, train, pair, same, dpsl_enabled=not is_warmup, darg_warmup=is_warmup)
    kind = runner.active_supervision_kind(config, epoch, warmup=is_warmup)
    coefficient = runner.configured_supervision_coefficient(config, latest["lambda_fold"])
    gradient = per_modality_gradient_probe(aux, labels, train, model, coefficient)
    if group == "N5":
        computed_lambda, calibration = runner.calibrate_aloss(aux, labels, train, model)
        if computed_lambda != latest["lambda_fold"]:
            raise RuntimeError(f"Stored N5 coefficient differs for fold {fold}")
        gradient["calibration"] = calibration
    if any(m["status"] == "INVALID_ZERO_GRADIENT"
           or not math.isfinite(m["aloss_gradient_norm"]) for m in gradient["per_modality"]):
        raise RuntimeError(f"Invalid modality gradient for {group}/{fold}")
    if epoch != reference["epoch"] or kind != reference["supervision_kind"] or (
            coefficient != reference["supervision_coefficient"]
            or gradient["aloss_value"] != reference["graph_supervision"]):
        raise RuntimeError(f"ALoss probe does not match formal training for {group}/{fold}")

    perturbed_labels = labels.clone()
    test = latest["test_idx"]
    perturbed_labels[test] = (perturbed_labels[test] + 1) % 3
    with torch.no_grad():
        true_loss, true_detail = runner.paper_aloss(aux, labels, train, model)
        perturbed_loss, perturbed_detail = runner.paper_aloss(aux, perturbed_labels, train, model)
    if not torch.equal(true_loss, perturbed_loss) or not torch.equal(true_detail["target"], perturbed_detail["target"]):
        raise RuntimeError(f"Held-out labels affected the ALoss target in {group}/{fold}")
    return {
        "group": group, "fold": fold, "epoch": epoch, "latest_sha256": latest_sha,
        "matches_formal_aloss": True, "held_out_label_perturbation_invariant": True,
        **gradient,
    }


def main():
    seal_path = AUDITS / "training_seal.json"
    if json.loads(seal_path.read_text())["files"] != runner.current_implementation_hashes():
        raise RuntimeError("Training source seal does not match")
    x_np, y_np, _, splits = runner.load_data()
    tasks = [(group, int(split["fold"]), 11, x_np, y_np, split) for group in GROUPS for split in splits]
    tasks.extend(("N4", int(split["fold"]), 1, x_np, y_np, split) for split in splits)
    with runner.lock(AUDITS / "training.lock"):
        with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=runner.configure_worker) as pool:
            records = list(pool.map(probe_fold, tasks))

    path = AUDITS / "gradient_mechanism.json"
    payload = json.loads(path.read_text())
    epoch_11 = [record for record in records if record["epoch"] == 11]
    check_summary(payload["aloss_gradients"]["groups"], summarize_gradients(epoch_11)["groups"])
    for group in GROUPS:
        payload["aloss_gradients"]["groups"][group]["epoch"] = 11
    payload["aloss_gradients"]["definition"] = "Train-only GraphLearn p.weight gradients before the indicated update"
    payload["aloss_gradients"]["fold_checks"] = records
    epoch_1 = [record for record in records if record["epoch"] == 1]
    ratios = [record["actual_gradient_ratio"] for record in epoch_1]
    payload["aloss_gradients"]["n4_epoch_1"] = {
        "folds": len(epoch_1), "epoch": 1, "actual_gradient_ratio_mean": sum(ratios) / len(ratios),
        "actual_gradient_ratio_min": min(ratios), "actual_gradient_ratio_max": max(ratios),
    }
    payload["aloss_gradients"]["fold_checks_source"] = {
        "script": str(Path(__file__).resolve()), "sha256": sha_file(Path(__file__)),
        "training_seal_sha256": sha_file(seal_path), "status": "PASS",
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(f"PASS: {len(records)} ALoss probes matched their formal training records and gradient summary", flush=True)


if __name__ == "__main__":
    main()
