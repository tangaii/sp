"""Independent Best/Latest checkpoint replay and deterministic Latest recovery."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as runner


GROUPS = runner.FORMAL_GROUPS
NEW_GROUPS = runner.TRAIN_GROUPS


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def stable_state_sha(value):
    """Hash checkpoint state by tensor values rather than archive metadata."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda value: repr(value)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b"tuple\0" if isinstance(item, tuple) else b"list\0")
            for child in item:
                visit(child)
        elif isinstance(item, np.ndarray):
            digest.update(b"array\0")
            digest.update(str(item.dtype).encode())
            digest.update(repr(item.shape).encode())
            digest.update(item.tobytes())
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def restore_model(group, fold, payload, device):
    config = runner.group_config(group)
    model = runner.build_model(config, fold, device, spectral_stats=payload["spectral_stats"])
    model.load_state_dict(payload["model"], strict=True)
    if model.use_darg:
        model.darg.enter_positive_dcr()
    return config, model


def inference(group, fold, payload, x, labels, split, device, *, dpsl_enabled=None):
    config, model = restore_model(group, fold, payload, device)
    model.eval()
    train = torch.as_tensor(split["train_idx"], dtype=torch.long, device=device)
    pair = payload["relation_pairs"]
    same = payload["relation_targets"]
    if pair is not None:
        pair = pair.to(device)
        same = same.to(device)
    enabled = bool(config.get("dpsl_enabled", model.dpsl is not None))
    if dpsl_enabled is not None:
        enabled = dpsl_enabled
    with torch.no_grad():
        full, aux = model(x, labels, train, pair, same, dpsl_enabled=enabled)
        native, _ = model(x, labels, train, pair, same, dpsl_enabled=False)
        if model.dpsl is not None:
            residual_off, _ = model(
                x, labels, train, pair, same, dpsl_enabled=True,
                dpsl_intervention="residual_off")
        else:
            residual_off = native
    return config, model, full, native, residual_off, aux


def optimizer_step_replay(group, fold, payload, x, labels, split, device):
    config, model = restore_model(group, fold, payload, device)
    optimizers, optimizer_config = runner.make_optimizers(model, config, formal=True)
    if optimizer_config != payload["optimizer_config"]:
        raise RuntimeError(f"{group} fold {fold}: Latest optimizer configuration differs")
    for name, state in payload["optimizers"].items():
        optimizers[name].load_state_dict(copy.deepcopy(state))
    if optimizers.keys() != payload["optimizers"].keys():
        raise RuntimeError(f"{group} fold {fold}: Latest optimizer family differs")

    train = torch.as_tensor(split["train_idx"], dtype=torch.long, device=device)
    pair = payload["relation_pairs"]
    same = payload["relation_targets"]
    if pair is not None:
        pair = pair.to(device)
        same = same.to(device)
    epoch = int(payload["epoch"]) + 1
    kind = runner.active_supervision_kind(config, epoch)
    coefficient = runner.configured_supervision_coefficient(
        config, payload.get("lambda_fold"))
    dpsl_enabled = bool(config.get("dpsl_enabled", model.dpsl is not None))

    runner.historical.restore_rng(payload["rng"])
    model.train()
    runner.zero_optimizers(optimizers)
    _, aux = model(x, labels, train, pair, same, dpsl_enabled=dpsl_enabled)
    _, historical_total, _, _ = runner.historical_loss(
        aux, labels, train, model, kind, coefficient)
    total = historical_total + aux["dpsl_loss"]
    if not torch.isfinite(total):
        raise FloatingPointError(f"{group} fold {fold}: Latest recovery loss is non-finite")
    if model.dpsl is not None and dpsl_enabled:
        runner.dpsl_optimizer_step(
            model, optimizers, historical_total, aux["scmr_objectives"],
            float(config.get("dpsl_shared_cap", 0.001)))
    else:
        historical_total.backward()
        runner.step_optimizers(optimizers)

    model.eval()
    with torch.no_grad():
        _, eval_aux = model(x, labels, train, pair, same, dpsl_enabled=dpsl_enabled)
    if model.dpsl is not None and dpsl_enabled:
        model.dpsl.advance_ema(eval_aux["cam"])
    model_state = cpu_tree(model.state_dict())
    optimizer_state = cpu_tree({name: optimizer.state_dict()
                                for name, optimizer in optimizers.items()})
    rng_state = runner.historical.capture_rng_state()
    return {
        "model_sha256": stable_state_sha(model_state),
        "optimizer_sha256": stable_state_sha(optimizer_state),
        "rng_sha256": stable_state_sha(rng_state),
        "next_epoch": epoch,
        "loss": float(total.detach()),
    }


def replay_fold(group, fold, x, y, ids, split, device, *, next_step=True):
    folder = ROOT / "groups" / group
    best_path = folder / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest_path = folder / "latest_checkpoints" / f"fold{fold:02d}.pt"
    prediction_path = folder / "predictions" / f"fold{fold:02d}.csv"
    fold_audit_path = folder / "audits" / f"fold{fold:02d}.json"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    prediction = pd.read_csv(prediction_path)
    fold_audit = json.loads(fold_audit_path.read_text())
    expected_train = np.asarray(split["train_idx"], dtype=np.int64)
    expected_test = np.asarray(split["test_idx"], dtype=np.int64)
    if (best.get("data_sha256") != runner.DATA_SHA
            or best.get("split_sha256") != runner.SPLIT_SHA
            or latest.get("data_sha256") != runner.DATA_SHA
            or latest.get("split_sha256") != runner.SPLIT_SHA):
        raise RuntimeError(f"{group} fold {fold}: locked data identity mismatch")
    if (not np.array_equal(best.get("train_idx"), expected_train)
            or not np.array_equal(best.get("test_idx"), expected_test)
            or not np.array_equal(latest.get("train_idx"), expected_train)
            or not np.array_equal(latest.get("test_idx"), expected_test)):
        raise RuntimeError(f"{group} fold {fold}: fold indices mismatch")
    if int(latest["epoch"]) != int(fold_audit["latest_epoch"]):
        raise RuntimeError(f"{group} fold {fold}: Latest epoch metadata differs")
    if int(best["epoch"]) != int(fold_audit["best_epoch"]):
        raise RuntimeError(f"{group} fold {fold}: Best epoch metadata differs")

    train_set = set(expected_train.tolist())
    test_set = set(expected_test.tolist())
    pair_np = best["relation_pairs"].detach().cpu().numpy()
    target_np = best["relation_targets"].detach().cpu().numpy().astype(np.int64)
    if pair_np.shape != (4096, 2):
        raise RuntimeError(f"{group} fold {fold}: relation-pair count is not 4096")
    if any(int(row) not in train_set for row in pair_np.ravel()):
        raise RuntimeError(f"{group} fold {fold}: relation pair contains non-training patient")
    expected_targets = (y[pair_np[:, 0]] == y[pair_np[:, 1]]).astype(np.int64)
    if not np.array_equal(target_np, expected_targets):
        raise RuntimeError(f"{group} fold {fold}: relation targets differ from train labels")
    if not test_set.isdisjoint(pair_np.ravel().tolist()):
        raise RuntimeError(f"{group} fold {fold}: test patient occurs in a training relation pair")

    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.long, device=device)
    config, model, full, native, residual_off, _ = inference(
        group, fold, best, x_tensor, y_tensor, split, device)
    if not torch.isfinite(full).all() or not torch.isfinite(native).all():
        raise FloatingPointError(f"{group} fold {fold}: Best replay logits are non-finite")
    test = torch.as_tensor(expected_test, dtype=torch.long, device=device)
    full_np = full.exp()[test].detach().cpu().numpy()
    native_np = native.exp()[test].detach().cpu().numpy()
    predicted = full.argmax(1)[test].detach().cpu().numpy()
    truth = y[expected_test]
    stored = prediction.sort_values("sample_index")
    if not np.array_equal(stored["sample_index"].to_numpy(), expected_test):
        raise RuntimeError(f"{group} fold {fold}: prediction order/index identity mismatch")
    if not np.array_equal(stored["sample_id"].to_numpy(), ids.iloc[expected_test]["RID"].to_numpy()):
        raise RuntimeError(f"{group} fold {fold}: prediction RID identity mismatch")
    if not np.array_equal(stored["true_label"].to_numpy(), truth):
        raise RuntimeError(f"{group} fold {fold}: prediction ground truth identity mismatch")
    if not np.array_equal(stored["pred_label"].to_numpy(), predicted):
        raise RuntimeError(f"{group} fold {fold}: independent Best prediction mismatch")
    expected_probability = stored[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy()
    max_probability_error = float(np.max(np.abs(expected_probability - full_np)))
    if max_probability_error > 5e-6:
        raise RuntimeError(f"{group} fold {fold}: Best probability replay error {max_probability_error}")
    native_prediction = native.argmax(1)[test].detach().cpu().numpy()
    if "native_pred_label" in stored and not np.array_equal(
            stored["native_pred_label"].to_numpy(), native_prediction):
        raise RuntimeError(f"{group} fold {fold}: Native prediction replay mismatch")
    if not torch.equal(residual_off, native):
        raise RuntimeError(f"{group} fold {fold}: DPSL Residual-Off is not exactly Native")

    metrics = runner.acc_auc(truth, predicted, 3)
    probabilities = full_np
    probability_auc = float(roc_auc_score(
        np.eye(3, dtype=np.float64)[truth], probabilities,
        average="macro", multi_class="ovr"))
    reference_metrics = best.get("metrics_at_best", {})
    if abs(float(metrics["ACC"]) - float(reference_metrics["ACC"])) > 1e-12:
        raise RuntimeError(f"{group} fold {fold}: Best ACC differs from checkpoint record")
    if abs(float(metrics["AUC"]) - float(reference_metrics["AUC"])) > 1e-12:
        raise RuntimeError(f"{group} fold {fold}: Best AUC_original differs from checkpoint record")
    if abs(probability_auc - float(reference_metrics["AUC_probability"])) > 1e-8:
        raise RuntimeError(f"{group} fold {fold}: Best probability AUC differs from checkpoint record")

    _, latest_model = restore_model(group, fold, latest, device)
    latest_model.eval()
    latest_train = torch.as_tensor(expected_train, dtype=torch.long, device=device)
    latest_pair = latest["relation_pairs"]
    latest_same = latest["relation_targets"]
    if latest_pair is not None:
        latest_pair, latest_same = latest_pair.to(device), latest_same.to(device)
    with torch.no_grad():
        latest_logp, _ = latest_model(
            x_tensor, y_tensor, latest_train, latest_pair, latest_same,
            dpsl_enabled=bool(config.get("dpsl_enabled", latest_model.dpsl is not None)))
    if not torch.isfinite(latest_logp).all():
        raise FloatingPointError(f"{group} fold {fold}: Latest replay logits are non-finite")
    if abs(float(latest_logp.exp().sum(1).sub(1).abs().max())) > 1e-6:
        raise RuntimeError(f"{group} fold {fold}: Latest probabilities do not sum to one")

    latest_next = None
    if next_step:
        if group not in NEW_GROUPS:
            raise ValueError("Next-step recovery is registered only for newly trained groups")
        first = optimizer_step_replay(group, fold, latest, x_tensor, y_tensor, split, device)
        second = optimizer_step_replay(group, fold, latest, x_tensor, y_tensor, split, device)
        latest_next = {
            "status": "PASS" if first == second else "FAIL",
            "exactly_equal": first == second,
            "first_replay": first,
            "second_replay": second,
        }
    return {
        "experiment": runner.EXPERIMENT, "group": group, "fold": fold,
        "status": "PASS", "checkpoint_source": (
            "server historical checkpoint archive" if group.startswith("B") else "formal training"),
        "best_sha256": runner.sha_file(best_path),
        "latest_sha256": runner.sha_file(latest_path),
        "prediction_sha256": runner.sha_file(prediction_path),
        "fold_audit_sha256": runner.sha_file(fold_audit_path),
        "best_epoch": int(best["epoch"]), "latest_epoch": int(latest["epoch"]),
        "best_metrics": {
            "ACC": float(metrics["ACC"]), "AUC_original": float(metrics["AUC"]),
            "AUC_probability": probability_auc,
        },
        "best_prediction_exact": True,
        "best_probability_max_abs_error": max_probability_error,
        "latest_finite": True,
        "residual_off_exact_native": True,
        "relation_pairs_train_only": True,
        "relation_pair_count": int(len(pair_np)),
        "next_step_recovery": latest_next,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-next-step", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("Audit worker count must be between one and eight")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Checkpoint replay is pinned to the single assigned GPU")
    runner.configure_worker()
    x, y, ids, splits = runner.load_data()
    splits_by_id = {int(item["fold"]): item for item in splits}
    device = torch.device("cuda:0")
    records = []
    failures = []
    for group in args.groups:
        for fold in range(1, 11):
            try:
                records.append(replay_fold(
                    group, fold, x, y, ids, splits_by_id[fold], device,
                    next_step=(not args.skip_next_step and group in NEW_GROUPS)))
            except Exception as exc:
                failures.append({"group": group, "fold": fold, "error": repr(exc)})
                print(f"FAIL {group} fold {fold:02d}: {exc!r}", flush=True)
            else:
                print(f"PASS {group} fold {fold:02d}", flush=True)
    checkpoint_status = (
        "PASS" if not failures and len(records) == 10 * len(args.groups) else "FAIL")
    recovery_failures = [
        {"group": record["group"], "fold": record["fold"],
         "first_replay": record["next_step_recovery"]["first_replay"],
         "second_replay": record["next_step_recovery"]["second_replay"]}
        for record in records
        if record["next_step_recovery"] is not None
        and record["next_step_recovery"]["status"] != "PASS"
    ]
    recovery_expected = 10 * len(set(args.groups) & set(NEW_GROUPS))
    recovery_performed = sum(
        record["next_step_recovery"] is not None for record in records)
    recovery_status = (
        "NOT_RUN" if args.skip_next_step
        else "PASS" if not recovery_failures and recovery_performed == recovery_expected
        else "FAIL")
    summary = {
        "experiment": runner.EXPERIMENT,
        "status": "PASS" if checkpoint_status == "PASS" and recovery_status in ("PASS", "NOT_RUN") else "FAIL",
        "best_latest_replay_status": checkpoint_status,
        "latest_next_step_recovery_status": recovery_status,
        "groups_requested": args.groups,
        "folds_expected": 10 * len(args.groups),
        "folds_passed": len(records),
        "best_latest_folds_passed": len(records),
        "latest_next_step_expected": recovery_expected if not args.skip_next_step else 0,
        "latest_next_step_passed": sum(
            record["next_step_recovery"] is not None
            and record["next_step_recovery"]["status"] == "PASS" for record in records),
        "records": records,
        "failures": failures,
        "latest_next_step_recovery_failures": recovery_failures,
        "script_sha256": runner.sha_file(Path(__file__)),
        "sealed_training_implementation_sha256": json.loads(
            (ROOT / "audits/training_seal.json").read_text())["implementation_sha256"],
        "device": torch.cuda.get_device_name(0),
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
    }
    destination = ROOT / "audits/checkpoint_replay.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "status", "best_latest_replay_status", "latest_next_step_recovery_status",
        "folds_expected", "folds_passed", "latest_next_step_expected",
        "latest_next_step_passed")}, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
