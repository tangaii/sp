"""Replay each Best and Latest checkpoint and verify deterministic continuation."""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
Z0_ROOT = ROOT.parent / "experiment83_8_4_tadpole_y3_darg_dpsl_factorial"
Z0_ARCHIVE = ROOT / "sources/reference/Z0"
Y3_ARCHIVE = ROOT / "sources/reference/Y3"
LEGACY = "--legacy-signatures" in sys.argv
CODE = (Z0_ROOT if LEGACY else ROOT) / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "common"))
if LEGACY:
    import run_e8384 as runner
else:
    import run_e8385 as runner
from metrics import acc_auc

Y3_ROOT = ROOT.parent / "experiment83_8_3_tadpole_n3_darg_dpsl_optimization"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite_optimizer(optimizers):
    for optimizer in optimizers.values():
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    check(torch.isfinite(value).all().item(), "Checkpoint optimizer contains non-finite state")


def build_loaded(group, fold, checkpoint, config, x, y, split):
    device = torch.device("cuda:0")
    runner.historical.seed_everything(runner.SEED)
    model = runner.build_model(config, fold, device, spectral_stats=checkpoint["spectral_stats"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.darg.enter_positive_dcr()
    optimizers, optimizer_config = runner.make_optimizers(model, config, formal=True)
    check(optimizer_config == checkpoint["optimizer_config"], f"{group} fold {fold} optimizer config differs")
    for name, state in checkpoint["optimizers"].items():
        # Adam keeps CPU step counters, which may alias the loaded checkpoint.
        optimizers[name].load_state_dict(copy.deepcopy(state))
    pair = checkpoint["relation_pairs"].to(device)
    same = checkpoint["relation_targets"].to(device)
    train_rows = torch.as_tensor(split["train_idx"], dtype=torch.long, device=device)
    x_tensor = torch.from_numpy(x).to(device)
    labels = torch.from_numpy(y).long().to(device)
    return model, optimizers, pair, same, train_rows, x_tensor, labels


def best_replay(group, fold, config, audit, best, frame, x, y, split):
    model, optimizers, pair, same, train_rows, x_tensor, labels = build_loaded(
        group, fold, best, config, x, y, split)
    model.eval()
    with torch.no_grad():
        logp, aux = model(x_tensor, labels, train_rows, pair, same, dpsl_enabled=True)
    test_rows = np.asarray(split["test_idx"], dtype=np.int64)
    probabilities = logp.exp().detach().cpu().numpy()[test_rows]
    prediction = logp.argmax(1).detach().cpu().numpy()[test_rows]
    check(np.array_equal(prediction, frame["pred_label"].to_numpy(dtype=np.int64)),
          f"{group} fold {fold} Best replay predictions differ")
    saved_probabilities = frame[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy(dtype=np.float64)
    max_probability_difference = float(np.max(np.abs(probabilities - saved_probabilities)))
    check(np.array_equal(probabilities, saved_probabilities.astype(probabilities.dtype)),
          f"{group} fold {fold} Best replay probabilities differ after CSV float32 round trip")
    with torch.no_grad():
        native = model(x_tensor, labels, train_rows, pair, same, dpsl_enabled=False)[0][test_rows]
    check(np.array_equal(native.argmax(1).cpu().numpy(), frame["native_pred_label"].to_numpy()),
          f"{group} fold {fold} Native predictions differ")
    native_saved = frame[["native_prob_NC", "native_prob_MCI", "native_prob_AD"]].to_numpy(dtype=np.float32)
    check(np.array_equal(native.exp().cpu().numpy(), native_saved), f"{group} fold {fold} Native probabilities differ")
    truth = y[test_rows]
    metrics = acc_auc(truth, prediction, 3)
    metrics["AUC_probability"] = float(runner.roc_auc_score(
        np.eye(3)[truth], probabilities, average="macro", multi_class="ovr"))
    check(metrics == best["metrics_at_best"]
          and metrics == {"ACC": audit["ACC"], "AUC": audit["AUC_original"],
                          "AUC_probability": audit["AUC_probability"]},
          f"{group} fold {fold} Best replay metrics differ")
    check(hashlib.sha256(prediction.astype(np.int64).tobytes()).hexdigest()
          == best["prediction_sha256"] == audit["prediction_sha256"],
          f"{group} fold {fold} Best prediction digest differs")
    check(int(best["epoch"]) == int(audit["best_epoch"]), f"{group} fold {fold} Best epoch differs")
    check(int(best["epoch"]) == int(best["metrics_at_best"].get("epoch", best["epoch"])),
          f"{group} fold {fold} Best metric epoch differs")
    finite_optimizer(optimizers)
    confidence = aux.get("dpsl", {}).get("confidence_gate")
    if config["dpsl_confidence_protection"]:
        check(confidence is not None, f"{group} fold {fold} confidence gate was not replayed")
        check(float(aux["dpsl"]["ratio"].max()) <= .08 + 1e-7,
              f"{group} fold {fold} replay residual exceeds the cap")
    return {"status": "PASS", "prediction_exact": True, "native_exact": True,
            "float32_probabilities_exact": True, "metrics": metrics,
            "probability_max_abs_difference": max_probability_difference,
            "best_epoch": int(best["epoch"]),
            "confidence_gate_range": ([float(confidence.min()), float(confidence.max())]
                                       if confidence is not None else None)}


def next_step_signature(group, fold, config, latest, x, y, split):
    model, optimizers, pair, same, train_rows, x_tensor, labels = build_loaded(
        group, fold, latest, config, x, y, split)
    runner.historical.restore_rng(latest["rng"])
    check(runner.sha_object(runner.historical.capture_rng_state()) == runner.sha_object(latest["rng"]),
          f"{group} fold {fold} RNG restore differs")
    model.train()
    runner.zero_optimizers(optimizers)
    next_epoch = int(latest["epoch"]) + 1
    logp, aux = model(x_tensor, labels, train_rows, pair, same, dpsl_enabled=True)
    graph_kind = runner.active_supervision_kind(config, next_epoch)
    coefficient = runner.configured_supervision_coefficient(config, latest.get("lambda_fold"))
    _, historical_total, _, _ = runner.historical_loss(
        aux, labels, train_rows, model, graph_kind, coefficient)
    total = historical_total + aux["dpsl_loss"]
    check(torch.isfinite(total).all().item(), f"{group} fold {fold} next-step loss is non-finite")
    if model.dpsl is not None:
        budget = runner.dpsl_optimizer_step(
            model, optimizers, historical_total, aux["scmr_objectives"],
            float(config["dpsl_shared_cap"]))
    else:
        historical_total.backward()
        runner.step_optimizers(optimizers)
        budget = {"families": {}, "shared_cap": 0.0}
    check(all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
          f"{group} fold {fold} next-step parameters are non-finite")
    finite_optimizer(optimizers)
    model.eval()
    with torch.no_grad():
        _, eval_aux = model(x_tensor, labels, train_rows, pair, same, dpsl_enabled=True)
    model.train()
    if model.dpsl is not None:
        model.dpsl.advance_ema(eval_aux["cam"])
    return {
        "model_sha256": runner.sha_object(model.state_dict()),
        "optimizer_sha256": runner.sha_object({key: value.state_dict() for key, value in optimizers.items()}),
        "rng_sha256": runner.sha_object(runner.historical.capture_rng_state()),
        "loss": float(total.detach()),
        "prediction_sha256": runner.sha_object(logp.detach()),
        "budget": budget,
    }


def replay_fold(group, fold, x, y, ids, split):
    group_dir = ROOT / "groups" / group
    config = runner.group_config(group)
    fold_name = f"fold{fold:02d}"
    audit_path = group_dir / "audits" / f"{fold_name}.json"
    pred_path = group_dir / "predictions" / f"{fold_name}.csv"
    audit = json.loads(audit_path.read_text())
    check(audit.get("status") == "TRAINED", f"{group} fold {fold} is incomplete")
    best_path = group_dir / "best_checkpoints" / f"{fold_name}.pt"
    latest_path = group_dir / "latest_checkpoints" / f"{fold_name}.pt"
    best_sha, latest_sha = runner.sha_file(best_path), runner.sha_file(latest_path)
    check(best_sha == audit["checkpoint_sha256"], f"{group} fold {fold} Best SHA differs")
    check(latest_sha == audit["latest_sha256"], f"{group} fold {fold} Latest SHA differs")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    check(latest["best_sha256"] == best_sha and latest["best_epoch"] == best["epoch"]
          and latest["epoch"] == audit["last_epoch"], f"{group} fold {fold} Latest linkage differs")
    history = latest["history"]
    check([item["epoch"] for item in history] == list(range(11, latest["epoch"] + 1)),
          f"{group} fold {fold} formal epoch history is incomplete")
    first_best = max(history, key=lambda row: row["test_acc"])
    check(first_best["epoch"] == best["epoch"] and first_best["test_acc"] == latest["best_acc"]
          and latest["wait"] == latest["epoch"] - best["epoch"]
          and (latest["epoch"] == runner.MAX_EPOCHS or latest["wait"] == runner.PATIENCE + 1),
          f"{group} fold {fold} Best selection or stopping rule differs")
    check(all(np.isfinite(row["loss"]) for row in history), f"{group} fold {fold} has a nonfinite epoch loss")
    check(set(latest["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
          and len(latest["rng"]["torch_cuda"]) == 1,
          f"{group} fold {fold} lacks complete RNG state")
    for checkpoint in (best, latest):
        check(checkpoint["train_idx"] == split["train_idx"]
              and checkpoint["test_idx"] == split["test_idx"], f"{group} fold {fold} split differs")
        pairs = checkpoint["relation_pairs"].numpy()
        check(pairs.shape == (4096, 2) and int(checkpoint["relation_targets"].sum()) == 2048
              and np.isin(pairs, split["train_idx"]).all() and np.array_equal(
            checkpoint["relation_targets"].numpy(), y[pairs[:, 0]] == y[pairs[:, 1]]),
            f"{group} fold {fold} relation supervision differs")
    if group == "R0":
        historical_group = Y3_ROOT / "groups/Y3"
        evidence_root = Z0_ARCHIVE if Z0_ARCHIVE.exists() else Z0_ROOT
        z0_evidence = json.loads((evidence_root / "audits/z0_y3_full_regression.json").read_text())
        archived = z0_evidence["folds"][fold - 1]
        for local_path in (best_path, latest_path, audit_path, pred_path):
            upstream_path = historical_group / local_path.parent.name / local_path.name
            z0_path = Z0_ROOT / "groups/Z0" / local_path.parent.name / local_path.name
            for reference in (upstream_path, z0_path):
                if reference.is_file():
                    check(runner.sha_file(local_path) == runner.sha_file(reference),
                          f"R0 fold {fold} Y3/Z0 identity differs for {local_path}")
        check(best_sha == archived["best_sha256"] and latest_sha == archived["latest_sha256"]
              and runner.sha_file(pred_path) == archived["prediction_sha256"],
              f"R0 fold {fold} differs from the historical hash manifest")
        y3_source = Y3_ARCHIVE if Y3_ARCHIVE.exists() else Y3_ROOT
        seal = json.loads((y3_source / "audits/training_seal.json").read_text())
        expected = {"implementation_sha256": seal["implementation_sha256"],
                    "group_config_sha256": runner.sha_file(y3_source / "configs/Y3.json"),
                    "group": "Y3", "experiment": "Experiment 83.8.3",
                    "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
                    "prompt_sha256": runner.sha_file(runner.OLD_PROMPT_SCMR)}
        check(expected["group_config_sha256"] == seal["files"]["configs/Y3.json"],
              f"Z0 fold {fold} historical Y3 config differs from its seal")
    else:
        expected = runner.identity(group, fold, split["train_idx"], split["test_idx"], config)
        check(set(best.get("rng", {})) == set(latest["rng"])
              and len(best["rng"]["torch_cuda"]) == 1, f"{group} fold {fold} Best RNG missing")
        r0 = torch.load(ROOT / "groups/R0/best_checkpoints" / f"{fold_name}.pt",
                        map_location="cpu", weights_only=False)
        check(torch.equal(latest["relation_pairs"], r0["relation_pairs"])
              and torch.equal(latest["relation_targets"], r0["relation_targets"]),
              f"{group} fold {fold} relation pairs differ from R0")
        warmup = latest["warmup_pair_audit"]
        registered = json.loads((ROOT / "audits/preflight_start.json").read_text())["warmup"][str(fold)]
        check(warmup["status"] == "PASS" and warmup["pre_epoch_11_state"] == registered
              and best["warmup_pair_audit"] == warmup and audit["warmup_pair_audit"] == warmup,
              f"{group} fold {fold} warmup audit differs")
    check(set(split["train_idx"]).isdisjoint(split["test_idx"])
          and len(set(split["train_idx"]) | set(split["test_idx"])) == 598,
          f"{group} fold {fold} outer partition differs")
    for name, value in expected.items():
        check(all(obj.get(name) == value for obj in (best, latest, audit)),
              f"{group} fold {fold} differs on checkpoint identity field {name}")
    frame = pd.read_csv(pred_path, float_precision="round_trip")
    rows = np.asarray(split["test_idx"], dtype=np.int64)
    check(np.array_equal(frame["sample_index"], rows) and np.array_equal(frame["true_label"], y[rows])
          and np.array_equal(frame["sample_id"], ids.iloc[rows]["RID"])
          and np.all(frame["fold"] == fold) and np.all(frame["seed"] == runner.SEED),
          f"{group} fold {fold} OOF patient identity differs")
    best_result = best_replay(group, fold, config, audit, best, frame, x, y, split)
    first = next_step_signature(group, fold, config, latest, x, y, split)
    second = next_step_signature(group, fold, config, latest, x, y, split)
    check(first == second, f"{group} fold {fold} Latest next-step continuation is not deterministic")
    latest_result = {"restore_status": "PASS", "next_step_deterministic": True,
                     "latest_epoch": int(latest["epoch"]), "next_step_loss": first["loss"],
                     **{f"next_step_{key}": first[key] for key in
                        ("model_sha256", "optimizer_sha256", "rng_sha256", "prediction_sha256")}}
    if group == "R0":
        old = json.loads((y3_source / "audits/checkpoint_replay.json").read_text())["groups"]["Y3"][str(fold)]
        check(all(latest_result[key] == old["latest"][key] for key in
                  ("next_step_model_sha256", "next_step_optimizer_sha256", "next_step_rng_sha256", "next_step_loss")),
              f"Z0 fold {fold} Latest next step differs from the sealed historical Y3 replay")
        latest_result["historical_next_step_exact"] = True
    return {"best": best_result, "latest": latest_result, "best_sha256": best_sha,
            "latest_sha256": latest_sha, "identity": expected,
            "prediction_csv_sha256": runner.sha_file(pred_path),
            "best_rng_present": "rng" in best, "first_strict_best_and_stop": "PASS",
            "warmup_exact": group == "R0" or warmup["status"] == "PASS"}


def baseline_signatures(path, group):
    runner.configure_worker()
    x, y, ids, splits = runner.load_data()
    signatures = {}
    for split in splits:
        fold = int(split["fold"])
        best = torch.load(path / f"fold{fold:02d}.pt", map_location="cpu", weights_only=False)
        config = runner.group_config(group)
        model, optimizers, pair, same, train_rows, x_tensor, labels = build_loaded(
            group, fold, best, config, x, y, split)
        model.eval()
        with torch.no_grad():
            logp, aux = model(x_tensor, labels, train_rows, pair, same)
            signatures[str(fold)] = {name: runner.sha_tensor(value) for name, value in {
                "logits": aux["logits"], "probability": logp.exp(), "prediction": logp.argmax(1),
                "native_logits": aux["native_logits"], "cam": aux["cam"],
                "enhanced_cam": aux["enhanced_cam"],
            }.items()}
            for key in ("adjacencies", "unified_adjacencies", "edge_rows", "edge_cols", "edge_q",
                        "pair_logits", "gates", "messages", "direct_messages"):
                signatures[str(fold)][key] = [runner.sha_tensor(value) for value in aux[key]]
        del logp, aux
        model.train()
        _, aux = model(x_tensor, labels, train_rows, pair, same)
        _, historical_total, _, _ = runner.historical_loss(
            aux, labels, train_rows, model, "paper_aloss", 0.01)
        loss = historical_total + aux["dpsl_loss"]
        loss.backward()
        signatures[str(fold)]["training_gradients"] = {
            name: runner.sha_tensor(parameter.grad) if parameter.grad is not None else None
            for name, parameter in model.named_parameters()
        }
        signatures[str(fold)]["training_loss"] = float(loss.detach())
        del aux, model, optimizers, loss, historical_total
    return signatures


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--r0", action="store_true")
    mode.add_argument("--legacy-signatures", action="store_true")
    parser.add_argument("--verify-one", nargs=2, metavar=("GROUP", "FOLD"))
    args = parser.parse_args()
    if args.legacy_signatures:
        signatures = baseline_signatures(Z0_ROOT / "groups/Z0/best_checkpoints", "Z0")
        print(json.dumps(signatures))
        return
    if args.r0:
        legacy = subprocess.check_output([sys.executable, "-B", __file__, "--legacy-signatures"],
                                         text=True)
        current = baseline_signatures(ROOT / "groups/R0/best_checkpoints", "R0")
        legacy = json.loads(legacy)
        for fold, signature in current.items():
            check(signature == legacy[fold],
                  f"R0 fold {fold} differs from Z0: "
                  f"{[key for key, value in signature.items() if value != legacy[fold][key]]}")
        runner.write_json(ROOT / "audits/r0_regression.json", {
            "status": "PASS_EXACT", "compared_folds": len(current), "signatures": current,
            "source": str(Z0_ROOT / "code/run_e8384.py"),
            "historical_source_sha256": runner.sha_file(Z0_ROOT / "code/run_e8384.py"),
        })
    else:
        runner.require_training_gate(list(runner.TRAIN_GROUPS))
    runner.configure_worker()
    x, y, ids, splits = runner.load_data()
    if args.verify_one:
        group, fold_text = args.verify_one
        fold = int(fold_text)
        split = next(row for row in splits if row["fold"] == fold)
        result = replay_fold(group, fold, x, y, ids, split)
        print(json.dumps({"group": group, "fold": fold, "status": result["best"]["status"]}))
        return
    results = {}
    for group in ("R0",) if args.r0 else runner.FORMAL_GROUPS:
        folds = {}
        for split in sorted(splits, key=lambda item: int(item["fold"])):
            fold = int(split["fold"])
            folds[str(fold)] = replay_fold(group, fold, x, y, ids, split)
            print(f"PASS {group} fold {fold:02d} Best/Latest", flush=True)
        results[group] = folds
        all_rows = np.concatenate([pd.read_csv(ROOT / "groups" / group / "predictions"
                                               / f"fold{fold:02d}.csv")["sample_index"]
                                   for fold in range(1, 11)])
        check(np.array_equal(np.sort(all_rows), np.arange(598)), f"{group} OOF coverage differs")
    report = {
        "experiment": runner.EXPERIMENT,
        "status": "PASS",
        "best_replayed": sum(item["best"]["status"] == "PASS" for group in results.values()
                              for item in group.values()),
        "latest_restored": sum(item["latest"]["restore_status"] == "PASS" for group in results.values()
                               for item in group.values()),
        "deterministic_next_steps": sum(item["latest"]["next_step_deterministic"] for group in results.values()
                                        for item in group.values()),
        "groups": results,
    }
    if args.r0:
        report["R0"] = "REUSED_VERIFIED_FORMAL"
    target = ROOT / "audits" / ("r0_identity.json" if args.r0 else "checkpoint_replay.json")
    runner.write_json(target, report)
    print(json.dumps({key: report[key] for key in
                      ("status", "best_replayed", "latest_restored", "deterministic_next_steps")}, indent=2))


if __name__ == "__main__":
    main()
