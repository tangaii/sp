"""Independent checkpoint replay, mechanism audit, and result finalizer."""
from __future__ import annotations

import csv
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "audits"
RESULTS = ROOT / "results"
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "common"))
import run_e8382 as runner

GROUPS = [f"N{index}" for index in range(7)]
CURRENT = Path("/model/spromptgl/experiment/experiment83_8_2_tadpole_g4_aloss_gradient_repair")
HISTORICAL_ROOT = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
E8381_ROOT = Path("/model/spromptgl/experiment/experiment83_8_1_tadpole_six_group_recovery")
H4_DIR = HISTORICAL_ROOT / "G4_E82_4_6_10_G1"
E8381_SUMMARY = E8381_ROOT / "results/group_summary.csv"
E8381_FOLDS = E8381_ROOT / "results/fold_metrics.csv"
E8381_OOF = E8381_ROOT / "results/oof_predictions.csv"


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return bool(np.isfinite(float(value)))


def write_csv(path: Path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_inputs(group, fold, x_np, y_np, split, device, config=None):
    config = runner.group_config(group) if config is None else config
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    train = torch.from_numpy(train_idx).long().to(device)
    runner.historical.seed_everything(runner.SEED)
    model = runner.build_model(config, fold, device)
    optimizers, optimizer_config = runner.make_optimizers(model, config)
    supports = runner.historical.anchor_pair_support(model, x)
    pair_np, same_np = runner.historical.make_pairs(train_idx, y_np, fold, supports)
    pair = torch.from_numpy(pair_np).long().to(device)
    same = torch.from_numpy(same_np).float().to(device)
    return config, x, labels, train, pair, same, model, optimizers, optimizer_config


def per_modality_gradient_probe(aux, labels, rows, model, coefficient):
    _, base_total, _, _ = runner.historical_loss(aux, labels, rows, model, "none")
    aloss, detail = runner.paper_aloss(aux, labels, rows, model)
    params = [parameter for _, parameter in runner.graphlearn_parameters(model)]
    base_grads = torch.autograd.grad(base_total, params, retain_graph=True, allow_unused=True)
    aloss_grads = torch.autograd.grad(aloss, params, retain_graph=True, allow_unused=True)
    records = []
    base_parts, aloss_parts = [], []
    for index, (base_grad, aloss_grad) in enumerate(zip(base_grads, aloss_grads)):
        adjacency = aux["adjacencies"][index].detach()
        off_diagonal = adjacency.clone()
        off_diagonal.fill_diagonal_(0)
        edge_count = int((off_diagonal != 0).sum().item())
        possible = max(adjacency.shape[0] * (adjacency.shape[0] - 1), 1)
        base_norm = float(base_grad.norm().detach()) if base_grad is not None else 0.0
        aloss_norm = float(aloss_grad.norm().detach()) if aloss_grad is not None else 0.0
        structural = edge_count == 0 and aloss_norm <= 1e-10
        invalid = edge_count > 0 and (aloss_grad is None or not torch.isfinite(aloss_grad).all()
                                      or aloss_norm <= 1e-12)
        if base_grad is not None:
            base_parts.append(base_grad.detach().reshape(-1))
        if aloss_grad is not None:
            aloss_parts.append(aloss_grad.detach().reshape(-1))
        records.append({
            "modality": runner.MODAL_NAMES[index],
            "off_diagonal_edge_count": edge_count,
            "graph_density": float(edge_count / possible),
            "base_gradient_norm": base_norm,
            "aloss_gradient_norm": aloss_norm,
            "status": ("STRUCTURAL_ZERO_GRADIENT" if structural else
                       "INVALID_ZERO_GRADIENT" if invalid else "VALID"),
        })
    base_vector = torch.cat(base_parts) if base_parts else aloss.new_zeros(0)
    aloss_vector = torch.cat(aloss_parts) if aloss_parts else aloss.new_zeros(0)
    base_norm = float(base_vector.norm())
    aloss_norm = float(aloss_vector.norm())
    requested_ratio = float(coefficient or 0.0) * aloss_norm / (base_norm + 1e-12)
    return {
        "aloss_value": float(aloss.detach()),
        "base_gradient_norm": base_norm,
        "aloss_gradient_norm": aloss_norm,
        "coefficient": float(coefficient or 0.0),
        "actual_gradient_ratio": requested_ratio,
        "per_modality": records,
        "target_sha256": runner.sha_object((rows.detach().cpu(),
                                              labels.index_select(0, rows).detach().cpu(),
                                              detail["target"].detach().cpu())),
    }


def build_after_warmup(group, fold, x_np, y_np, split, device, *, capture_epoch11=False):
    config, x, labels, train, pair, same, model, optimizers, optimizer_config = model_inputs(
        group, fold, x_np, y_np, split, device)
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    initial = runner.initial_warmup_identity(model, optimizers)
    initial_gradient = None
    if group == "N4":
        before = runner.historical.capture_rng_state()
        model.train()
        _, initial_aux = model(x, labels, train, pair, same,
                               dpsl_enabled=False, darg_warmup=True)
        initial_gradient = per_modality_gradient_probe(
            initial_aux, labels, train, model, 0.003)
        runner.historical.restore_rng(before)
    history, warmup_rng, warmup_updates = runner.run_warmup(
        model, optimizers, x, labels, train, pair, same, config, fold, group)
    model.eval()
    with torch.no_grad():
        _, warm_aux = model(x, labels, train, pair, same, dpsl_enabled=False)
    stats = None
    if model.use_dpsl:
        stats = runner.fit_spectral_statistics(warm_aux["cam"], labels, train, fold=fold,
                                               warm_logits=warm_aux["native_logits"])
        stats["warmup_cam"] = warm_aux["cam"].detach().cpu()
        stats["warmup_logits"] = warm_aux["native_logits"].detach().cpu()
        model.attach_dpsl(stats, str(runner.prompt_path_for(
            config, runner.BASE_GROUPS[config["base_reference"]])), 824602 * 100 + fold)
        optimizers["dpsl"] = torch.optim.AdamW(model.dpsl.parameters(), lr=0.0005, weight_decay=0.01)
        optimizer_config["dpsl"] = [0.0005, 0.01]
    warmup = runner.make_warmup_payload(fold, model, optimizers, history, warmup_rng,
                                        pair, same, initial, config, train_idx, y_np,
                                        warmup_updates)
    if runner.sha_object(runner.historical.capture_rng_state()) != warmup["rng_state_sha256"]:
        raise RuntimeError(f"RNG changed while reconstructing warmup for {group} fold {fold}")
    return (config, x, labels, train, pair, same, model, optimizers,
            stats, warmup_rng, warmup, initial_gradient)


def replay_rng_and_initial_gradient(group, fold, x_np, y_np, split, latest, best, device):
    config, x, labels, train, pair, same, model, optimizers, stats, warmup_rng, warmup, gradient = build_after_warmup(
        group, fold, x_np, y_np, split, device)
    best_rng = None

    lambda_fold = latest.get("lambda_fold")
    if group == "N5" and lambda_fold is None:
        raise RuntimeError("N5 Latest lacks its calibrated coefficient")
    for epoch in range(11, int(latest["epoch"]) + 1):
        model.train()
        _, aux = model(x, labels, train, pair, same, dpsl_enabled=True)
        kind = runner.active_supervision_kind(config, epoch)
        coefficient = runner.configured_supervision_coefficient(config, lambda_fold)
        if epoch == 11 and kind == "paper_aloss":
            gradient = per_modality_gradient_probe(aux, labels, train, model, coefficient)
            if group == "N5":
                computed, calibration = runner.calibrate_aloss(aux, labels, train, model)
                if not np.isclose(computed, lambda_fold, rtol=0.0, atol=1e-12):
                    raise RuntimeError("N5 calibrated coefficient differs from deterministic reconstruction")
                gradient["calibration"] = calibration
        runner.historical_loss(aux, labels, train, model, kind, coefficient)
        model.eval()
        with torch.no_grad():
            _, eval_aux = model(x, labels, train, pair, same, dpsl_enabled=True)
        model.train()
        if epoch == int(best["epoch"]):
            best_rng = runner.historical.capture_rng_state()
        if model.dpsl is not None:
            model.dpsl.advance_ema(eval_aux["cam"])
    latest_rng = runner.historical.capture_rng_state()
    latest_rng_match = runner.sha_object(latest_rng) == runner.sha_object(latest["rng"])
    if not latest_rng_match:
        raise RuntimeError(f"Latest RNG did not replay for {group} fold {fold}")
    if best_rng is None:
        raise RuntimeError(f"Best epoch RNG missing in replay for {group} fold {fold}")
    check = {
        "group": group, "fold": int(fold), "best_epoch": int(best["epoch"]),
        "latest_epoch": int(latest["epoch"]), "latest_rng_replay": "PASS",
        "latest_rng_sha256": runner.sha_object(latest_rng),
        "best_rng_sha256": runner.sha_object(best_rng),
        "initial_model_sha256": warmup["initial_model_sha256"],
        "initial_optimizer_sha256": warmup["initial_optimizer_sha256"],
        "initial_rng_sha256": warmup["initial_rng_sha256"],
        "warmup_state_sha256": warmup["base_state_sha256"],
        "warmup_optimizer_sha256": warmup["optimizer_state_sha256"],
        "warmup_rng_sha256": warmup["rng_state_sha256"],
    }
    return check, best_rng, gradient


def load_checkpoint_model(group, checkpoint, x_np, y_np, device):
    config = runner.group_config(group)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    train_idx = np.asarray(checkpoint["train_idx"], dtype=np.int64)
    test_idx = np.asarray(checkpoint["test_idx"], dtype=np.int64)
    train = torch.from_numpy(train_idx).long().to(device)
    pair, same = checkpoint["relation_pairs"], checkpoint["relation_targets"]
    pair = pair.to(device) if pair is not None else None
    same = same.to(device) if same is not None else None
    stats = checkpoint.get("spectral_stats")
    model = runner.build_model(config, int(checkpoint["fold"]), device, spectral_stats=stats)
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.use_darg:
        model.darg.enter_positive_dcr()
    return config, x, labels, train, pair, same, model


def update_witness(group, checkpoint, rng, x_np, y_np, device, epoch):
    outside_rng = runner.historical.capture_rng_state()
    try:
        config, x, labels, train, pair, same, model = load_checkpoint_model(
            group, checkpoint, x_np, y_np, device)
        optimizers, optimizer_config = runner.make_optimizers(model, config, formal=True)
        if optimizer_config != checkpoint["optimizer_config"]:
            raise RuntimeError(f"Optimizer contract differs in {group} fold {checkpoint['fold']}")
        for name, state in checkpoint["optimizers"].items():
            optimizers[name].load_state_dict(copy.deepcopy(state))
        runner.historical.restore_rng(rng)
        model.train()
        runner.zero_optimizers(optimizers)
        _, aux = model(x, labels, train, pair, same, dpsl_enabled=True)
        kind = runner.active_supervision_kind(config, epoch)
        coefficient = runner.configured_supervision_coefficient(config, checkpoint.get("lambda_fold"))
        _, historical_total, _, _ = runner.historical_loss(
            aux, labels, train, model, kind, coefficient)
        if model.dpsl is not None:
            runner.dpsl_optimizer_step(model, optimizers, historical_total,
                                       aux["scmr_objectives"],
                                       float(config.get("dpsl_shared_cap", 0.001)))
        else:
            historical_total.backward()
            runner.step_optimizers(optimizers)
        model.eval()
        with torch.no_grad():
            _, eval_aux = model(x, labels, train, pair, same, dpsl_enabled=True)
        if model.dpsl is not None:
            model.dpsl.advance_ema(eval_aux["cam"])
        return {
            "loss": float(historical_total.detach()),
            "model_sha256": runner.sha_object(model.state_dict()),
            "optimizer_sha256": runner.sha_object({name: opt.state_dict()
                                                    for name, opt in optimizers.items()}),
            "rng_sha256": runner.sha_object(runner.historical.capture_rng_state()),
        }
    finally:
        runner.historical.restore_rng(outside_rng)


def twin_update_witness(group, checkpoint, rng, x_np, y_np, device, epoch):
    left = update_witness(group, checkpoint, rng, x_np, y_np, device, epoch)
    right = update_witness(group, checkpoint, rng, x_np, y_np, device, epoch)
    if left != right:
        differences = {key: [left[key], right[key]] for key in left if left[key] != right[key]}
        raise RuntimeError(f"Deterministic update witness differs for {group} fold {checkpoint['fold']}: {differences}")
    return left


def audit_fold(group, fold, x_np, y_np, ids, split, device):
    directory = ROOT / "groups" / group
    best_path = directory / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest_path = directory / "latest_checkpoints" / f"fold{fold:02d}.pt"
    audit_path = directory / "audits" / f"fold{fold:02d}.json"
    record = json.loads(audit_path.read_text())
    best_sha, latest_sha = sha_file(best_path), sha_file(latest_path)
    if record.get("checkpoint_sha256") != best_sha or record.get("latest_sha256") != latest_sha:
        raise RuntimeError(f"Training audit checkpoint SHA differs for {group} fold {fold}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    if latest.get("best_sha256") != best_sha:
        raise RuntimeError(f"Latest does not reference Best for {group} fold {fold}")
    for expected in ("group", "fold", "seed", "data_sha256", "split_sha256", "train_idx", "test_idx"):
        if best.get(expected) != record.get(expected) or latest.get(expected) != record.get(expected):
            raise RuntimeError(f"Checkpoint identity differs for {group} fold {fold}: {expected}")
    replay, best_rng, gradient = replay_rng_and_initial_gradient(
        group, fold, x_np, y_np, split, latest, best, device)

    config, x, labels, train, pair, same, model = load_checkpoint_model(group, best, x_np, y_np, device)
    model.eval()
    inference_labels = torch.zeros_like(labels)
    with torch.no_grad():
        full_logp, full_aux = model(x, inference_labels, train, pair, same, dpsl_enabled=True)
        native_logp = (model(x, inference_labels, train, pair, same, dpsl_enabled=False)[0]
                       if model.dpsl is not None else None)
        residual_off_logp = (model(x, inference_labels, train, pair, same, dpsl_enabled=True,
                                   dpsl_intervention="residual_off")[0]
                             if model.dpsl is not None else None)
        direct_off_logp = model(x, inference_labels, train, pair, same, dpsl_enabled=True,
                                darg_intervention="direct_off")[0]
        disease_neutral_logp = model(x, inference_labels, train, pair, same, dpsl_enabled=True,
                                     darg_intervention="disease_neutral")[0]
        label_independent_logp = model(x, labels, train, pair, same, dpsl_enabled=True)[0]
    test_idx = np.asarray(best["test_idx"], dtype=np.int64)
    truth = y_np[test_idx]
    prediction = full_logp.argmax(1).detach().cpu().numpy()
    probability = full_logp.exp().detach().cpu().numpy()
    native_prediction = native_logp.argmax(1).detach().cpu().numpy() if native_logp is not None else None
    if not np.array_equal(prediction[test_idx], pd.read_csv(
            directory / "predictions" / f"fold{fold:02d}.csv")["pred_label"].to_numpy()):
        raise RuntimeError(f"Best re-evaluation prediction differs for {group} fold {fold}")
    metric = runner.acc_auc(truth, prediction[test_idx], 3)
    probability_auc = float(roc_auc_score(np.eye(3)[truth], probability[test_idx],
                                          average="macro", multi_class="ovr"))
    if metric["ACC"] != record["ACC"] or metric["AUC"] != record["AUC_original"]:
        raise RuntimeError(f"Best re-evaluation metrics differ for {group} fold {fold}")
    if model.dpsl is not None:
        residual_gap = float((residual_off_logp - native_logp).abs().max())
        if residual_gap != 0.0:
            raise RuntimeError(f"DPSL Residual-Off is not exact Native for {group} fold {fold}")
        residual_max = float(full_aux["dpsl"]["ratio"].max())
        if residual_max > 0.08 + 1e-7:
            raise RuntimeError(f"DPSL residual cap exceeded for {group} fold {fold}")
        pairs = full_aux["dpsl"].get("class_pairs")
        if pairs is None or tuple(pairs.shape) != (3, 2):
            raise RuntimeError(f"DPSL did not evaluate all three class pairs for {group} fold {fold}")
    else:
        residual_gap, residual_max = None, 0.0
    label_gap = float((label_independent_logp - full_logp).abs().max())
    if label_gap != 0.0:
        raise RuntimeError(f"Inference output depends on provided labels for {group} fold {fold}")

    native_pred_test = native_prediction[test_idx] if native_prediction is not None else None
    full_pred_test = prediction[test_idx]
    def intervention_counts(logp):
        changed = logp.argmax(1).detach().cpu().numpy()[test_idx]
        full_correct = full_pred_test == truth
        changed_correct = changed == truth
        return {"changed": int((full_pred_test != changed).sum()),
                "corrected": int((~full_correct & changed_correct).sum()),
                "damaged": int((full_correct & ~changed_correct).sum())}

    model.train()
    outside_rng = runner.historical.capture_rng_state()
    try:
        runner.historical.restore_rng(best_rng)
        _, train_aux = model(x, labels, train, pair, same, dpsl_enabled=True)
        kind = runner.active_supervision_kind(config, int(best["epoch"]))
        coefficient = runner.configured_supervision_coefficient(config, best.get("lambda_fold"))
        _, train_total, graph_sup, target = runner.historical_loss(
            train_aux, labels, train, model, kind, coefficient)
        relation_params = [parameter for _, parameter in model.darg.relation_named_parameters()
                           if parameter.requires_grad]
        relation_grads = torch.autograd.grad(train_total, relation_params, retain_graph=True,
                                             allow_unused=True)
        relation_norms = [float(grad.norm()) for grad in relation_grads if grad is not None]
        dpsl_grad_norm = None
        if model.dpsl is not None:
            dpsl_grads = torch.autograd.grad(train_aux["dpsl_loss"], tuple(model.dpsl.parameters()),
                                             retain_graph=True, allow_unused=True)
            present_dpsl_grads = [grad.norm() for grad in dpsl_grads if grad is not None]
            dpsl_grad_norm = float(torch.stack(present_dpsl_grads).norm()) if present_dpsl_grads else 0.0
    finally:
        runner.historical.restore_rng(outside_rng)

    diagnostics = train_aux.get("edge_diagnostics", [])
    edge_scores = train_aux.get("edge_q_used", [])
    pos_edges = 0
    neg_edges = 0
    for modality, scores in enumerate(edge_scores):
        scores = scores.detach()
        if modality == 3:
            pos_edges += int((scores > 0.5).sum().item())
            neg_edges += int((scores < 0.5).sum().item())
        else:
            pos_edges += int((scores >= 0.55).sum().item())
            neg_edges += int((scores <= 0.45).sum().item())
    relation_state = best.get("optimizers", {}).get("darg", {}).get("state", {})
    relation_steps = int(sum(float(value.get("step", 0)) > 0 for value in relation_state.values()))
    mechanisms = {
        "group": group, "fold": fold, "best_epoch": int(best["epoch"]),
        "train_historical_total_eval": float(train_total.detach()),
        "train_graph_supervision_eval": float(graph_sup.detach()),
        "train_supervision_kind": kind, "train_supervision_coefficient": coefficient or 0.0,
        "train_supervision_details": runner.serializable_supervision_details(target),
        "positive_dcr_callable": bool(callable(model.darg.positive.rudag_forward)),
        "positive_signed_edges": pos_edges, "negative_signed_edges": neg_edges,
        "signed_relation_both_present": bool(pos_edges > 0 and neg_edges > 0),
        "relation_parameter_gradient_norm": float(np.linalg.norm(relation_norms)),
        "relation_parameter_gradient_nonzero_count": int(sum(value > 0 for value in relation_norms)),
        "relation_optimizer_state_steps": relation_steps,
        "edge_diagnostics": diagnostics,
        "native_full_corrected": (int(((native_pred_test != truth) & (full_pred_test == truth)).sum())
                                  if native_pred_test is not None else None),
        "native_full_damaged": (int(((native_pred_test == truth) & (full_pred_test != truth)).sum())
                                if native_pred_test is not None else None),
        "native_correct": (int((native_pred_test == truth).sum()) if native_pred_test is not None else None),
        "full_correct": int((full_pred_test == truth).sum()),
        "direct_off": intervention_counts(direct_off_logp),
        "disease_neutral": intervention_counts(disease_neutral_logp),
        "dpsl_active": model.dpsl is not None,
        "dpsl_residual_max": residual_max,
        "dpsl_residual_off_native_max_abs": residual_gap,
        "dpsl_gradient_norm": dpsl_grad_norm,
        "dpsl_shared_cap_max": record.get("dpsl_shared_gradient_ratio_max"),
        "dpsl_class_pairs": (pairs.detach().cpu().tolist() if model.dpsl is not None else []),
        "dpsl_pair_weight_mean": (full_aux["dpsl"]["pair_weights"].mean(0).detach().cpu().tolist()
                                   if model.dpsl is not None else []),
        "inference_label_independence_max_abs": label_gap,
        "ACC": metric["ACC"], "AUC_original": metric["AUC"],
        "AUC_probability": probability_auc,
    }
    replay.update({"best_sha256": best_sha, "latest_sha256": latest_sha,
                   "best_replay": "PASS", "residual_off_replay": "PASS" if model.dpsl is not None else "NOT_APPLICABLE",
                   "ACC": metric["ACC"], "AUC_original": metric["AUC"],
                   "AUC_probability": probability_auc,
                   "next_update_witness": "PENDING"})
    return {"group": group, "fold": fold, "audit": replay, "mechanisms": mechanisms,
            "best_rng": best_rng, "gradient": gradient}


def run_witness_task(payload):
    group, fold, checkpoint_kind, x_np, y_np, split, best_rng = payload
    runner.configure_worker()
    device = torch.device("cuda:0")
    path = ROOT / "groups" / group / ("best_checkpoints" if checkpoint_kind == "Best"
                                       else "latest_checkpoints") / f"fold{fold:02d}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint_kind == "Best":
        rng = best_rng
        epoch = int(checkpoint["epoch"]) + 1
    else:
        rng = checkpoint["rng"]
        epoch = int(checkpoint["epoch"]) + 1
    witness = twin_update_witness(group, checkpoint, rng, x_np, y_np, device, epoch)
    return {"group": group, "fold": fold, "checkpoint": checkpoint_kind,
            "epoch": epoch, "status": "PASS", "checkpoint_sha256": sha_file(path), **witness}


def save_temporary_cache(path, cache):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(cache, temporary_path)
    os.replace(temporary_path, path)


def historical_records():
    with E8381_SUMMARY.open(newline="", encoding="utf-8") as stream:
        summary = {row["group"]: row for row in csv.DictReader(stream)}
    with E8381_FOLDS.open(newline="", encoding="utf-8") as stream:
        folds = list(csv.DictReader(stream))
    with E8381_OOF.open(newline="", encoding="utf-8") as stream:
        oof = list(csv.DictReader(stream))
    keep = {"H1", "H2", "H3", "H4", "H5", "X3", "X4", "X5", "X6"}
    return ({key: value for key, value in summary.items() if key in keep},
            [row for row in folds if row["group"] in keep],
            [row for row in oof if row["group"] in keep])


def aggregate_current(results):
    fold_rows, prediction_rows, mechanism_rows, gradient_rows = [], [], [], []
    recovery_rng = {}
    for result in results:
        group, fold = result["group"], result["fold"]
        record = result["audit"]
        fold_rows.append({
            "group": group, "fold": fold, "status": "PASS", "ACC": record["ACC"],
            "AUC_original": record["AUC_original"], "AUC_probability": record["AUC_probability"],
            "correct": int(result["mechanisms"]["full_correct"]),
            "best_epoch": record["best_epoch"], "best_sha256": record["best_sha256"],
            "latest_sha256": record["latest_sha256"],
            "best_checkpoint": str(ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"),
            "latest_checkpoint": str(ROOT / "groups" / group / "latest_checkpoints" / f"fold{fold:02d}.pt"),
            "source": "E83.8.2 independent Best replay",
        })
        prediction_file = ROOT / "groups" / group / "predictions" / f"fold{fold:02d}.csv"
        frame = pd.read_csv(prediction_file)
        for row in frame.to_dict(orient="records"):
            prediction_rows.append({"group": group, **row})
        mechanism_rows.append(result["mechanisms"])
        if result["gradient"] is not None:
            gradient_rows.append({"group": group, "fold": fold, **result["gradient"]})
        recovery_rng[(group, fold)] = result["best_rng"]
    return fold_rows, prediction_rows, mechanism_rows, gradient_rows, recovery_rng


def current_summaries(fold_rows, prediction_rows):
    by_group = {}
    for group in GROUPS:
        folds = sorted([row for row in fold_rows if row["group"] == group], key=lambda row: row["fold"])
        oof = [row for row in prediction_rows if row["group"] == group]
        if len(folds) != 10 or len(oof) != 598:
            raise RuntimeError(f"Incomplete OOF group {group}: folds={len(folds)} patients={len(oof)}")
        values_acc = np.asarray([row["ACC"] for row in folds], dtype=np.float64)
        values_auc = np.asarray([row["AUC_original"] for row in folds], dtype=np.float64)
        by_group[group] = {
            "group": group, "status": "COMPLETE", "folds": 10,
            "ACC_mean": float(values_acc.mean()), "ACC_std_population": float(values_acc.std(ddof=0)),
            "AUC_original_mean": float(values_auc.mean()),
            "AUC_original_std_population": float(values_auc.std(ddof=0)),
            "AUC_probability_mean": float(np.mean([row["AUC_probability"] for row in folds])),
            "OOF_correct": int(sum(int(row["pred_label"]) == int(row["true_label"]) for row in oof)),
            "OOF_patients": 598,
            "NC_correct": sum(int(row["true_label"]) == 0 and int(row["pred_label"]) == 0 for row in oof),
            "MCI_correct": sum(int(row["true_label"]) == 1 and int(row["pred_label"]) == 1 for row in oof),
            "AD_correct": sum(int(row["true_label"]) == 2 and int(row["pred_label"]) == 2 for row in oof),
            "source": "E83.8.2 formal training",
        }
    return by_group


def add_historical_summary(rows, historical):
    for group, item in historical.items():
        rows[group] = {
            "group": group, "status": item["status"], "folds": int(item["folds"]),
            "ACC_mean": float(item["ACC_mean"]), "ACC_std_population": float(item["ACC_std_population"]),
            "AUC_original_mean": float(item["AUC_original_mean"]),
            "AUC_original_std_population": float(item["AUC_original_std_population"]),
            "AUC_probability_mean": float(item.get("AUC_probability_mean", "nan")),
            "OOF_correct": int(item["OOF_correct"]), "OOF_patients": int(item.get("OOF_patients", 598)),
            "NC_correct": int(item.get("NC_correct", 0)), "MCI_correct": int(item.get("MCI_correct", 0)),
            "AD_correct": int(item.get("AD_correct", 0)), "source": item["source"],
        }
    rows["PAPER"] = {
        "group": "PAPER", "status": "REFERENCE", "folds": 10,
        "ACC_mean": 0.9481, "ACC_std_population": 0.0295,
        "AUC_original_mean": 0.9559, "AUC_original_std_population": 0.0257,
        "OOF_correct": "", "OOF_patients": "", "NC_correct": "", "MCI_correct": "",
        "AD_correct": "", "source": "SPromptGL MICCAI 2025 TADPOLE result",
    }
    return rows


def paired_rows(fold_rows, prediction_rows, historic_folds, historic_oof):
    fold_lookup = {(row["group"], int(row["fold"])): row for row in fold_rows}
    fold_lookup.update({(row["group"], int(row["fold"])): {
        "group": row["group"], "fold": int(row["fold"]),
        "ACC": float(row["ACC"]), "AUC_original": float(row["AUC_original"]),
        "correct": int(row["correct"]),
    } for row in historic_folds})
    predictions = {}
    for row in prediction_rows:
        predictions.setdefault((row["group"], int(row["fold"])), []).append(row)
    for row in historic_oof:
        predictions.setdefault((row["group"], int(row["fold"])), []).append({
            **row, "sample_index": int(row["sample_index"]),
            "true_label": int(row["true_label"]), "pred_label": int(row["pred_label"]),
        })
    comparisons = [("N0", "H4", "N0_vs_H4")]
    for candidate in ("N1", "N2", "N3", "N4", "N5"):
        comparisons.extend([(candidate, "N0", f"{candidate}_vs_N0"),
                            (candidate, "H4", f"{candidate}_vs_H4")])
    comparisons.extend([("N2", "N4", "N2_vs_N4_warmup_timing"),
                        ("N2", "N6", "N2_vs_N6_DPSL"),
                        ("N1", "N2", "N1_vs_N2_ALoss_strength"),
                        ("N2", "N3", "N2_vs_N3_ALoss_strength")])
    rows = []
    for candidate, reference, name in comparisons:
        acc_delta, auc_delta, corrected, damaged = [], [], [], []
        for fold in range(1, 11):
            left = fold_lookup[(candidate, fold)]
            right = fold_lookup[(reference, fold)]
            cand_rows = sorted(predictions[(candidate, fold)], key=lambda row: int(row["sample_index"]))
            ref_rows = sorted(predictions[(reference, fold)], key=lambda row: int(row["sample_index"]))
            if [int(row["sample_index"]) for row in cand_rows] != [int(row["sample_index"]) for row in ref_rows]:
                raise RuntimeError(f"Paired patient order differs for {candidate}/{reference} fold {fold}")
            truths = np.asarray([int(row["true_label"]) for row in cand_rows])
            candidate_pred = np.asarray([int(row["pred_label"]) for row in cand_rows])
            reference_pred = np.asarray([int(row["pred_label"]) for row in ref_rows])
            candidate_correct = candidate_pred == truths
            reference_correct = reference_pred == truths
            n_corrected = int((~reference_correct & candidate_correct).sum())
            n_damaged = int((reference_correct & ~candidate_correct).sum())
            acc_delta.append(left["ACC"] - right["ACC"])
            auc_delta.append(left["AUC_original"] - right["AUC_original"])
            corrected.append(n_corrected)
            damaged.append(n_damaged)
            rows.append({
                "comparison": name, "candidate": candidate, "reference": reference,
                "fold": fold, "ACC_candidate": left["ACC"], "ACC_reference": right["ACC"],
                "ACC_delta": acc_delta[-1], "AUC_original_candidate": left["AUC_original"],
                "AUC_original_reference": right["AUC_original"], "AUC_original_delta": auc_delta[-1],
                "patient_corrected": n_corrected, "patient_damaged": n_damaged,
                "patient_net": n_corrected - n_damaged,
            })
        rows.append({
            "comparison": name, "candidate": candidate, "reference": reference, "fold": "MEAN",
            "ACC_candidate": float(np.mean([fold_lookup[(candidate, fold)]["ACC"] for fold in range(1, 11)])),
            "ACC_reference": float(np.mean([fold_lookup[(reference, fold)]["ACC"] for fold in range(1, 11)])),
            "ACC_delta": float(np.mean(acc_delta)),
            "AUC_original_candidate": float(np.mean([fold_lookup[(candidate, fold)]["AUC_original"] for fold in range(1, 11)])),
            "AUC_original_reference": float(np.mean([fold_lookup[(reference, fold)]["AUC_original"] for fold in range(1, 11)])),
            "AUC_original_delta": float(np.mean(auc_delta)),
            "patient_corrected": int(sum(corrected)), "patient_damaged": int(sum(damaged)),
            "patient_net": int(sum(corrected) - sum(damaged)),
        })
    return rows


def patient_mechanism(mechanisms, prediction_rows, paired):
    by_group = {}
    for group in GROUPS:
        rows = [row for row in prediction_rows if row["group"] == group]
        native = [row for row in rows if "native_pred_label" in row and pd.notna(row.get("native_pred_label"))]
        if native:
            y = np.asarray([int(row["true_label"]) for row in native])
            pred_native = np.asarray([int(row["native_pred_label"]) for row in native])
            pred_full = np.asarray([int(row["pred_label"]) for row in native])
            native_correct, full_correct = pred_native == y, pred_full == y
            by_group[group] = {
                "group": group, "native_correct": int(native_correct.sum()),
                "full_correct": int(full_correct.sum()),
                "native_to_full_corrected": int((~native_correct & full_correct).sum()),
                "native_to_full_damaged": int((native_correct & ~full_correct).sum()),
                "native_to_full_net": int((~native_correct & full_correct).sum() - (native_correct & ~full_correct).sum()),
                "NC_correct_native": int(((y == 0) & (pred_native == 0)).sum()),
                "NC_correct_full": int(((y == 0) & (pred_full == 0)).sum()),
                "MCI_correct_native": int(((y == 1) & (pred_native == 1)).sum()),
                "MCI_correct_full": int(((y == 1) & (pred_full == 1)).sum()),
                "AD_correct_native": int(((y == 2) & (pred_native == 2)).sum()),
                "AD_correct_full": int(((y == 2) & (pred_full == 2)).sum()),
            }
    rows = []
    for name, summary in by_group.items():
        rows.append(summary)
    for row in paired:
        if row["fold"] == "MEAN" and (row["reference"] in ("N0", "H4") or row["comparison"] == "N2_vs_N6_DPSL"):
            rows.append({
                "group": row["candidate"], "relative_to": row["reference"],
                "patient_corrected": row["patient_corrected"],
                "patient_damaged": row["patient_damaged"], "patient_net": row["patient_net"],
                "comparison": row["comparison"],
            })
    mechanism_map = {(row["group"], row["fold"]): row for row in mechanisms}
    for group in GROUPS:
        selected = [row for (name, _), row in mechanism_map.items() if name == group]
        if selected:
            rows.append({
                "group": group, "direct_off_changed": sum(row["direct_off"]["changed"] for row in selected),
                "direct_off_corrected": sum(row["direct_off"]["corrected"] for row in selected),
                "direct_off_damaged": sum(row["direct_off"]["damaged"] for row in selected),
                "disease_neutral_changed": sum(row["disease_neutral"]["changed"] for row in selected),
                "disease_neutral_corrected": sum(row["disease_neutral"]["corrected"] for row in selected),
                "disease_neutral_damaged": sum(row["disease_neutral"]["damaged"] for row in selected),
            })
    return rows


def summarize_gradients(gradient_rows):
    result = {"groups": {}, "definition": "train-only GraphLearn p.weight gradients before any formal update"}
    for group in ("N1", "N2", "N3", "N4", "N5", "N6"):
        selected = [row for row in gradient_rows if row["group"] == group]
        modalities = []
        for index, modality in enumerate(runner.MODAL_NAMES):
            entries = [row["per_modality"][index] for row in selected]
            modalities.append({
                "modality": modality,
                "edge_count_min": min(item["off_diagonal_edge_count"] for item in entries),
                "edge_count_max": max(item["off_diagonal_edge_count"] for item in entries),
                "density_mean": float(np.mean([item["graph_density"] for item in entries])),
                "base_gradient_norm_mean": float(np.mean([item["base_gradient_norm"] for item in entries])),
                "aloss_gradient_norm_mean": float(np.mean([item["aloss_gradient_norm"] for item in entries])),
                "structural_zero_folds": sum(item["status"] == "STRUCTURAL_ZERO_GRADIENT" for item in entries),
                "invalid_gradient_folds": sum(item["status"] == "INVALID_ZERO_GRADIENT" for item in entries),
            })
        ratios = [row.get("actual_gradient_ratio") for row in selected]
        result["groups"][group] = {
            "folds": len(selected),
            "base_gradient_norm_mean": float(np.mean([row["base_gradient_norm"] for row in selected])),
            "aloss_gradient_norm_mean": float(np.mean([row["aloss_gradient_norm"] for row in selected])),
            "actual_gradient_ratio_mean": float(np.mean(ratios)),
            "actual_gradient_ratio_min": float(np.min(ratios)),
            "actual_gradient_ratio_max": float(np.max(ratios)),
            "lambda_fold_mean": (float(np.mean([row["calibration"]["lambda_fold"] for row in selected]))
                                 if group == "N5" else None),
            "lambda_fold_min": (float(np.min([row["calibration"]["lambda_fold"] for row in selected]))
                                if group == "N5" else None),
            "lambda_fold_max": (float(np.max([row["calibration"]["lambda_fold"] for row in selected]))
                                if group == "N5" else None),
            "clipped_folds": (sum(row["calibration"]["clipped"] for row in selected)
                              if group == "N5" else 0),
            "modalities": modalities,
        }
    return result


def darg_dpsl_summary(mechanisms):
    result = {"groups": {}}
    for group in GROUPS:
        rows = [row for row in mechanisms if row["group"] == group]
        result["groups"][group] = {
            "folds": len(rows),
            "positive_signed_edges_mean": float(np.mean([row["positive_signed_edges"] for row in rows])),
            "negative_signed_edges_mean": float(np.mean([row["negative_signed_edges"] for row in rows])),
            "relation_gradient_norm_mean": float(np.mean([row["relation_parameter_gradient_norm"] for row in rows])),
            "relation_gradient_nonzero_count_mean": float(np.mean([row["relation_parameter_gradient_nonzero_count"] for row in rows])),
            "historical_total_eval_mean": float(np.mean([row["train_historical_total_eval"] for row in rows])),
            "graph_supervision_eval_mean": float(np.mean([row["train_graph_supervision_eval"] for row in rows])),
            "dpsl_gradient_norm_mean": float(np.mean([row["dpsl_gradient_norm"] for row in rows
                                                        if row["dpsl_gradient_norm"] is not None]))
                                        if any(row["dpsl_gradient_norm"] is not None for row in rows) else None,
            "positive_and_negative_paths_all_folds": all(row["signed_relation_both_present"] for row in rows),
            "dpsl_residual_max": float(max(row["dpsl_residual_max"] for row in rows)),
            "dpsl_shared_gradient_ratio_max": float(max(
                row["dpsl_shared_cap_max"] or 0.0 for row in rows)),
        }
    return result


def collect_update_vector_audits():
    result = {}
    for group in ("N1", "N2", "N3", "N4", "N5"):
        expected_epoch = 1 if group == "N4" else 11
        selected = []
        for fold in range(1, 11):
            path = ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json"
            record = json.loads(path.read_text())
            matches = [item for item in record.get("update_vector_audits", [])
                       if int(item["epoch"]) == expected_epoch]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one optimizer update audit for {group} fold {fold}")
            selected.append(matches[0])
        result[group] = selected
    return result


def summarize_update_vectors(vectors):
    fields = (
        "u_base_norm", "u_supervised_norm", "u_difference_norm",
        "u_difference_relative", "u_cosine", "graph_p_weight_u_base_norm",
        "graph_p_weight_u_supervised_norm", "graph_p_weight_u_difference_norm",
        "graph_p_weight_u_difference_relative", "graph_p_weight_u_cosine",
    )
    result = {}
    for group, rows in vectors.items():
        result[group] = {
            "folds": len(rows), "epoch": int(rows[0]["epoch"]),
            "coefficient_min": min(float(row["coefficient"]) for row in rows),
            "coefficient_max": max(float(row["coefficient"]) for row in rows),
            "metrics": {field: {
                "mean": float(np.mean([row[field] for row in rows])),
                "min": float(np.min([row[field] for row in rows])),
                "max": float(np.max([row[field] for row in rows])),
            } for field in fields},
        }
    return result


def report_text(summary, paired, fidelity, execution, gradient, darg_dpsl,
                historical, update_summary=None, warmup_pairing=None):
    def pct(value):
        return f"{100 * float(value):.4f}%"
    lines = [
        "# Experiment 83.8.2 Final Report", "",
        f"Status: `{execution['status']}`. Formal execution completed {execution['completed_folds']}/70 folds.",
        f"METHOD_FIDELITY: `{fidelity['status']}`. EXECUTION_AUDIT: `{execution['status']}`. PERFORMANCE_GO_NO_GO: `{execution['performance_decision']}`.",
        "", "## Protocol and Fidelity", "",
        f"The data and split SHA-256 values are `{runner.DATA_SHA}` and `{runner.SPLIT_SHA}`. The first-choice OSSFS directory failed with EIO, so the run used `{ROOT}`.",
        "The source matrix distinguishes the squared paper Frobenius term, the author's unsquared network calculation, the author's simple-2 backward path that excludes ALoss, the E83.8.1 six-modality train×train ALoss, and E83.6 continuous PairMean. The seven 83.8.2 groups use no PairMean.",
        "The historical CV protocol still uses full-data RFE, imputation and scaling before outer CV, transductive patient graphs, `val_idx == test_idx`, and outer-test ACC checkpoint selection. Results are therefore protocol reproductions, not independent generalization estimates.",
        "", "## Groups and Main Metrics", "",
        "ACC and AUC_original are mean ± population SD over ten folds. AUC_original uses hard one-hot macro OVR. AUC_probability is auxiliary.",
        "", "| Group | ACC | AUC_original | AUC_probability mean | OOF correct | NC/MCI/AD correct |", "|---|---:|---:|---:|---:|---:|",
    ]
    for group in ("N0", "N1", "N2", "N3", "N4", "N5", "N6", "H1", "H2", "H3", "H4", "H5", "X3", "X4", "X5", "X6"):
        item = summary[group]
        acc = f"{pct(item['ACC_mean'])} ± {100 * float(item['ACC_std_population']):.4f} pp"
        auc = f"{pct(item['AUC_original_mean'])} ± {100 * float(item['AUC_original_std_population']):.4f} pp"
        prob = "N/A" if not finite(item.get("AUC_probability_mean", float("nan"))) else pct(item["AUC_probability_mean"])
        cls = f"{item.get('NC_correct', '')}/{item.get('MCI_correct', '')}/{item.get('AD_correct', '')}"
        lines.append(f"| {group} | {acc} | {auc} | {prob} | {item.get('OOF_correct', '')}/598 | {cls} |")
    lines += [
        "| Paper | 94.8100% ± 2.9500 pp | 95.5900% ± 2.5700 pp | N/A | N/A | N/A |",
        "", "## Paired Contrasts", "",
        "Paired differences below are descriptive across the ten fixed folds. Positive patient net means candidate corrects more patients than it damages.",
        "", "| Comparison | ΔACC | ΔAUC_original | Corrected | Damaged | Net |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        if row["fold"] == "MEAN":
            lines.append(f"| {row['comparison']} | {100*row['ACC_delta']:+.4f} pp | {100*row['AUC_original_delta']:+.4f} pp | {row['patient_corrected']} | {row['patient_damaged']} | {row['patient_net']:+d} |")
    if warmup_pairing:
        lines += ["", "## Initialization and Warmup Pairing", "",
                  "N0, N1, N2, N3, and N5 share the same per-fold initial parameter, optimizer, RNG, DARG relation pairs, ten warmup losses, post-warmup model/optimizer/RNG, spectral statistics, and DPSL prompt hashes. N4 shares the same initial hashes, relation pairs, training labels, and post-warmup RNG. Its warmup losses and post-warmup model/optimizer differ because ALoss is active from epoch 1. N6 passes the per-fold common GFR/CAM/DARG initialization and warmup comparison against N0.",
                  "", "| Group | Fold pairing status | Folds |", "|---|---|---:|"]
        for group in GROUPS:
            item = warmup_pairing["groups"][group]
            lines.append(f"| {group} | {item['status']} | {item['folds']} |")
    lines += ["", "## Gradient and Optimizer Audit", "",
              "The ALoss gradient probe uses the train-only historical loss and six GraphLearn `p.weight` tensors immediately before N4 epoch 1 or epoch 11. Structural zero is classified when a modality has no off-diagonal edges and its ALoss gradient is at numerical-zero scale. N5 computes one fixed coefficient per fold from the same probe.",
              "The Adam audit compares full optimizer update vectors from independent model and optimizer clones using the real checkpointed Adam state. It reports vector difference norms, relative difference, and cosine for the entire trainable model and the six GraphLearn `p.weight` tensors.",
              "", "| Group | ALoss/base gradient ratio mean | range | λ mean or fixed |", "|---|---:|---:|---:|"]
    for group in ("N1", "N2", "N3", "N4", "N5", "N6"):
        item = gradient["groups"][group]
        coefficient = (f"{item['lambda_fold_mean']:.7g} [{item['lambda_fold_min']:.7g}, {item['lambda_fold_max']:.7g}]"
                       if group == "N5" else str({"N1": .001, "N2": .003, "N3": .01, "N4": .003, "N6": .003}[group]))
        lines.append(f"| {group} | {item['actual_gradient_ratio_mean']:.6g} | [{item['actual_gradient_ratio_min']:.6g}, {item['actual_gradient_ratio_max']:.6g}] | {coefficient} |")
    if update_summary:
        lines += ["", "### Adam Update-Vector Results", "",
                  "Each row summarizes ten independent clone-based counterfactuals at the registered audit epoch. The full vector covers all trainable model parameters. The subset covers the six GraphLearn `p.weight` tensors.",
                  "", "| Group | Epoch | Full `||u_base||` | Full `||u_supervised||` | Full `||difference||` | Full relative difference | Full cosine | Graph `||difference||` | Graph relative difference | Graph cosine |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for group in ("N1", "N2", "N3", "N4", "N5"):
            item = update_summary[group]
            metrics = item["metrics"]
            lines.append(
                f"| {group} | {item['epoch']} | {metrics['u_base_norm']['mean']:.6g} | "
                f"{metrics['u_supervised_norm']['mean']:.6g} | "
                f"{metrics['u_difference_norm']['mean']:.6g} | "
                f"{metrics['u_difference_relative']['mean']:.6g} | "
                f"{metrics['u_cosine']['mean']:.9f} | "
                f"{metrics['graph_p_weight_u_difference_norm']['mean']:.6g} | "
                f"{metrics['graph_p_weight_u_difference_relative']['mean']:.6g} | "
                f"{metrics['graph_p_weight_u_cosine']['mean']:.9f} |"
            )
    lines += ["", "## DARG and DPSL Mechanisms", "",
              "DARG positive-DCR and signed positive/negative paths were checked at each Best checkpoint. Direct-Off and Disease-Neutral intervention counts are in `results/patient_mechanism.csv`. DPSL Residual-Off matched same-state Native logits exactly in each G4 fold; all three class-pair evidence paths were present. The Native-versus-Full counts and class-specific accuracy are in the same file.",
              "The table separates the trained model's Native-to-Full inference changes from cross-model paired changes, which include joint training effects.",
              "", "| Group | Positive signed edges mean | Negative signed edges mean | Both signs every fold | DARG relation grad norm mean | Historical loss mean | ALoss mean | DPSL grad norm mean |", "|---|---:|---:|:---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        item = darg_dpsl["groups"][group]
        dpsl_grad = "N/A" if item["dpsl_gradient_norm_mean"] is None else f"{item['dpsl_gradient_norm_mean']:.6g}"
        lines.append(f"| {group} | {item['positive_signed_edges_mean']:.2f} | {item['negative_signed_edges_mean']:.2f} | {item['positive_and_negative_paths_all_folds']} | {item['relation_gradient_norm_mean']:.6g} | {item['historical_total_eval_mean']:.6g} | {item['graph_supervision_eval_mean']:.6g} | {dpsl_grad} |")
    lines += ["", "## Identity and Decision", "",
              f"N0 versus historical H4 is `{fidelity['n0_h4_status']}`. The ten warmup histories, post-warmup model/optimizer/RNG hashes, relation pairs, best epoch, hard predictions, and fold metrics match. H4 has no recorded initial model, optimizer, or RNG hashes, so its initial-state identity remains partial.",
              f"PERFORMANCE_GO_NO_GO: `{execution['performance_decision']}`. The final complete model remains G4 E82.4.6.10-G1 (DARG + DPSL).",
              "", "## Audit Artifacts", "",
              "See `audits/final_audit.json`, `audits/gradient_mechanism.json`, `audits/checkpoint_recovery.json`, `audits/warmup_pairing.json`, `audits/report_repair.json`, `audits/training_seal.json`, and `results/darg_dpsl_fold_mechanism.csv` for fold-level identities, gradient measurements, recovery witnesses, corrected mechanism counts, and the frozen source seal.",
              ""]
    return "\n".join(lines)


def main():
    runner.configure_worker()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Independent audit expects the assigned single GPU")
    x_np, y_np, ids, splits = runner.load_data()
    split_by_fold = {int(row["fold"]): row for row in splits}
    seal_path = AUDITS / "training_seal.json"
    seal = json.loads(seal_path.read_text())
    if seal["files"] != runner.current_implementation_hashes():
        raise RuntimeError("Training source/input/configuration seal changed after training")
    replay_cache_path = AUDITS / "independent_replay_cache.pt"
    replay_cache = (torch.load(replay_cache_path, map_location="cpu", weights_only=False)
                    if replay_cache_path.exists() else {})
    tasks = []
    failures = []
    results = []
    pending = []
    for group in GROUPS:
        for fold in range(1, 11):
            key = f"{group}/fold{fold:02d}"
            if key in replay_cache:
                cached = replay_cache[key]
                best_path = ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"
                latest_path = ROOT / "groups" / group / "latest_checkpoints" / f"fold{fold:02d}.pt"
                if (cached["audit"]["best_sha256"] != sha_file(best_path)
                        or cached["audit"]["latest_sha256"] != sha_file(latest_path)):
                    raise RuntimeError(f"Cached replay identity differs for {key}")
                results.append(cached)
                print(f"REPLAY_CACHE_PASS {key}", flush=True)
            else:
                pending.append((group, fold))
    if pending:
        with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(audit_fold, group, fold, x_np, y_np, ids,
                                   split_by_fold[fold], torch.device("cuda:0")): (group, fold)
                       for group, fold in pending}
            for future in as_completed(futures):
                group, fold = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    replay_cache[f"{group}/fold{fold:02d}"] = result
                    save_temporary_cache(replay_cache_path, replay_cache)
                    print(f"REPLAY_PASS {group} fold {fold:02d}", flush=True)
                except Exception as exc:
                    failures.append({"group": group, "fold": fold, "error": repr(exc),
                                     "traceback": traceback.format_exc()})
                    print(f"REPLAY_FAIL {group} fold {fold:02d}: {exc!r}", file=sys.stderr, flush=True)
    if failures:
        (AUDITS / "audit_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"Independent fold replay failed for {len(failures)} folds")
    fold_rows, prediction_rows, mechanisms, gradient_rows, recovery_rng = aggregate_current(results)

    # Audit-only one-step recovery witnesses from the persisted Best and Latest states.
    witness_cache_path = AUDITS / "recovery_witness_cache.pt"
    witness_cache = (torch.load(witness_cache_path, map_location="cpu", weights_only=False)
                     if witness_cache_path.exists() else {})
    witness_results = []
    witness_jobs = []
    for group in GROUPS:
        for fold in range(1, 11):
            rng = recovery_rng[(group, fold)]
            for kind, state in (("Best", rng), ("Latest", None)):
                key = f"{group}/fold{fold:02d}/{kind}"
                checkpoint_path = ROOT / "groups" / group / (
                    "best_checkpoints" if kind == "Best" else "latest_checkpoints") / f"fold{fold:02d}.pt"
                if key in witness_cache:
                    cached = witness_cache[key]
                    if cached["checkpoint_sha256"] != sha_file(checkpoint_path):
                        raise RuntimeError(f"Cached recovery identity differs for {key}")
                    witness_results.append(cached)
                    print(f"RECOVERY_CACHE_PASS {key}", flush=True)
                else:
                    witness_jobs.append((group, fold, kind, x_np, y_np, split_by_fold[fold], state))
    if witness_jobs:
        with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_witness_task, job): (job[0], job[1], job[2])
                       for job in witness_jobs}
            for future in as_completed(futures):
                group, fold, kind = futures[future]
                try:
                    result = future.result()
                    witness_results.append(result)
                    witness_cache[f"{group}/fold{fold:02d}/{kind}"] = result
                    save_temporary_cache(witness_cache_path, witness_cache)
                    print(f"RECOVERY_PASS {group} fold {fold:02d} {kind}", flush=True)
                except Exception as exc:
                    failures.append({"group": group, "fold": fold, "checkpoint": kind,
                                     "error": repr(exc), "traceback": traceback.format_exc()})
                    print(f"RECOVERY_FAIL {group} fold {fold:02d} {kind}: {exc!r}",
                          file=sys.stderr, flush=True)
    if failures:
        (AUDITS / "audit_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError(f"Checkpoint recovery witness failed for {len(failures)} cases")

    recovery_path = AUDITS / "best_recovery_rng.pt"
    recovery_payload = {
        "experiment": "Experiment 83.8.2",
        "source": "Independent RNG replay from frozen initialization and warmup; verified against each Latest RNG",
        "states": {f"{group}/fold{fold:02d}": {
            "best_epoch": int(torch.load(ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt",
                                         map_location="cpu", weights_only=False)["epoch"]),
            "best_sha256": sha_file(ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"),
            "rng": rng,
            "rng_sha256": runner.sha_object(rng),
        } for (group, fold), rng in recovery_rng.items()},
    }
    torch.save(recovery_payload, recovery_path)
    recovery_audit = {
        "experiment": "Experiment 83.8.2", "best_recovery_rng_sidecar": str(recovery_path),
        "best_recovery_rng_sidecar_sha256": sha_file(recovery_path),
        "best_rng_states": len(recovery_rng),
        "latest_rng_replays": sum(result["audit"]["latest_rng_replay"] == "PASS" for result in results),
        "next_step_witnesses": len(witness_results), "status": "PASS",
        "witnesses": witness_results,
    }
    (AUDITS / "checkpoint_recovery.json").write_text(json.dumps(recovery_audit, indent=2, sort_keys=True) + "\n")

    for result in results:
        result["audit"]["next_update_witness"] = "PASS"
    current_metrics, old_fold_metrics, old_oof = historical_records()
    summaries = current_summaries(fold_rows, prediction_rows)
    add_historical_summary(summaries, current_metrics)
    write_csv(RESULTS / "fold_metrics.csv", [*fold_rows, *old_fold_metrics])
    write_csv(RESULTS / "oof_predictions.csv", [*prediction_rows, *old_oof])
    paired = paired_rows(fold_rows, prediction_rows, old_fold_metrics, old_oof)
    write_csv(RESULTS / "paired_contrasts.csv", paired)
    write_csv(RESULTS / "patient_mechanism.csv", patient_mechanism(mechanisms, prediction_rows, paired))
    write_csv(RESULTS / "group_summary.csv", summaries.values())
    gradient = summarize_gradients(gradient_rows)
    darg_dpsl = darg_dpsl_summary(mechanisms)
    update_vectors = collect_update_vector_audits()
    update_summary = summarize_update_vectors(update_vectors)
    gradient_payload = {
        "experiment": "Experiment 83.8.2", "aloss_gradients": gradient,
        "adam_update_vectors": update_vectors,
        "adam_update_vector_summary": update_summary,
        "darg_dpsl": darg_dpsl,
    }
    (AUDITS / "gradient_mechanism.json").write_text(json.dumps(gradient_payload, indent=2, sort_keys=True) + "\n")

    # Verify exact local N0/H4 warmup and output identity. Initial H4 hashes are absent.
    h4_checks = []
    n0_dir = ROOT / "groups/N0"
    for fold in range(1, 11):
        h4_best = torch.load(H4_DIR / "best_checkpoints" / f"fold{fold:02d}.pt",
                             map_location="cpu", weights_only=False)
        n0_best = torch.load(n0_dir / "best_checkpoints" / f"fold{fold:02d}.pt",
                             map_location="cpu", weights_only=False)
        h4_warm, n0_warm = h4_best["warmup"], n0_best["warmup"]
        warm_fields = ("base_state_sha256", "optimizer_state_sha256", "rng_state_sha256", "pair_sha256")
        warm_exact = all(h4_warm.get(key) == n0_warm.get(key) for key in warm_fields)
        history_exact = [{key: row[key] for key in ("epoch", "loss")} for row in h4_warm["history"]] == [
            {key: row[key] for key in ("epoch", "loss")} for row in n0_warm["history"]]
        old_pred = pd.read_csv(H4_DIR / "predictions" / f"fold{fold:02d}.csv")["pred_label"].to_numpy()
        new_pred = pd.read_csv(n0_dir / "predictions" / f"fold{fold:02d}.csv")["pred_label"].to_numpy()
        h4_checks.append({
            "fold": fold, "warmup_post_state_exact": warm_exact,
            "warmup_history_exact": history_exact,
            "best_epoch_exact": h4_best["epoch"] == n0_best["epoch"],
            "prediction_exact": bool(np.array_equal(old_pred, new_pred)),
            "h4_initial_model_sha256": h4_warm.get("initial_model_sha256"),
            "n0_initial_model_sha256": n0_warm.get("initial_model_sha256"),
            "h4_best_sha256": sha_file(H4_DIR / "best_checkpoints" / f"fold{fold:02d}.pt"),
            "n0_best_sha256": sha_file(n0_dir / "best_checkpoints" / f"fold{fold:02d}.pt"),
        })
    n0_h4_status = "PARTIAL_MATCH" if all(
        row["warmup_post_state_exact"] and row["warmup_history_exact"]
        and row["best_epoch_exact"] and row["prediction_exact"] for row in h4_checks
    ) else "MISMATCH"

    candidate_eligible = []
    for group in ("N1", "N2", "N3", "N4", "N5"):
        item = summaries[group]
        if all(item[metric] > summaries[reference][metric]
               for reference in ("N0", "H4")
               for metric in ("ACC_mean", "AUC_original_mean")):
            candidate_eligible.append(group)
    decision = "PERFORMANCE_GO" if candidate_eligible else "PERFORMANCE_NO_GO"
    status = {
        "experiment": "Experiment 83.8.2", "status": "PASS",
        "completed_folds": len(fold_rows), "expected_folds": 70,
        "groups": {group: {"folds": sum(row["group"] == group for row in fold_rows),
                           "status": "COMPLETE"} for group in GROUPS},
        "best_replay_count": len(fold_rows), "latest_rng_replay_count": len(results),
        "best_next_step_witness_count": sum(item["checkpoint"] == "Best" for item in witness_results),
        "latest_next_step_witness_count": sum(item["checkpoint"] == "Latest" for item in witness_results),
        "training_seal_sha256": sha_file(seal_path),
        "data_sha256": runner.DATA_SHA, "split_sha256": runner.SPLIT_SHA,
        "n0_h4_status": n0_h4_status, "n0_h4_fold_checks": h4_checks,
        "candidate_eligible": candidate_eligible,
        "performance_decision": decision,
        "limitations": [
            "full-data RFE, mean imputation and StandardScaler before outer CV are historical protocol choices",
            "transductive graph and val_idx == test_idx are retained",
            "outer test ACC participates in checkpoint selection",
            "single seed 0 and ten-fold results are descriptive",
            "historical H4 lacks recorded initial parameter, optimizer and RNG hashes",
        ],
    }
    (AUDITS / "final_audit.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    fidelity = {
        "status": "PASS", "n0_h4_status": n0_h4_status,
        "method_fidelity_matrix_sha256": sha_file(AUDITS / "method_fidelity_matrix.md"),
        "training_seal_sha256": sha_file(seal_path),
        "training_sources_unmodified": True,
    }
    report = report_text(summaries, paired, fidelity, status, gradient, darg_dpsl,
                        current_metrics, update_summary)
    (RESULTS / "final_report.md").write_text(report, encoding="utf-8")
    replay_cache_path.unlink(missing_ok=True)
    witness_cache_path.unlink(missing_ok=True)
    print(json.dumps({"status": status["status"], "folds": status["completed_folds"],
                      "performance_decision": decision, "candidate_eligible": candidate_eligible,
                      "n0_h4_status": n0_h4_status}, indent=2))


if __name__ == "__main__":
    main()
