"""Train and resume the sealed Experiment 83.8.3 TADPOLE groups."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
from collections import Counter
import hashlib
import io
import json
import multiprocessing
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "common"))
sys.path.insert(0, str(CODE_ROOT))
from fidelity_model import FidelityModel, fit_spectral_statistics, normalize_adj
from fold_checkpoint import atomic_write, lock, publish_epoch, recover_best, write_checkpoint, write_json
from metrics import acc_auc
import run_e833 as historical


DATA_SHA = "0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc"
SPLIT_SHA = "7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5"
SEED = 0
MAX_EPOCHS = 450
PATIENCE = 200
BASE_GROUPS = {"H4": "G4_E82_4_6_10_G1"}
HISTORICAL_ROOT = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
N3_REFERENCE_ROOT = Path("/model/spromptgl/experiment/experiment83_8_2_tadpole_g4_aloss_gradient_repair")
OLD_PROMPT_RN50 = ROOT / "inputs/prompts/mmGPL12_rn50.pt"
OLD_PROMPT_SCMR = ROOT / "inputs/prompts/mmGPL12_scmr.pt"
CLASS_PROMPT_RN50 = ROOT / "inputs/prompts/mmGPL12_class_rn50.pt"
CLASS_PROMPT_SCMR = ROOT / "inputs/prompts/mmGPL12_class_scmr.pt"
PROMPT8 = CODE_ROOT / "sources/E82_4_6_G2_SCMR_DA_SRPE/inputs/prompts/prompts.pt"
MODAL_DIMS = (145, 148, 3, 36, 24, 9)
MODAL_NAMES = ("UCSFFSX", "UCBERKELEYAV45", "UPENNBIOMK9", "RISK_FACTOR",
               "COGNITIVE_TEST", "ROI_AVERAGE")
FORMAL_GROUPS = ("Y0", "Y1", "Y2", "Y3", "Y4", "Y5")
PAPER_ALOSS_MATHEMATICAL_COEFFICIENT = 1.0
PAIR_MEAN_COEFFICIENT = 0.1
PAIR_WARMUP_EPOCHS = 10


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_object(value) -> str:
    stream = io.BytesIO()
    torch.save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def sha_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def current_implementation_hashes():
    paths = []
    for base in (CODE_ROOT, ROOT / "sources", ROOT / "inputs", ROOT / "groups/S0/sp/data"):
        paths.extend(path for path in base.rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    paths.extend(sorted((ROOT / "configs").glob("*.json")))
    paths.extend(ROOT / name for name in ("protocol.json", "source_manifest.json", "data_manifest.json",
                                          "prompt_manifest.json", "audits/method_fidelity_matrix.md"))
    unique = {path for path in paths if path.is_file()}
    source_comparison = ROOT / "audits/source_comparison.json"
    if source_comparison.is_file():
        unique.add(source_comparison)
    return {str(path.relative_to(ROOT)): sha_file(path) for path in sorted(unique)}


def implementation_fingerprint(hashes=None):
    hashes = current_implementation_hashes() if hashes is None else hashes
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def require_training_gate(groups):
    seal_path = ROOT / "audits/training_seal.json"
    preflight_path = ROOT / "audits/preflight_start.json"
    resource_path = ROOT / "audits/resource_probe.json"
    if not seal_path.is_file() or not preflight_path.is_file() or not resource_path.is_file():
        raise RuntimeError("Training requires the E83.8.3 seal, preflight, and resource probe")
    seal = json.loads(seal_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    resource_probe = json.loads(resource_path.read_text())
    hashes = current_implementation_hashes()
    if (seal.get("status") != "SEALED_BEFORE_TRAINING"
            or seal.get("experiment") != "Experiment 83.8.3"
            or seal.get("files") != hashes
            or seal.get("implementation_sha256") != implementation_fingerprint(hashes)
            or preflight.get("experiment") != "Experiment 83.8.3"
            or preflight.get("status") != "PASS"
            or preflight.get("formal_training_allowed") is not True
            or resource_probe.get("experiment") != "Experiment 83.8.3"
            or resource_probe.get("status") != "PASS"
            or resource_probe.get("training_seal_sha256") != sha_file(seal_path)
            or resource_probe.get("worker_count") != 8):
        raise RuntimeError("E83.8.3 source, input, configuration, preflight, or resource identity differs from its seal")
    for group in groups:
        if (group not in preflight.get("groups", {})
                or preflight["groups"][group].get("formal_training_allowed") is not True
                or preflight["groups"][group].get("status") != "PASS"):
            raise RuntimeError(f"Training is not registered for {group}")


def configure_worker():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        if "after parallel work has started" not in str(exc):
            raise
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Experiment 83.8.3 is registered for exactly one physical GPU")
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)


def load_data():
    data_dir = ROOT / "inputs"
    data_path = data_dir / "processed_standard_data.csv"
    split_path = data_dir / "split_indices.json"
    if sha_file(data_path) != DATA_SHA or sha_file(split_path) != SPLIT_SHA:
        raise RuntimeError("Locked TADPOLE data or split SHA differs")
    frame = pd.read_csv(data_path)
    ids = pd.read_csv(data_dir / "sample_ids.csv")
    x = frame.drop(columns=["LABEL"]).to_numpy(np.float32)
    y = frame["LABEL"].to_numpy(np.int64) - 1
    splits = json.loads(split_path.read_text())["splits"]
    if x.shape != (598, 365) or len(ids) != 598 or np.bincount(y, minlength=3).tolist() != [209, 315, 74]:
        raise RuntimeError("Locked TADPOLE cohort dimensions or class counts differ")
    return x, y, ids, splits


def group_config(group: str):
    path = ROOT / "configs" / f"{group}.json"
    config = json.loads(path.read_text())
    if (config.get("experiment") != "Experiment 83.8.3" or config.get("group") != group
            or config.get("data_sha256") != DATA_SHA or config.get("split_sha256") != SPLIT_SHA):
        raise RuntimeError(f"Unregistered group identity {group}")
    return config


def prompt_path_for(config, base_group):
    prefixed = bool(config.get("class_prefix_prompts"))
    if base_group == BASE_GROUPS["H4"]:
        return CLASS_PROMPT_SCMR if prefixed else OLD_PROMPT_SCMR
    return PROMPT8


def build_model(config, fold: int, device, spectral_stats=None):
    base_group = BASE_GROUPS[config["base_reference"]]
    prompt_path = prompt_path_for(config, base_group)
    return FidelityModel(
        historical.DIMS, base_group, 3, str(prompt_path), spectral_stats,
        824602 * 100 + fold, fold=fold, device=device,
        res_only=bool(config.get("res_only")),
        encoder_attention_dropout=(0.1 if config.get("encoder_attention_dropout") is None
                                   else float(config["encoder_attention_dropout"])),
        darg_gate_scales=tuple(config["darg_gate_scales"]),
        dpsl_confidence_protection=bool(config["dpsl_confidence_protection"]),
    ).to(device)


def make_optimizers(model, config, formal=False):
    initial_decay = float(config.get("initial_base_weight_decay",
                                    0.11 if model.use_darg else 0.0036))
    intermediate_decay = config.get("intermediate_base_weight_decay")
    base_decay = (float(intermediate_decay) if formal and intermediate_decay is not None
                  else initial_decay)
    base_options = {"lr": 0.012, "weight_decay": base_decay}
    optimizers = {
        "base": torch.optim.Adam(model.graph_parameters(), **base_options),
        "cam": torch.optim.Adam(model.cam.parameters(), **base_options),
    }
    optimizer_config = {
        "class": "Adam/AdamW" if model.use_darg else "Adam",
        "gfr": [0.012, base_decay], "cam": [0.012, base_decay],
    }
    if intermediate_decay is not None:
        optimizer_config["base_weight_decay_phase"] = "intermediate" if formal else "warmup"
    if model.use_darg:
        decay, lower_rank = [], []
        for name, parameter in model.darg.relation_named_parameters():
            bucket = lower_rank if parameter.ndim <= 1 or name.endswith("bias") or "norm" in name.lower() else decay
            bucket.append(parameter)
        optimizers["darg"] = torch.optim.AdamW(
            [{"params": decay}, {"params": lower_rank}], lr=0.001, weight_decay=0.01)
        optimizer_config["darg"] = [0.001, 0.01]
        optimizer_config["dpsl"] = None
    if model.dpsl is not None:
        optimizers["dpsl"] = torch.optim.AdamW(model.dpsl.parameters(), lr=0.0005, weight_decay=0.01)
        optimizer_config["dpsl"] = [0.0005, 0.01]
    return optimizers, optimizer_config


def apply_intermediate_weight_decay(optimizers, config, optimizer_config):
    """Change only the GFR/CAM Adam groups after the registered warmup."""
    target = config.get("intermediate_base_weight_decay")
    if target is None:
        return optimizer_config
    target = float(target)
    for name in ("base", "cam"):
        for parameter_group in optimizers[name].param_groups:
            parameter_group["weight_decay"] = target
    if "darg" in optimizers:
        darg_decay = [float(group["weight_decay"]) for group in optimizers["darg"].param_groups]
        if darg_decay != [0.01, 0.01]:
            raise RuntimeError("Intermediate WD changed the DARG optimizer")
    if "dpsl" in optimizers:
        if any(float(group["weight_decay"]) != 0.01 for group in optimizers["dpsl"].param_groups):
            raise RuntimeError("Intermediate WD changed the DPSL optimizer")
    updated = copy.deepcopy(optimizer_config)
    updated["gfr"] = [0.012, target]
    updated["cam"] = [0.012, target]
    updated["base_weight_decay_phase"] = "intermediate"
    return updated


def dpsl_optimizer_step(model, optimizers, historical_total, objectives, shared_cap):
    """Apply the sealed DPSL optimizer with only its registered shared cap changed."""
    optimization = historical.load_scmr_optimization()
    old_cap = optimization.CLASS["shared_cap"]
    optimization.CLASS["shared_cap"] = float(shared_cap)
    try:
        return historical.scmr_optimizer_step(model, optimizers, historical_total, objectives)
    finally:
        optimization.CLASS["shared_cap"] = old_cap


def zero_optimizers(optimizers):
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)


def step_optimizers(optimizers):
    for optimizer in optimizers.values():
        optimizer.step()


def _masked_label_target(labels: torch.Tensor, train_rows: torch.Tensor,
                         dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the train-only label graph without comparing unknown rows."""
    train_labels = labels.index_select(0, train_rows).detach()
    same = train_labels[:, None].eq(train_labels[None, :])
    target = torch.zeros((len(labels), len(labels)), dtype=dtype, device=labels.device)
    target_train = same.to(dtype=dtype)
    target[train_rows[:, None], train_rows[None, :]] = target_train
    eye = torch.eye(len(labels), dtype=dtype, device=labels.device)
    return normalize_adj(target + eye), same


def paper_aloss(aux, labels, train_rows, model):
    """Eq.(1) label graph term with squared Frobenius norm and train-only mask.

    GraphLearn creates each actual modality graph on all transductive nodes. The
    supervision target is normalized on the same node set, while the Frobenius
    difference is evaluated only on the train-by-train block. This retains the
    graph normalization used by the source and excludes every unknown test label.
    """
    if len(aux["adjacencies"]) != len(MODAL_DIMS):
        raise RuntimeError("Paper-ALoss requires the six actual TADPOLE modality graphs")
    target_full, same = _masked_label_target(labels, train_rows, aux["raw"].dtype)
    train_block = (train_rows[:, None], train_rows[None, :])
    modal_losses = []
    for adjacency in aux["adjacencies"]:
        eye = torch.eye(len(adjacency), dtype=adjacency.dtype, device=adjacency.device)
        normalized = normalize_adj(adjacency + eye)
        selected = normalized[train_block]
        target_selected = target_full[train_block]
        modal_losses.append((selected - target_selected).square().sum())
    aloss = torch.stack(modal_losses).mean()
    return aloss, {
        "target": same,
        "per_modality_loss": [float(value.detach()) for value in modal_losses],
        "train_count": int(train_rows.numel()),
        "train_pair_count_including_diagonal": int(train_rows.numel() ** 2),
        "formula": "mean_m ||Ahat_train - normalize(label_same_train + I_full)||_F^2",
        "coefficient": PAPER_ALOSS_MATHEMATICAL_COEFFICIENT,
        "diagonal_included": True,
    }


def _pair_groups(train_labels: torch.Tensor):
    """Return six unique class-pair index groups in train-local coordinates."""
    groups = []
    for class_id in range(3):
        positions = torch.nonzero(train_labels == class_id, as_tuple=False).flatten()
        if positions.numel() < 2:
            left = positions.new_empty((0,))
            right = positions.new_empty((0,))
        else:
            rows, cols = torch.triu_indices(positions.numel(), positions.numel(), offset=1,
                                             device=positions.device)
            left, right = positions[rows], positions[cols]
        groups.append((f"class_{class_id}-class_{class_id}", left, right, 1.0))
    for left_class, right_class in ((0, 1), (0, 2), (1, 2)):
        left_positions = torch.nonzero(train_labels == left_class, as_tuple=False).flatten()
        right_positions = torch.nonzero(train_labels == right_class, as_tuple=False).flatten()
        left = left_positions.repeat_interleave(right_positions.numel())
        right = right_positions.repeat(left_positions.numel())
        groups.append((f"class_{left_class}-class_{right_class}", left, right, 0.0))
    return groups


def pair_mean_graph_supervision(scores, labels, train_rows, native_logits=None):
    """Class-balanced continuous pair mean, with optional detached uncertainty weights."""
    train_labels = labels.index_select(0, train_rows).detach()
    groups = _pair_groups(train_labels)
    if len(scores) != len(MODAL_DIMS):
        raise RuntimeError("Pair-mean supervision requires all six continuous modality scores")
    per_modal = []
    per_group_values = {name: [] for name, *_ in groups}
    per_group_weight_sums = {name: [] for name, *_ in groups}
    entropy = None
    if native_logits is not None:
        probabilities = F.softmax(native_logits.detach(), dim=-1)
        entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(-1)
        entropy = (entropy / np.log(3.0)).clamp(0.0, 1.0).detach()
    for score in scores:
        train_score = score.index_select(0, train_rows).index_select(1, train_rows)
        group_losses = []
        for name, left, right, target_value in groups:
            if left.numel() == 0:
                raise RuntimeError(f"pair supervision group {name} is empty")
            selected = train_score[left, right]
            target = selected.new_full(selected.shape, target_value)
            error = (selected - target).square()
            if entropy is None:
                weights = torch.ones_like(error)
            else:
                uncertainty = entropy.index_select(0, train_rows)
                weights = (0.5 + 0.5 * (uncertainty[left] + uncertainty[right]) / 2.0).detach()
            loss = (weights * error).sum() / (weights.sum() + 1e-8)
            group_losses.append(loss)
            per_group_values[name].append(loss.detach())
            per_group_weight_sums[name].append(float(weights.sum().detach()))
        per_modal.append(torch.stack(group_losses).mean())
    details = {
        "group_counts": {name: int(left.numel()) for name, left, _, _ in groups},
        "group_target": {name: float(target) for name, _, _, target in groups},
        "per_modality_loss": [float(value.detach()) for value in per_modal],
        "per_group_loss": {name: float(torch.stack(values).mean()) for name, values in per_group_values.items()},
        "per_group_weight_sum": {name: float(np.mean(values)) for name, values in per_group_weight_sums.items()},
        "weighted": entropy is not None,
        "weights_detached": entropy is not None,
        "coefficient": PAIR_MEAN_COEFFICIENT,
        "formula": "mean_m mean_s sum(w*(S-T)^2)/(sum(w)+eps)",
    }
    return torch.stack(per_modal).mean(), details


def historical_loss(aux, labels, rows, model, graph_supervision_kind="none",
                    supervision_coefficient=None):
    native = F.log_softmax(aux["native_logits"], dim=-1)
    classification = historical.weighted_nll(native, labels, rows)
    graph = aux["graph_loss"]
    total = classification + graph
    if model.use_darg:
        total = total + 0.05 * aux["pair_loss"] + 0.01 * aux["adv_loss"] + 0.01 * aux["preserve_loss"]
    supervision = total.new_zeros(())
    supervision_details = None
    target = None
    aloss_details = None
    if graph_supervision_kind == "paper_aloss":
        supervision, aloss_details = paper_aloss(aux, labels, rows, model)
        aloss_details["target_sha256"] = sha_object((
            rows.detach().cpu(), labels.index_select(0, rows).detach().cpu(),
            aloss_details["target"].detach().cpu()))
        target = aloss_details
        coefficient = (PAPER_ALOSS_MATHEMATICAL_COEFFICIENT
                       if supervision_coefficient is None else float(supervision_coefficient))
        total = total + coefficient * supervision
    elif graph_supervision_kind in ("cb_pair_mean", "cb_uncertainty_pair_mean"):
        logits = aux["native_logits"] if graph_supervision_kind == "cb_uncertainty_pair_mean" else None
        supervision, supervision_details = pair_mean_graph_supervision(
            aux["continuous_scores"], labels, rows, native_logits=logits)
        supervision_details["target_sha256"] = sha_object((
            rows.detach().cpu(), labels.index_select(0, rows).detach().cpu(),
            supervision_details["group_target"], supervision_details["group_counts"]))
        target = supervision_details
        total = total + PAIR_MEAN_COEFFICIENT * supervision
    return classification, total, supervision, target


def warmup_loss(aux, labels, rows, model, graph_supervision_kind="none",
                supervision_coefficient=None):
    classification, total, supervision, _ = historical_loss(
        aux, labels, rows, model, graph_supervision_kind, supervision_coefficient)
    return classification, total, supervision


def run_warmup(model, optimizers, x, labels, train_rows, pair, same, config,
               fold, group, test_rows):
    history = []
    gradient_budget_history = []
    update_audits = []
    log_records = []
    for epoch in range(1, 11):
        model.train()
        zero_optimizers(optimizers)
        rng_before = historical.capture_rng_state()
        _, aux = model(x, labels, train_rows, pair, same, dpsl_enabled=False, darg_warmup=True)
        kind = active_supervision_kind(config, epoch, warmup=True)
        coefficient = configured_supervision_coefficient(config)
        classification, loss, _ = warmup_loss(aux, labels, train_rows, model, kind, coefficient)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DARG warmup loss at epoch {epoch}")
        loss.backward()
        step_optimizers(optimizers)
        eval_logp = evaluate_without_training_side_effects(
            model, x, labels, train_rows, pair, same, darg_warmup=True)
        train_metrics = author_epoch_metrics(eval_logp, train_rows, labels)
        val_metrics = author_epoch_metrics(eval_logp, test_rows, labels)
        log_records.append({
            "epoch": epoch, "trn_loss_mf_1": float(classification.detach()),
            "trn_loss_mf_2": float(classification.detach()),
            "trn_loss_mf_3": author_unsquared_aloss(aux, labels, train_rows),
            "trn_auc": train_metrics["AUC"], "trn_acc": train_metrics["ACC"],
            "val_auc": val_metrics["AUC"], "val_acc": val_metrics["ACC"],
        })
        history.append({"epoch": epoch, "loss": float(loss.detach()),
                        "graph_supervision": float(_graph_supervision_value(
                            aux, labels, train_rows, model, kind, coefficient).detach()),
                        "supervision_kind": kind,
                        "supervision_coefficient": coefficient or 0.0,
                        "graph_gradient_audit": graph_gradient_audit(model, aux)})
    darg_opt = optimizers["darg"]
    for gate in (model.gfr.edge_gate_logits, model.gfr.adaptive_gate_logits):
        if gate in darg_opt.state:
            raise RuntimeError("A DARG gate received an optimizer update during E61 warmup")
    model.darg.enter_positive_dcr()
    return history, historical.capture_rng_state(), update_audits, log_records


def author_epoch_metrics(logp, rows, labels):
    selected = rows.detach().cpu().numpy().astype(np.int64)
    truth = labels.detach().cpu().numpy()[selected]
    prediction = logp.detach().argmax(1).cpu().numpy()[selected]
    return acc_auc(truth, prediction, 3)


def author_unsquared_aloss(aux, labels, train_rows):
    """Recompute the original unsquared observation with six train-known graphs."""
    with torch.no_grad():
        y = labels.detach()
        rows = train_rows.detach().long()
        masked = torch.zeros_like(y)
        train_labels = y.index_select(0, rows)
        masked[rows] = torch.where(train_labels == 0, -1, train_labels)
        target = masked[:, None].eq(masked[None, :]).to(aux["raw"].dtype)
        eye = torch.eye(len(y), device=y.device, dtype=target.dtype)
        target = normalize_adj(target + eye)
        values = []
        for adjacency in aux["adjacencies"]:
            graph_eye = torch.eye(len(adjacency), device=adjacency.device, dtype=adjacency.dtype)
            normalized = normalize_adj(adjacency.detach() + graph_eye)
            values.append(torch.linalg.norm(normalized - target))
        if len(values) != len(MODAL_DIMS):
            raise RuntimeError("Author ALoss observation must cover all six TADPOLE modalities")
        return float(torch.stack(values).mean().item())


def evaluate_without_training_side_effects(model, x, labels, train_rows, pair, same,
                                           *, darg_warmup):
    rng = historical.capture_rng_state()
    training_flags = [(module, module.training) for module in model.modules()]
    graph_attributes = []
    if model.darg is not None:
        for graph in model.gfr.graphs:
            graph_attributes.append((graph, {
                name: getattr(graph, name) for name in ("score431", "last_score", "last_mask")
                if hasattr(graph, name)
            }))
        core_last = model.darg.core.last
    else:
        core_last = None
    try:
        model.eval()
        with torch.no_grad():
            result = model(x, labels, train_rows, pair, same, dpsl_enabled=False,
                           darg_warmup=darg_warmup)[0]
        return result
    finally:
        if model.darg is not None:
            for graph, saved in graph_attributes:
                for name in ("score431", "last_score", "last_mask"):
                    if hasattr(graph, name):
                        delattr(graph, name)
                for name, value in saved.items():
                    setattr(graph, name, value)
            model.darg.core.last = core_last
        for module, training in training_flags:
            module.training = training
        historical.restore_rng(rng)


def _graph_supervision_value(aux, labels, rows, model, kind, coefficient=None):
    if kind == "none":
        return aux["raw"].new_zeros(())
    return historical_loss(aux, labels, rows, model, kind, coefficient)[2]


def initial_warmup_identity(model, optimizers):
    shared_prefixes = ("gfr.", "cam.", "darg.")
    shared_state = {key: value for key, value in model.state_dict().items()
                    if key.startswith(shared_prefixes)}
    shared_optimizers = {key: optimizers[key].state_dict() for key in ("base", "cam", "darg")
                         if key in optimizers}
    return {
        "initial_model_sha256": sha_object(model.state_dict()),
        "initial_optimizer_sha256": sha_object({key: value.state_dict() for key, value in optimizers.items()}),
        "initial_rng_sha256": sha_object(historical.capture_rng_state()),
        "initial_shared_darg_cam_gfr_sha256": sha_object(shared_state),
        "initial_shared_optimizer_sha256": sha_object(shared_optimizers),
    }


def configured_supervision_coefficient(config, lambda_fold=None):
    value = config.get("supervision_coefficient", 0.0) if lambda_fold is None else lambda_fold
    if value in (None, "fold_calibrated"):
        return None
    return float(value)


def graphlearn_parameters(model):
    parameters = []
    named = dict(model.named_parameters())
    for index, graph in enumerate(model.gfr.graphs):
        name = next(name for name, parameter in named.items() if parameter is graph.p.weight)
        parameters.append((name, graph.p.weight))
    return parameters


def graph_gradient_audit(model, aux):
    records = []
    for index, (name, parameter) in enumerate(graphlearn_parameters(model)):
        adjacency = aux["adjacencies"][index].detach()
        off_diagonal = adjacency.clone()
        off_diagonal.fill_diagonal_(0)
        edge_count = int((off_diagonal != 0).sum().item())
        possible = max(adjacency.shape[0] * (adjacency.shape[0] - 1), 1)
        grad_norm = float(parameter.grad.detach().norm().item()) if parameter.grad is not None else 0.0
        records.append({
            "modality": MODAL_NAMES[index], "parameter": name,
            "off_diagonal_edge_count": edge_count,
            "graph_density": float(edge_count / possible),
            "gradient_norm": grad_norm,
            "gradient_finite": bool(parameter.grad is not None and torch.isfinite(parameter.grad).all()),
            "status": ("STRUCTURAL_ZERO_GRADIENT" if edge_count == 0 and grad_norm == 0.0
                       else "VALID" if grad_norm > 0.0 and torch.isfinite(parameter.grad).all()
                       else "INVALID_ZERO_GRADIENT"),
        })
    return records


def validate_aloss_gradient_audit(records):
    invalid = [item["modality"] for item in records if item["status"] == "INVALID_ZERO_GRADIENT"]
    if invalid:
        raise RuntimeError(f"ALoss gradient missing for effective graph modalities: {invalid}")


def calibrate_aloss(aux, labels, train_rows, model, target_ratio=0.1,
                    lower=1e-5, upper=1e-2):
    params = [parameter for _, parameter in graphlearn_parameters(model)]
    _, base_total, _, _ = historical_loss(aux, labels, train_rows, model, "none")
    aloss, detail = paper_aloss(aux, labels, train_rows, model)
    base_grads = torch.autograd.grad(base_total, params, retain_graph=True, allow_unused=True)
    aloss_grads = torch.autograd.grad(aloss, params, retain_graph=True, allow_unused=True)
    base_parts, aloss_parts, per_modality = [], [], []
    for index, (base_grad, aloss_grad) in enumerate(zip(base_grads, aloss_grads)):
        adjacency = aux["adjacencies"][index].detach()
        off_diagonal = adjacency.clone()
        off_diagonal.fill_diagonal_(0)
        edge_count = int((off_diagonal != 0).sum().item())
        base_norm = float(base_grad.detach().norm()) if base_grad is not None else 0.0
        aloss_norm = float(aloss_grad.detach().norm()) if aloss_grad is not None else 0.0
        if edge_count > 0 and (aloss_grad is None or not torch.isfinite(aloss_grad).all()
                               or aloss_norm == 0.0):
            raise RuntimeError(f"ALoss has an invalid gradient for {MODAL_NAMES[index]}")
        if base_grad is not None:
            base_parts.append(base_grad.detach().reshape(-1))
        if aloss_grad is not None:
            aloss_parts.append(aloss_grad.detach().reshape(-1))
        per_modality.append({
            "modality": MODAL_NAMES[index], "off_diagonal_edge_count": edge_count,
            "base_gradient_norm": base_norm, "aloss_gradient_norm": aloss_norm,
            "status": ("STRUCTURAL_ZERO_GRADIENT" if edge_count == 0 and aloss_norm == 0.0
                       else "VALID"),
        })
    base_vector = torch.cat(base_parts) if base_parts else aloss.new_zeros(0)
    aloss_vector = torch.cat(aloss_parts) if aloss_parts else aloss.new_zeros(0)
    base_norm = float(base_vector.norm())
    aloss_norm = float(aloss_vector.norm())
    if not np.isfinite(base_norm) or not np.isfinite(aloss_norm) or aloss_norm <= 0.0:
        raise RuntimeError("ALoss calibration gradient is non-finite or zero")
    lambda_raw = float(target_ratio * base_norm / (aloss_norm + 1e-12))
    lambda_fold = float(np.clip(lambda_raw, lower, upper))
    actual_ratio = float(lambda_fold * aloss_norm / (base_norm + 1e-12))
    return lambda_fold, {
        "target_ratio": float(target_ratio), "lambda_raw": lambda_raw,
        "lambda_fold": lambda_fold, "clip_lower": float(lower), "clip_upper": float(upper),
        "base_gradient_norm": base_norm, "aloss_gradient_norm": aloss_norm,
        "actual_initial_gradient_ratio": actual_ratio,
        "clipped": bool(lambda_fold != lambda_raw),
        "per_modality": per_modality,
        "formula": "lambda_raw=target_ratio*||g_base||/(||g_ALoss||+1e-12)",
        "target_sha256": sha_object((train_rows.detach().cpu(),
                                      labels.index_select(0, train_rows).detach().cpu(),
                                      detail["target"].detach().cpu())),
    }


def _vector_from_parameters(before, after, names):
    values = []
    for name in names:
        delta = after[name].detach().reshape(-1) - before[name].detach().reshape(-1)
        values.append(delta)
    return torch.cat(values) if values else torch.zeros(0)


def _cosine(left, right):
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def counterfactual_update_audit(model, optimizers, x, labels, train_rows, pair, same,
                                config, group, fold, epoch, kind, coefficient,
                                rng_before, *, darg_warmup, dpsl_enabled):
    """Compare complete Adam vectors from independent clones without touching training state."""
    outside_rng = historical.capture_rng_state()
    parameter_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    graph_names = [name for name, _ in graphlearn_parameters(model)]
    # GraphLearn keeps the latest score and mask as non-leaf diagnostic tensors.
    # They are not model state and cannot be deep-copied by current PyTorch.
    for graph in model.gfr.graphs:
        for attribute in ("score431", "last_score", "last_mask"):
            if hasattr(graph, attribute):
                delattr(graph, attribute)
    if model.use_darg:
        model.darg.core.last = {}
    base_model = copy.deepcopy(model).to(x.device)
    supervised_model = copy.deepcopy(model).to(x.device)
    base_opts, _ = make_optimizers(base_model, config, formal=not darg_warmup)
    supervised_opts, _ = make_optimizers(supervised_model, config, formal=not darg_warmup)
    for target, source in ((base_opts, optimizers), (supervised_opts, optimizers)):
        for name, optimizer in source.items():
            if name not in target:
                raise RuntimeError(f"Counterfactual optimizer family missing {name}")
            target[name].load_state_dict(copy.deepcopy(optimizer.state_dict()))
    snapshots = []
    try:
        for variant, clone, clone_opts, variant_kind, variant_coeff in (
                ("base", base_model, base_opts, "none", None),
                ("supervised", supervised_model, supervised_opts, kind, coefficient)):
            historical.restore_rng(rng_before)
            clone.train()
            zero_optimizers(clone_opts)
            _, clone_aux = clone(x, labels, train_rows, pair, same,
                                 dpsl_enabled=dpsl_enabled, darg_warmup=darg_warmup)
            _, clone_total, _, _ = historical_loss(
                clone_aux, labels, train_rows, clone, variant_kind, variant_coeff)
            if clone.dpsl is not None and dpsl_enabled:
                dpsl_optimizer_step(clone, clone_opts, clone_total,
                                    clone_aux["scmr_objectives"],
                                    float(config.get("dpsl_shared_cap", 0.001)))
            else:
                clone_total.backward()
                step_optimizers(clone_opts)
            snapshots.append((variant, {name: value.detach().clone()
                                         for name, value in clone.named_parameters()}))
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        base_after = snapshots[0][1]
        supervised_after = snapshots[1][1]
        u_base = _vector_from_parameters(before, base_after, parameter_names)
        u_supervised = _vector_from_parameters(before, supervised_after, parameter_names)
        delta = u_supervised - u_base
        graph_base = _vector_from_parameters(before, base_after, graph_names)
        graph_supervised = _vector_from_parameters(before, supervised_after, graph_names)
        graph_delta = graph_supervised - graph_base
        eps = 1e-12
        return {
            "epoch": int(epoch), "group": group, "fold": int(fold),
            "kind": kind, "coefficient": coefficient or 0.0,
            "darg_warmup": bool(darg_warmup), "dpsl_enabled": bool(dpsl_enabled),
            "optimizer_state_sha256": sha_object({name: value.state_dict() for name, value in optimizers.items()}),
            "u_base_norm": float(u_base.norm()),
            "u_supervised_norm": float(u_supervised.norm()),
            "u_difference_norm": float(delta.norm()),
            "u_difference_relative": float(delta.norm() / (u_base.norm() + eps)),
            "u_cosine": _cosine(u_supervised, u_base),
            "graph_p_weight_u_base_norm": float(graph_base.norm()),
            "graph_p_weight_u_supervised_norm": float(graph_supervised.norm()),
            "graph_p_weight_u_difference_norm": float(graph_delta.norm()),
            "graph_p_weight_u_difference_relative": float(graph_delta.norm() / (graph_base.norm() + eps)),
            "graph_p_weight_u_cosine": _cosine(graph_supervised, graph_base),
            "parameter_count": int(u_base.numel()),
            "graph_p_weight_count": int(graph_base.numel()),
        }
    finally:
        del base_model, supervised_model, base_opts, supervised_opts
        historical.restore_rng(outside_rng)


def make_warmup_payload(fold, model, optimizers, history, rng_after, pair, same,
                        initial, config, train_idx, labels, update_audits, log_records):
    kind = active_supervision_kind(config, 1, warmup=True)
    return {
        "fold": int(fold), "source": "E83.8.3 exact 10-step original-graph warmup",
        "epochs": 10, "history": history, **initial,
        "author_log_records": log_records,
        "base_state_sha256": sha_object(model.state_dict()),
        "optimizer_state_sha256": sha_object({key: value.state_dict() for key, value in optimizers.items()}),
        "shared_darg_cam_gfr_state_sha256": sha_object({
            key: value for key, value in model.state_dict().items()
            if key.startswith(("gfr.", "cam.", "darg."))}),
        "shared_optimizer_state_sha256": sha_object({
            key: optimizers[key].state_dict() for key in ("base", "cam", "darg")
            if key in optimizers}),
        "rng_state_sha256": sha_object(rng_after),
        "pair_sha256": sha_object((pair.detach().cpu(), same.detach().cpu())),
        "train_index_sha256": hashlib.sha256(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest(),
        "train_labels_sha256": hashlib.sha256(
            np.asarray(labels, dtype=np.int64)[np.asarray(train_idx, dtype=np.int64)].tobytes()
        ).hexdigest(),
        "graph_supervision": kind != "none",
        "supervision_kind": kind,
        "supervision_coefficient": configured_supervision_coefficient(config) or 0.0,
        "graph_supervision_schedule": config.get("graph_supervision_schedule", "none"),
        "res_only": bool(config.get("res_only")),
        "base_weight_decay": float(config.get("initial_base_weight_decay",
                                              0.0036 if config.get("low_base_weight_decay") else 0.11)),
        "fallback_modal": model.gfr.fallback_modal.tolist(),
        "update_vector_audits": update_audits,
    }


def compare_warmup(group, fold, config, warmup, pair, same, train_idx, labels):
    reference = config.get("warmup_reference")
    if not reference:
        return {"status": "NOT_REQUIRED"}
    source_dir = (N3_REFERENCE_ROOT / "groups" / "N3" if group == "Y0"
                  else ROOT / "groups" / reference)
    best_path = source_dir / "best_checkpoints" / f"fold{fold:02d}.pt"
    if not best_path.is_file():
        raise RuntimeError(f"Paired warmup source {reference} is missing for fold {fold}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    source_warmup = best.get("warmup")
    if source_warmup is None:
        raise RuntimeError(f"Paired warmup source {reference} lacks a complete warmup payload")
    source_history = [{key: item[key] for key in ("epoch", "loss")}
                      for item in source_warmup.get("history", [])]
    current_history = [{key: item[key] for key in ("epoch", "loss")}
                       for item in warmup.get("history", [])]
    initial_fields = ("initial_model_sha256", "initial_optimizer_sha256", "initial_rng_sha256")
    differences = [key for key in initial_fields if warmup.get(key) != source_warmup.get(key)]
    source_train_idx = np.asarray(best.get("train_idx", []), dtype=np.int64)
    requested_train_idx = np.asarray(train_idx, dtype=np.int64)
    source_labels_sha = hashlib.sha256(
        np.asarray(labels, dtype=np.int64)[source_train_idx].tobytes()
    ).hexdigest()
    if not np.array_equal(source_train_idx, requested_train_idx):
        differences.append("train_idx")
    if source_warmup.get("train_labels_sha256") is not None and warmup.get("train_labels_sha256") != source_labels_sha:
        differences.append("train_labels_sha256")
    pairs_exact = (best.get("relation_pairs") is not None
                   and torch.equal(best["relation_pairs"].cpu(), pair.detach().cpu())
                   and torch.equal(best["relation_targets"].cpu(), same.detach().cpu()))
    if not pairs_exact:
        differences.append("relation_pairs")
    if differences:
        return {"status": "FAIL", "reference": reference,
                "field_differences": differences, "relation_pairs_exact": pairs_exact,
                "reference_checkpoint_sha256": sha_file(best_path)}
    shared_fields = ("shared_darg_cam_gfr_state_sha256", "shared_optimizer_state_sha256",
                     "rng_state_sha256", "pair_sha256")
    shared_differences = [key for key in shared_fields
                          if warmup.get(key) != source_warmup.get(key)]
    if current_history != source_history:
        shared_differences.append("history")
    if warmup.get("base_state_sha256") != source_warmup.get("base_state_sha256"):
        shared_differences.append("base_state_sha256")
    if warmup.get("optimizer_state_sha256") != source_warmup.get("optimizer_state_sha256"):
        shared_differences.append("optimizer_state_sha256")
    if shared_differences:
        return {"status": "FAIL", "reference": reference,
                "field_differences": shared_differences, "relation_pairs_exact": True,
                "reference_checkpoint_sha256": sha_file(best_path)}
    return {"status": "PASS", "reference": reference,
            "initial_model_sha256": warmup["initial_model_sha256"],
            "initial_optimizer_sha256": warmup["initial_optimizer_sha256"],
            "initial_rng_sha256": warmup["initial_rng_sha256"],
            "post_warmup_model_sha256": warmup["base_state_sha256"],
            "post_warmup_optimizer_sha256": warmup["optimizer_state_sha256"],
            "post_warmup_rng_sha256": warmup["rng_state_sha256"],
            "relation_pairs_exact": True,
            "training_labels_sha256": warmup.get("train_labels_sha256", source_labels_sha),
            "history_exact": current_history == source_history,
            "shared_state_exact": not shared_differences,
            "initial_fields_missing_from_reference": [],
            "reference_checkpoint_sha256": sha_file(best_path)}


def compare_warmup_spectral_state(group, fold, config, spectral_stats):
    reference = config.get("warmup_reference")
    source_dir = (N3_REFERENCE_ROOT / "groups" / "N3" if group == "Y0"
                  else ROOT / "groups" / reference)
    best_path = source_dir / "best_checkpoints" / f"fold{fold:02d}.pt"
    if not best_path.is_file():
        raise RuntimeError(f"Paired spectral reference {reference} is missing for fold {fold}")
    source_best = torch.load(best_path, map_location="cpu", weights_only=False)
    source_stats = source_best.get("spectral_stats")

    maximum_difference = 0.0

    def equivalent(left, right, path="spectral_stats"):
        nonlocal maximum_difference
        if torch.is_tensor(left):
            if not torch.is_tensor(right):
                return False
            left_cpu = left.detach().cpu()
            right_cpu = right.detach().cpu()
            if left_cpu.shape != right_cpu.shape or left_cpu.dtype != right_cpu.dtype:
                return False
            if torch.equal(left_cpu, right_cpu):
                return True
            if left_cpu.is_floating_point():
                difference = float((left_cpu - right_cpu).abs().max())
                maximum_difference = max(maximum_difference, difference)
                return difference <= 1e-12
            return False
        if isinstance(left, np.ndarray):
            if not isinstance(right, np.ndarray) or left.shape != right.shape or left.dtype != right.dtype:
                return False
            if np.array_equal(left, right):
                return True
            if np.issubdtype(left.dtype, np.floating):
                difference = float(np.max(np.abs(left - right)))
                maximum_difference = max(maximum_difference, difference)
                return difference <= 1e-12
            return False
        if isinstance(left, dict):
            return (isinstance(right, dict) and left.keys() == right.keys()
                    and all(equivalent(left[key], right[key], f"{path}.{key}") for key in left))
        if isinstance(left, (list, tuple)):
            return (isinstance(right, type(left)) and len(left) == len(right)
                    and all(equivalent(a, b, f"{path}[{index}]")
                            for index, (a, b) in enumerate(zip(left, right))))
        return type(left) is type(right) and left == right

    def normalized(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalized(item) for item in value]
        if isinstance(value, tuple):
            return tuple(normalized(item) for item in value)
        return value

    prompt_path = prompt_path_for(config, BASE_GROUPS[config["base_reference"]])
    prompt_sha = sha_file(prompt_path)
    if not equivalent(spectral_stats, source_stats):
        raise RuntimeError(f"{group} fold {fold} warmup DPSL spectral statistics differ from {reference}")
    if source_best.get("prompt_sha256") != prompt_sha:
        raise RuntimeError(f"{group} fold {fold} warmup DPSL prompt vectors differ from {reference}")
    canonical_stats = normalized(source_stats)
    report = {
        "spectral_statistics_exact": True,
        "spectral_statistics_sha256": sha_object(canonical_stats),
        "recomputed_spectral_statistics_within_tolerance": True,
        "recomputed_spectral_statistics_max_abs_difference": maximum_difference,
        "canonical_spectral_source": reference,
        "prompt_vector_file_sha256": prompt_sha,
        "prompt_vectors_exact": True,
        "reference": reference,
        "reference_checkpoint_sha256": sha_file(best_path),
    }
    return canonical_stats, report


def identity(group, fold, train_idx, test_idx, config):
    config_path = ROOT / "configs" / f"{group}.json"
    seal = json.loads((ROOT / "audits/training_seal.json").read_text())
    prompt = prompt_path_for(config, BASE_GROUPS[config["base_reference"]])
    return {
        "experiment": "Experiment 83.8.3", "group": group, "fold": int(fold), "seed": SEED,
        "model_type": "FidelityModel", "base_reference": config["base_reference"],
        "implementation_sha256": seal["implementation_sha256"],
        "group_config_sha256": sha_file(config_path), "prompt_sha256": sha_file(prompt),
        "data_sha256": DATA_SHA, "split_sha256": SPLIT_SHA,
        "dpsl_shared_cap": float(config.get("dpsl_shared_cap", 0.0)),
        "supervision_kind": config.get("supervision_kind", "none"),
        "supervision_coefficient": (config.get("supervision_coefficient")
                                     if isinstance(config.get("supervision_coefficient"), str)
                                     else float(config.get("supervision_coefficient", 0.0))),
        "graph_supervision_schedule": config.get("graph_supervision_schedule", "none"),
        "intermediate_base_weight_decay": config.get("intermediate_base_weight_decay"),
        "train_idx": np.asarray(train_idx, dtype=np.int64).tolist(),
        "test_idx": np.asarray(test_idx, dtype=np.int64).tolist(),
    }


def make_dirs(group):
    directory = ROOT / "groups" / group
    for name in ("best_checkpoints", "latest_checkpoints", "predictions", "audits"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    return directory


def active_supervision_kind(config, epoch: int, *, warmup: bool = False) -> str:
    kind = config.get("supervision_kind", "none")
    if kind == "paper_aloss":
        schedule = config.get("graph_supervision_schedule", "epoch_11_onward")
        if schedule == "epoch_1_onward_including_warmup":
            return "paper_aloss"
        if warmup or epoch <= PAIR_WARMUP_EPOCHS:
            return "none"
        return "paper_aloss"
    return "none"


def supervision_target_sha(target):
    if target is None:
        return None
    if isinstance(target, dict) and "target_sha256" in target:
        return target["target_sha256"]
    if isinstance(target, torch.Tensor):
        return sha_tensor(target)
    return sha_object(target)


def serializable_supervision_details(target):
    if target is None:
        return None
    result = {}
    for key, value in target.items():
        if key == "target":
            continue
        result[key] = value
    return result


def graph_gradient_norms(model):
    values = []
    for graph in model.gfr.graphs:
        gradients = [parameter.grad.detach().norm() for parameter in graph.parameters()
                     if parameter.grad is not None]
        values.append(float(torch.stack(gradients).norm()) if gradients else 0.0)
    return values


def train_fold(group, fold, x_np, y_np, ids, split, device, log):
    config = group_config(group)
    directory = make_dirs(group)
    best_path = directory / "best_checkpoints" / f"fold{fold:02d}.pt"
    latest_path = directory / "latest_checkpoints" / f"fold{fold:02d}.pt"
    audit_path = directory / "audits" / f"fold{fold:02d}.json"
    train_idx = np.asarray(split["train_idx"], dtype=np.int64)
    test_idx = np.asarray(split["test_idx"], dtype=np.int64)
    artifact = identity(group, fold, train_idx, test_idx, config)
    saved = torch.load(latest_path, map_location="cpu", weights_only=False) if latest_path.is_file() else None
    if saved is not None:
        differences = [key for key, value in artifact.items() if saved.get(key) != value]
        if differences:
            raise RuntimeError(f"Latest identity differs for {group} fold {fold}: {differences}")
        recover_best(best_path, saved["best_sha256"])
    elif best_path.exists():
        raise RuntimeError(f"Unsealed Best exists without its Latest state at {best_path}")

    historical.seed_everything(SEED)
    x = torch.from_numpy(x_np).to(device)
    labels = torch.from_numpy(y_np).long().to(device)
    train_t = torch.from_numpy(train_idx).long().to(device)
    stats = saved["spectral_stats"] if saved is not None else None
    model = build_model(config, fold, device, spectral_stats=stats)
    optimizers, optimizer_config = make_optimizers(model, config, formal=saved is not None)
    elapsed_before = float(saved.get("elapsed_training_seconds", 0.0)) if saved is not None else 0.0
    started = time.monotonic()
    pair = same = warmup = None
    warmup_pair_audit = {"status": "NOT_APPLICABLE"}
    history = []
    author_epoch_log_records = []
    gradient_budget_history = []
    update_vector_audits = []
    lambda_fold = configured_supervision_coefficient(config)
    aloss_calibration = None
    best_acc, best_epoch, wait, best_sha = -1.0, -1, 0, None
    start_epoch = 1

    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        if model.use_darg:
            model.darg.enter_positive_dcr()
        if optimizer_config != saved["optimizer_config"] or optimizers.keys() != saved["optimizers"].keys():
            raise RuntimeError("Latest optimizer contract differs from the sealed group config")
        for name, state in saved["optimizers"].items():
            optimizers[name].load_state_dict(state)
        pair, same = saved["relation_pairs"], saved["relation_targets"]
        if pair is not None:
            pair, same = pair.to(device), same.to(device)
        warmup, history = saved["warmup"], saved["history"]
        author_epoch_log_records = saved.get("author_epoch_log_records", warmup.get("author_log_records", []))
        warmup_pair_audit = saved["warmup_pair_audit"]
        gradient_budget_history = saved.get("gradient_budget_history", [])
        update_vector_audits = saved.get("update_vector_audits", [])
        lambda_fold = saved.get("lambda_fold", lambda_fold)
        aloss_calibration = saved.get("aloss_calibration")
        best_acc, best_epoch, wait, best_sha = (saved[key] for key in
                                                ("best_acc", "best_epoch", "wait", "best_sha256"))
        start_epoch = int(saved["epoch"]) + 1
        historical.restore_rng(saved["rng"])
        log.write(f"RESUME group={group} fold={fold:02d} next_epoch={start_epoch}\n")
    elif model.use_darg:
        supports = historical.anchor_pair_support(model, x)
        pair_np, same_np = historical.make_pairs(train_idx, y_np, fold, supports)
        pair = torch.from_numpy(pair_np).long().to(device)
        same = torch.from_numpy(same_np).float().to(device)
        initial = initial_warmup_identity(model, optimizers)
        test_t = torch.from_numpy(test_idx).long().to(device)
        warmup_history, rng_after, warmup_updates, warmup_logs = run_warmup(
            model, optimizers, x, labels, train_t, pair, same, config, fold, group, test_t)
        warmup = make_warmup_payload(fold, model, optimizers, warmup_history, rng_after,
                                     pair, same, initial, config, train_idx, y_np,
                                     warmup_updates, warmup_logs)
        author_epoch_log_records.extend(warmup_logs)
        update_vector_audits.extend(warmup_updates)
        warmup_pair_audit = compare_warmup(group, fold, config, warmup, pair, same,
                                           train_idx, y_np)
        if warmup_pair_audit["status"] == "FAIL":
            raise RuntimeError(f"Warmup pairing failed for {group} fold {fold}")
        if model.use_dpsl:
            model.eval()
            with torch.no_grad():
                _, warm_aux = model(x, labels, train_t, pair, same, dpsl_enabled=False)
            stats = fit_spectral_statistics(warm_aux["cam"], labels, train_t, fold=fold,
                                            warm_logits=warm_aux["native_logits"])
            stats["warmup_cam"] = warm_aux["cam"].detach().cpu()
            stats["warmup_logits"] = warm_aux["native_logits"].detach().cpu()
            stats, spectral_audit = compare_warmup_spectral_state(group, fold, config, stats)
            warmup_pair_audit.update(spectral_audit)
            model.attach_dpsl(stats, str(prompt_path_for(config, BASE_GROUPS[config["base_reference"]])),
                              824602 * 100 + fold)
            optimizers["dpsl"] = torch.optim.AdamW(model.dpsl.parameters(), lr=0.0005, weight_decay=0.01)
            optimizer_config["dpsl"] = [0.0005, 0.01]
        if config.get("intermediate_base_weight_decay") is not None:
            optimizer_config = apply_intermediate_weight_decay(optimizers, config, optimizer_config)
        if historical.sha_object(historical.capture_rng_state()) != warmup["rng_state_sha256"]:
            raise RuntimeError(f"RNG changed after warmup before formal training in {group} fold {fold}")
        start_epoch = 11

    def persist(epoch, new_best=None):
        state = {
            **artifact, "epoch": int(epoch), "model": model.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
            "optimizer_config": optimizer_config, "rng": historical.capture_rng_state(),
            "best_epoch": best_epoch, "best_acc": best_acc, "best_sha256": best_sha,
            "wait": wait, "history": history, "spectral_stats": stats,
            "gradient_budget_history": gradient_budget_history,
            "update_vector_audits": update_vector_audits,
            "lambda_fold": lambda_fold,
            "aloss_calibration": aloss_calibration,
            "warmup": warmup, "warmup_pair_audit": warmup_pair_audit,
            "relation_pairs": pair, "relation_targets": same,
            "author_epoch_log_records": author_epoch_log_records,
            "elapsed_training_seconds": elapsed_before + time.monotonic() - started,
        }
        return publish_epoch(latest_path, best_path, state, new_best)

    if saved is None:
        persist(start_epoch - 1)
    model.train()
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if wait > PATIENCE:
            break
        zero_optimizers(optimizers)
        rng_before = historical.capture_rng_state()
        _, aux = model(x, labels, train_t, pair, same, dpsl_enabled=True)
        graph_kind = active_supervision_kind(config, epoch)
        effective_coefficient = configured_supervision_coefficient(config, lambda_fold)
        classification, historical_total, graph_sup, target = historical_loss(
            aux, labels, train_t, model, graph_kind, effective_coefficient)
        total = historical_total + aux["dpsl_loss"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite loss group={group} fold={fold} epoch={epoch}")
        audit_epoch = epoch == 11
        if audit_epoch and not any(item.get("epoch") == epoch for item in update_vector_audits):
            update_vector_audits.append(counterfactual_update_audit(
                model, optimizers, x, labels, train_t, pair, same, config,
                group, fold, epoch, graph_kind, effective_coefficient, rng_before,
                darg_warmup=False, dpsl_enabled=True))
        if model.dpsl is not None:
            budget = dpsl_optimizer_step(
                model, optimizers, historical_total, aux["scmr_objectives"],
                float(config.get("dpsl_shared_cap", 0.001)),
            )
        else:
            historical_total.backward()
            step_optimizers(optimizers)
            budget = {"families": {}, "shared_cap": 0.0}
        current_graph_gradient_audit = graph_gradient_audit(model, aux)
        if graph_kind == "paper_aloss":
            validate_aloss_gradient_audit(current_graph_gradient_audit)
        model.eval()
        with torch.no_grad():
            eval_logp, eval_aux = model(x, labels, train_t, pair, same, dpsl_enabled=True)
            prediction_all = eval_logp.argmax(1)
            test_acc = float((prediction_all[test_idx] == labels[test_idx]).sum()) / len(test_idx)
        model.train()
        train_metrics = author_epoch_metrics(eval_logp, train_t, labels)
        val_metrics = author_epoch_metrics(eval_logp, torch.from_numpy(test_idx).to(device), labels)
        author_epoch_log_records.append({
            "epoch": epoch, "trn_loss_mf_1": float(classification.detach()),
            "trn_loss_mf_2": float(classification.detach()),
            "trn_loss_mf_3": author_unsquared_aloss(aux, labels, train_t),
            "trn_auc": train_metrics["AUC"], "trn_acc": train_metrics["ACC"],
            "val_auc": val_metrics["AUC"], "val_acc": val_metrics["ACC"],
        })
        gradient_budget_history.append({"epoch": epoch, **budget})
        history.append({
            "epoch": epoch, "loss": float(total.detach()), "classification": float(classification.detach()),
            "graph": float(aux["graph_loss"].detach()), "graph_supervision": float(graph_sup.detach()),
            "pair": float(aux["pair_loss"].detach()), "adv": float(aux["adv_loss"].detach()),
            "preserve": float(aux["preserve_loss"].detach()), "dpsl": float(aux["dpsl_loss"].detach()),
            "test_acc": test_acc, "supervision_kind": graph_kind,
            "supervision_coefficient": effective_coefficient or 0.0,
            "aloss_lambda_fold": lambda_fold,
            "graph_gradient_norms": graph_gradient_norms(model),
            "graph_gradient_audit": current_graph_gradient_audit,
            "aloss_calibration": aloss_calibration,
            "update_vector_audit_epoch": (epoch if audit_epoch else None),
            "supervision_details": serializable_supervision_details(target),
        })
        new_best = None
        if test_acc > best_acc:
            best_acc, best_epoch, wait = test_acc, epoch, 0
            best_predictions = prediction_all[test_idx].detach().cpu().numpy().astype(np.int64)
            best_probabilities = eval_logp[test_idx].exp().detach().cpu().numpy()
            best_metrics = acc_auc(y_np[test_idx], best_predictions, 3)
            best_metrics["AUC_probability"] = float(roc_auc_score(
                np.eye(3, dtype=np.float64)[y_np[test_idx]], best_probabilities,
                average="macro", multi_class="ovr"))
            new_best = {
                **artifact, "epoch": epoch, "best_val_acc": best_acc,
                "model": copy.deepcopy(model.state_dict()),
                "optimizers": {name: copy.deepcopy(opt.state_dict()) for name, opt in optimizers.items()},
                "optimizer_config": optimizer_config, "history_at_best": history[-1],
                "spectral_stats": stats, "warmup": warmup,
                "warmup_pair_audit": warmup_pair_audit,
                "relation_pairs": pair, "relation_targets": same,
                "dpsl_ema_updates": int(model.dpsl.ema_updates) if model.dpsl is not None else 0,
                "gradient_budget": budget,
                "gradient_budget_history": copy.deepcopy(gradient_budget_history),
                "update_vector_audits": copy.deepcopy(update_vector_audits),
                "lambda_fold": lambda_fold,
                "aloss_calibration": copy.deepcopy(aloss_calibration),
                "metrics_at_best": best_metrics,
                "prediction_sha256": hashlib.sha256(best_predictions.tobytes()).hexdigest(),
                "graph_supervision_target_sha256": supervision_target_sha(target),
                "supervision_kind": graph_kind,
                "supervision_details": serializable_supervision_details(target),
            }
        else:
            wait += 1
        if model.dpsl is not None:
            model.dpsl.advance_ema(eval_aux["cam"])
        if new_best is not None or epoch % 10 == 0 or epoch == MAX_EPOCHS or wait > PATIENCE:
            best_sha = persist(epoch, new_best)
        if wait > PATIENCE:
            break

    if best_sha is None or best_epoch < 1:
        raise RuntimeError("No Best checkpoint was selected")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    differences = [key for key, value in artifact.items() if best.get(key) != value]
    if differences:
        raise RuntimeError(f"Best checkpoint identity differs: {differences}")
    model.load_state_dict(best["model"], strict=True)
    model.eval()
    with torch.no_grad():
        logp, aux = model(x, labels, train_t, pair, same, dpsl_enabled=True)
        native_logp = (model(x, labels, train_t, pair, same, dpsl_enabled=False)[0]
                       if model.dpsl is not None else None)
        direct_off_logp = (model(x, labels, train_t, pair, same, dpsl_enabled=True,
                                 darg_intervention="direct_off")[0]
                           if model.use_darg else None)
        disease_neutral_logp = (model(x, labels, train_t, pair, same, dpsl_enabled=True,
                                      darg_intervention="disease_neutral")[0]
                                if model.use_darg else None)
    probabilities = logp.exp().detach().cpu().numpy()
    predictions = logp.argmax(1).cpu().numpy()
    metrics = acc_auc(y_np[test_idx], predictions[test_idx], 3)
    metrics["AUC_probability"] = float(roc_auc_score(
        np.eye(3, dtype=np.float64)[y_np[test_idx]], probabilities[test_idx],
        average="macro", multi_class="ovr"))
    pred_sha = hashlib.sha256(predictions[test_idx].astype(np.int64).tobytes()).hexdigest()
    if metrics != best["metrics_at_best"] or pred_sha != best["prediction_sha256"]:
        raise RuntimeError("Best checkpoint re-evaluation differs from the selected prediction")
    test_truth = y_np[test_idx]
    full_pred = predictions[test_idx]
    native_pred = native_logp.argmax(1).cpu().numpy() if native_logp is not None else None
    direct_pred = direct_off_logp.argmax(1).cpu().numpy() if direct_off_logp is not None else None
    neutral_pred = disease_neutral_logp.argmax(1).cpu().numpy() if disease_neutral_logp is not None else None
    def intervention_counts(changed):
        if changed is None:
            return {"changed": None, "corrected": None, "damaged": None}
        full_correct = full_pred == test_truth
        changed_correct = changed[test_idx] == test_truth
        return {"changed": int((full_pred != changed[test_idx]).sum()),
                "corrected": int((~full_correct & changed_correct).sum()),
                "damaged": int((full_correct & ~changed_correct).sum())}
    def top_margin(logits):
        values = logits.detach().cpu()
        top = values.topk(2, dim=1).values
        return float((top[:, 0] - top[:, 1])[test_idx].mean())
    message_norms = []
    if aux.get("messages") is not None:
        raw_norm = aux["raw"].norm(dim=1).clamp_min(1e-12)
        message_norms = [float((message.norm(dim=1) / raw_norm).mean()) for message in aux["messages"]]
    budget_caps = [float(item.get("shared_cap", 0.0)) for item in gradient_budget_history]
    budget_actual = []
    for item in gradient_budget_history:
        family = item.get("families", {}).get("cam", {})
        if "total_applied" in family and "historical" in family:
            budget_actual.append(float(family["total_applied"]) / max(float(family["historical"]), 1e-30))
    table = {
        "fold": fold, "seed": SEED, "sample_index": test_idx,
        "sample_id": ids.iloc[test_idx]["RID"].to_numpy(), "true_label": y_np[test_idx],
        "pred_label": predictions[test_idx], "prob_NC": probabilities[test_idx, 0],
        "prob_MCI": probabilities[test_idx, 1], "prob_AD": probabilities[test_idx, 2],
    }
    if native_logp is not None:
        native_prob = native_logp.exp().detach().cpu().numpy()
        native_pred = native_logp.argmax(1).cpu().numpy()
        table.update(native_pred_label=native_pred[test_idx], native_prob_NC=native_prob[test_idx, 0],
                     native_prob_MCI=native_prob[test_idx, 1], native_prob_AD=native_prob[test_idx, 2])
    prediction_path = directory / "predictions" / f"fold{fold:02d}.csv"
    pd.DataFrame(table).to_csv(prediction_path, index=False)
    audit = {
        **artifact, "status": "TRAINED", "checkpoint_sha256": sha_file(best_path),
        "latest_sha256": sha_file(latest_path), "best_epoch": best_epoch,
        "latest_epoch": int(torch.load(latest_path, map_location="cpu", weights_only=False)["epoch"]),
        "ACC": metrics["ACC"], "AUC_original": metrics["AUC"],
        "AUC_probability": metrics["AUC_probability"],
        "OOF_correct": int((predictions[test_idx] == y_np[test_idx]).sum()),
        "prediction_sha256": pred_sha,
        "native_ACC": (acc_auc(y_np[test_idx], native_pred[test_idx], 3)["ACC"]
                       if native_logp is not None else None),
        "pair_mean_graph_supervision": config.get("supervision_kind") in ("cb_pair_mean", "cb_uncertainty_pair_mean"),
        "graph_supervision_final": float(graph_sup.detach()) if graph_kind != "none" else 0.0,
        "supervision_kind_final": graph_kind,
        "supervision_coefficient": (config.get("supervision_coefficient")
                                     if isinstance(config.get("supervision_coefficient"), str)
                                     else float(config.get("supervision_coefficient", 0.0))),
        "lambda_fold": lambda_fold,
        "aloss_calibration": aloss_calibration,
        "update_vector_audits": update_vector_audits,
        "aloss_formula": ("mean_m squared Frobenius norm on train-train normalized graphs"
                           if config.get("supervision_kind") == "paper_aloss" else None),
        "aloss_observation_author_unsquared": author_unsquared_aloss(aux, labels, train_t),
        "graph_gradient_norms_final": (history[-1]["graph_gradient_norms"] if history else []),
        "graph_gradient_norms_at_best": best.get("history_at_best", {}).get("graph_gradient_norms", []),
        "graph_gradient_audit_at_best": best.get("history_at_best", {}).get("graph_gradient_audit", []),
        "train_indices_sha256": hashlib.sha256(train_idx.tobytes()).hexdigest(),
        "test_indices_sha256": hashlib.sha256(test_idx.tobytes()).hexdigest(),
        "relation_pairs_sha256": sha_object((pair.detach().cpu(), same.detach().cpu()))
        if pair is not None else None,
        "res_only": bool(config.get("res_only")), "pen_present": bool(model.cam.use_pen),
        "darg_active": bool(model.use_darg), "dpsl_active": bool(model.dpsl is not None),
        "dpsl_residual_max": (float(aux["dpsl"].get("residual_ratio", torch.zeros(1)).max())
                              if model.dpsl is not None else 0.0),
        "dpsl_confidence_protection": bool(config.get("dpsl_confidence_protection")),
        "dpsl_confidence_gate_mean": (float(aux["dpsl"]["confidence_gate"].mean())
                                      if model.dpsl is not None and "confidence_gate" in aux["dpsl"] else None),
        "dpsl_confidence_gate_min": (float(aux["dpsl"]["confidence_gate"].min())
                                     if model.dpsl is not None and "confidence_gate" in aux["dpsl"] else None),
        "dpsl_confidence_gate_max": (float(aux["dpsl"]["confidence_gate"].max())
                                     if model.dpsl is not None and "confidence_gate" in aux["dpsl"] else None),
        "darg_modalities": list(model.darg.modalities) if model.darg is not None else [],
        "darg_gate_scales": list(model.darg.gate_scales) if model.darg is not None else [],
        "dpsl_ema_updates": int(model.dpsl.ema_updates) if model.dpsl is not None else 0,
        "native_full_logit_max_abs": (float((native_logp - logp).abs()[test_idx].max())
                                       if native_logp is not None else None),
        "native_full_flip_count": (int((native_pred[test_idx] != full_pred).sum())
                                    if native_pred is not None else None),
        "native_full_corrected": (int(((native_pred[test_idx] != test_truth)
                                        & (full_pred == test_truth)).sum())
                                   if native_pred is not None else None),
        "native_full_damaged": (int(((native_pred[test_idx] == test_truth)
                                      & (full_pred != test_truth)).sum())
                                 if native_pred is not None else None),
        "native_top1_top2_margin_mean": (top_margin(native_logp) if native_logp is not None else None),
        "full_top1_top2_margin_mean": top_margin(logp),
        "direct_off": intervention_counts(direct_pred),
        "disease_neutral": intervention_counts(neutral_pred),
        "dpsl_shared_cap_configured": float(config.get("dpsl_shared_cap", 0.0)),
        "dpsl_shared_cap_observed_max": max(budget_caps) if budget_caps else 0.0,
        "dpsl_shared_gradient_ratio_max": max(budget_actual) if budget_actual else 0.0,
        "graph_message_relative_norm_mean_by_modality": message_norms,
        "graph_supervision_schedule": config.get("graph_supervision_schedule", "none"),
        "formal_graph_supervision_enabled": graph_kind != "none",
        "weight_decay_schedule": {
            "warmup": float(warmup.get("base_weight_decay", 0.11)) if warmup else None,
            "formal": float(optimizer_config["gfr"][1]),
        },
        "warmup": warmup, "warmup_pair_audit": warmup_pair_audit,
        "author_epoch_log_records": author_epoch_log_records,
        "pair_count": int(len(pair)) if pair is not None else 0,
        "same_pairs": int(same.sum()) if same is not None else 0,
        "different_pairs": int((same == 0).sum()) if same is not None else 0,
        "train_count": int(len(train_idx)), "test_count": int(len(test_idx)),
        "last_epoch": int(torch.load(latest_path, map_location="cpu", weights_only=False)["epoch"]),
        "stop_wait": wait, "optimizer_config": optimizer_config,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "train_seconds": elapsed_before + time.monotonic() - started,
        "author_epoch_log_count": len(author_epoch_log_records),
        "author_epoch_log": author_epoch_log_records,
        "checkpoint": str(best_path), "latest_checkpoint": str(latest_path),
        "prediction": str(prediction_path),
    }
    write_json(audit_path, audit)
    return audit


def execute_fold(group, fold):
    require_training_gate([group])
    configure_worker()
    x, y, ids, splits = load_data()
    split = next(item for item in splits if int(item["fold"]) == int(fold))
    directory = make_dirs(group)
    with lock(directory / "latest_checkpoints" / f"fold{fold:02d}.lock"):
        audit_path = directory / "audits" / f"fold{fold:02d}.json"
        if audit_path.is_file():
            record = json.loads(audit_path.read_text())
            if (record.get("status") == "TRAINED"
                    and record.get("implementation_sha256") == json.loads(
                        (ROOT / "audits/training_seal.json").read_text())["implementation_sha256"]
                    and sha_file(directory / "best_checkpoints" / f"fold{fold:02d}.pt") == record["checkpoint_sha256"]
                    and sha_file(directory / "latest_checkpoints" / f"fold{fold:02d}.pt") == record["latest_sha256"]):
                return {"group": group, "fold": fold, "status": "VERIFIED_PREVIOUS"}
            raise RuntimeError(f"Existing audit for {group} fold {fold} does not match its checkpoints")
        torch.cuda.reset_peak_memory_stats(0)
        return train_fold(group, fold, x, y, ids, split, torch.device("cuda:0"), NullWriter())


class NullWriter:
    def write(self, value):
        return len(value)

    def flush(self):
        return None


def summarize_group(group):
    directory = ROOT / "groups" / group
    rows = [json.loads(path.read_text()) for path in sorted((directory / "audits").glob("fold*.json"))]
    rows = [row for row in rows if row.get("status") == "TRAINED"]
    fields = ["group", "fold", "seed", "train_count", "test_count", "best_epoch", "ACC",
              "AUC_original", "OOF_correct", "checkpoint", "latest_checkpoint", "prediction", "train_seconds"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(directory / "fold_metrics.csv", lambda output: output.write(stream.getvalue().encode()))
    summary = {"group": group, "folds": len(rows), "data_sha256": DATA_SHA,
               "split_sha256": SPLIT_SHA, "status": "COMPLETE" if len(rows) == 10 else "INCOMPLETE"}
    if len(rows) == 10:
        for metric in ("ACC", "AUC_original"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std_population"] = float(values.std(ddof=0))
        summary["OOF_correct"] = int(sum(row["OOF_correct"] for row in rows))
        summary["AUC_probability_mean"] = float(np.mean([row.get("AUC_probability", np.nan) for row in rows]))
    write_json(directory / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--folds", nargs="*", type=int)
    args = parser.parse_args()
    allowed = set(FORMAL_GROUPS)
    if not set(args.groups) <= allowed:
        parser.error("Main-run groups must be registered E83.8.3 variants")
    if not 1 <= args.workers <= 8:
        parser.error("The global GPU worker limit is eight")
    folds = args.folds or list(range(1, 11))
    if not set(folds) <= set(range(1, 11)):
        parser.error("Fold IDs must be between 1 and 10")
    require_training_gate(args.groups)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Formal execution requires the assigned single GPU")
    failures = []
    with lock(ROOT / "audits/training.lock"):
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            tasks = {pool.submit(execute_fold, group, fold): (group, fold)
                     for group in args.groups for fold in folds}
            for future in as_completed(tasks):
                group, fold = tasks[future]
                try:
                    result = future.result()
                    print(f"{result['status']} {group} fold {fold:02d}", flush=True)
                except Exception as exc:
                    failures.append((group, fold))
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                    print(f"ERROR {group} fold {fold:02d}: {exc!r}", file=sys.stderr, flush=True)
        for group in args.groups:
            summarize_group(group)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
