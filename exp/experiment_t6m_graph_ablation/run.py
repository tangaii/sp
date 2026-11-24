"""Train one SPromptGL TADPOLE variant and save fold-level evidence."""

import argparse
from collections import Counter
import importlib.metadata
import json
import hashlib
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import (
    CLASS_NAMES, DEFAULTS, MMGPL_CLASS_PROMPTS, MMGPL_CONCEPTS, MMGPL_PAPER, PROVENANCE,
    PUBLIC_PROMPTS, TADPOLE_PROMPTS, VARIANTS,
)
from prepare import sha256, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / 'src'))
from network import CLIP_WEIGHTS, GFR, PLN, prompt_features
from utils import ClsLoss, GraphConstructLoss, my_weight_init

CLASSES = np.arange(len(CLASS_NAMES))


def read_json(path):
    return json.loads(path.read_text())


def initialize_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(config, device):
    data_dir = HERE / 'inputs'
    csv_path = data_dir / 'processed_standard_data.csv'
    dict_path = data_dir / 'modal_feat_dict.npy'
    frame = pd.read_csv(csv_path)
    modal_features = np.load(dict_path, allow_pickle=True).item()
    columns = [feature for names in modal_features.values() for feature in names]
    if columns != frame.columns[:-1].tolist() or frame.columns[-1] != 'LABEL':
        raise ValueError('The feature matrix and modality dictionary disagree')
    labels = frame.LABEL.to_numpy(dtype=np.int64) - 1
    if len(frame) != 598 or np.bincount(labels).tolist() != [209, 315, 74]:
        raise ValueError('Unexpected TADPOLE cohort')
    features = torch.from_numpy(frame[columns].to_numpy()).float().to(device)
    dims = [len(names) for names in modal_features.values()]
    info = {
        'paths': {str(path): sha256(path) for path in (csv_path, dict_path)},
        'modalities': list(modal_features), 'dimensions': dims,
        'samples': len(labels), 'label_counts': dict(zip(CLASS_NAMES, np.bincount(labels).tolist())),
        'row_metadata': str(HERE / 'data/samples.csv'),
    }
    return features, labels, dims, info


def all_folds(labels, config):
    splitter = StratifiedKFold(n_splits=config.folds, random_state=config.split_seed, shuffle=True)
    return list(splitter.split(labels, labels))


def experiment_manifest(variant, config, source_prompts, device, data_info):
    sources = [HERE / name for name in ('run.py', 'config.py', 'prepare.py')]
    sources += sorted(
        path for path in (HERE / 'src').rglob('*')
        if path.is_file() and path.suffix in {'.py', '.gz'}
    )
    packages = ['torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'torchvision']
    return {
        'variant': variant,
        'hyperparameters': vars(config),
        'provenance': PROVENANCE,
        'environment': {
            'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name) for name in packages},
            'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'cuda_build': torch.version.cuda,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'torch_cpu_threads': torch.get_num_threads(),
        },
        'inputs': data_info,
        'mmgpl_prompt_reference': {
            'path': str(HERE / MMGPL_PAPER),
            'sha256': sha256(HERE / MMGPL_PAPER),
            'figure': 'Figure 6',
        },
        'prompts': {name: list(group) for name, group in source_prompts.items()},
        'clip_weights': {'path': str(CLIP_WEIGHTS), 'sha256': sha256(CLIP_WEIGHTS)},
        'source_sha256': {str(path.relative_to(HERE)): sha256(path) for path in sources},
        'metric_definitions': {
            'acc': 'argmax accuracy',
            'auc_hard': 'macro one-vs-rest ROC AUC from one-hot argmax predictions',
            'auc_probability': 'macro one-vs-rest ROC AUC from softmax probabilities',
            'aggregate': 'unweighted mean and population standard deviation over outer folds',
        },
    }


def save_or_check(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Existing configuration or inputs differ at {path}')
    else:
        write_json(path, value)


def evaluate(labels, predictions, probabilities):
    targets = np.eye(len(CLASSES))[labels]
    decisions = np.eye(len(CLASSES))[predictions]
    confusion = confusion_matrix(labels, predictions, labels=CLASSES)
    true_positive = confusion.diagonal()
    support = confusion.sum(axis=1)
    negative_support = confusion.sum() - support
    false_positive = confusion.sum(axis=0) - true_positive
    return {
        'acc': float(np.mean(predictions == labels)),
        'auc_hard': float(roc_auc_score(targets, decisions)),
        'auc_probability': float(roc_auc_score(targets, probabilities)),
        'sensitivity': dict(zip(CLASS_NAMES, (true_positive / support).tolist())),
        'specificity': dict(zip(CLASS_NAMES, (1 - false_positive / negative_support).tolist())),
        'confusion': confusion.tolist(),
    }


RELATIONS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def relation_masks(train_labels):
    masks = {}
    for left, right in RELATIONS:
        mask = train_labels[:, None].eq(left) & train_labels[None, :].eq(right)
        if left != right:
            mask |= train_labels[:, None].eq(right) & train_labels[None, :].eq(left)
        mask.fill_diagonal_(False)
        masks[(left, right)] = mask
    return masks


def relation_losses(scores, train_index, train_labels):
    masks = relation_masks(train_labels)
    values = {}
    for score in scores:
        selected = score[train_index[:, None], train_index[None, :]]
        for relation, mask in masks.items():
            target = 1.0 if relation[0] == relation[1] else 0.0
            values.setdefault(f'{relation[0]}_{relation[1]}', []).append(
                (selected[mask] - target).square().mean()
            )
    return {key: torch.stack(value).mean() for key, value in values.items()}


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def named_parameter_hashes(module):
    return {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in sorted(module.named_parameters())
    }


def shared_parameter_hashes(module):
    prefixes = ('InputLayer.', 'encode_', 'feed_', 'Outputlayer.')
    return {name: value for name, value in named_parameter_hashes(module).items()
            if name.startswith(prefixes)}


def graph_supervision(scores, train_index, targets, mode, train_labels=None):
    if mode == 'class_balanced':
        masks = relation_masks(train_labels)
        modal_values = []
        for score in scores:
            train_score = score[train_index[:, None], train_index[None, :]]
            relation_values = {}
            for relation, mask in masks.items():
                target = 1.0 if relation[0] == relation[1] else 0.0
                relation_values[relation] = (train_score[mask] - target).square().mean()
            positive = torch.stack([relation_values[key] for key in RELATIONS[:3]]).mean()
            negative = torch.stack([relation_values[key] for key in RELATIONS[3:]]).mean()
            modal_values.append(0.5 * (positive + negative))
        cb = torch.stack(modal_values).mean()
        return cb
    squared_errors = []
    for score in scores:
        train_score = score[train_index[:, None], train_index[None, :]]
        difference = train_score - targets
        squared_errors.append(difference.square().sum())
    squared_frobenius = torch.stack(squared_errors).mean()
    return squared_frobenius / targets.numel()


def relation_audit(gfr, adjacency, cosine, train_index, targets, all_labels):
    weights = gfr.fusion_weight
    weights = weights.softmax(dim=0) if gfr.fusion == 'softmax_sum' else weights.sigmoid()
    relations = []
    labels = torch.as_tensor(all_labels, device=adjacency[0].device)
    off_diagonal = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same = labels[:, None].eq(labels[None, :]) & off_diagonal
    different = labels[:, None].ne(labels[None, :]) & off_diagonal
    for index, (graph, score) in enumerate(zip(adjacency, cosine)):
        train_graph = graph[train_index[:, None], train_index[None, :]]
        train_score = score[train_index[:, None], train_index[None, :]]
        propagation = gfr.last_propagation_graphs[index]
        positives = targets.bool()
        density = int((graph.bool() & off_diagonal).sum())
        propagation_density = int((propagation.bool() & off_diagonal).sum())
        relations.append({
            'index': index,
            'weight': weights[index].item(),
            'used_for_fusion': not (gfr.fusion == 'legacy' and index == 0),
            'threshold': float(getattr(gfr, f'GraphConstruct{index}').last_threshold),
            'learned_off_diagonal_edge_density': density / (len(graph) * (len(graph) - 1)),
            'propagation_off_diagonal_edge_density': propagation_density / (len(graph) * (len(graph) - 1)),
            'learned_same_label_edge_rate': float(graph[same].bool().float().mean()),
            'learned_different_label_edge_rate': float(graph[different].bool().float().mean()),
            'propagation_same_label_edge_rate': float(propagation[same].bool().float().mean()),
            'propagation_different_label_edge_rate': float(propagation[different].bool().float().mean()),
            'train_pair_mse': (train_score - targets).square().mean().item(),
            'mean_train_cosine_same_label': train_score[positives].mean().item(),
            'mean_train_cosine_different_label': train_score[~positives].mean().item(),
            'train_positive_edge_rate': (train_graph[positives] > 0).float().mean().item(),
            'train_negative_edge_rate': (train_graph[~positives] > 0).float().mean().item(),
            'message_norm': gfr.last_message_norms[index],
            'fusion_weight': gfr.last_fusion_weights[index],
        })
    return relations


def train_fold(fold, config, prompts, features, y, dims, labels, train, test, output, rng_state):
    start = time.monotonic()
    device = features.device
    train_tensor = torch.as_tensor(train, device=device)
    train_labels = y[train_tensor]
    target_edges = train_labels[:, None].eq(train_labels[None, :]).float()
    modalities = list(features.split(dims, dim=1))
    # The original GFR uses a contiguous concatenation of the modality slices.
    raw = torch.cat(modalities, dim=1)
    eye = torch.eye(len(features), device=device, dtype=features.dtype)
    counts = Counter(labels[train])
    if list(counts) != [0, 1, 2]:
        raise AssertionError(f'Legacy class weight order changed: {list(counts)}')
    weights = len(train) / np.array(list(counts.values())) / len(CLASSES)
    class_weights = torch.from_numpy(weights).float().to(device)
    gfr = GFR(dims, config.th, config.fusion, config.alpha, config.graph_mode,
              config.adaptive_quantile, config.st_temperature, config.biomarker_mode).to(device) if config.use_gfr else None
    pln = PLN(dims, config, prompts, device).to(device)
    pln.apply(my_weight_init)
    init_file = HERE / 'initial_states' / f'fold_{fold:02d}.pt'
    if init_file.exists():
        ref = torch.load(init_file, weights_only=True, map_location='cpu')
        if gfr is not None:
            gfr.load_state_dict(ref['gfr'], strict=True)
        common = {k:v for k,v in ref['pln'].items() if k in pln.state_dict() and pln.state_dict()[k].shape == v.shape}
        missing, unexpected = pln.load_state_dict(common, strict=False)
        if any(k.startswith(('InputLayer','encode_','feed_','Outputlayer')) for k in missing):
            raise RuntimeError(f'common initialization missing in {config.variant}: {missing}')
        torch.set_rng_state(ref['cpu_rng'])
        torch.cuda.set_rng_state_all(ref['cuda_rng'])
    else:
        raise FileNotFoundError(f'missing shared initialization {init_file}')
    initial_hashes = {'gfr': state_hash(gfr) if gfr is not None else None, 'pln': state_hash(pln)}
    if gfr is not None:
        optimizer_gfr = torch.optim.Adam(gfr.parameters(), lr=config.lr, weight_decay=config.reg)
    else:
        optimizer_gfr = None
    optimizer_pln = torch.optim.Adam(pln.parameters(), lr=config.lr, weight_decay=config.reg)
    history = []
    best_correct = -1
    best_epoch = None
    model_file = output / 'best.pt'

    for epoch in range(config.epochs):
        if gfr is not None: gfr.train()
        pln.train()
        if optimizer_gfr is not None:
            optimizer_gfr.zero_grad()
        optimizer_pln.zero_grad()
        if gfr is not None:
            fused, graphs, scores = gfr(raw, modalities, eye, train_tensor)
        else:
            fused, graphs, scores = raw, [], []
        log_probs = pln(fused)
        classification = ClsLoss(log_probs, y, train_tensor, class_weights)
        structure = classification.new_zeros(())
        if gfr is not None:
            structure = sum(GraphConstructLoss(x, adj, config.theta_smooth, config.theta_degree, config.theta_sparsity) for x, adj in zip(modalities, graphs)) / 6
        supervision = classification.new_zeros(())
        pair_supervision = classification.new_zeros(())
        cb_supervision = classification.new_zeros(())
        per_relation = {f'{a}_{b}': classification.new_zeros(()) for a,b in RELATIONS}
        if gfr is not None and config.graph_supervision != 'none':
            per_relation = relation_losses(scores, train_tensor, train_labels)
            pair_supervision = graph_supervision(scores, train_tensor, target_edges, 'pair_mean')
            cb_supervision = graph_supervision(scores, train_tensor, target_edges, 'class_balanced', train_labels)
            supervision = (1 - config.beta) * pair_supervision + config.beta * cb_supervision
        loss = config.eta_classification * classification + config.lambda_graph * (structure + supervision)
        loss.backward()
        if epoch == 0:
            gradient_audit = []
            for index in range(6):
                parameter = getattr(gfr, f'GraphConstruct{index}').p.weight if gfr is not None else None
                gradient_audit.append({
                    'relation': index,
                    'finite': bool(parameter is None or (parameter.grad is not None and torch.isfinite(parameter.grad).all())),
                    'mean_abs': float(parameter.grad.detach().abs().mean()) if parameter is not None and parameter.grad is not None else 0.0,
                })
        if optimizer_gfr is not None:
            optimizer_gfr.step()
        optimizer_pln.step()

        if gfr is not None: gfr.eval()
        pln.eval()
        with torch.no_grad():
            fused_eval = gfr(raw, modalities, eye, train_tensor)[0] if gfr is not None else raw
            post_update = pln(fused_eval)
            predictions = post_update.argmax(1).cpu().numpy()
        correct = int(np.count_nonzero(predictions[test] == labels[test]))
        loss_values = torch.stack((classification, structure, pair_supervision, cb_supervision,
                                   supervision, loss, *per_relation.values())).detach().cpu().tolist()
        if not np.isfinite(loss_values).all():
            raise FloatingPointError(f'Non-finite loss at fold {fold} epoch {epoch}')
        trn_acc = float(np.mean(predictions[train] == labels[train]))
        test_acc = correct / len(test)
        history.append([epoch, *loss_values, trn_acc, test_acc])
        if correct > best_correct:
            best_correct = correct
            best_epoch = epoch
            torch.save({
                'fold': fold, 'epoch': epoch, 'gfr': gfr.state_dict() if gfr is not None else None, 'pln': pln.state_dict(),
            }, model_file)

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f'fold {fold:02d} epoch {epoch + 1:03d}/{config.epochs} '
                  f'total_loss={float(loss.detach()):.5f} test_acc={test_acc:.4f}', flush=True)

    checkpoint = torch.load(model_file, weights_only=True, map_location=device)
    if gfr is not None: gfr.load_state_dict(checkpoint['gfr'])
    pln.load_state_dict(checkpoint['pln'])
    with torch.no_grad():
        if gfr is not None:
            fused, adjacency, scores = gfr(raw, modalities, eye, train_tensor)
        else:
            fused, adjacency, scores = raw, [], []
        log_probs = pln(fused)
        predictions = log_probs.argmax(1).cpu().numpy()[test]
        probabilities = log_probs.exp().cpu().numpy()[test]
        relations = relation_audit(gfr, adjacency, scores, train_tensor, target_edges, labels) if gfr is not None else []
        best_relation_losses = relation_losses(scores, train_tensor, train_labels) if gfr is not None and config.graph_supervision != 'none' else {}
        best_structure_losses = [float(GraphConstructLoss(x, adj, config.theta_smooth, config.theta_degree, config.theta_sparsity))
                                 for x, adj in zip(modalities, adjacency)] if gfr is not None else []
    metrics = evaluate(labels[test], predictions, probabilities)
    if metrics['acc'] != best_correct / len(test):
        raise RuntimeError(f'Checkpoint accuracy changed during reload of fold {fold}')
    metrics.update({
        'fold': fold,
        'best_epoch_zero_based': best_epoch,
        'train_size': len(train),
        'test_size': len(test),
        'class_weight_counter_order': list(map(int, counts)),
        'class_weights_in_loss_index_order': weights.tolist(),
        'seconds': time.monotonic() - start,
        'graph_relations': relations,
        'gradient_audit': gradient_audit,
        'initial_state_sha256': initial_hashes,
        'shared_parameter_sha256': shared_parameter_hashes(pln),
        'parameter_count': sum(parameter.numel() for parameter in pln.parameters()) + (sum(parameter.numel() for parameter in gfr.parameters()) if gfr is not None else 0),
        'module_names': list(dict(pln.named_modules())) + ([f'GFR.{name}' for name in dict(gfr.named_modules())] if gfr is not None else []),
        'parameter_names': [f'PLN.{name}' for name, _ in pln.named_parameters()] + ([f'GFR.{name}' for name, _ in gfr.named_parameters()] if gfr is not None else []),
        'buffer_names': [f'PLN.{name}' for name, _ in pln.named_buffers()] + ([f'GFR.{name}' for name, _ in gfr.named_buffers()] if gfr is not None else []),
        'pen_physically_present': any(name.startswith('MLP.') for name, _ in pln.named_parameters()),
        'gfr_physically_present': gfr is not None,
        'best_graph_structure_loss_by_modality': best_structure_losses,
        'best_relation_supervision_mse': {key: float(value.detach()) for key, value in best_relation_losses.items()},
        'latest_relation_supervision_mse': {key: float(value.detach()) for key, value in per_relation.items()},
        'best_checkpoint_sha256': sha256(model_file),
        'rng_origin': rng_state,
    })
    pd.DataFrame(history, columns=[
        'epoch_zero_based', 'classification_loss', 'graph_structure_loss', 'pair_supervision_loss',
        'cb_supervision_loss', 'mixed_supervision_loss', 'total_loss',
        *[f'relation_{left}_{right}_mse' for left, right in RELATIONS],
        'train_acc', 'outer_test_acc',
    ]).to_csv(output / 'history.csv', index=False)
    frame = pd.DataFrame({
        'fold': fold, 'row_index': test, 'label': labels[test], 'prediction': predictions,
    })
    for index, name in enumerate(CLASS_NAMES):
        frame[f'p_{name}'] = probabilities[:, index]
    frame.to_csv(output / 'predictions.csv', index=False, float_format='%.9g')
    torch.save({
        'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
    }, output / 'next_fold_rng.pt')
    write_json(output / 'metrics.json', metrics)
    return metrics


def aggregate(output, results):
    overall = {}
    for metric in ('acc', 'auc_hard', 'auc_probability'):
        values = [result[metric] for result in results]
        overall[metric] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
    for metric in ('sensitivity', 'specificity'):
        overall[metric] = {}
        for name in CLASS_NAMES:
            values = [result[metric][name] for result in results]
            overall[metric][name] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
    predictions = pd.concat(
        [pd.read_csv(output / f'fold_{result["fold"]:02d}' / 'predictions.csv') for result in results],
        ignore_index=True,
    ).sort_values('row_index')
    predictions.to_csv(output / 'out_of_fold_predictions.csv', index=False, float_format='%.9g')
    write_json(output / 'summary.json', {'overall': overall, 'folds': results})


def run(args):
    epochs = args.epochs
    variant = args.variant
    config = SimpleNamespace(**(DEFAULTS | VARIANTS[variant])); config.variant = variant
    if epochs is not None:
        config.epochs = epochs
    folds_to_run = args.folds_to_run or config.folds
    if not 1 <= folds_to_run <= config.folds or config.epochs <= 0:
        raise ValueError('Use positive epoch and fold counts within the configured split')
    full = config.epochs == DEFAULTS['epochs'] and folds_to_run == DEFAULTS['folds']
    output = HERE / 'results' / variant
    config.executed_folds = folds_to_run
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initialize_rng(config.seed)
    features, labels, dims, info = load_data(config, device)
    y = torch.from_numpy(labels).to(device)
    prompt_sets = {
        'public': PUBLIC_PROMPTS,
        'tadpole': TADPOLE_PROMPTS,
        'mmgpl_concepts': MMGPL_CONCEPTS,
        'mmgpl_class_prompts': MMGPL_CLASS_PROMPTS,
    }
    prompts = prompt_sets[config.prompts]
    output.mkdir(parents=True, exist_ok=True)
    manifest = experiment_manifest(variant, config, prompts, device, info)
    config_path = output / 'config.json'
    original_runner = HERE / 'run.py'
    if full and original_runner.is_file() and (output / 'summary.json').is_file():
        # A completed run keeps the exact source hash recorded during training.
        recorded = read_json(config_path)['source_sha256']['run.py']
        if recorded == sha256(original_runner) and all(
            (output / f'fold_{fold:02d}' / 'metrics.json').is_file()
            for fold in range(1, folds_to_run + 1)
        ):
            manifest['source_sha256']['run.py'] = recorded
    save_or_check(config_path, manifest)
    folds = all_folds(labels, config)
    save_or_check(output / 'splits.json', [
        {'fold': index, 'train': train.tolist(), 'test': test.tolist()}
        for index, (train, test) in enumerate(folds, start=1)
    ])

    if args.fold_index:
        fold = args.fold_index
        if not 1 <= fold <= config.folds:
            raise ValueError('fold_index must be in 1..10')
        fold_output = output / f'fold_{fold:02d}'
        if (fold_output / 'metrics.json').is_file():
            print(f'{variant} fold {fold:02d} already complete', flush=True)
            return
        if fold_output.exists():
            raise FileExistsError(f'Incomplete fold directory requires explicit recovery: {fold_output}')
        initialize_rng(config.seed)
        rng_origin = f'shared_reference_fold_{fold:02d}'
        staging = output / f'.fold_{fold:02d}.working'
        staging.mkdir(exist_ok=False)
        metrics = train_fold(fold, config, prompts, features, y, dims, labels,
                             *folds[fold - 1], staging, rng_origin)
        staging.rename(fold_output)
        print(f'{variant} fold {fold:02d} complete epoch={metrics["best_epoch_zero_based"]} '
              f'acc={metrics["acc"]:.4f} hard_auc={metrics["auc_hard"]:.4f}', flush=True)
        return

    if args.parallel:
        pending = [fold for fold in range(1, folds_to_run + 1)
                   if not (output / f'fold_{fold:02d}' / 'metrics.json').is_file()]
        commands = [[sys.executable, str(HERE / 'run.py'), variant, '--fold-index', str(fold),
                     '--epochs', str(config.epochs)] for fold in pending[:10]]
        processes = [subprocess.Popen(command, cwd=HERE) for command in commands]
        failures = [process.wait() for process in processes if process.wait() != 0]
        if failures:
            raise RuntimeError(f'{len(failures)} fold workers failed')
    results = [read_json(output / f'fold_{fold:02d}' / 'metrics.json')
               for fold in range(1, folds_to_run + 1)]
    aggregate(output, results)
    print(f'{variant} complete  {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=tuple(VARIANTS))
    parser.add_argument('--epochs', type=int, help='Smoke test only, unless the full 300 epochs are requested')
    parser.add_argument('--folds-to-run', type=int, help='Run the first N folds for a smoke test')
    parser.add_argument('--parallel', action='store_true', help='Run independent folds concurrently')
    parser.add_argument('--fold-index', type=int, help=argparse.SUPPRESS)
    run(parser.parse_args())
