"""Independent checkpoint reload and OOF audit for all CBSR groups."""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from config import CLASS_NAMES, DEFAULTS, MMGPL_CLASS_PROMPTS, VARIANTS
from prepare import sha256, write_json
from run import (HERE, all_folds, evaluate, initialize_rng, load_data, read_json,
                graph_supervision)
from network import GFR, PLN, prompt_features
from utils import ClsLoss, GraphConstructLoss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = SimpleNamespace(**(DEFAULTS | VARIANTS['G0_BASELINE']))
    initialize_rng(0)
    features, labels, dims, info = load_data(config, device)
    y = torch.from_numpy(labels).to(device)
    folds = all_folds(labels, config)
    historical_splits = read_json(HERE / 'inputs/splits.json')
    assert len(historical_splits) == 10
    assert len(labels) == 598 and np.bincount(labels).tolist() == [209, 315, 74]
    assert dims == [145, 148, 3, 36, 24, 9]

    prompts = MMGPL_CLASS_PROMPTS
    texts = [text for group in prompts.values() for text in group]
    prompt_features(texts, device, 'frozen_pretrained')
    groups = ['G0_BASELINE', 'G1_CB_FULL_FIXED', 'G2_CB_HALF_FIXED', 'G3_CB_QUARTER_FIXED']
    summaries = {}
    base_initial = {}
    for group in groups:
        root = HERE / 'results' / group
        assert json.loads((root / 'config.json').read_text())['inputs']['paths'] == info['paths']
        assert read_json(root / 'splits.json') == historical_splits
        fold_files = []
        for index, (train, test) in enumerate(folds, start=1):
            folder = root / f'fold_{index:02d}'
            metrics = read_json(folder / 'metrics.json')
            assert metrics['best_epoch_zero_based'] < 300
            assert len(pd.read_csv(folder / 'history.csv')) == 300
            assert len(pd.read_csv(folder / 'predictions.csv')) == len(test)
            if group == 'G0_BASELINE':
                base_initial[index] = metrics['initial_state_sha256']
            else:
                assert metrics['initial_state_sha256'] == base_initial[index]
            checkpoint = torch.load(folder / 'best.pt', map_location=device, weights_only=True)
            gfr = GFR(dims, config.th, config.fusion, config.alpha, config.graph_mode,
                      config.adaptive_quantile, config.st_temperature).to(device)
            pln = PLN(dims, config, prompts, device).to(device)
            gfr.load_state_dict(checkpoint['gfr'])
            pln.load_state_dict(checkpoint['pln'])
            modalities = list(features.split(dims, dim=1))
            raw = torch.cat(modalities, dim=1)
            eye = torch.eye(len(features), device=device, dtype=features.dtype)
            train_tensor = torch.as_tensor(train, device=device)
            train_labels = y[train_tensor]
            edge_targets = train_labels[:, None].eq(train_labels[None, :]).float()
            gfr.zero_grad(set_to_none=True)
            pln.zero_grad(set_to_none=True)
            fused, graphs, scores = gfr(raw, modalities, eye, train_tensor)
            counts = Counter(labels[train])
            weight_values = len(train) / np.array(list(counts.values())) / len(CLASS_NAMES)
            class_weights = torch.from_numpy(weight_values).float().to(device)
            classification = ClsLoss(pln(fused), y, train_tensor, class_weights)
            structure = sum(GraphConstructLoss(x, adj, config.theta_smooth,
                                               config.theta_degree, config.theta_sparsity)
                            for x, adj in zip(modalities, graphs)) / 6
            pair_loss = graph_supervision(scores, train_tensor, edge_targets, 'pair_mean')
            cb_loss = graph_supervision(scores, train_tensor, edge_targets,
                                        'class_balanced', train_labels)
            beta = {'G0_BASELINE': 0.0, 'G1_CB_FULL_FIXED': 1.0,
                    'G2_CB_HALF_FIXED': 0.5, 'G3_CB_QUARTER_FIXED': 0.25}[group]
            (classification + structure + (1 - beta) * pair_loss + beta * cb_loss).backward()
            graphlearn_norms = {}
            for i in range(6):
                grad = getattr(gfr, f'GraphConstruct{i}').p.weight.grad
                assert grad is not None and torch.isfinite(grad).all()
                graphlearn_norms[str(i)] = float(torch.linalg.vector_norm(grad).item())
            metrics['independent_best_checkpoint_graphlearn_gradient_l2'] = graphlearn_norms
            write_json(folder / 'metrics.json', metrics)
            with torch.no_grad():
                fused, _, _ = gfr(raw, modalities, eye, torch.as_tensor(train, device=device))
                log_probs = pln(fused)
                pred = log_probs.argmax(1).cpu().numpy()[test]
                prob = log_probs.exp().cpu().numpy()[test]
            saved = pd.read_csv(folder / 'predictions.csv')
            assert np.array_equal(saved.row_index.to_numpy(), test)
            assert np.array_equal(saved.prediction.to_numpy(), pred)
            assert np.allclose(saved[[f'p_{n}' for n in CLASS_NAMES]].to_numpy(), prob, atol=2e-7)
            assert np.allclose(prob.sum(axis=1), 1.0, atol=1e-6)
            assert np.array_equal(pred, prob.argmax(1))
            fold_files.append(saved.assign(_prob=list(prob)))

        oof = pd.concat(fold_files, ignore_index=True).sort_values('row_index')
        assert len(oof) == 598 and oof.row_index.nunique() == 598
        assert oof.row_index.tolist() == list(range(598))
        y_true = oof.label.to_numpy()
        y_pred = oof.prediction.to_numpy()
        probs = oof[[f'p_{n}' for n in CLASS_NAMES]].to_numpy()
        recalculated = evaluate(y_true, y_pred, probs)
        fold_metrics = [read_json(root / f'fold_{i:02d}/metrics.json') for i in range(1, 11)]
        for metric in ('acc', 'auc_hard', 'auc_probability'):
            values = [m[metric] for m in fold_metrics]
            recalculated[f'{metric}_fold_mean'] = float(np.mean(values))
            recalculated[f'{metric}_fold_std'] = float(np.std(values))
        assert sum(int(m['acc'] * m['test_size']) for m in fold_metrics) == int((y_true == y_pred).sum())
        summaries[group] = {'oof': recalculated, 'folds': fold_metrics}
        write_json(root / 'independent_audit.json', recalculated)

    base_pred = pd.read_csv(HERE / 'results/G0_BASELINE/out_of_fold_predictions.csv').sort_values('row_index')
    comparison = []
    for group in groups:
        root = HERE / 'results' / group
        oof = pd.concat([pd.read_csv(root / f'fold_{i:02d}/predictions.csv') for i in range(1, 11)], ignore_index=True).sort_values('row_index')
        assert np.array_equal(oof.row_index, base_pred.row_index)
        beta = {'G0_BASELINE': 0.0, 'G1_CB_FULL_FIXED': 1.0,
                'G2_CB_HALF_FIXED': 0.5, 'G3_CB_QUARTER_FIXED': 0.25}[group]
        comparison.append({'group': group, 'beta': beta,
                           'acc_mean': summaries[group]['oof']['acc_fold_mean'],
                           'acc_std': summaries[group]['oof']['acc_fold_std'],
                           'hard_auc_mean': summaries[group]['oof']['auc_hard_fold_mean'],
                           'hard_auc_std': summaries[group]['oof']['auc_hard_fold_std'],
                           'prob_auc_mean': summaries[group]['oof']['auc_probability_fold_mean'],
                           'prob_auc_std': summaries[group]['oof']['auc_probability_fold_std'],
                           'oof_correct': int((oof.prediction == oof.label).sum()),
                           'net_correct_vs_g0': int((oof.prediction == oof.label).sum() - (base_pred.prediction == base_pred.label).sum()),
                           'best_epoch_mean': float(np.mean([m['best_epoch_zero_based'] for m in summaries[group]['folds']])),
                           **{f'{metric}_{name}': summaries[group]['oof'][metric][name]
                              for metric in ('sensitivity', 'specificity') for name in CLASS_NAMES}})
    pd.DataFrame(comparison).to_csv(HERE / 'summary.csv', index=False)
    write_json(HERE / 'audit_results.json', {'status': 'PASS', 'groups': comparison,
               'input_hashes': info['paths'], 'source_commit_verified': False,
               'all_folds': 40, 'prediction_reload_match': True})
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
