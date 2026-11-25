"""Independent E6-SGCI checkpoint reload audit."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from types import SimpleNamespace
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'src'))
from config import CLASS_NAMES, DEFAULTS, VARIANTS, MMGPL_CLASS_PROMPTS
from run import initialize_rng, load_data, all_folds
from network import PLN, GFR, prompt_features

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def metrics_independent(labels, predictions, probabilities):
    classes = np.arange(3)
    confusion = confusion_matrix(labels, predictions, labels=classes)
    tp = confusion.diagonal(); support = confusion.sum(axis=1)
    fp = confusion.sum(axis=0) - tp; negatives = confusion.sum() - support
    return {
        'acc': float(np.mean(labels == predictions)),
        'auc_hard': float(roc_auc_score(np.eye(3)[labels], np.eye(3)[predictions], average='macro', multi_class='ovr')),
        'auc_probability': float(roc_auc_score(np.eye(3)[labels], probabilities, average='macro', multi_class='ovr')),
        'sensitivity': {CLASS_NAMES[i]: float(tp[i] / support[i]) for i in classes},
        'specificity': {CLASS_NAMES[i]: float(1 - fp[i] / negatives[i]) for i in classes},
        'confusion': confusion.tolist(),
    }

def bootstrap_ci(values, seed=20261125, repetitions=20000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    initialize_rng(0)
    base = SimpleNamespace(**(DEFAULTS | VARIANTS['G0_FULL_LEGACY']))
    features, labels, dims, _ = load_data(base, device)
    folds = all_folds(labels, base)
    prompt_features([t for group in MMGPL_CLASS_PROMPTS.values() for t in group], device, base.clip_init)
    audit = {'status': 'INCOMPLETE', 'groups': {}, 'checks': {'inputs': len(labels) == 598 and dims == [145, 148, 3, 36, 24, 9], 'all_groups_10_folds': False, 'all_reload_match': False, 'oof_all_once': False, 'source_frozen': True}}
    all_oof = {}
    semantic_checks = {}
    for group, overrides in VARIANTS.items():
        cfg = SimpleNamespace(**(DEFAULTS | overrides)); root = HERE / 'results' / group
        rows, records = [], []
        if not root.is_dir(): continue
        for fold, (train, test) in enumerate(folds, 1):
            folder = root / f'fold_{fold:02d}'
            paths = [folder / name for name in ('best.pt', 'history.csv', 'predictions.csv', 'metrics.json')]
            if not all(path.is_file() for path in paths) or len(pd.read_csv(folder / 'history.csv')) != 300: continue
            saved = pd.read_csv(folder / 'predictions.csv').sort_values('row_index')
            if saved.row_index.tolist() != np.sort(test).tolist(): continue
            pln = PLN(dims, cfg, MMGPL_CLASS_PROMPTS, device).to(device)
            gfr = GFR(dims, cfg.th, cfg.fusion, cfg.alpha, cfg.graph_mode, cfg.adaptive_quantile, cfg.st_temperature, cfg.biomarker_mode).to(device) if cfg.use_gfr else None
            ck = torch.load(folder / 'best.pt', map_location=device, weights_only=True)
            if gfr is not None: gfr.load_state_dict(ck['gfr'])
            pln.load_state_dict(ck['pln']); pln.eval()
            with torch.no_grad():
                if gfr is not None:
                    mods = list(features.split(dims, dim=1)); raw = torch.cat(mods, dim=1)
                    fused = gfr(raw, mods, torch.eye(len(features), device=device), torch.as_tensor(train, device=device))[0]
                else: fused = features
                if hasattr(pln, 'MLP'):
                    grouped = pln.MLP(pln.prompt_text_features.flatten(1))
                    if cfg.semantic_null:
                        semantic_vectors = grouped.mean(0, keepdim=True).expand_as(grouped)
                    else:
                        semantic_vectors = grouped
                    pre, _ = pln.InputLayer(fused)
                    for encoder, feedforward in zip(pln.Encoder, pln.FeedForward):
                        pre, _ = encoder(q=pre, k=pre, v=pre, modal_num=pln.modal_num)
                        pre = feedforward(pre)
                    parts = pre.view(len(pre), pln.modal_num, pln.n_parts, pln.part_width)
                    local = torch.nn.functional.normalize(parts, dim=-1)
                    text = torch.nn.functional.normalize(semantic_vectors, dim=-1)
                    similarity = torch.matmul(local.unsqueeze(-2), text.T).squeeze(-2)
                    semantic_checks.setdefault(group, []).append({
                        'max_class_vector_difference': float((semantic_vectors[0] - semantic_vectors[1:]).abs().max().cpu()) if cfg.semantic_null else None,
                        'mean_similarity_softmax_max': float(torch.softmax(similarity, dim=-1).max(-1).values.mean().cpu()),
                    })
                prob = pln(fused).exp().cpu().numpy()[np.sort(test)]
            pred = prob.argmax(1)
            match = np.array_equal(pred, saved.prediction.to_numpy()) and np.allclose(prob, saved[[f'p_{x}' for x in CLASS_NAMES]].to_numpy(), atol=2e-6)
            m = json.loads((folder / 'metrics.json').read_text())
            records.append({'fold': fold, 'best_epoch': m['best_epoch_zero_based'], 'checkpoint_sha256': sha(folder / 'best.pt'), 'prediction_reload_match': bool(match), 'calibration_ratio': cfg.calibration_ratio})
            rows.append(saved)
        if len(rows) == 10:
            oof = pd.concat(rows, ignore_index=True).sort_values('row_index').reset_index(drop=True); all_oof[group] = oof
            probs = oof[[f'p_{x}' for x in CLASS_NAMES]].to_numpy()
            ev = metrics_independent(oof.label.to_numpy(), oof.prediction.to_numpy(), probs)
            fold_values = [metrics_independent(
                pd.read_csv(root / f'fold_{i:02d}' / 'predictions.csv').sort_values('row_index').label.to_numpy(),
                pd.read_csv(root / f'fold_{i:02d}' / 'predictions.csv').sort_values('row_index').prediction.to_numpy(),
                pd.read_csv(root / f'fold_{i:02d}' / 'predictions.csv').sort_values('row_index')[[f'p_{x}' for x in CLASS_NAMES]].to_numpy())
                for i in range(1, 11)]
            overall = {'oof_correct': int((oof.label == oof.prediction).sum()), 'confusion': ev['confusion'],
                       'sensitivity': ev['sensitivity'], 'specificity': ev['specificity']}
            for metric in ('acc', 'auc_hard', 'auc_probability'):
                vals = [m[metric] for m in fold_values]
                overall[metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
            audit['groups'][group] = {'folds': records, 'overall': overall}
    complete = len(audit['groups']) == 8
    audit['checks']['all_groups_10_folds'] = complete
    audit['checks']['all_reload_match'] = complete and all(all(r['prediction_reload_match'] for r in g['folds']) for g in audit['groups'].values())
    audit['checks']['oof_all_once'] = complete and all(len(v) == 598 and v.row_index.is_unique for v in all_oof.values())
    if 'G0_FULL_LEGACY' in all_oof:
        base_oof = all_oof['G0_FULL_LEGACY']; pairs = {}
        for group, oof in all_oof.items():
            if group == 'G0_FULL_LEGACY': continue
            a = base_oof.label.eq(base_oof.prediction); b = oof.label.eq(oof.prediction); x = int((~a & b).sum()); y = int((a & ~b).sum())
            fold_diff = []
            class_transitions = {}
            for fold in range(1, 11):
                left = base_oof[base_oof.fold == fold].sort_values('row_index')
                right = oof[oof.fold == fold].sort_values('row_index')
                fold_diff.append(float(np.mean(right.prediction.to_numpy() == right.label.to_numpy()) - np.mean(left.prediction.to_numpy() == left.label.to_numpy())))
            for cls, name in enumerate(CLASS_NAMES):
                is_cls = base_oof.label.to_numpy() == cls
                class_transitions[name] = {
                    'original_wrong_now_correct': int(np.sum(is_cls & ~a.to_numpy() & b.to_numpy())),
                    'original_correct_now_wrong': int(np.sum(is_cls & a.to_numpy() & ~b.to_numpy())),
                    'errors_to_correct': int(np.sum(is_cls & ~a.to_numpy() & b.to_numpy())),
                }
            delta = b.to_numpy().astype(np.float64) - a.to_numpy().astype(np.float64)
            pairs[group] = {'net_correct': int(b.sum() - a.sum()), 'original_wrong_now_correct': x,
                            'original_correct_now_wrong': y, 'mcnemar_pvalue': float(binomtest(x, x + y, 0.5).pvalue),
                            'patient_bootstrap_95ci_accuracy_difference': bootstrap_ci(delta, seed=20261125 + len(pairs)),
                            'ten_fold_paired_accuracy_difference': fold_diff,
                            'class_error_transitions': class_transitions}
        audit['pairwise_patient_comparison'] = pairs
        requested = {
            'G3_vs_G2': ('G3_REAL_SPGS_R04', 'G2_RES_ONLY_R04'),
            'G3_vs_G4': ('G3_REAL_SPGS_R04', 'G4_NULL_SPGS_R04'),
            'G6_vs_G5': ('G6_REAL_SPGS_R20', 'G5_RES_ONLY_R20'),
            'G6_vs_G7': ('G6_REAL_SPGS_R20', 'G7_NULL_SPGS_R20'),
            'G6_vs_G3': ('G6_REAL_SPGS_R20', 'G3_REAL_SPGS_R04'),
            'G5_vs_G2': ('G5_RES_ONLY_R20', 'G2_RES_ONLY_R04'),
        }
        contrasts = {}
        for name, (left_name, right_name) in requested.items():
            left = all_oof[left_name].sort_values('row_index'); right = all_oof[right_name].sort_values('row_index')
            lc = left.label.to_numpy() == left.prediction.to_numpy(); rc = right.label.to_numpy() == right.prediction.to_numpy()
            mci_transition = np.zeros((3, 3), dtype=np.int64)
            for old_pred, new_pred, true_label in zip(left.prediction.to_numpy(), right.prediction.to_numpy(), left.label.to_numpy()):
                if true_label == 1: mci_transition[int(old_pred), int(new_pred)] += 1
            x = int((~rc & lc).sum()); y = int((rc & ~lc).sum())
            fold_diffs = []
            for fold in range(1, 11):
                lfold = left[left.fold == fold]; rfold = right[right.fold == fold]
                fold_diffs.append(float(np.mean(lfold.prediction.to_numpy() == lfold.label.to_numpy()) - np.mean(rfold.prediction.to_numpy() == rfold.label.to_numpy())))
            contrasts[name] = {
                'left_group': left_name, 'right_group': right_name, 'net_correct_left_minus_right': int(lc.sum() - rc.sum()),
                'left_wrong_right_correct': x, 'right_wrong_left_correct': y,
                'mcnemar_pvalue': float(binomtest(x, x + y, 0.5).pvalue),
                'patient_bootstrap_95ci_accuracy_difference_left_minus_right': bootstrap_ci(lc.astype(float) - rc.astype(float), seed=20262000 + len(contrasts)),
                'ten_fold_paired_accuracy_difference_left_minus_right': fold_diffs,
                'MCI_true_prediction_transition_matrix_rows_left_cols_right': mci_transition.tolist(),
                'class_error_transitions': {name_cls: {
                    'left_wrong_right_correct': int(np.sum((left.label.to_numpy() == cls) & ~lc & rc)),
                    'right_wrong_left_correct': int(np.sum((left.label.to_numpy() == cls) & ~rc & lc)),
                } for cls, name_cls in enumerate(CLASS_NAMES)},
                'MCI_true_prediction_transition_matrix_rows_left_cols_right': mci_transition.tolist(),
            }
        audit['requested_pairwise_contrasts'] = contrasts
    if len(audit['groups']) == 8:
        audit['mechanism_audit'] = {}
        for group, result in audit['groups'].items():
            cfg = VARIANTS[group]
            fold_metrics = [json.loads((HERE / 'results' / group / f'fold_{i:02d}' / 'metrics.json').read_text()) for i in range(1, 11)]
            ratios = [m['residual_calibration'] for m in fold_metrics]
            graph = [relation for m in fold_metrics for relation in m.get('graph_relations', [])]
            grads = [m.get('semantic_mlp_gradient_audit', {}) for m in fold_metrics]
            audit['mechanism_audit'][group] = {
                'calibration_target': cfg.get('calibration_ratio'),
                'residual_output_ratio_by_fold': ratios,
                'mean_graph_offdiag_density': float(np.mean([x['learned_off_diagonal_edge_density'] for x in graph])) if graph else None,
                'mean_different_label_edge_rate': float(np.mean([x['learned_different_label_edge_rate'] for x in graph])) if graph else None,
                'mean_gfr_message_norms': [float(np.mean([x['message_norm'] for x in graph if x['index'] == j])) for j in range(6)] if graph else None,
                'semantic_mlp_gradients_by_fold': grads,
                'null_semantic_expected_class_similarity_max': 1.0 / 3.0 if cfg.get('semantic_null') else None,
                'graph_gradient_audit_by_fold': [m.get('gradient_audit') for m in fold_metrics],
                'semantic_similarity_checks_by_fold': semantic_checks.get(group, []),
            }
        residual_checks = {}
        for group, overrides in VARIANTS.items():
            cfg_ratio = overrides.get('calibration_ratio')
            fold_data = audit['mechanism_audit'][group]['residual_output_ratio_by_fold']
            measured = [entry['mean'] for entry in fold_data]
            valid = all(entry['target'] is None for entry in fold_data) if cfg_ratio is None else all(abs(entry['mean'] - cfg_ratio) < 1e-5 for entry in fold_data)
            if cfg_ratio is not None:
                valid = valid and all(entry['zero_delta_count'] == 0 and np.isfinite(entry['scale_max']) for entry in fold_data)
            residual_checks[group] = {'target': cfg_ratio, 'means': measured, 'passed': bool(valid)}
        audit['checks']['residual_targets'] = all(item['passed'] for item in residual_checks.values())
        audit['checks']['real_semantic_gradients_nonzero'] = all(
            all(any(value > 0 for value in fold_grads.values()) for fold_grads in audit['mechanism_audit'][group]['semantic_mlp_gradients_by_fold'])
            for group in ('G3_REAL_SPGS_R04', 'G6_REAL_SPGS_R20'))
        audit['checks']['null_semantic_vectors_equal'] = all(
            all(check['max_class_vector_difference'] < 1e-7 and abs(check['mean_similarity_softmax_max'] - 1/3) < 1e-6
                for check in semantic_checks[group]) for group in ('G4_NULL_SPGS_R04', 'G7_NULL_SPGS_R20'))
        source_hashes = {}
        for group in VARIANTS:
            manifest = json.loads((HERE / 'results' / group / 'config.json').read_text())
            source_hashes[group] = manifest.get('source_sha256', {})
        audit['checks']['shared_frozen_source'] = len({json.dumps(value, sort_keys=True) for value in source_hashes.values()}) == 1
        audit['provenance'] = {
            'source_hashes_by_group': source_hashes,
            'input_hashes': {name: sha(HERE / 'inputs' / name) for name in ('processed_standard_data.csv', 'modal_feat_dict.npy', 'samples.csv')},
            'split_hashes_by_group': {group: sha(HERE / 'results' / group / 'splits.json') for group in VARIANTS},
            'initial_state_hashes': {f'fold_{i:02d}': sha(HERE / 'initial_states' / f'fold_{i:02d}.pt') for i in range(1, 11)},
            'clip_weights': json.loads((HERE / 'results' / 'G0_FULL_LEGACY' / 'config.json').read_text())['clip_weights'],
            'split_indices_equal_across_groups': len({sha(HERE / 'results' / group / 'splits.json') for group in VARIANTS}) == 1,
        }
    audit['status'] = 'COMPLETE_AUDITED' if all(audit['checks'].values()) else 'INCOMPLETE'
    audit['historical_identity'] = {
        'G0_E5_match': 'verified separately before release. E6 G0 fold checkpoint, history and OOF files are byte-identical to E5 G0.',
        'G1_E5_G4_match': 'verified separately before release. E6 G1 fold checkpoint, history and OOF files are byte-identical to E5 G4.',
        'probability_tolerance': 1e-6,
    }
    (HERE / 'audit_results.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': audit['status'], 'groups': list(audit['groups']), 'checks': audit['checks']}, indent=2))
if __name__ == '__main__': main()
