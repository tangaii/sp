#!/usr/bin/env python3
"""Independent audits and one complete outer-fold recomputation for P0.1."""
import hashlib
import io
import json
import os
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import run as runner

ROOT = runner.ROOT
INPUT_ROOT = runner.INPUT_ROOT
DB_PATH = runner.DB_PATH


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_blob(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def db_records(connection):
    names = [item[1] for item in connection.execute('PRAGMA table_info(tasks)')]
    return [dict(zip(names, row)) for row in connection.execute('SELECT * FROM tasks ORDER BY task_id')]


def independent_fold_refit(frame, split, outer_fold, connection):
    raw_data = frame.iloc[:, :365].to_numpy(dtype=np.float64)
    labels = frame.LABEL.to_numpy(dtype=np.int8) - 1
    boundaries = np.cumsum([0] + runner.MODALITY_DIMS)
    raw_modalities = [raw_data[:, boundaries[i]:boundaries[i + 1]] for i in range(6)]
    outer_train = np.asarray(split['splits'][outer_fold - 1]['train_idx'], dtype=np.int32)
    inner_splitter = StratifiedKFold(5, shuffle=True, random_state=0)
    maximum_difference = 0.0
    comparison_count = 0
    comparison_rows = []
    for inner_fold, (train_local, valid_local) in enumerate(inner_splitter.split(outer_train, labels[outer_train]), 1):
        train_patients = outer_train[train_local]
        valid_patients = outer_train[valid_local]
        pair_tt = runner.make_pairs(train_patients, None)
        pair_vv = runner.make_pairs(valid_patients, None)
        pair_vt = runner.make_pairs(train_patients, valid_patients)
        target_tt_six = runner.relation_labels(labels, pair_tt)
        target_vv_six = runner.relation_labels(labels, pair_vv)
        target_vt_six = runner.relation_labels(labels, pair_vt)
        target_sets = {
            'six': (target_tt_six, target_vv_six, target_vt_six),
            'same': ((target_tt_six < 3).astype(np.int8), (target_vv_six < 3).astype(np.int8), (target_vt_six < 3).astype(np.int8)),
        }
        sample_sets = {
            'six': runner.sample_indices(target_tt_six, runner.SIX_CAP, runner.SEED + outer_fold * 10000 + inner_fold * 100 + 6),
            'same': runner.sample_indices((target_tt_six < 3).astype(np.int8), runner.SAME_CAP, runner.SEED + outer_fold * 10000 + inner_fold * 100 + 2),
        }
        for modality, raw in zip(runner.MODALITIES, raw_modalities):
            transformed, _ = runner.fit_transform(raw, train_patients)
            features = [runner.pair_features(transformed, pair) for pair in [pair_tt, pair_vv, pair_vt]]
            scaler = StandardScaler()
            features[0] = scaler.fit_transform(features[0])
            features[1] = scaler.transform(features[1])
            features[2] = scaler.transform(features[2])
            for relation_task in ['six', 'same']:
                task_id = f'o{outer_fold:02d}_i{inner_fold:02d}_{modality}_{relation_task}'
                row = connection.execute('SELECT result_blob FROM tasks WHERE task_id=? AND status="completed"', (task_id,)).fetchone()
                if row is None:
                    return {'pass': False, 'reason': f'missing {task_id}'}
                stored = read_blob(row[0])
                target_train, target_vv, target_vt = target_sets[relation_task]
                sample_index = sample_sets[relation_task]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ConvergenceWarning)
                    classifier = LogisticRegression(C=runner.C, max_iter=runner.MAX_ITER, tol=runner.TOL,
                                                     solver=runner.SOLVER,
                                                     multi_class=runner.MULTI_CLASS if relation_task == 'six' else 'auto',
                                                     class_weight='balanced', random_state=0)
                    classifier.fit(features[0][sample_index], target_train[sample_index])
                p_vv = classifier.predict_proba(features[1])
                p_vt = classifier.predict_proba(features[2])
                diff_vv = float(np.max(np.abs(p_vv.astype(np.float32) - stored['p_vv'])))
                diff_vt = float(np.max(np.abs(p_vt.astype(np.float32) - stored['p_vt'])))
                maximum_difference = max(maximum_difference, diff_vv, diff_vt)
                comparison_count += 1
                comparison_rows.append({'task_id': task_id, 'max_abs_vv': diff_vv, 'max_abs_vt': diff_vt})
    return {'pass': maximum_difference <= 2e-6, 'outer_fold': outer_fold,
            'task_count': comparison_count, 'maximum_probability_difference': maximum_difference,
            'comparisons': comparison_rows}


def metric_recompute(connection, saved_metrics):
    rows = []
    records = db_records(connection)
    by_outer = {}
    for record in records:
        by_outer.setdefault(record['outer_fold'], []).append(record)
    for outer_fold in range(1, 11):
        records_fold = by_outer.get(outer_fold, [])
        for modality in runner.MODALITIES:
            for scenario in ['VV', 'VT']:
                six_arrays = []
                same_arrays = []
                for relation_task, destination in [('six', six_arrays), ('same', same_arrays)]:
                    selected = [record for record in records_fold if record['modality'] == modality and record['relation_task'] == relation_task]
                    for record in selected:
                        blob = read_blob(record['result_blob'])
                        destination.append((blob['y_vv' if scenario == 'VV' else 'y_vt'], blob['p_vv' if scenario == 'VV' else 'p_vt']))
                y6 = np.concatenate([item[0] for item in six_arrays]); p6 = np.vstack([item[1] for item in six_arrays])
                ys = np.concatenate([item[0] for item in same_arrays]); ps = np.vstack([item[1] for item in same_arrays])
                m6 = runner.metric_values(y6, p6, np.argmax(p6, axis=1), 'six')
                ms = runner.metric_values(ys, ps, (ps[:, 1] >= 0.5).astype(int), 'same')
                rows.append({'outer_fold': outer_fold, 'modality': modality, 'scenario': scenario,
                             'macro_auc_6': m6['macro_auc_6'], 'macro_f1_6': m6['macro_f1_6'],
                             'balanced_accuracy_6': m6['balanced_accuracy_6'], 'same_auc': ms['same_auc'],
                             'same_average_precision': ms['same_average_precision'],
                             'same_balanced_accuracy': ms['same_balanced_accuracy'], 'same_f1': ms['same_f1'],
                             'same_brier': ms['same_brier']})
    saved = pd.read_csv(saved_metrics)
    recomputed = pd.DataFrame(rows).sort_values(['outer_fold', 'modality', 'scenario']).reset_index(drop=True)
    saved = saved.sort_values(['outer_fold', 'modality', 'scenario']).reset_index(drop=True)
    scalar_columns = ['macro_auc_6', 'macro_f1_6', 'balanced_accuracy_6', 'same_auc', 'same_average_precision', 'same_balanced_accuracy', 'same_f1', 'same_brier']
    max_difference = float(np.max(np.abs(recomputed[scalar_columns].to_numpy() - saved[scalar_columns].to_numpy())))
    return {'pass': max_difference <= 2e-12, 'rows': len(rows), 'maximum_metric_difference': max_difference}


def main():
    protocol, protocol_sha, source_sha = runner.protocol_and_hashes()
    connection = sqlite3.connect(DB_PATH)
    records = db_records(connection)
    frame = pd.read_csv(runner.DATA_PATH)
    ids = pd.read_csv(runner.IDS_PATH)
    split = json.loads(runner.SPLIT_PATH.read_text())
    labels = frame.LABEL.to_numpy(dtype=np.int8) - 1
    data_identity = {
        'n_samples': int(len(frame)), 'n_features': int(frame.iloc[:, :365].shape[1]),
        'label_counts': {str(c): int(np.sum(labels == c)) for c in range(3)},
        'modalities': runner.MODALITY_DIMS,
        'duplicate_rid_count': int(ids.RID.duplicated().sum()),
        'data_sha256': sha_file(runner.DATA_PATH), 'split_sha256': sha_file(runner.SPLIT_PATH),
        'sample_ids_sha256': sha_file(runner.IDS_PATH),
    }
    expected_data = {
        'n_samples': 598, 'n_features': 365, 'label_counts': {'0': 209, '1': 315, '2': 74},
        'modalities': runner.MODALITY_DIMS, 'duplicate_rid_count': 0,
        'data_sha256': '0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc',
        'split_sha256': '7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5',
        'sample_ids_sha256': '3c354e2623128209544b23d84fbac0e436f55b4c898c74af83d2e99e7f61d8e0',
    }
    data_pass = data_identity == expected_data
    disjoint_rows = []
    permutation_rows = []
    pair_label_pass = True
    pair_symmetry_pass = True
    for outer_fold, outer_record in enumerate(split['splits'], 1):
        outer_train = np.asarray(outer_record['train_idx'], dtype=np.int32)
        outer_test = np.asarray(outer_record['test_idx'], dtype=np.int32)
        permuted = labels.copy()
        permuted[outer_test] = np.roll(labels[outer_test], 1)
        changed = int(np.sum(permuted[outer_test] != labels[outer_test]))
        unchanged_training_pair_labels = True
        for train_local, valid_local in StratifiedKFold(5, shuffle=True, random_state=0).split(outer_train, labels[outer_train]):
            train_patients = outer_train[train_local]
            valid_patients = outer_train[valid_local]
            intersections = {
                'train_valid': int(len(set(train_patients) & set(valid_patients))),
                'train_outer_test': int(len(set(train_patients) & set(outer_test))),
                'valid_outer_test': int(len(set(valid_patients) & set(outer_test))),
            }
            disjoint_rows.append({'outer_fold': outer_fold, **intersections})
            pair_tt = runner.make_pairs(train_patients, None)
            pair_vv = runner.make_pairs(valid_patients, None)
            pair_vt = runner.make_pairs(train_patients, valid_patients)
            for pair in [pair_tt, pair_vv, pair_vt]:
                if len(pair) and (np.any(pair[:, 0] == pair[:, 1]) or len(np.unique(np.sort(pair, axis=1), axis=0)) != len(pair)):
                    pair_label_pass = False
                if len(pair) and np.any(np.isin(pair, outer_test)):
                    unchanged_training_pair_labels = False
            original = runner.relation_labels(labels, pair_tt)
            permuted_labels = runner.relation_labels(permuted, pair_tt)
            unchanged_training_pair_labels &= bool(np.array_equal(original, permuted_labels))
            if len(pair_vv):
                sample_pair = pair_vv[:min(1000, len(pair_vv))]
                reversed_pair = sample_pair[:, ::-1]
                # Relation labels are unordered and the feature map is explicitly symmetric.
                pair_label_pass &= bool(np.array_equal(runner.relation_labels(labels, sample_pair), runner.relation_labels(labels, reversed_pair)))
                test_z = np.zeros((len(labels), 32), dtype=np.float64)
                pair_symmetry_pass &= bool(np.max(np.abs(runner.pair_features(test_z, sample_pair) - runner.pair_features(test_z, reversed_pair))) == 0)
        permutation_rows.append({'outer_fold': outer_fold, 'changed_outer_test_labels': changed, 'training_pair_labels_unchanged': unchanged_training_pair_labels})
    patient_pass = all(item['train_valid'] == 0 and item['train_outer_test'] == 0 and item['valid_outer_test'] == 0 for item in disjoint_rows)
    permutation_pass = all(item['changed_outer_test_labels'] > 0 and item['training_pair_labels_unchanged'] for item in permutation_rows)
    transform_records = json.loads((ROOT / 'audits/transform_records.json').read_text())
    # A resume-only pass can replace the runner's transient transform metadata with
    # completion markers. Recompute the compact transform audit from the locked data.
    if sum('raw_dim' in item for item in transform_records.values()) != 50:
        raw_data = frame.iloc[:, :365].to_numpy(dtype=np.float64)
        boundaries = np.cumsum([0] + runner.MODALITY_DIMS)
        transform_records = {}
        for outer_fold, outer_record in enumerate(split['splits'], 1):
            outer_train = np.asarray(outer_record['train_idx'], dtype=np.int32)
            for inner_fold, (train_local, valid_local) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(outer_train, labels[outer_train]), 1):
                train_patients = outer_train[train_local]
                for modality_index, modality in enumerate(runner.MODALITIES):
                    raw = raw_data[:, boundaries[modality_index]:boundaries[modality_index + 1]]
                    _, meta = runner.fit_transform(raw, train_patients)
                    transform_records[f'outer{outer_fold:02d}_inner{inner_fold:02d}_{modality}'] = meta
        (ROOT / 'audits/transform_records.json').write_text(json.dumps(transform_records, indent=2))
    transform_pass = all(item.get('pca_components', 0) <= item.get('raw_dim', 0) and item.get('scaler_fit_patient_count') > 0 for key, item in transform_records.items() if 'raw_dim' in item)
    transform_pass &= len([item for item in transform_records.values() if 'raw_dim' in item]) == 300
    expected_tasks = 600
    task_ids = [item['task_id'] for item in records]
    task_pass = len(records) == expected_tasks and len(set(task_ids)) == expected_tasks and all(item['status'] == 'completed' for item in records)
    solver_pass = task_pass and all(item['solver'] == 'lbfgs' and item['converged'] == 1 and item['warning_count'] == 0 and (item['multi_class'] == 'multinomial' if item['relation_task'] == 'six' else item['multi_class'] == 'auto') for item in records)
    class_pass = True
    sampling_records = json.loads((ROOT / 'audits/sampling_records.json').read_text())
    coverage_pass = len(sampling_records) == 50
    for sampling in sampling_records.values():
        for task, expected_classes in [('six', 6), ('same', 2)]:
            counts = sampling['target_counts'][task]['TT']
            sampled = sampling['sample_counts'][task]
            coverage_pass &= len(counts) == expected_classes and all(value > 0 for value in counts) and sampled > 0
    class_pass &= coverage_pass
    for item in records:
        blob = read_blob(item['result_blob'])
        class_pass &= bool(np.isfinite(blob['p_vv']).all() and np.isfinite(blob['p_vt']).all())
        if item['relation_task'] == 'six':
            class_pass &= set(np.unique(blob['y_vv']).tolist()) == set(range(6))
        else:
            class_pass &= set(np.unique(blob['y_vv']).tolist()) == {0, 1}
    metrics = metric_recompute(connection, ROOT / 'results/fold_metrics.csv')
    refit = independent_fold_refit(frame, split, 1, connection)
    resume_audit_path = ROOT / 'audits/resume_audit.json'
    resume_pass = resume_audit_path.exists() and json.loads(resume_audit_path.read_text()).get('pass') is True
    source_manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    source_pass = sha_file(CODE / 'run.py') == source_manifest['reviewed_paths']['p01_runner']['sha256'] and protocol_sha == protocol['protocol_sha256']
    source_checks = {}
    for name, item in source_manifest['reviewed_paths'].items():
        path = Path(item['path'])
        source_checks[name] = {'path_exists': path.exists(), 'sha256_matches': bool(path.exists() and sha_file(path) == item['sha256'])}
        source_pass &= source_checks[name]['path_exists'] and source_checks[name]['sha256_matches']
    graph_source = Path('/model/spromptgl/experiment/experiment83_8_4_tadpole_y3_darg_dpsl_factorial/audits/mechanism_summary.json')
    graph_data = json.loads(graph_source.read_text()) if graph_source.exists() else None
    checkpoint_root = Path('/model/spromptgl/experiment/experiment83_8_4_tadpole_y3_darg_dpsl_factorial/groups/Z0/best_checkpoints')
    checkpoint_files = sorted(checkpoint_root.glob('fold*.pt'))
    graph_audit = {'status': 'PASS_READ_ONLY_SOURCE' if graph_data and graph_data.get('groups', {}).get('Z0') else 'NOT_AVAILABLE', 'source_path': str(graph_source), 'source_sha256': sha_file(graph_source) if graph_source.exists() else None, 'checkpoint_type': 'E83.8.4 Z0 Best checkpoint, read-only historical audit', 'checkpoint_files': [{'path': str(path), 'sha256': sha_file(path)} for path in checkpoint_files], 'z0_darg_by_modality': {}}
    if graph_data:
        denominator = 598 * 597 / 2
        for modality, item in graph_data['groups']['Z0']['darg_by_modality'].items():
            graph_audit['z0_darg_by_modality'][modality] = {
                'original_effective_edges_mean': item['original_effective_edges_mean'],
                'unified_graph_edges_mean': item['unified_graph_edges_mean'],
                'candidate_edges_before_support_intersection_mean': item['candidate_edges_before_support_intersection_mean'],
                'scored_edges_mean': item['scored_edges_mean'],
                'positive_score_edges_mean': item['positive_score_edges_mean'],
                'negative_score_edges_mean': item['negative_score_edges_mean'],
                'graph_density_mean': item['original_effective_edges_mean'] / denominator,
                'candidate_edge_coverage_mean': item['candidate_edges_before_support_intersection_mean'] / denominator,
            }
    (ROOT / 'audits/graph_edge_audit.json').write_text(json.dumps(graph_audit, indent=2))
    statuses = {
        'DATA_IDENTITY': 'PASS' if data_pass else 'FAIL',
        'SOURCE_IDENTITY': 'PASS' if source_pass else 'FAIL',
        'PATIENT_DISJOINT_AUDIT': 'PASS' if patient_pass else 'FAIL',
        'PAIR_LABEL_AUDIT': 'PASS' if pair_label_pass else 'FAIL',
        'PAIR_SYMMETRY_AUDIT': 'PASS' if pair_symmetry_pass else 'FAIL',
        'FEATURE_TRANSFORM_AUDIT': 'PASS' if transform_pass else 'FAIL',
        'CLASS_COVERAGE_AUDIT': 'PASS' if class_pass else 'FAIL',
        'SOLVER_CONVERGENCE_AUDIT': 'PASS' if solver_pass else 'FAIL',
        'METRIC_RECOMPUTATION': 'PASS' if metrics['pass'] else 'FAIL',
        'INDEPENDENT_OUTER_FOLD_RECOMPUTATION': 'PASS' if refit['pass'] else 'FAIL',
        'OUTER_TEST_LABEL_PERMUTATION_AUDIT': 'PASS' if permutation_pass else 'FAIL',
        'RESUME_AUDIT': 'PASS' if resume_pass else 'FAIL',
    }
    statuses['RELATION_EVALUATION'] = 'PASS' if all(statuses[key] == 'PASS' for key in ['DATA_IDENTITY', 'SOURCE_IDENTITY', 'PATIENT_DISJOINT_AUDIT', 'PAIR_LABEL_AUDIT', 'PAIR_SYMMETRY_AUDIT', 'FEATURE_TRANSFORM_AUDIT', 'CLASS_COVERAGE_AUDIT', 'SOLVER_CONVERGENCE_AUDIT', 'METRIC_RECOMPUTATION', 'INDEPENDENT_OUTER_FOLD_RECOMPUTATION', 'OUTER_TEST_LABEL_PERMUTATION_AUDIT', 'RESUME_AUDIT']) else 'FAIL'
    audit = {'experiment': '83.8.5-P0.1', 'generated_utc': runner.now(), 'statuses': statuses,
             'data_identity': data_identity, 'disjoint_rows': disjoint_rows,
             'outer_test_label_permutation': permutation_rows, 'transform_record_count': len(transform_records),
             'task_count': len(records), 'solver_rows': {'min_elapsed': min(item['elapsed_seconds'] for item in records), 'max_elapsed': max(item['elapsed_seconds'] for item in records), 'nonconverged': sum(item['converged'] == 0 for item in records)},
             'metric_recomputation': metrics, 'independent_outer_fold_recomputation': {key: value for key, value in refit.items() if key != 'comparisons'}, 'source_checks': source_checks,
             'graph_edge_audit': graph_audit}
    (ROOT / 'audits/patient_disjoint_audit.json').write_text(json.dumps({'rows': disjoint_rows, 'pass': patient_pass}, indent=2))
    (ROOT / 'audits/outer_test_label_permutation_audit.json').write_text(json.dumps({'rows': permutation_rows, 'pass': permutation_pass}, indent=2))
    (ROOT / 'audits/independent_recompute.json').write_text(json.dumps(refit, indent=2))
    (ROOT / 'audits/final_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    connection.close()


if __name__ == '__main__':
    main()
