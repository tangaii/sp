#!/usr/bin/env python3
"""Experiment 83.8.5-P0.1, patient-disjoint relation probe with resumable tasks."""
import argparse
import hashlib
import io
import json
import os
import sqlite3
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = Path('/model/spromptgl/experiment/experiment83_8_4_tadpole_y3_darg_dpsl_factorial/inputs')
DATA_PATH = INPUT_ROOT / 'processed_standard_data.csv'
SPLIT_PATH = INPUT_ROOT / 'split_indices.json'
IDS_PATH = INPUT_ROOT / 'sample_ids.csv'
DB_PATH = ROOT / 'progress.sqlite'
MODALITIES = ['UCSFFSX', 'UCBERKELEYAV45', 'UPENNBIOMK9', 'RISK_FACTOR', 'COGNITIVE_TEST', 'ROI_AVERAGE']
MODALITY_DIMS = [145, 148, 3, 36, 24, 9]
REL_NAMES = ['NC-NC', 'MCI-MCI', 'AD-AD', 'NC-MCI', 'NC-AD', 'MCI-AD']
SOLVER = 'lbfgs'
MULTI_CLASS = 'multinomial'
MAX_ITER = 1000
TOL = 1e-4
C = 1.0
SIX_CAP = 4000
SAME_CAP = 12000
SEED = 0


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def sha_json(value):
    return sha_bytes(canonical(value))


def now():
    return datetime.now(timezone.utc).isoformat()


def save_array(value):
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def load_array(value):
    return np.load(io.BytesIO(value), allow_pickle=False)


def relation_labels(labels, pair_index):
    a = np.minimum(labels[pair_index[:, 0]], labels[pair_index[:, 1]])
    b = np.maximum(labels[pair_index[:, 0]], labels[pair_index[:, 1]])
    table = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]], dtype=np.int8)
    return table[a, b]


def make_pairs(train_patients, valid_patients):
    if valid_patients is None:
        ii, jj = np.triu_indices(len(train_patients), k=1)
        return np.column_stack((train_patients[ii], train_patients[jj])).astype(np.int32)
    return np.column_stack((np.repeat(valid_patients, len(train_patients)),
                            np.tile(train_patients, len(valid_patients)))).astype(np.int32)


def sample_indices(target, cap, seed):
    rng = np.random.RandomState(seed)
    selected = []
    for label in np.unique(target):
        available = np.flatnonzero(target == label)
        if len(available) > cap:
            available = np.sort(rng.choice(available, size=cap, replace=False))
        selected.append(available)
    return np.sort(np.concatenate(selected).astype(np.int64))


def pair_features(z, pair_index):
    a = z[pair_index[:, 0]]
    b = z[pair_index[:, 1]]
    difference = np.abs(a - b)
    product = a * b
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.divide(np.sum(product, axis=1), denominator,
                       out=np.zeros(len(pair_index), dtype=np.float64),
                       where=denominator > 1e-12)
    features = np.hstack((difference, product, cosine[:, None]))
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def fit_transform(raw, train_patients):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(raw[train_patients])
    rank = int(np.linalg.matrix_rank(train_scaled))
    components = min(32, raw.shape[1], max(1, rank))
    pca = PCA(n_components=components, svd_solver='full', random_state=0)
    pca.fit(train_scaled)
    transformed = pca.transform(scaler.transform(raw))
    if components < 32:
        transformed = np.pad(transformed, ((0, 0), (0, 32 - components)))
    return transformed, {
        'raw_dim': int(raw.shape[1]),
        'pca_components': int(components),
        'train_rank': rank,
        'scaler_fit_patient_count': int(len(train_patients)),
        'pca_svd_solver': 'full',
        'pca_random_state': 0,
    }


def metric_values(y_true, probability, relation_prediction, task):
    if task == 'six':
        prediction = np.argmax(probability, axis=1)
        aucs = [float(roc_auc_score((y_true == c).astype(int), probability[:, c]))
                if np.sum(y_true == c) and np.sum(y_true != c) else np.nan for c in range(6)]
        return {
            'macro_auc_6': float(np.nanmean(aucs)),
            'macro_f1_6': float(f1_score(y_true, prediction, average='macro')),
            'balanced_accuracy_6': float(balanced_accuracy_score(y_true, prediction)),
            'ovr_auc': aucs,
            'support': np.bincount(y_true, minlength=6).tolist(),
            'confusion_matrix': confusion_matrix(y_true, prediction, labels=np.arange(6)).tolist(),
        }
    same = (y_true == 1).astype(int)
    probability_same = probability[:, 1]
    return {
        'same_auc': float(roc_auc_score(same, probability_same)),
        'same_average_precision': float(average_precision_score(same, probability_same)),
        'same_balanced_accuracy': float(balanced_accuracy_score(same, relation_prediction)),
        'same_f1': float(f1_score(same, relation_prediction)),
        'same_brier': float(brier_score_loss(same, probability_same)),
        'support': np.bincount(same, minlength=2).tolist(),
    }


def init_db(connection):
    connection.execute('''CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        outer_fold INTEGER NOT NULL,
        inner_fold INTEGER NOT NULL,
        modality TEXT NOT NULL,
        relation_task TEXT NOT NULL,
        status TEXT NOT NULL,
        input_sha256 TEXT NOT NULL,
        protocol_sha256 TEXT NOT NULL,
        source_sha256 TEXT NOT NULL,
        sample_sha256 TEXT NOT NULL,
        n_train INTEGER,
        n_vv INTEGER,
        n_vt INTEGER,
        solver TEXT,
        multi_class TEXT,
        n_iter TEXT,
        converged INTEGER,
        warning_count INTEGER,
        elapsed_seconds REAL,
        peak_memory_mb REAL,
        result_blob BLOB,
        error TEXT,
        completed_at TEXT
    )''')
    connection.commit()


def completed_row(connection, task_id, input_sha, protocol_sha, source_sha):
    row = connection.execute('SELECT * FROM tasks WHERE task_id=?', (task_id,)).fetchone()
    if row is None:
        return False
    names = [item[1] for item in connection.execute('PRAGMA table_info(tasks)')]
    item = dict(zip(names, row))
    if item['status'] != 'completed' or item['result_blob'] is None:
        return False
    if item['input_sha256'] != input_sha or item['protocol_sha256'] != protocol_sha or item['source_sha256'] != source_sha:
        return False
    try:
        result = np.load(io.BytesIO(item['result_blob']), allow_pickle=False)
        result.close()
    except Exception:
        return False
    return True


def insert_result(connection, result):
    columns = ['task_id', 'outer_fold', 'inner_fold', 'modality', 'relation_task', 'status',
               'input_sha256', 'protocol_sha256', 'source_sha256', 'sample_sha256',
               'n_train', 'n_vv', 'n_vt', 'solver', 'multi_class', 'n_iter', 'converged',
               'warning_count', 'elapsed_seconds', 'peak_memory_mb', 'result_blob', 'error', 'completed_at']
    values = [result.get(column) for column in columns]
    connection.execute('INSERT OR REPLACE INTO tasks (' + ','.join(columns) + ') VALUES (' + ','.join('?' for _ in columns) + ')', values)
    connection.commit()


def task_result_blob(y_vv, p_vv, y_vt, p_vt, metrics_vv, metrics_vt):
    stream = io.BytesIO()
    np.savez_compressed(stream, y_vv=y_vv.astype(np.int8), p_vv=p_vv.astype(np.float32),
                        y_vt=y_vt.astype(np.int8), p_vt=p_vt.astype(np.float32),
                        metrics_vv=np.array(json.dumps(metrics_vv), dtype='U'),
                        metrics_vt=np.array(json.dumps(metrics_vt), dtype='U'))
    return stream.getvalue()


def read_blob(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def protocol_and_hashes():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    source_sha = sha_file(Path(__file__))
    payload = {key: value for key, value in protocol.items() if key != 'protocol_sha256'}
    protocol_sha = sha_json(payload)
    if protocol.get('protocol_sha256') != protocol_sha:
        raise RuntimeError('protocol SHA mismatch')
    if protocol.get('source_sha256') != source_sha:
        raise RuntimeError('runner source SHA mismatch')
    return protocol, protocol_sha, source_sha


def build_inner(outer_train, labels):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return [(outer_train[tr], outer_train[va]) for tr, va in splitter.split(outer_train, labels[outer_train])]


def task_fit(connection, task_id, outer_fold, inner_fold, modality, relation_task,
             raw, labels, train_patients, valid_patients, pair_sets, target_sets,
             sample_sets, feature_sets, protocol_sha, source_sha, input_sha):
    sample_index = sample_sets[relation_task]
    train_features, vv_features, vt_features = feature_sets
    target_train, target_vv, target_vt = target_sets[relation_task]
    selected = train_features[sample_index]
    process = psutil.Process()
    start_rss = process.memory_info().rss
    start = time.perf_counter()
    warning_count = 0
    warning_text = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            classifier = LogisticRegression(C=C, max_iter=MAX_ITER, tol=TOL,
                                             solver=SOLVER,
                                             multi_class=MULTI_CLASS if relation_task == 'six' else 'auto',
                                             class_weight='balanced', random_state=0)
            classifier.fit(selected, target_train[sample_index])
        warning_count = sum(isinstance(item.message, ConvergenceWarning) for item in caught)
        warning_text = [str(item.message) for item in caught if isinstance(item.message, ConvergenceWarning)]
        if relation_task == 'six' and classifier.coef_.shape[0] != 6:
            raise RuntimeError('six-class classifier did not expose six probability columns')
        if relation_task == 'six' and classifier.multi_class != 'multinomial':
            raise RuntimeError('six-class classifier was not configured as multinomial')
        p_vv = classifier.predict_proba(vv_features)
        p_vt = classifier.predict_proba(vt_features)
        if not np.isfinite(p_vv).all() or not np.isfinite(p_vt).all():
            raise RuntimeError('non-finite prediction probability')
        prediction_vv = np.argmax(p_vv, axis=1) if relation_task == 'six' else (p_vv[:, 1] >= 0.5).astype(int)
        prediction_vt = np.argmax(p_vt, axis=1) if relation_task == 'six' else (p_vt[:, 1] >= 0.5).astype(int)
        metrics_vv = metric_values(target_vv, p_vv, prediction_vv, relation_task)
        metrics_vt = metric_values(target_vt, p_vt, prediction_vt, relation_task)
        elapsed = time.perf_counter() - start
        result = {
            'task_id': task_id, 'outer_fold': outer_fold, 'inner_fold': inner_fold,
            'modality': modality, 'relation_task': relation_task, 'status': 'completed',
            'input_sha256': input_sha, 'protocol_sha256': protocol_sha, 'source_sha256': source_sha,
            'sample_sha256': sha_bytes(sample_index.astype(np.int64).tobytes()),
            'n_train': int(len(sample_index)), 'n_vv': int(len(target_vv)), 'n_vt': int(len(target_vt)),
            'solver': SOLVER, 'multi_class': classifier.multi_class,
            'n_iter': json.dumps(np.asarray(classifier.n_iter_).astype(int).tolist()),
            'converged': int(warning_count == 0), 'warning_count': int(warning_count),
            'elapsed_seconds': float(elapsed),
            'peak_memory_mb': float(max(0, process.memory_info().rss - start_rss) / 2**20),
            'result_blob': task_result_blob(target_vv, p_vv, target_vt, p_vt, metrics_vv, metrics_vt),
            'error': json.dumps({'warnings': warning_text}), 'completed_at': now(),
        }
    except Exception as error:
        result = {
            'task_id': task_id, 'outer_fold': outer_fold, 'inner_fold': inner_fold,
            'modality': modality, 'relation_task': relation_task, 'status': 'failed',
            'input_sha256': input_sha, 'protocol_sha256': protocol_sha, 'source_sha256': source_sha,
            'sample_sha256': sha_bytes(sample_index.astype(np.int64).tobytes()),
            'n_train': int(len(sample_index)), 'n_vv': int(len(target_vv)), 'n_vt': int(len(target_vt)),
            'solver': SOLVER, 'multi_class': MULTI_CLASS if relation_task == 'six' else 'auto',
            'n_iter': None, 'converged': 0, 'warning_count': warning_count,
            'elapsed_seconds': float(time.perf_counter() - start),
            'peak_memory_mb': float(max(0, process.memory_info().rss - start_rss) / 2**20),
            'result_blob': None, 'error': repr(error), 'completed_at': now(),
        }
    insert_result(connection, result)
    return result


def aggregate_outer(connection, outer_fold, protocol_sha, source_sha, expected_tasks):
    rows = connection.execute('SELECT * FROM tasks WHERE outer_fold=? AND status="completed" ORDER BY task_id', (outer_fold,)).fetchall()
    names = [item[1] for item in connection.execute('PRAGMA table_info(tasks)')]
    records = [dict(zip(names, row)) for row in rows]
    if len(records) != expected_tasks:
        raise RuntimeError(f'outer fold {outer_fold} has {len(records)} complete tasks, expected {expected_tasks}')
    metrics = []
    pair_rows = []
    conf_rows = []
    for modality in MODALITIES:
        for scenario in ['VV', 'VT']:
            six_parts = []
            same_parts = []
            for relation_task in ['six', 'same']:
                task_records = [r for r in records if r['modality'] == modality and r['relation_task'] == relation_task]
                if len(task_records) != 5:
                    raise RuntimeError(f'missing inner tasks for {outer_fold}/{modality}/{relation_task}')
                arrays = [read_blob(r['result_blob']) for r in task_records]
                if relation_task == 'six':
                    y = np.concatenate([a['y_vv' if scenario == 'VV' else 'y_vt'] for a in arrays])
                    p = np.vstack([a['p_vv' if scenario == 'VV' else 'p_vt'] for a in arrays])
                    six_parts = (y, p)
                else:
                    y = np.concatenate([a['y_vv' if scenario == 'VV' else 'y_vt'] for a in arrays])
                    p = np.vstack([a['p_vv' if scenario == 'VV' else 'p_vt'] for a in arrays])
                    same_parts = (y, p)
            y6, p6 = six_parts
            ys, ps = same_parts
            six_metrics = metric_values(y6, p6, np.argmax(p6, axis=1), 'six')
            same_metrics = metric_values(ys, ps, (ps[:, 1] >= 0.5).astype(int), 'same')
            row = {'outer_fold': outer_fold, 'modality': modality, 'scenario': scenario,
                   **{key: value for key, value in six_metrics.items() if key not in ['ovr_auc', 'support', 'confusion_matrix']},
                   **{key: value for key, value in same_metrics.items() if key != 'support'},
                   'n_pairs': int(len(y6)), 'six_support': six_metrics['support'], 'confusion_matrix': six_metrics['confusion_matrix']}
            metrics.append(row)
            for relation, auc, support in zip(REL_NAMES, six_metrics['ovr_auc'], six_metrics['support']):
                pair_rows.append({'outer_fold': outer_fold, 'modality': modality, 'scenario': scenario,
                                  'relation': relation, 'ovr_auc': auc, 'support': support})
            conf_rows.append({'outer_fold': outer_fold, 'modality': modality, 'scenario': scenario,
                              'matrix': six_metrics['confusion_matrix']})
    return metrics, pair_rows, conf_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol, protocol_sha, source_sha = protocol_and_hashes()
    data_sha = sha_file(DATA_PATH)
    split_sha = sha_file(SPLIT_PATH)
    sample_sha = sha_file(IDS_PATH)
    input_sha = sha_json({'data': data_sha, 'split': split_sha, 'sample_ids': sample_sha})
    frame = pd.read_csv(DATA_PATH)
    ids = pd.read_csv(IDS_PATH)
    split = json.loads(SPLIT_PATH.read_text())
    raw_data = frame.iloc[:, :365].to_numpy(dtype=np.float64)
    labels = frame.LABEL.to_numpy(dtype=np.int8) - 1
    rid = ids.RID.to_numpy()
    boundaries = np.cumsum([0] + MODALITY_DIMS)
    raw_modalities = [raw_data[:, boundaries[i]:boundaries[i + 1]] for i in range(6)]
    connection = sqlite3.connect(DB_PATH)
    init_db(connection)
    completed_before = connection.execute('SELECT count(*) FROM tasks WHERE status="completed"').fetchone()[0]
    transform_records = {}
    sampling_records = {}
    fold_metrics = []
    fold_pairs = []
    fold_confusions = []
    failed = []
    process_start = time.time()
    expected_total = 10 * 5 * 6 * 2
    for outer_fold, outer_record in enumerate(split['splits'], 1):
        outer_train = np.asarray(outer_record['train_idx'], dtype=np.int32)
        outer_test = np.asarray(outer_record['test_idx'], dtype=np.int32)
        inner_folds = build_inner(outer_train, labels)
        for inner_fold, (train_patients, valid_patients) in enumerate(inner_folds, 1):
            pair_tt = make_pairs(train_patients, None)
            pair_vv = make_pairs(valid_patients, None)
            pair_vt = make_pairs(train_patients, valid_patients)
            target_tt_six = relation_labels(labels, pair_tt)
            target_vv_six = relation_labels(labels, pair_vv)
            target_vt_six = relation_labels(labels, pair_vt)
            target_sets = {
                'six': (target_tt_six, target_vv_six, target_vt_six),
                'same': ((target_tt_six < 3).astype(np.int8), (target_vv_six < 3).astype(np.int8), (target_vt_six < 3).astype(np.int8)),
            }
            sample_sets = {
                'six': sample_indices(target_tt_six, SIX_CAP, SEED + outer_fold * 10000 + inner_fold * 100 + 6),
                'same': sample_indices((target_tt_six < 3).astype(np.int8), SAME_CAP, SEED + outer_fold * 10000 + inner_fold * 100 + 2),
            }
            sampling_records[f'outer{outer_fold:02d}_inner{inner_fold:02d}'] = {
                'train_patient_sha256': sha_bytes(train_patients.tobytes()),
                'valid_patient_sha256': sha_bytes(valid_patients.tobytes()),
                'outer_test_patient_sha256': sha_bytes(outer_test.tobytes()),
                'pair_counts': {'TT': len(pair_tt), 'VV': len(pair_vv), 'VT': len(pair_vt)},
                'target_counts': {task: {scene: np.bincount(value, minlength=(6 if task == 'six' else 2)).tolist() for scene, value in zip(['TT', 'VV', 'VT'], values)} for task, values in target_sets.items()},
                'sample_counts': {task: int(len(value)) for task, value in sample_sets.items()},
                'sample_sha256': {task: sha_bytes(value.astype(np.int64).tobytes()) for task, value in sample_sets.items()},
                'pair_index_sha256': {scene: sha_bytes(value.tobytes()) for scene, value in [('TT', pair_tt), ('VV', pair_vv), ('VT', pair_vt)]},
                'patient_intersection_train_valid': int(len(set(train_patients) & set(valid_patients))),
                'patient_intersection_train_test': int(len(set(train_patients) & set(outer_test))),
                'patient_intersection_valid_test': int(len(set(valid_patients) & set(outer_test))),
            }
            for modality, raw in zip(MODALITIES, raw_modalities):
                task_ids = [f'o{outer_fold:02d}_i{inner_fold:02d}_{modality}_{task}' for task in ['six', 'same']]
                missing = [not completed_row(connection, task_id, input_sha, protocol_sha, source_sha) for task_id in task_ids]
                if not any(missing):
                    continue
                transformed, transform_meta = fit_transform(raw, train_patients)
                transform_records[f'outer{outer_fold:02d}_inner{inner_fold:02d}_{modality}'] = transform_meta
                train_features = pair_features(transformed, pair_tt)
                valid_features = pair_features(transformed, pair_vv)
                cross_features = pair_features(transformed, pair_vt)
                feature_scaler = StandardScaler()
                train_features = feature_scaler.fit_transform(train_features)
                valid_features = feature_scaler.transform(valid_features)
                cross_features = feature_scaler.transform(cross_features)
                feature_sets = (train_features, valid_features, cross_features)
                for relation_task, is_missing in zip(['six', 'same'], missing):
                    if not is_missing:
                        continue
                    task_id = f'o{outer_fold:02d}_i{inner_fold:02d}_{modality}_{relation_task}'
                    result = task_fit(connection, task_id, outer_fold, inner_fold, modality, relation_task,
                                      raw, labels, train_patients, valid_patients,
                                      (pair_tt, pair_vv, pair_vt), target_sets, sample_sets,
                                      feature_sets, protocol_sha, source_sha, input_sha)
                    completed = connection.execute('SELECT count(*) FROM tasks WHERE status="completed"').fetchone()[0]
                    if result['status'] != 'completed':
                        failed.append({'task_id': task_id, 'error': result['error']})
                    elapsed = max(time.time() - process_start, 1e-6)
                    rate = (completed - completed_before) / elapsed
                    eta = (expected_total - completed) / rate if rate > 0 else None
                    print(f'P0.1 progress {completed}/{expected_total} rate={rate:.2f}/s eta_seconds={eta if eta is not None else "NA"} failed={len(failed)}', flush=True)
            transform_records.setdefault(f'outer{outer_fold:02d}_inner{inner_fold:02d}', {'all_modality_tasks_previously_complete': True})
        metrics, pairs_out, confusions = aggregate_outer(connection, outer_fold, protocol_sha, source_sha, 5 * 6 * 2)
        fold_metrics.extend(metrics)
        fold_pairs.extend(pairs_out)
        fold_confusions.extend(confusions)
        pd.DataFrame(fold_metrics).to_csv(ROOT / 'results/fold_metrics.csv', index=False)
        pd.DataFrame(fold_pairs).to_csv(ROOT / 'results/class_pair_metrics.csv', index=False)
        (ROOT / 'results/confusion_matrices.json').write_text(json.dumps(fold_confusions, indent=2))
    (ROOT / 'audits/sampling_records.json').write_text(json.dumps(sampling_records, indent=2))
    (ROOT / 'audits/transform_records.json').write_text(json.dumps(transform_records, indent=2))
    failed_rows = connection.execute('SELECT task_id,error FROM tasks WHERE status!="completed"').fetchall()
    summary = []
    metrics_frame = pd.DataFrame(fold_metrics)
    for modality in MODALITIES:
        for scenario in ['VV', 'VT']:
            subset = metrics_frame[(metrics_frame.modality == modality) & (metrics_frame.scenario == scenario)]
            row = {'modality': modality, 'scenario': scenario}
            for metric in ['macro_auc_6', 'macro_f1_6', 'balanced_accuracy_6', 'same_auc', 'same_average_precision', 'same_balanced_accuracy', 'same_f1', 'same_brier']:
                row[metric + '_mean'] = float(subset[metric].mean())
                row[metric + '_std'] = float(subset[metric].std(ddof=0))
            summary.append(row)
    pd.DataFrame(summary).to_csv(ROOT / 'results/modality_summary.csv', index=False)
    ranking = []
    for scenario in ['VV', 'VT']:
        for metric in ['same_auc', 'macro_auc_6']:
            for outer_fold in range(1, 11):
                subset = metrics_frame[(metrics_frame.scenario == scenario) & (metrics_frame.outer_fold == outer_fold)]
                for rank, (_, row) in enumerate(subset.sort_values(metric, ascending=False).iterrows(), 1):
                    ranking.append({'outer_fold': outer_fold, 'scenario': scenario, 'metric': metric, 'modality': row.modality, 'value': row[metric], 'rank': rank})
    pd.DataFrame(ranking).to_csv(ROOT / 'results/anchor_ranking.csv', index=False)
    database_count = connection.execute('SELECT count(*) FROM tasks WHERE status="completed"').fetchone()[0]
    run_manifest = {
        'status': 'RUN_COMPLETE' if database_count == expected_total and not failed_rows else 'RUN_INCOMPLETE',
        'completed_tasks': int(database_count), 'expected_tasks': expected_total,
        'failed_tasks': int(len(failed_rows)), 'input_sha256': input_sha, 'protocol_sha256': protocol_sha, 'source_sha256': source_sha,
        'data_sha256': data_sha, 'split_sha256': split_sha, 'sample_ids_sha256': sample_sha,
        'last_run_utc': now(), 'resume': bool(args.resume),
    }
    (ROOT / 'audits/run_manifest.json').write_text(json.dumps(run_manifest, indent=2))
    if failed_rows:
        (ROOT / 'audits/failed_tasks.json').write_text(json.dumps([{'task_id': row[0], 'error': row[1]} for row in failed_rows], indent=2))
    print(json.dumps(run_manifest, indent=2))
    connection.close()


if __name__ == '__main__':
    main()
