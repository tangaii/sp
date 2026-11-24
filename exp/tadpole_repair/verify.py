"""Verify the saved fold artifacts and write the TADPOLE comparison tables."""

import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from config import (
    AUTHOR_LOG, CLASS_NAMES, DEFAULTS, MMGPL_CLASS_PROMPTS, MMGPL_CONCEPTS,
    MMGPL_PAPER, PAPER, VARIANTS,
)
from prepare import sha256, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAMES = {
    'previous_control': '上次设置复跑',
    'paper_config': '论文超参数基准',
    'clip_xavier': 'CLIP 线性层 Xavier 重置',
    'tadpole_prompts': '三类 prompt',
    'mmgpl_concepts': 'MMGPL Figure 6 概念文本',
    'mmgpl_class_prompts': 'MMGPL Figure 6 概念 + 类别名',
    'mmgpl_class_graph_sup': 'MMGPL 类别名 + 配对均方图监督',
    'mmgpl_class_pen_graph_sup': 'MMGPL 类别名 + PEN + 配对均方图监督',
    'mmgpl_class_full_graph_sup': 'MMGPL 类别名 + PEN + 六关系 GFR + 配对均方图监督',
    'pen_equations': '三类 prompt + PEN 式 (5)–(6)',
    'gfr_sum': '六关系 GFR + sigmoid 加权和',
    'gfr_softmax': '六关系 GFR + softmax 加权和',
    'graph_supervision': '图监督 + 配对均方误差',
    'graph_frobenius': '图监督 + Frobenius 平方和',
    'combined': 'PEN + 六关系 GFR + 配对均方图监督',
}
METRICS = ('acc', 'auc_hard', 'auc_probability')


def read_json(path):
    return json.loads(path.read_text())


def verify_hashes(config):
    for relative_path, digest in config['source_sha256'].items():
        candidates = [HERE / relative_path]
        snapshots = {
            'run.py': ['references/run_as_executed.py'],
            'config.py': [
                'references/config_as_executed.py',
                'references/config_mmgpl_as_executed.py',
            ],
        }
        candidates.extend(HERE / path for path in snapshots.get(relative_path, []))
        assert any(path.is_file() and sha256(path) == digest for path in candidates), relative_path
    for path, digest in config['inputs']['paths'].items():
        assert sha256(path) == digest, path
    weight = config['clip_weights']
    assert sha256(weight['path']) == weight['sha256'], weight['path']


def from_log(path, test_sizes):
    source = path.read_text()
    pairs = re.findall(r'^tst-auc: ([\d.]+)  acc: ([\d.]+)', source, re.M)
    assert len(pairs) == len(test_sizes), path
    sens = np.asarray(ast.literal_eval(re.findall(r'^sens: (\[.+\])\s*$', source, re.M)[-1]))
    spec = np.asarray(ast.literal_eval(re.findall(r'^spes: (\[.+\])\s*$', source, re.M)[-1]))
    fold_acc = [round(float(acc) * size) / size for (_, acc), size in zip(pairs, test_sizes)]
    fold_auc = ((sens + spec) / 2).mean(axis=1)
    np.testing.assert_allclose(fold_auc, [float(auc) for auc, _ in pairs], atol=5.01e-5)
    return {
        'file': str(path), 'sha256': sha256(path), 'fold_acc': fold_acc,
        'fold_auc_hard_reconstructed_from_sen_spe': fold_auc.tolist(),
        'acc': {'mean': float(np.mean(fold_acc)), 'std': float(np.std(fold_acc))},
        'auc_hard': {'mean': float(np.mean(fold_auc)), 'std': float(np.std(fold_auc))},
    }


def verify_previous_history(path):
    pattern = (
        r'^\s*(\d+)/300 trn-loss-MF: ([\d.\-]+) trn-loss-GC: ([\d.\-]+) '
        r'ALoss: [\d.\-]+ trn-auc: [\d.]+  acc: ([\d.]+) val-auc: [\d.]+  acc: ([\d.]+)'
    )
    old = np.asarray(re.findall(pattern, path.read_text(), re.M), dtype=float)
    history = pd.concat([
        pd.read_csv(file) for file in sorted((HERE / 'runs/previous_control').glob('fold_*/history.csv'))
    ])
    columns = ['epoch_zero_based', 'classification_loss', 'graph_structure_loss', 'train_acc', 'outer_test_acc']
    current = history[columns].to_numpy()
    assert old.shape == (3000, len(columns))
    np.testing.assert_allclose(current, old, rtol=0, atol=5.01e-5)
    return dict(zip(columns, np.abs(old - current).max(axis=0).tolist()))


def verify_fold(run_dir, split, truth, config):
    fold = split['fold']
    fold_dir = run_dir / f'fold_{fold:02d}'
    metrics = read_json(fold_dir / 'metrics.json')
    pred = pd.read_csv(fold_dir / 'predictions.csv')
    history = pd.read_csv(fold_dir / 'history.csv')
    indices = pred.row_index.to_numpy()
    class_ids = np.arange(len(CLASS_NAMES))
    np.testing.assert_array_equal(indices, split['test'])
    np.testing.assert_array_equal(pred.label, truth[indices])
    assert pred.fold.eq(fold).all()
    scores = pred[[f'p_{name}' for name in CLASS_NAMES]].to_numpy()
    assert np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()
    np.testing.assert_allclose(scores.sum(axis=1), 1, atol=5e-7)
    np.testing.assert_array_equal(scores[np.arange(len(scores)), pred.prediction], scores.max(axis=1))
    matrix = confusion_matrix(pred.label, pred.prediction, labels=class_ids)
    sens = matrix.diagonal() / matrix.sum(axis=1)
    spec = 1 - (matrix.sum(axis=0) - matrix.diagonal()) / (matrix.sum() - matrix.sum(axis=1))
    computed = {
        'acc': float(np.trace(matrix) / matrix.sum()),
        'auc_hard': float((sens + spec).mean() / 2),
        'auc_probability': float(roc_auc_score(
            pred.label, scores, labels=class_ids, multi_class='ovr', average='macro',
        )),
    }
    assert matrix.tolist() == metrics['confusion'], (run_dir, fold)
    np.testing.assert_allclose(list(computed.values()), [metrics[m] for m in METRICS], atol=1e-10)
    np.testing.assert_allclose(sens, [metrics['sensitivity'][name] for name in CLASS_NAMES], atol=1e-10)
    np.testing.assert_allclose(spec, [metrics['specificity'][name] for name in CLASS_NAMES], atol=1e-10)

    epochs = config['epochs']
    np.testing.assert_array_equal(history.epoch_zero_based, np.arange(epochs))
    assert np.isfinite(history.to_numpy()).all()
    best_epoch = int(history.outer_test_acc.argmax())
    assert best_epoch == metrics['best_epoch_zero_based'], (run_dir, fold, best_epoch)
    np.testing.assert_allclose(history.outer_test_acc.iloc[best_epoch], computed['acc'], atol=1e-10)
    np.testing.assert_allclose(
        history.total_loss,
        config['eta_classification'] * history.classification_loss
        + config['lambda_graph'] * (history.graph_structure_loss + history.graph_supervision_loss),
        rtol=1e-6, atol=5e-7,
    )
    if config['graph_supervision'] == 'none':
        assert history.graph_supervision_loss.eq(0).all()
    else:
        assert history.graph_supervision_loss.gt(0).all()

    relations = metrics['graph_relations']
    assert [relation['index'] for relation in relations] == list(range(config['nmodal']))
    expected = [False, True, True, True, True, True] if config['fusion'] == 'legacy' else [True] * 6
    assert [relation['used_for_fusion'] for relation in relations] == expected
    if config['fusion'] == 'softmax_sum':
        np.testing.assert_allclose(sum(relation['weight'] for relation in relations), 1, atol=1e-6)

    artifacts = {
        name: sha256(fold_dir / name) for name in
        ('best.pt', 'history.csv', 'metrics.json', 'next_fold_rng.pt', 'predictions.csv')
    }
    return computed, pred, artifacts


def verify_variant(variant, splits, truth):
    directory = HERE / 'runs' / variant
    config = read_json(directory / 'config.json')
    assert config['variant'] == variant
    expected_hyperparameters = DEFAULTS | VARIANTS[variant] | {'executed_folds': DEFAULTS['folds']}
    if 'prompt_format' not in config['hyperparameters']:
        expected_hyperparameters.pop('prompt_format')
    assert config['hyperparameters'] == expected_hyperparameters
    verify_hashes(config)
    if 'mmgpl_prompt_reference' in config:
        paper_ref = config['mmgpl_prompt_reference']
        assert paper_ref['path'].endswith(MMGPL_PAPER)
        assert paper_ref['sha256'] == sha256(paper_ref['path'])
        if variant == 'mmgpl_concepts':
            assert config['prompts'] == {name: list(values) for name, values in MMGPL_CONCEPTS.items()}
    assert read_json(directory / 'splits.json') == splits
    summary = read_json(directory / 'summary.json')
    results = [verify_fold(directory, split, truth, config['hyperparameters']) for split in splits]
    observed_preds = pd.read_csv(directory / 'out_of_fold_predictions.csv')
    expected_preds = pd.concat([result[1] for result in results], ignore_index=True).sort_values('row_index')
    pd.testing.assert_frame_equal(observed_preds, expected_preds.reset_index(drop=True))
    np.testing.assert_array_equal(expected_preds.row_index, np.arange(len(truth)))
    for metric in METRICS:
        values = [result[0][metric] for result in results]
        np.testing.assert_allclose(
            [summary['overall'][metric]['mean'], summary['overall'][metric]['std']],
            [np.mean(values), np.std(values)], atol=1e-10,
        )
    assert [item['fold'] for item in summary['folds']] == [item['fold'] for item in splits]
    hashes = {f'fold_{split["fold"]:02d}': result[2] for split, result in zip(splits, results)}
    for name in ('config.json', 'splits.json', 'summary.json', 'out_of_fold_predictions.csv'):
        hashes[name] = sha256(directory / name)
    return summary, hashes


def table_row(name, summary, paper):
    row = {'experiment': name}
    for metric in METRICS:
        record = summary.get(metric, {})
        row[f'{metric}_mean_pct'] = record.get('mean', np.nan) * 100
        row[f'{metric}_std_pct'] = record.get('std', np.nan) * 100
    row['acc_delta_paper_pp'] = row['acc_mean_pct'] - paper['acc']['mean'] * 100
    row['auc_hard_delta_paper_pp'] = row['auc_hard_mean_pct'] - paper['auc_hard']['mean'] * 100
    return row


def pct_cell(row, metric):
    mean = row[f'{metric}_mean_pct']
    std = row[f'{metric}_std_pct']
    return '—' if np.isnan(mean) else f'{mean:.2f} ± {std:.2f}'


def main():
    prompt_audit = read_json(HERE / 'mmgpl_prompts.json')
    assert prompt_audit['source']['sha256'] == sha256(ROOT / MMGPL_PAPER)
    assert prompt_audit['concepts'] == {name: list(values) for name, values in MMGPL_CONCEPTS.items()}
    assert prompt_audit['class_prompts'] == {name: list(values) for name, values in MMGPL_CLASS_PROMPTS.items()}

    preprocessing = read_json(HERE / 'data/preprocessing.json')
    determinism = read_json(HERE / 'data/determinism.json')
    for filename, digest in determinism['sha256_equal'].items():
        assert sha256(HERE / 'data' / filename) == digest, filename
    for filename, digest in preprocessing['raw_sources'].items():
        assert sha256(filename) == digest, filename
    diagnostics = read_json(HERE / 'diagnostics.json')
    assert diagnostics['clip']['frozen_cached_vs_frozen_live']['exactly_equal']
    assert diagnostics['graph_supervision']['no_direct_supervision_gradient_for_pairs_with_outer_test_node']
    resume_path = HERE / 'smoke/resume_check.json'
    resume_check = read_json(resume_path)
    assert resume_check['continuous_and_resumed_model_equal']

    frame = pd.read_csv(HERE / 'data/processed_standard_data.csv')
    truth = frame.LABEL.to_numpy(dtype=int) - 1
    samples = pd.read_csv(HERE / 'data/samples.csv')
    assert samples.RID.is_unique
    np.testing.assert_array_equal(samples.LABEL.to_numpy() - 1, truth)
    np.testing.assert_array_equal(samples.row_index, np.arange(len(truth)))
    splits = read_json(HERE / 'runs/paper_config/splits.json')
    assert len(splits) == DEFAULTS['folds']
    for split in splits:
        assert set(split['train']).isdisjoint(split['test'])
        assert sorted(split['train'] + split['test']) == list(range(len(truth)))
    np.testing.assert_array_equal(
        sorted(index for split in splits for index in split['test']), np.arange(len(truth)),
    )

    sizes = [len(split['test']) for split in splits]
    author = from_log(HERE / 'references/author_Tadpole.log', sizes)
    old = from_log(ROOT / 'reproduction_TADPOLE_300.log', sizes)
    paper = {'acc': {'mean': .9481, 'std': .0295}, 'auc_hard': {'mean': .9559, 'std': .0257}}
    for metric in paper:
        for component in ('mean', 'std'):
            assert abs(author[metric][component] - paper[metric][component]) < 5.01e-5
    results = [('论文 Table 2', paper), ('上次复现日志', old)]
    hashes = {}
    for variant in VARIANTS:
        summary, hashes[variant] = verify_variant(variant, splits, truth)
        results.append((NAMES[variant], summary['overall']))
        if variant == 'previous_control':
            np.testing.assert_allclose(
                [metric['acc'] for metric in summary['folds']], old['fold_acc'], atol=1e-10,
            )
            np.testing.assert_allclose(
                [metric['auc_hard'] for metric in summary['folds']],
                old['fold_auc_hard_reconstructed_from_sen_spe'], atol=2e-9,
            )

    rows = [table_row(name, data, paper) for name, data in results]
    pd.DataFrame(rows).to_csv(HERE / 'comparison.csv', index=False, float_format='%.6f')
    lines = [
        '# TADPOLE 复现与论文对比', '',
        '10 折均值 ± 折间总体标准差，单位为百分比。各组沿用外层测试折 ACC 首次达峰值的轮次。',
        '', '| 实验 | ACC | 硬标签 AUC | 概率 AUC | ACC 与论文差值（百分点） |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        gap = '—' if row['experiment'] == '论文 Table 2' else f'{row["acc_delta_paper_pp"]:+.2f}'
        cells = [row['experiment'], *(pct_cell(row, metric) for metric in METRICS), gap]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += [
        '', f'论文 AUC 与作者日志的硬标签 AUC 一致。概率 AUC 由三类概率按宏平均 OvR 计算。'
        f' [SPromptGL 论文]({PAPER}) · [作者日志]({AUTHOR_LOG}) · '
        f'[MMGPL 论文](https://doi.org/10.1016/j.media.2024.103225)',
        '', '每组的 `runs/<variant>/` 目录保存配置、拆分、每轮损失、最佳权重、外层折预测与混淆矩阵。'
        ' `verification.json` 保存复算及文件哈希。',
    ]
    (HERE / 'comparison.md').write_text('\n'.join(lines) + '\n')
    write_json(HERE / 'verification.json', {
        'paper': {'url': PAPER, 'table_2': paper, 'author_log_url': AUTHOR_LOG, 'log_verification': author},
        'previous_log': old, 'all_fold_metrics_recomputed': True, 'previous_log_fold_metrics_match': True,
        'previous_epoch_history_max_absolute_diff': verify_previous_history(Path(old['file'])),
        'all_best_epochs_verified': True, 'artifact_hashes': hashes,
        'preprocessing': {'determinism': determinism, 'raw_input_hashes_verified': True},
        'diagnostics_sha256': sha256(HERE / 'diagnostics.json'),
        'mmgpl_prompt_audit_sha256': sha256(HERE / 'mmgpl_prompts.json'),
        'training_runner_snapshot_sha256': sha256(HERE / 'references/run_as_executed.py'),
        'resume_check': {'path': str(resume_path), 'sha256': sha256(resume_path), 'result': resume_check},
    })
    print(f'Verified {len(VARIANTS)} variants and {len(splits)} folds and wrote {HERE / "comparison.md"}')


if __name__ == '__main__':
    main()
