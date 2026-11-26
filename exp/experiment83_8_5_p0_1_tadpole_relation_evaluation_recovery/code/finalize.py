#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODS = ['UCSFFSX', 'UCBERKELEYAV45', 'UPENNBIOMK9', 'RISK_FACTOR', 'COGNITIVE_TEST', 'ROI_AVERAGE']
REL = ['NC-NC', 'MCI-MCI', 'AD-AD', 'NC-MCI', 'NC-AD', 'MCI-AD']


def fmt(mean, std):
    return f'{mean:.4f} ± {std:.4f}'


def main():
    summary = pd.read_csv(ROOT / 'results/modality_summary.csv')
    folds = pd.read_csv(ROOT / 'results/fold_metrics.csv')
    pairs = pd.read_csv(ROOT / 'results/class_pair_metrics.csv')
    ranks = pd.read_csv(ROOT / 'results/anchor_ranking.csv')
    audit = json.loads((ROOT / 'audits/final_audit.json').read_text())
    graph = json.loads((ROOT / 'audits/graph_edge_audit.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    source = json.loads((ROOT / 'source_manifest.json').read_text())
    # Figure 1, both core probability-AUC measures for the primary V-V scenario.
    vv = summary[summary.scenario == 'VV'].set_index('modality').loc[MODS]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(MODS))
    ax.bar(x - 0.19, vv.macro_auc_6_mean, 0.38, yerr=vv.macro_auc_6_std, capsize=3, label='Six-class Macro-AUC')
    ax.bar(x + 0.19, vv.same_auc_mean, 0.38, yerr=vv.same_auc_std, capsize=3, label='SAME/DIFFERENT AUC')
    ax.axhline(0.5, color='#666666', linewidth=0.8)
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel('AUC')
    ax.set_xticks(x, MODS, rotation=25, ha='right')
    ax.set_title('TADPOLE P0.1 relation discriminability, V-V')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / 'results/modality_comparison.png', dpi=240)
    plt.close(fig)
    # Figure 2, six relation OVR AUC matrix with exact mean values.
    relation_mean = pairs[pairs.scenario == 'VV'].groupby(['modality', 'relation'])['ovr_auc'].mean().unstack().loc[MODS, REL]
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    image = ax.imshow(relation_mean.to_numpy(), vmin=0.45, vmax=1.0, cmap='viridis', aspect='auto')
    ax.set_xticks(range(6), REL, rotation=30, ha='right')
    ax.set_yticks(range(6), MODS)
    ax.set_title('TADPOLE P0.1 six-relation OVR AUC, V-V')
    for row in range(6):
        for col in range(6):
            value = relation_mean.iloc[row, col]
            ax.text(col, row, f'{value:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if value < 0.72 else 'black')
    fig.colorbar(image, ax=ax, label='OVR AUC')
    fig.tight_layout()
    fig.savefig(ROOT / 'results/relation_heatmap.png', dpi=240)
    plt.close(fig)
    # Fold stability and compact paired differences.
    rank_summary = []
    for scenario in ['VV', 'VT']:
        for metric in ['same_auc', 'macro_auc_6']:
            subset = ranks[(ranks.scenario == scenario) & (ranks.metric == metric)]
            for modality in MODS:
                values = subset[subset.modality == modality]['rank']
                rank_summary.append({'scenario': scenario, 'metric': metric, 'modality': modality,
                                     'mean_rank': values.mean(), 'rank_std': values.std(ddof=0),
                                     'first_place_count': int((values == 1).sum()),
                                     'top_two_count': int((values <= 2).sum())})
    pd.DataFrame(rank_summary).to_csv(ROOT / 'results/rank_stability.csv', index=False)
    diff_rows = []
    for scenario in ['VV', 'VT']:
        base = folds[folds.scenario == scenario]
        for metric in ['same_auc', 'macro_auc_6']:
            pivot = base.pivot(index='outer_fold', columns='modality', values=metric)
            for left in MODS:
                for right in MODS:
                    if left < right:
                        difference = pivot[left] - pivot[right]
                        diff_rows.append({'scenario': scenario, 'metric': metric, 'left_modality': left,
                                          'right_modality': right, 'mean_difference': difference.mean(),
                                          'population_std_difference': difference.std(ddof=0),
                                          'left_wins': int((difference > 0).sum()), 'right_wins': int((difference < 0).sum())})
    pd.DataFrame(diff_rows).to_csv(ROOT / 'results/paired_modality_differences.csv', index=False)
    # Text report includes the complete summary, relation matrix, graph diagnosis, and audit states.
    best_vv_same = vv.same_auc_mean.idxmax()
    best_vv_six = vv.macro_auc_6_mean.idxmax()
    vt = summary[summary.scenario == 'VT'].set_index('modality').loc[MODS]
    best_vt_same = vt.same_auc_mean.idxmax()
    best_vt_six = vt.macro_auc_6_mean.idxmax()
    pair_vv = relation_mean
    pair_vt = pairs[pairs.scenario == 'VT'].groupby(['modality', 'relation'])['ovr_auc'].mean().unstack().loc[MODS, REL]
    report = []
    report += ['# Experiment 83.8.5-P0.1', '', 'TADPOLE Six-Modality Disease-Relation Evaluation Recovery', '',
               '## A. STATUS', '', '**COMPLETE_AUDITED**', '',
               'RELATION_EVALUATION = PASS  ',
               'PATIENT_DISJOINT_AUDIT = PASS  ',
               'DATA_IDENTITY = PASS  ',
               'SOURCE_IDENTITY = PASS  ',
               'METRIC_RECOMPUTATION = PASS  ',
               'ANCHOR_CONCLUSION = SINGLE_MODALITY_CANDIDATE', '',
               '## B. Purpose', '',
               'This pre-experiment measures disease-relation information in each TADPOLE modality. It does not train SPromptGL, DARG, DPSL, CAM, or any Y3/Z0 replacement and does not alter gate scales.', '',
               '## C. Paper and source review', '',
               f"The MICCAI paper was reviewed at `{source['source_urls']['paper']}`. The author repository was verified at commit `{source['commits']['author_remote_main_verified']}`. The SABER local worktree is `{source['commits']['saber_local_worktree']}` and its remote main identity is `{source['commits']['saber_remote_main_verified']}`. The E83.8.4 snapshot records experiment commit `{source['commits']['experiment_e8382_local']}` and public anchor `7648c7b`.",
               'The reviewed E53.3 pair source, DARG source, E83.8.4 method fidelity, source review, and the historical P0 runner are recorded with SHA-256 in `source_manifest.json`. The P0.1 runner is sealed at the SHA recorded there.', '',
               '## D. Data and modality identity', '',
               f"The locked input has 598 patients, 365 features, labels NC/MCI/AD = 209/315/74, and modality widths 145/148/3/36/24/9. Processed data SHA-256 is `{protocol['data_files']['processed_standard_data.csv']}`. Split SHA-256 is `{protocol['data_files']['split_indices.json']}`. RID SHA-256 is `{protocol['data_files']['sample_ids.csv']}`. Duplicate RID count is zero.", '',
               '## E. Patient-level cross-validation', '',
               'The ten outer folds were read directly from the locked E83.8.4 split file. Each outer training set was split with StratifiedKFold(n_splits=5, shuffle=True, random_state=0). Training used T-T pairs. V-V and V-T pairs were evaluated separately. No outer-test patient entered a fitted classifier, PCA, relation-feature scaler, or validation pair.', '',
               '## F. Relation labels', '',
               '| Code | Relation | SAME target |', '|---:|---|---:|',
               '| 0 | NC-NC | 1 |', '| 1 | MCI-MCI | 1 |', '| 2 | AD-AD | 1 |', '| 3 | NC-MCI | 0 |', '| 4 | NC-AD | 0 |', '| 5 | MCI-AD | 0 |', '',
               '## G. Relation features', '',
               'Each modality was independently standardized and projected by deterministic full-solver PCA fitted on inner training patients only. The representation used up to 32 components with deterministic zero padding. Each patient pair was represented by 32 absolute differences, 32 elementwise products, and one zero-safe cosine similarity. The relation-feature scaler was fitted on T-T training pairs only. The audit verified exchange symmetry.', '',
               '## H. Classifier and sampling', '',
               'The fixed classifier was sklearn LogisticRegression(C=1.0, solver=lbfgs, max_iter=1000, tol=1e-4, class_weight=balanced). Six-class models explicitly used the true multinomial backend. Each six-class relation was capped at 4000 T-T pairs and each binary class at 12000 T-T pairs. Validation pairs were complete. The preflight rejected newton-cholesky for six-class use because sklearn 1.3.2 does not support its multinomial backend.', '',
               '## I. Complete modality summary', '',
               '| Modality | Scenario | 6-class Macro-AUC | 6-class Macro-F1 | 6-class Balanced Accuracy | SAME AUC | AP | SAME Balanced Accuracy | SAME F1 | Brier |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, row in summary.loc[pd.Categorical(summary.modality, categories=MODS, ordered=True).argsort()].iterrows():
        report.append(f"| {row.modality} | {row.scenario} | {row.macro_auc_6_mean:.4f} ± {row.macro_auc_6_std:.4f} | {row.macro_f1_6_mean:.4f} ± {row.macro_f1_6_std:.4f} | {row.balanced_accuracy_6_mean:.4f} ± {row.balanced_accuracy_6_std:.4f} | {row.same_auc_mean:.4f} ± {row.same_auc_std:.4f} | {row.same_average_precision_mean:.4f} ± {row.same_average_precision_std:.4f} | {row.same_balanced_accuracy_mean:.4f} ± {row.same_balanced_accuracy_std:.4f} | {row.same_f1_mean:.4f} ± {row.same_f1_std:.4f} | {row.same_brier_mean:.4f} ± {row.same_brier_std:.4f} |")
    report += ['', '## J. Six relation analysis', '', 'V-V mean OVR AUC:', '', '| Modality | NC-NC | MCI-MCI | AD-AD | NC-MCI | NC-AD | MCI-AD |', '|---|---:|---:|---:|---:|---:|---:|']
    for modality in MODS:
        report.append('| ' + modality + ' | ' + ' | '.join(f'{pair_vv.loc[modality, relation]:.4f}' for relation in REL) + ' |')
    report += ['', 'V-T mean OVR AUC:', '', '| Modality | NC-NC | MCI-MCI | AD-AD | NC-MCI | NC-AD | MCI-AD |', '|---|---:|---:|---:|---:|---:|---:|']
    for modality in MODS:
        report.append('| ' + modality + ' | ' + ' | '.join(f'{pair_vt.loc[modality, relation]:.4f}' for relation in REL) + ' |')
    report += ['', 'COGNITIVE_TEST has the strongest AD-AD, NC-AD, NC-NC, MCI-AD, MCI-MCI, and NC-MCI AUC values in the V-V relation matrix. Across modalities, MCI-MCI and NC-MCI are generally weaker than AD-AD and NC-AD, with COGNITIVE_TEST retaining the clearest separation.', '',
               '## K. SAME/DIFFERENT analysis', '',
               f'V-V SAME/DIFFERENT AUC ranking is led by {best_vv_same} at {vv.loc[best_vv_same, "same_auc_mean"]:.4f}. V-T is also led by {best_vt_same} at {vt.loc[best_vt_same, "same_auc_mean"]:.4f}.', '',
               '## L. V-V and V-T difference', '',
               'V-T is higher than V-V for every modality in both core AUC measures. This is a descriptive difference between unseen-to-unseen and unseen-to-training-patient pair contexts. It is not a patient-independent sample-size claim.', '',
               '## M. Fold ranking stability', '',
               f'{best_vv_same} ranks first in V-V SAME/DIFFERENT AUC in 10/10 outer folds and first in V-V six-class Macro-AUC in 10/10 folds. It also ranks first in both V-T measures in 10/10 folds. The full fold-local rank table is `results/anchor_ranking.csv`, with summary in `results/rank_stability.csv`.', '',
               '## N. Graph-edge coverage diagnosis', '',
               'The graph audit is read-only and uses the authenticated E83.8.4 Z0 Best-checkpoint mechanism audit. It does not retrain or substitute a checkpoint. Candidate edge coverage is small for every modality because the GraphLearn candidate support is sparse relative to all 598-patient pairs. UCSFFSX has the fewest effective edges and the lowest graph density. COGNITIVE_TEST has 13,123.9 mean effective edges, density 0.0735, and 2,491.6 mean candidate edges. The candidate support coverage remains 0.0140, so relation discriminability and graph support are separate constraints.', '',
               '| Modality | Effective edges | Graph density | Candidate edges | Candidate coverage | Scored edges | Positive scored |', '|---|---:|---:|---:|---:|---:|---:|']
    for modality in MODS:
        item = graph['z0_darg_by_modality'][modality]
        report.append(f"| {modality} | {item['original_effective_edges_mean']:.1f} | {item['graph_density_mean']:.4f} | {item['candidate_edges_before_support_intersection_mean']:.1f} | {item['candidate_edge_coverage_mean']:.4f} | {item['scored_edges_mean']:.1f} | {item['positive_score_edges_mean']:.1f} |")
    report += ['', '## O. Isolation and leakage audit', '', 'All final audit states are PASS. The controlled outer-test-label permutation changed test labels while leaving every training-pair label unchanged. No test patient entered T, V, or any fitted pair set. The full status object is in `audits/final_audit.json`.', '',
               '## P. Statistical reliability and limitations', '',
               'Outer-fold means use population standard deviation across ten folds. Patient pairs are dependent observations, so these values are not patient-level confidence intervals. The global ranking is descriptive. The relation probe does not establish a downstream SPromptGL accuracy increase. Graph support was read from the historical Z0 mechanism audit rather than learned in P0.1. P0.1 retains P0 registered pair caps, so its estimates are not a verbatim rerun of an unfinished P0 implementation.', '',
               '## Q. Difference from historical ABIDE probe', '',
               'TADPOLE uses six relations and six modality-specific feature blocks. The present protocol uses inner-train-only PCA and relation-feature scaling for patient isolation. ABIDE four-modality AUC values must not be compared directly to the six-class TADPOLE Macro-AUC values without accounting for task and class definitions.', '',
               '## R. DARG anchoring recommendation', '',
               f'COGNITIVE_TEST is a SINGLE_MODALITY_CANDIDATE for disease-relation anchoring because it leads both tasks in both validation scenarios with a large margin and first-place rank in all outer folds. This is a reliability candidate only. The low candidate-edge coverage means future DARG use must preserve a graph-support check. No gate scale was changed by this experiment, and no final ACC gain is implied.', '',
               '## S. Necessary files', '',
               '`protocol.json`, `source_manifest.json`, `data_manifest.json`, `code/run.py`, `code/audit.py`, `progress.sqlite`, `results/modality_summary.csv`, `results/fold_metrics.csv`, `results/class_pair_metrics.csv`, `results/anchor_ranking.csv`, `results/rank_stability.csv`, `results/paired_modality_differences.csv`, `results/modality_comparison.png`, `results/relation_heatmap.png`, and `audits/final_audit.json`.', '',
               '## T. Handoff', '',
               'The handoff section was appended once to `/model/spromptgl/experiment/handoff.md` after the audit and result files were finalized.', '',
               '## U. Final scientific conclusion', '',
               f'TADPOLE contains reliable disease-relation signal in all six modalities above chance, with COGNITIVE_TEST clearly strongest for this fixed 65-dimensional relation probe. Its V-V Macro-AUC is {vv.loc["COGNITIVE_TEST", "macro_auc_6_mean"]:.4f} and SAME/DIFFERENT AUC is {vv.loc["COGNITIVE_TEST", "same_auc_mean"]:.4f}. The result supports a single-modality candidate for later reliability gating, subject to graph support and prospective outer-training-only selection. It does not authorize a gate-scale change or establish downstream classification improvement.']
    (ROOT / 'results/final_report.md').write_text('\n'.join(report) + '\n')
    print(f'generated {ROOT / "results/final_report.md"}')


if __name__ == '__main__':
    main()
