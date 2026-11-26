# Experiment 83.8.5-P0.1

TADPOLE Six-Modality Disease-Relation Evaluation Recovery

## A. STATUS

**COMPLETE_AUDITED**

RELATION_EVALUATION = PASS  
PATIENT_DISJOINT_AUDIT = PASS  
DATA_IDENTITY = PASS  
SOURCE_IDENTITY = PASS  
METRIC_RECOMPUTATION = PASS  
ANCHOR_CONCLUSION = SINGLE_MODALITY_CANDIDATE

## B. Purpose

This pre-experiment measures disease-relation information in each TADPOLE modality. It does not train SPromptGL, DARG, DPSL, CAM, or any Y3/Z0 replacement and does not alter gate scales.

## C. Paper and source review

The MICCAI paper was reviewed at `https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf`. The author repository was verified at commit `ba3b9446fa451fde6864d6555bdf324de0c4b64a`. The SABER local worktree is `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4` and its remote main identity is `5e88a9dfeff47735e0a5e09a06672161c3b9c1da`. The E83.8.4 snapshot records experiment commit `252fa96b5ff2ac89250cda378c777fc36a99d316` and public anchor `7648c7b`.
The reviewed E53.3 pair source, DARG source, E83.8.4 method fidelity, source review, and the historical P0 runner are recorded with SHA-256 in `source_manifest.json`. The P0.1 runner is sealed at the SHA recorded there.

## D. Data and modality identity

The locked input has 598 patients, 365 features, labels NC/MCI/AD = 209/315/74, and modality widths 145/148/3/36/24/9. Processed data SHA-256 is `0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc`. Split SHA-256 is `7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5`. RID SHA-256 is `3c354e2623128209544b23d84fbac0e436f55b4c898c74af83d2e99e7f61d8e0`. Duplicate RID count is zero.

## E. Patient-level cross-validation

The ten outer folds were read directly from the locked E83.8.4 split file. Each outer training set was split with StratifiedKFold(n_splits=5, shuffle=True, random_state=0). Training used T-T pairs. V-V and V-T pairs were evaluated separately. No outer-test patient entered a fitted classifier, PCA, relation-feature scaler, or validation pair.

## F. Relation labels

| Code | Relation | SAME target |
|---:|---|---:|
| 0 | NC-NC | 1 |
| 1 | MCI-MCI | 1 |
| 2 | AD-AD | 1 |
| 3 | NC-MCI | 0 |
| 4 | NC-AD | 0 |
| 5 | MCI-AD | 0 |

## G. Relation features

Each modality was independently standardized and projected by deterministic full-solver PCA fitted on inner training patients only. The representation used up to 32 components with deterministic zero padding. Each patient pair was represented by 32 absolute differences, 32 elementwise products, and one zero-safe cosine similarity. The relation-feature scaler was fitted on T-T training pairs only. The audit verified exchange symmetry.

## H. Classifier and sampling

The fixed classifier was sklearn LogisticRegression(C=1.0, solver=lbfgs, max_iter=1000, tol=1e-4, class_weight=balanced). Six-class models explicitly used the true multinomial backend. Each six-class relation was capped at 4000 T-T pairs and each binary class at 12000 T-T pairs. Validation pairs were complete. The preflight rejected newton-cholesky for six-class use because sklearn 1.3.2 does not support its multinomial backend.

## I. Complete modality summary

| Modality | Scenario | 6-class Macro-AUC | 6-class Macro-F1 | 6-class Balanced Accuracy | SAME AUC | AP | SAME Balanced Accuracy | SAME F1 | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UCSFFSX | VV | 0.5762 ± 0.0159 | 0.1713 ± 0.0068 | 0.2269 ± 0.0104 | 0.5351 ± 0.0064 | 0.4331 ± 0.0057 | 0.5261 ± 0.0045 | 0.5042 ± 0.0079 | 0.2522 ± 0.0009 |
| UCSFFSX | VT | 0.6293 ± 0.0085 | 0.2005 ± 0.0040 | 0.2783 ± 0.0069 | 0.5546 ± 0.0040 | 0.4541 ± 0.0041 | 0.5390 ± 0.0029 | 0.5105 ± 0.0049 | 0.2488 ± 0.0005 |
| UCBERKELEYAV45 | VV | 0.6058 ± 0.0118 | 0.1832 ± 0.0058 | 0.2773 ± 0.0131 | 0.5520 ± 0.0071 | 0.4468 ± 0.0046 | 0.5385 ± 0.0056 | 0.4899 ± 0.0077 | 0.2491 ± 0.0011 |
| UCBERKELEYAV45 | VT | 0.6498 ± 0.0050 | 0.2119 ± 0.0034 | 0.3210 ± 0.0064 | 0.5680 ± 0.0037 | 0.4646 ± 0.0034 | 0.5495 ± 0.0031 | 0.5072 ± 0.0040 | 0.2468 ± 0.0006 |
| UPENNBIOMK9 | VV | 0.6190 ± 0.0056 | 0.1513 ± 0.0082 | 0.2810 ± 0.0073 | 0.5469 ± 0.0039 | 0.4391 ± 0.0039 | 0.5360 ± 0.0029 | 0.4985 ± 0.0057 | 0.2485 ± 0.0004 |
| UPENNBIOMK9 | VT | 0.6270 ± 0.0050 | 0.1565 ± 0.0083 | 0.2890 ± 0.0054 | 0.5514 ± 0.0032 | 0.4484 ± 0.0026 | 0.5387 ± 0.0023 | 0.5044 ± 0.0047 | 0.2480 ± 0.0003 |
| RISK_FACTOR | VV | 0.5657 ± 0.0113 | 0.1658 ± 0.0067 | 0.2196 ± 0.0086 | 0.5097 ± 0.0077 | 0.4181 ± 0.0067 | 0.5069 ± 0.0058 | 0.4581 ± 0.0146 | 0.2532 ± 0.0011 |
| RISK_FACTOR | VT | 0.6135 ± 0.0070 | 0.1865 ± 0.0053 | 0.2575 ± 0.0063 | 0.5314 ± 0.0055 | 0.4405 ± 0.0049 | 0.5217 ± 0.0042 | 0.4759 ± 0.0089 | 0.2504 ± 0.0006 |
| COGNITIVE_TEST | VV | 0.8325 ± 0.0055 | 0.4570 ± 0.0114 | 0.5725 ± 0.0118 | 0.7153 ± 0.0060 | 0.5807 ± 0.0078 | 0.6661 ± 0.0057 | 0.6384 ± 0.0057 | 0.2154 ± 0.0020 |
| COGNITIVE_TEST | VT | 0.8501 ± 0.0048 | 0.4857 ± 0.0107 | 0.5985 ± 0.0094 | 0.7238 ± 0.0046 | 0.5988 ± 0.0059 | 0.6722 ± 0.0035 | 0.6478 ± 0.0035 | 0.2118 ± 0.0015 |
| ROI_AVERAGE | VV | 0.6552 ± 0.0066 | 0.1975 ± 0.0034 | 0.3270 ± 0.0073 | 0.5624 ± 0.0039 | 0.4497 ± 0.0035 | 0.5491 ± 0.0025 | 0.5135 ± 0.0042 | 0.2474 ± 0.0005 |
| ROI_AVERAGE | VT | 0.6692 ± 0.0039 | 0.2083 ± 0.0018 | 0.3403 ± 0.0037 | 0.5677 ± 0.0025 | 0.4602 ± 0.0020 | 0.5515 ± 0.0021 | 0.5194 ± 0.0028 | 0.2467 ± 0.0003 |

## J. Six relation analysis

V-V mean OVR AUC:

| Modality | NC-NC | MCI-MCI | AD-AD | NC-MCI | NC-AD | MCI-AD |
|---|---:|---:|---:|---:|---:|---:|
| UCSFFSX | 0.5223 | 0.5333 | 0.6979 | 0.5369 | 0.6007 | 0.5659 |
| UCBERKELEYAV45 | 0.5767 | 0.5156 | 0.7672 | 0.5430 | 0.6674 | 0.5650 |
| UPENNBIOMK9 | 0.6243 | 0.5217 | 0.7489 | 0.5587 | 0.6356 | 0.6246 |
| RISK_FACTOR | 0.6216 | 0.5186 | 0.6080 | 0.5349 | 0.5325 | 0.5784 |
| COGNITIVE_TEST | 0.8745 | 0.7333 | 0.9708 | 0.6926 | 0.8966 | 0.8274 |
| ROI_AVERAGE | 0.6606 | 0.5145 | 0.8323 | 0.5840 | 0.6945 | 0.6455 |

V-T mean OVR AUC:

| Modality | NC-NC | MCI-MCI | AD-AD | NC-MCI | NC-AD | MCI-AD |
|---|---:|---:|---:|---:|---:|---:|
| UCSFFSX | 0.5882 | 0.5853 | 0.7841 | 0.5589 | 0.6442 | 0.6150 |
| UCBERKELEYAV45 | 0.6315 | 0.5679 | 0.8339 | 0.5626 | 0.6974 | 0.6055 |
| UPENNBIOMK9 | 0.6317 | 0.5328 | 0.7628 | 0.5619 | 0.6423 | 0.6308 |
| RISK_FACTOR | 0.6624 | 0.5673 | 0.7003 | 0.5550 | 0.5847 | 0.6113 |
| COGNITIVE_TEST | 0.8898 | 0.7564 | 0.9808 | 0.7080 | 0.9140 | 0.8514 |
| ROI_AVERAGE | 0.6744 | 0.5382 | 0.8508 | 0.5901 | 0.7028 | 0.6589 |

COGNITIVE_TEST has the strongest AD-AD, NC-AD, NC-NC, MCI-AD, MCI-MCI, and NC-MCI AUC values in the V-V relation matrix. Across modalities, MCI-MCI and NC-MCI are generally weaker than AD-AD and NC-AD, with COGNITIVE_TEST retaining the clearest separation.

## K. SAME/DIFFERENT analysis

V-V SAME/DIFFERENT AUC ranking is led by COGNITIVE_TEST at 0.7153. V-T is also led by COGNITIVE_TEST at 0.7238.

## L. V-V and V-T difference

V-T is higher than V-V for every modality in both core AUC measures. This is a descriptive difference between unseen-to-unseen and unseen-to-training-patient pair contexts. It is not a patient-independent sample-size claim.

## M. Fold ranking stability

COGNITIVE_TEST ranks first in V-V SAME/DIFFERENT AUC in 10/10 outer folds and first in V-V six-class Macro-AUC in 10/10 folds. It also ranks first in both V-T measures in 10/10 folds. The full fold-local rank table is `results/anchor_ranking.csv`, with summary in `results/rank_stability.csv`.

## N. Graph-edge coverage diagnosis

The graph audit is read-only and uses the authenticated E83.8.4 Z0 Best-checkpoint mechanism audit. It does not retrain or substitute a checkpoint. Candidate edge coverage is small for every modality because the GraphLearn candidate support is sparse relative to all 598-patient pairs. UCSFFSX has the fewest effective edges and the lowest graph density. COGNITIVE_TEST has 13,123.9 mean effective edges, density 0.0735, and 2,491.6 mean candidate edges. The candidate support coverage remains 0.0140, so relation discriminability and graph support are separate constraints.

| Modality | Effective edges | Graph density | Candidate edges | Candidate coverage | Scored edges | Positive scored |
|---|---:|---:|---:|---:|---:|---:|
| UCSFFSX | 406.8 | 0.0023 | 1293.9 | 0.0072 | 268.7 | 140.6 |
| UCBERKELEYAV45 | 5102.8 | 0.0286 | 2084.6 | 0.0117 | 1843.0 | 911.3 |
| UPENNBIOMK9 | 75545.6 | 0.4232 | 2875.0 | 0.0161 | 2875.0 | 295.4 |
| RISK_FACTOR | 6296.7 | 0.0353 | 2623.9 | 0.0147 | 2458.0 | 1297.7 |
| COGNITIVE_TEST | 13123.9 | 0.0735 | 2491.6 | 0.0140 | 2491.6 | 1913.7 |
| ROI_AVERAGE | 27585.4 | 0.1545 | 2475.7 | 0.0139 | 2475.7 | 1510.4 |

## O. Isolation and leakage audit

All final audit states are PASS. The controlled outer-test-label permutation changed test labels while leaving every training-pair label unchanged. No test patient entered T, V, or any fitted pair set. The full status object is in `audits/final_audit.json`.

## P. Statistical reliability and limitations

Outer-fold means use population standard deviation across ten folds. Patient pairs are dependent observations, so these values are not patient-level confidence intervals. The global ranking is descriptive. The relation probe does not establish a downstream SPromptGL accuracy increase. Graph support was read from the historical Z0 mechanism audit rather than learned in P0.1. P0.1 retains P0 registered pair caps, so its estimates are not a verbatim rerun of an unfinished P0 implementation.

## Q. Difference from historical ABIDE probe

TADPOLE uses six relations and six modality-specific feature blocks. The present protocol uses inner-train-only PCA and relation-feature scaling for patient isolation. ABIDE four-modality AUC values must not be compared directly to the six-class TADPOLE Macro-AUC values without accounting for task and class definitions.

## R. DARG anchoring recommendation

COGNITIVE_TEST is a SINGLE_MODALITY_CANDIDATE for disease-relation anchoring because it leads both tasks in both validation scenarios with a large margin and first-place rank in all outer folds. This is a reliability candidate only. The low candidate-edge coverage means future DARG use must preserve a graph-support check. No gate scale was changed by this experiment, and no final ACC gain is implied.

## S. Necessary files

`protocol.json`, `source_manifest.json`, `data_manifest.json`, `code/run.py`, `code/audit.py`, `progress.sqlite`, `results/modality_summary.csv`, `results/fold_metrics.csv`, `results/class_pair_metrics.csv`, `results/anchor_ranking.csv`, `results/rank_stability.csv`, `results/paired_modality_differences.csv`, `results/modality_comparison.png`, `results/relation_heatmap.png`, and `audits/final_audit.json`.

## T. Handoff

The handoff section was appended once to `/model/spromptgl/experiment/handoff.md` after the audit and result files were finalized.

## U. Final scientific conclusion

TADPOLE contains reliable disease-relation signal in all six modalities above chance, with COGNITIVE_TEST clearly strongest for this fixed 65-dimensional relation probe. Its V-V Macro-AUC is 0.8325 and SAME/DIFFERENT AUC is 0.7153. The result supports a single-modality candidate for later reliability gating, subject to graph support and prospective outer-training-only selection. It does not authorize a gate-scale change or establish downstream classification improvement.
