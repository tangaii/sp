# TADPOLE six-modal graph adaptation summary

## A. Completion and protocol

All five formal groups completed 10/10 folds with 300 epochs per fold. The input snapshot is 598 subjects with six modalities `[145, 148, 3, 36, 24, 9]`, class counts NC 209, MCI 315, AD 74. Every group uses the same fold manifest, prompt features, optimizer settings, training protocol, and `val=test` first-maximal-ACC checkpoint rule. This selection rule uses the outer test fold and is retained only for compatibility with the historical experiment.

The copied implementation has six GraphLearn and six GCN modules. Graph structure loss is averaged over six modalities. The legacy GFR path remains unchanged, so relation 0 is overwritten by relation 1 in the fusion expression. This is a known difference from a literal six-relation weighted-sum reading of the paper and was intentionally held fixed to isolate the requested graph-construction mechanisms.

## B. Results

| Method | ACC | Hard-AUC | Prob-AUC | NC SEN | MCI SEN | AD SEN | OOF correct | Δ correct vs G0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper TADPOLE | 94.81 ± 2.95 | 95.59 ± 2.57 | N/A | N/A | N/A | N/A | N/A | N/A |
| Historical best `mmgpl_class_graph_sup` | 91.81 ± 1.88 | 93.56 ± 2.45 | 96.59 ± 1.89 | N/A | N/A | N/A | N/A | N/A |
| G0 baseline rerun | 91.81 ± 1.88 | 93.56 ± 2.45 | 96.59 ± 1.89 | 93.31 ± 4.35 | 91.11 ± 3.43 | 90.71 ± 10.47 | 549 | 0 |
| G1 modal-adaptive graph | 91.80 ± 2.87 | 93.07 ± 2.63 | 96.10 ± 2.02 | 90.90 ± 5.41 | 93.35 ± 5.68 | 88.57 ± 14.22 | 549 | 0 |
| G2 gradient alignment | 83.79 ± 6.08 | 88.36 ± 4.36 | 93.68 ± 2.32 | 88.43 ± 9.12 | 79.36 ± 10.98 | 89.64 ± 11.00 | 501 | -48 |
| G3 class-balanced graph | 81.76 ± 3.70 | 86.81 ± 3.36 | 90.79 ± 3.31 | 85.07 ± 7.64 | 78.05 ± 6.77 | 88.39 ± 10.45 | 489 | -60 |
| G4 combined GAR | 81.77 ± 3.71 | 85.81 ± 4.04 | 90.32 ± 2.91 | 83.71 ± 4.92 | 80.60 ± 7.04 | 81.61 ± 18.75 | 489 | -60 |

The OOF counts are based on all 598 predictions. Relative to G0, G1 has the same number of correct OOF predictions while reducing MCI errors from 28 to 21. G2, G3, and G4 lose 48, 60, and 60 correct predictions respectively.

## C. Mechanism and graph audit

G1 produced approximately 3% off-diagonal graph density in every modality. Mean densities by relation were `[0.0298, 0.0296, 0.0300, 0.0301, 0.0300, 0.0301]`. This confirms the adaptive quantile mechanism, while same-class and different-class edge rates remained modality dependent. Relation 4 retained the strongest same-class enrichment, with mean same-class rate 0.0672 and different-class rate 0.0067.

G2 preserved the 0.9 forward threshold but increased propagation density, especially relation 2 at 0.2910 and relation 5 at 0.1200. The first-epoch gradient audit was finite and nonzero for all six GraphLearn projections. The performance decrease indicates that the added gradient path destabilized the current legacy training objective under the fixed learning rate.

G3 preserved the original hard graphs. Its class-relation supervision was computed from training-node pairs only, with diagonal pairs excluded and positive and negative relation groups averaged separately. It increased several graph densities during training and reduced MCI sensitivity to 78.05%.

G4 combined the three mechanisms and passed finite-gradient checks, but its class-balanced objective plus adaptive straight-through propagation produced 81.77% ACC. It did not show complementary gains.

Relation 0 has no effective contribution in the legacy fusion path because the relation-1 update overwrites the earlier message. The recorded sigmoid fusion weights therefore describe parameters, not an active relation-0 contribution to the final fused tensor.

## D. Reproduction correctness and paper gap

G0 reproduces the historical best group at the reported precision, validating the isolated input copy, prompt setup, fold split, checkpoint rule, and shared training implementation. The best formal group by ACC is G0 at 91.81%. G1 is effectively tied in OOF correct count and improves MCI error count, but its fold mean is 0.01 percentage point lower than G0.

Against the paper's 94.81% ACC and 95.59% hard-label AUC, G0 is lower by 3.00 and 2.03 percentage points. G1 is lower by 3.01 and 2.52 points. No requested adaptation closes the paper gap.

## E. Scientific decision

**GO**: retain G1 as a controlled candidate for follow-up because it equalized nominal graph density across modalities and reduced MCI OOF errors without changing the total number of correct subjects.

**NO-GO**: do not retain G2, G3, or G4 as current training mechanisms. Their fixed-protocol ten-fold results are substantially below G0, and G4 does not provide complementary benefit.

The immediate unresolved issue is interaction between graph objectives and the legacy GFR fusion path. Adaptive density alone is not sufficient to improve ACC, and the straight-through path causes substantial propagation-density expansion under the fixed threshold and learning rate. The next single research direction should isolate a gradient-preserving graph objective with explicit monitoring of relation-specific propagation density while keeping the legacy fusion and class supervision unchanged.

## F. Audit artifacts

`audit/audit.json` contains independent fold and OOF metric recomputation, confusion matrices, MCI error counts, relation densities, thresholds, fusion weights, and metric-file hashes. Each group directory contains `best.pt`, `history.csv`, `predictions.csv`, `metrics.json`, `next_fold_rng.pt`, `splits.json`, `config.json`, and `summary.json`.

The relation audit's density statistic excludes diagonal entries. Same-class and different-class rates retain the historical diagnostic convention, which includes diagonal same-class entries in the reported train-pair mask. This convention is preserved for direct comparison with `tadpole_repair`.
