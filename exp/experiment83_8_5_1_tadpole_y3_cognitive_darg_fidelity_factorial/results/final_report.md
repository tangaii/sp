# Experiment 83.8.5.1 Final Report

## Status

**NO_GO_PERFORMANCE**. N1–N4 completed 40/40 formal folds. H0–H2 were verified and replayed for 30/30 historical folds. No added group exceeded H0 in both registered primary metrics. Probability AUC was higher than H0 for N2, N3, N4, which is secondary and does not change the primary NO-GO decision.

The final report describes the historic `val=test` and pre-CV preprocessing protocol. Its fold intervals are descriptive because training sets overlap and checkpoint selection used the test-fold ACC.

## Environment and sources

Training used one PPU-ZW810E, four workers, seed 0, PyTorch 2.9.0. The sealed training implementation fingerprint is `b1cbd93aa6b973f7bdb6c12f5bd2510ac9f101d0ab24e85d1eca650494e3686d`. The verified data SHA-256 is `0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc` and the split SHA-256 is `7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5`.

SPromptGL paper reference: ACC 94.81 ± 2.95%, AUC 95.59 ± 2.57%. Original-author source identity is pinned to `ba3b9446fa451fde6864d6555bdf324de0c4b64a`. The inspected SABER checkout is `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4`. GitHub identifies the requested TADPOLE reference `ae225e5` as full commit `ae225e5135046d1d9622968b7d011f6aa55fa60d`. The server E83.8.5 source runtime, historical runner, and R0 config match that commit's raw-file SHA-256 values. The local `fidelity_model.py` differs from the historical file only by the registered configurable GRL scalar, retaining default 0.03. The training-time sealed source manifest contains an incorrect E83.8.2 pointer, corrected by [source_provenance_correction.json](../audits/source_provenance_correction.json). Its original hash and the training seal remain preserved.

The paper reports ACC 94.81 ± 2.95% and AUC 95.59 ± 2.57%. The table below compares against those means for reference. This complete DARG+DPSL model and its historical protocol are not identical to the paper's original model, so this is not a strict reproduction claim.

## Protocol and fidelity

The six modality dimensions, NC/MCI/AD mapping, historical ten splits, seed, RFE, mean imputation, pre-CV StandardScaler, transductive graph, `val_idx == test_idx`, test-ACC Best selection, 450-epoch ceiling, `wait > 200`, 10-epoch warmup, and Paper-ALoss coefficient 0.01 were retained. Test-fold labels were excluded from relation pairs, ALoss, graph supervision, DPSL objectives, and the masked-label checkpoint replays. The test-label permutation check passed. Interpret the metrics as exploratory results under the historical protocol, not unbiased independent-generalization estimates.

DARG retained the ABIDE E74.4-G2 Positive-DCR and E82.4.6.10-G1 conditional relation alignment, BCE, preservation loss, signed edges, source graph support, and double gate. Only the registered cognitive anchor, cognitive gate scalar, and GRL scalar varied. DPSL retained the E82.4.6.10-G1 frozen MMGPL-12 prompt path, old confidence mode, 0.08 residual cap, 0.001 shared-gradient cap, class-pair objectives, and AdamW settings. Fixed-tensor ABIDE parity and the available ABIDE G1 Fold 8 checkpoint replay passed. The fixed-tensor regression covers the historical four-modality/eight-prompt route. No full ABIDE fold retraining was performed.

## Seven-group results

Metrics are mean ± population standard deviation across ten folds. AUC_original is hard one-hot macro OVR. The OOF correctness column is Full output with NC/MCI/AD counts.

| Group | ACC % | AUC_original % | AUC_probability % | OOF correct | ΔACC vs paper pp | ΔAUC vs paper pp |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 93.6441 ± 2.2192 | 94.4265 ± 2.1648 | 95.8290 ± 2.3408 | 560/598 (205/291/64) | -1.166 | -1.164 |
| H1 | 92.6384 ± 2.2781 | 94.1104 ± 2.1786 | 96.4483 ± 1.9012 | 554/598 (201/287/66) | -2.172 | -1.480 |
| H2 | 92.6328 ± 3.3887 | 93.7182 ± 3.1780 | 96.7077 ± 1.6548 | 554/598 (202/288/64) | -2.177 | -1.872 |
| N1 | 92.4689 ± 2.4097 | 93.7285 ± 2.6081 | 95.6975 ± 2.4111 | 553/598 (200/288/65) | -2.341 | -1.862 |
| N2 | 92.4746 ± 3.4370 | 94.1033 ± 2.8752 | 96.2194 ± 2.4589 | 553/598 (200/286/67) | -2.335 | -1.487 |
| N3 | 92.8079 ± 3.3468 | 94.0774 ± 3.1354 | 96.0405 ± 2.1173 | 555/598 (203/287/65) | -2.002 | -1.513 |
| N4 | 92.4689 ± 3.2931 | 94.1991 ± 2.6475 | 95.9608 ± 2.2407 | 553/598 (202/284/67) | -2.341 | -1.391 |

The OOF cohort contains 598 patients exactly once per group. `group_summary.csv`, `fold_metrics.csv`, and `oof_predictions.csv` contain fold values, Native/Full results, class counts, identifiers, and patient predictions. Deltas to paper means are in `group_summary.csv`.

## Paired comparisons

Each cell reports Full-mode mean paired fold change in percentage points, its 95% Student-t interval over ten paired folds, fold wins/ties/losses, and pooled OOF corrected/damaged counts with net change. The CSV includes Native comparisons, AUC_probability, exact McNemar tests, Holm-adjusted fold p-values, and same-checkpoint interventions.

| Contrast | ΔACC pp [95% CI] | W/T/L | ΔAUC_original pp [95% CI] | W/T/L | OOF corrected/damaged (net) |
|---|---:|---:|---:|---:|---:|
| N1-H1 | -0.169 [-1.216, +0.877] | 2/6/2 | -0.382 [-1.240, +0.476] | 3/3/4 | 18/19 (-1) |
| N1-H0 | -1.175 [-2.563, +0.213] | 1/4/5 | -0.698 [-1.778, +0.382] | 3/2/5 | 13/20 (-7) |
| N2-N1 | +0.006 [-2.023, +2.035] | 5/1/4 | +0.375 [-0.679, +1.428] | 7/1/2 | 23/23 (+0) |
| N3-N1 | +0.339 [-1.334, +2.012] | 4/4/2 | +0.349 [-1.030, +1.728] | 5/3/2 | 18/16 (+2) |
| N4-N2 | -0.006 [-1.500, +1.489] | 2/5/3 | +0.096 [-0.852, +1.043] | 5/1/4 | 15/15 (+0) |
| N4-N3 | -0.339 [-1.099, +0.421] | 1/6/3 | +0.122 [-1.019, +1.262] | 3/3/4 | 15/17 (-2) |
| N4-H0 | -1.175 [-2.673, +0.322] | 2/2/6 | -0.227 [-1.403, +0.948] | 4/2/4 | 16/23 (-7) |
| N2-H2 | -0.158 [-1.807, +1.491] | 2/4/4 | +0.385 [-0.966, +1.737] | 3/2/5 | 18/19 (-1) |
| N2-H0 | -1.169 [-2.553, +0.214] | 1/4/5 | -0.323 [-1.285, +0.639] | 2/2/6 | 15/22 (-7) |
| N3-H0 | -0.836 [-2.350, +0.678] | 2/4/4 | -0.349 [-1.803, +1.105] | 3/4/3 | 16/21 (-5) |

The intervals and tests are descriptive because the ten CV training sets overlap and Best selection used `val=test`. [paired_contrasts.csv](paired_contrasts.csv) retains all registered comparisons and corrections.

The 2×2 factor contrasts use paired fold outcomes and show the registered scalar changes only.

| Factor contrast | ΔACC pp [95% CI] | ΔAUC_original pp [95% CI] |
|---|---:|---:|
| Gate .01→.03 at GRL .03 | +0.006 [-2.023, +2.035] | +0.375 [-0.679, +1.428] |
| Gate .01→.03 at GRL .015 | -0.339 [-1.099, +0.421] | +0.122 [-1.019, +1.262] |
| GRL .03→.015 at Gate .01 | +0.339 [-1.334, +2.012] | +0.349 [-1.030, +1.728] |
| GRL .03→.015 at Gate .03 | -0.006 [-1.500, +1.489] | +0.096 [-0.852, +1.043] |

## Mechanism results

The relation AUCs use each trained DARG RelationEncoder and RelationScorer on held-out patient pairs. The three AUC columns are all test pairs, pairs inside original graph support, and pairs outside support as a diagnostic only.

| Group | Cognitive relation AUC all/support/outside | Effective Gate | GCN message change vs disease-neutral | Adversarial loss / discriminator accuracy |
|---|---:|---:|---:|---:|
| H0 | 0.7986 / 0.7465 / 0.7984 | 0.00100 | 0.000028 | 1.5397 / 0.4588 |
| H1 | 0.7985 / 0.7562 / 0.7981 | 0.00100 | 0.000027 | 1.5237 / 0.4632 |
| H2 | 0.8191 / 0.7863 / 0.8195 | 0.00982 | 0.000689 | 1.5001 / 0.4712 |
| N1 | 0.7832 / 0.7239 / 0.7833 | 0.00100 | 0.000060 | 1.5584 / 0.4522 |
| N2 | 0.7908 / 0.7359 / 0.7895 | 0.00298 | 0.000151 | 1.5533 / 0.4627 |
| N3 | 0.7942 / 0.7226 / 0.7940 | 0.00100 | 0.000055 | 1.5246 / 0.4733 |
| N4 | 0.7589 / 0.6686 / 0.7604 | 0.00298 | 0.000084 | 1.6114 / 0.4470 |

The trained COGNITIVE_TEST RelationEncoder/RelationScorer had the highest held-out same/different AUC among the six modalities in all seven groups, ranging from 0.7589 to 0.8191. The relation advantage was therefore present in formal DARG scoring. The six-modality per-fold AUC, same/different train-pair score distributions, conditional centroid distances, signed edge counts, graph support coverage, message norms, and positive/negative propagation counterfactuals are in [mechanism_summary.json](../audits/mechanism_summary.json). Candidate and scored edges remained within the registered original graph support. Cognitive Gate 0.03 produced a larger effective Gate and larger graph perturbation than 0.01, while absolute GCN message changes stayed small. Direct-DCR message norms are separately recorded and are much smaller than the raw representation scale.

At the trained checkpoints, the cognitive anchor has zero encoder gradient from conditional adversarial alignment and nonzero relation-BCE encoder/scorer gradients. The other five encoders and the discriminator show finite gradients. This confirms anchor detach is branch-specific, not a frozen cognitive encoder. The configured GRL comparison uses N1 vs N3 and N2 vs N4. Their paired classification and relation-score changes do not support a uniform improvement from GRL 0.015.

Native is the DARG-trained classifier readout. Full versus Native is the same-checkpoint DPSL inference effect. The table reports per-group OOF corrections and harms. Residual-Off reproduced Native, while Full retained a nonzero bounded residual. The unchanged DPSL design produced small, group-dependent direct correction counts.

| Group | Native→Full corrected/damaged (net) | Direct-Off→Full corrected/damaged | Disease-Neutral→Full corrected/damaged | Cognitive-Graph-Off→Full corrected/damaged | Residual ratio max / shared-gradient ratio max |
|---|---:|---:|---:|---:|---:|
| H0 | 1/0 (+1) | 0/0 | 0/0 | 0/0 | 0.0519 / 0.001000 |
| H1 | 0/0 (+0) | 0/0 | 0/0 | 0/0 | 0.0472 / 0.001000 |
| H2 | 3/0 (+3) | 0/0 | 0/0 | 0/0 | 0.0406 / 0.001000 |
| N1 | 2/0 (+2) | 0/0 | 0/0 | 0/0 | 0.0529 / 0.001000 |
| N2 | 1/0 (+1) | 0/0 | 0/0 | 0/0 | 0.0333 / 0.001000 |
| N3 | 0/0 (+0) | 0/0 | 0/0 | 0/0 | 0.0380 / 0.001000 |
| N4 | 0/0 (+0) | 0/0 | 0/0 | 0/0 | 0.0285 / 0.001000 |

## Checkpoint and log audits

Historical and new Best checkpoints independently replayed 70/70. Latest states restored 70/70 and deterministic next steps matched on 70/70. Historical R0 Best checkpoints lack embedded RNG state in the archive for 10/10 folds; their real Latest states contain RNG and passed restore/continuation. No missing historical RNG was synthesized. ABIDE checkpoint replay and fixed-tensor parity are in [abide_checkpoint_replay.json](../audits/abide_checkpoint_replay.json) and [abide_tadpole_parity.json](../audits/abide_tadpole_parity.json).

The four author-format logs were reconstructed from stored epoch histories and fold predictions, with no training rerun or RNG use. All four passed exact line-by-line metric and format checks against `/data/adodas/SPromptGL-main/result_Tadpole.txt`; the epoch denominator is `/450` and the three `trn-loss-MF` fields retain their audited display definitions. See [log_format_audit.json](../audits/log_format_audit.json).

## Research judgement

1. N1 isolates the cognitive edge-anchor change relative to H1. The scorer uses the cognitive relation signal, but N1 does not show a paired performance gain over H1 or H0.
2. Gate 0.03 increases graph perturbation, but factor contrasts are mixed and the resulting message changes remain small. It is not more reasonable as a default from this experiment.
3. GRL 0.015 modestly improves some paired outcomes, especially N3 versus N1, but the gate 0.03 pair does not show consistent ACC and AUC_original gain.
4. N4 does not show complementary benefit. It does not exceed H0 on either registered primary metric.
5. The trained COGNITIVE_TEST scorer has the highest same/different held-out AUC among the six modalities in every group. The relation advantage reaches DARG scoring, but not the final classifier performance.
6. DARG changes supported cognitive graph weights and messages, with larger changes at Gate 0.03. Cognitive-Graph-Off caused zero argmax flips in all seven groups and at most 6.1e-5 absolute logit change, so its direct classification effect is negligible under these scales.
7. DPSL remains method-faithful, within its residual and shared-gradient caps, with reproducible Native/Full differences and modest corrections.
8. No new group simultaneously exceeds Y3/Z0 ACC and AUC_original.
9. No new group has paired evidence supporting replacement of Y3/Z0. Keep H0/Y3/Z0 as incumbent.
10. ABIDE E82.4.6.10-G1 and the TADPOLE adaptation retain the same generic DARG+DPSL core, with required modality, anchor, class-count, and prompt adaptations.

**Final decision: NO_GO_PERFORMANCE. Preserve Y3/Z0 as the complete-model base.**

Artifacts: [final_audit.json](../audits/final_audit.json), [protocol.json](../protocol.json), [source_manifest.json](../source_manifest.json), and [checkpoint_replay.json](../audits/checkpoint_replay.json).
