# Experiment 83.8.4 Results

Status: **STATISTICAL_EVIDENCE_INSUFFICIENT**. Next candidate to retain: **Z0**.

## Fold Results

| Group | ACC mean ± pop. SD | AUC_original mean ± pop. SD | AUC_probability mean ± pop. SD | OOF correct | NC | MCI | AD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Z0 | 93.6441% ± 2.2192% | 94.4265% ± 2.1648% | 95.8290% ± 2.3408% | 560/598 | 205/209 | 291/315 | 64/74 |
| Z1 | 92.4718% ± 3.4460% | 93.6815% ± 3.2352% | 96.1087% ± 2.0806% | 553/598 | 203/209 | 286/315 | 64/74 |
| Z2 | 92.9689% ± 3.0846% | 92.9926% ± 3.4814% | 96.1549% ± 2.9815% | 556/598 | 199/209 | 297/315 | 60/74 |
| Z3 | 92.1356% ± 3.2730% | 93.4549% ± 3.1404% | 96.1636% ± 1.3123% | 551/598 | 200/209 | 287/315 | 64/74 |
| Z4 | 92.9746% ± 2.7816% | 93.7534% ± 2.9948% | 95.5786% ± 3.0523% | 556/598 | 202/209 | 291/315 | 63/74 |
| Z5 | 91.8023% ± 3.7696% | 92.5507% ± 3.7959% | 95.2070% ± 2.6046% | 549/598 | 198/209 | 290/315 | 61/74 |

## Fixed References

| Reference | ACC | AUC_original | AUC_probability |
|---|---:|---:|---:|
| E83.8.2 G4 / N0 | 92.9746% | 94.0967% | 96.2710% |
| E83.8.2 N3 | 93.3079% | 94.1819% | 95.8360% |
| Paper | 94.81% ± 2.95% | 95.59% ± 2.57% | not reported |

Highest ACC: **Z0** (93.6441%). Highest AUC_original: **Z0** (94.4265%). Highest probability AUC: **Z3** (96.1636%).

## Paired Contrasts

| Comparison | ΔACC pp | ΔAUC_original pp | ΔAUC_probability pp | OOF corrections / harms | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Z1 vs Z0 | -1.1723 | -0.7450 | +0.2797 | 6/13 | 0.16707 |
| Z2 vs Z0 | -0.6751 | -1.4338 | +0.3260 | 17/21 | 0.62710 |
| Z3 vs Z0 | -1.5085 | -0.9715 | +0.3347 | 9/18 | 0.12208 |
| Z4 vs Z0 | -0.6695 | -0.6731 | -0.2504 | 11/15 | 0.55720 |
| Z5 vs Z0 | -1.8418 | -1.8758 | -0.6220 | 17/28 | 0.13516 |
| Z2 vs Z1 | +0.4972 | -0.6889 | +0.0462 | 23/20 | 0.76079 |
| Z4 vs Z1 | +0.5028 | +0.0719 | -0.5301 | 16/13 | 0.71107 |
| Z5 vs Z2 | -1.1667 | -0.4419 | -0.9480 | 16/23 | 0.33678 |
| Z4 vs Z3 | +0.8390 | +0.2984 | -0.5850 | 18/13 | 0.47313 |
| Z5 vs Z3 | -0.3333 | -0.9043 | -0.9566 | 19/21 | 0.87463 |
| Z0 vs E83.8.2-N3 | +0.3362 | +0.2445 | -0.0070 | 16/14 | 0.85554 |
| Z0 vs E83.8.2-G4 | +0.6695 | +0.3298 | -0.4420 | 26/22 | 0.66547 |
| Z1 vs E83.8.2-N3 | -0.8362 | -0.5005 | +0.2727 | 13/18 | 0.47313 |
| Z1 vs E83.8.2-G4 | -0.5028 | -0.4152 | -0.1623 | 25/28 | 0.78385 |
| Z2 vs E83.8.2-N3 | -0.3390 | -1.1893 | +0.3190 | 18/20 | 0.87141 |
| Z2 vs E83.8.2-G4 | -0.0056 | -1.1040 | -0.1161 | 27/27 | 1.00000 |
| Z3 vs E83.8.2-N3 | -1.1723 | -0.7270 | +0.3277 | 11/18 | 0.26493 |
| Z3 vs E83.8.2-G4 | -0.8390 | -0.6417 | -0.1074 | 22/27 | 0.56817 |
| Z4 vs E83.8.2-N3 | -0.3333 | -0.4286 | -0.2574 | 13/15 | 0.85055 |
| Z4 vs E83.8.2-G4 | +0.0000 | -0.3433 | -0.6924 | 24/24 | 1.00000 |
| Z5 vs E83.8.2-N3 | -1.5056 | -1.6313 | -0.6290 | 13/22 | 0.17547 |
| Z5 vs E83.8.2-G4 | -1.1723 | -1.5460 | -1.0640 | 24/31 | 0.41875 |
| DARG_scale_x_DPSL_mode_interaction | -1.6695 | -0.5138 | -0.4179 | n/a | n/a |

Each Δ column is the mean paired ten-fold difference in percentage points. `paired_contrasts.csv` contains all ten fold differences and patient OOF counts. Exact McNemar p-values use paired OOF predictions. The folds reuse overlapping training sets, so these patient comparisons do not establish independent-sample significance.

## Mechanism

| Group | Native ACC / AUC_original / probability AUC | Full ACC / AUC_original / probability AUC | Native→Full corrected / damaged | Protection-vs-unprotected corrected / damaged |
|---|---:|---:|---:|---:|
| Z0 | 93.4746% / 94.3406% / 95.8294% | 93.6441% / 94.4265% / 95.8290% | 1 / 0 | 1 / 0 |
| Z1 | 92.4718% / 93.7256% / 96.1045% | 92.4718% / 93.6815% / 96.1087% | 1 / 1 | 0 / 0 |
| Z2 | 92.4661% / 92.7155% / 96.1312% | 92.9689% / 92.9926% / 96.1549% | 3 / 0 | 0 / 0 |
| Z3 | 91.9689% / 93.3714% / 96.1560% | 92.1356% / 93.4549% / 96.1636% | 1 / 0 | 0 / 0 |
| Z4 | 92.8051% / 93.6585% / 95.5629% | 92.9746% / 93.7534% / 95.5786% | 1 / 0 | 0 / 0 |
| Z5 | 91.4633% / 92.3215% / 95.2073% | 91.8023% / 92.5507% / 95.2070% | 2 / 0 | 0 / 0 |

The groupwise Native results measure changes caused by joint training. Native→Full counts measure direct DPSL inference changes. The protection comparison changes only the deterministic inference gate at the same Best checkpoint.

Per-fold, per-class corrections and harms, six-modality edge/gate/message diagnostics, Direct-Off and Disease-Neutral interventions, relation gradients, confidence-gate distributions, residual ratios, and class-pair participation are in `audits/mechanism_summary.json` and `results/patient_mechanism.csv`.

All six groups retain the N3 squared Frobenius ALoss, lambda 0.010 from epoch 11, with training-fold labels only. Per-modality graph gradients and M1 structural-zero status are included in the mechanism audit.

## Method Interpretation

DARG gate effects are interpreted from six per-modality graph and signed-edge diagnostics, direct-message norms, and Direct-Off/Disease-Neutral interventions. Paired group-level differences retain the factorial comparison labels and are not attributed to an isolated module when another module also differs.

The data set has already been used across multiple rounds of method selection. Small mean differences are not treated as stable gains. All groups retain full G4, the fixed TADPOLE protocol, N3 ALoss, the original prompts, and the N3 DPSL optimization limits.

Paper reference reached on both requested primary means: **False**.

## Registered Mechanism Interpretation

The historical Positive-DCR path uses `delta = old * (REWEIGHT_STRENGTH * gate * relation)`. Its bounded direct residual is computed from the resulting `delta_adjacency`, then the historical direct path applies `direct = direct_scale * gate * bounded`. This is the registered double-gate condition in Z0 and Z3. Z1, Z2, Z4, and Z5 remove only the second direct-path multiplication. Z2 and Z5 use the preregistered `direct_scale=0.10`, while Z1 and Z4 use `0.02`.

Z0, Z1, and Z2 use `g_old = 4 * p1 * p2`. Z3, Z4, and Z5 use `g_new = 4 * p1 * p2 / ((p1 + p2)^2 + 1e-12)` with detached native logits and clipping to `[0, 1]`. Direct-Off and Disease-Neutral changed no Best-checkpoint predictions in the audited runs, so the observed group differences are joint training outcomes rather than isolated inference-only DARG effects.

The post-seal corrections were limited to audit and reporting utilities under `audits/`. The sealed training implementation SHA remained unchanged. Final status is `COMPLETE_AUDITED` for execution and audit coverage, with `STATISTICAL_EVIDENCE_INSUFFICIENT` performance evidence and `NO-GO` for replacing Y3.
