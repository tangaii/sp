# Experiment 83.8.3 Results

Status: **MIXED**. Priority follow-up candidate: **Y3**. Current incumbent pending confirmation: **Y0 / E83.8.2-N3**.

## Fold Results

| Group | ACC mean ± pop. SD | AUC_original mean ± pop. SD | AUC_probability mean ± pop. SD | OOF correct | NC | MCI | AD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Y0 | 93.3079% ± 2.9019% | 94.1819% ± 2.6392% | 95.8360% ± 1.6518% | 558/598 | 202/209 | 292/315 | 64/74 |
| Y1 | 92.9689% ± 3.8098% | 94.0296% ± 3.8172% | 95.7871% ± 1.9374% | 556/598 | 201/209 | 290/315 | 65/74 |
| Y2 | 92.8079% ± 3.2569% | 94.1127% ± 3.3428% | 94.8626% ± 3.6299% | 555/598 | 200/209 | 289/315 | 66/74 |
| Y3 | 93.6441% ± 2.2192% | 94.4265% ± 2.1648% | 95.8290% ± 2.3408% | 560/598 | 205/209 | 291/315 | 64/74 |
| Y4 | 92.4661% ± 2.6455% | 94.1331% ± 2.1418% | 95.1359% ± 2.5699% | 553/598 | 199/209 | 287/315 | 67/74 |
| Y5 | 92.8079% ± 3.2628% | 93.8962% ± 2.9859% | 96.2889% ± 1.6992% | 555/598 | 203/209 | 288/315 | 64/74 |

## Fixed References

| Reference | ACC | AUC_original | AUC_probability |
|---|---:|---:|---:|
| E83.8.2 G4 / N0 | 92.9746% | 94.0967% | 96.2710% |
| E83.8.2 N3 | 93.3079% | 94.1819% | 95.8360% |
| Paper | 94.81% ± 2.95% | 95.59% ± 2.57% | not reported |

Highest ACC: **Y3** (93.6441%). Highest AUC_original: **Y3** (94.4265%). Highest probability AUC: **Y5** (96.2889%).

## Paired Contrasts

| Comparison | ΔACC pp | ΔAUC_original pp | ΔAUC_probability pp | OOF corrections / harms | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Y1 vs Y0 | -0.3390 | -0.1523 | -0.0489 | 14/16 | 0.85554 |
| Y2 vs Y0 | -0.5000 | -0.0692 | -0.9734 | 18/21 | 0.74926 |
| Y3 vs Y0 | +0.3362 | +0.2445 | -0.0070 | 16/14 | 0.85554 |
| Y4 vs Y0 | -0.8418 | -0.0489 | -0.7001 | 12/17 | 0.45826 |
| Y5 vs Y0 | -0.5000 | -0.2858 | +0.4529 | 14/17 | 0.72010 |
| Y2 vs Y1 | -0.1610 | +0.0832 | -0.9245 | 20/21 | 1.00000 |
| Y4 vs Y1 | -0.5028 | +0.1035 | -0.6512 | 13/16 | 0.71107 |
| Y5 vs Y2 | +0.0000 | -0.2166 | +1.4263 | 18/18 | 1.00000 |
| Y4 vs Y3 | -1.1780 | -0.2934 | -0.6931 | 13/20 | 0.29621 |
| Y5 vs Y3 | -0.8362 | -0.5303 | +0.4599 | 9/14 | 0.40487 |
| Y0 vs E83.8.2-N3 | +0.0000 | +0.0000 | +0.0000 | 0/0 | 1.00000 |
| Y0 vs E83.8.2-G4 | +0.3333 | +0.0853 | -0.4350 | 25/23 | 0.88543 |
| Y1 vs E83.8.2-N3 | -0.3390 | -0.1523 | -0.0489 | 14/16 | 0.85554 |
| Y1 vs E83.8.2-G4 | -0.0056 | -0.0671 | -0.4839 | 26/26 | 1.00000 |
| Y2 vs E83.8.2-N3 | -0.5000 | -0.0692 | -0.9734 | 18/21 | 0.74926 |
| Y2 vs E83.8.2-G4 | -0.1667 | +0.0161 | -1.4084 | 27/28 | 1.00000 |
| Y3 vs E83.8.2-N3 | +0.3362 | +0.2445 | -0.0070 | 16/14 | 0.85554 |
| Y3 vs E83.8.2-G4 | +0.6695 | +0.3298 | -0.4420 | 26/22 | 0.66547 |
| Y4 vs E83.8.2-N3 | -0.8418 | -0.0489 | -0.7001 | 12/17 | 0.45826 |
| Y4 vs E83.8.2-G4 | -0.5085 | +0.0364 | -1.1351 | 24/27 | 0.77977 |
| Y5 vs E83.8.2-N3 | -0.5000 | -0.2858 | +0.4529 | 14/17 | 0.72010 |
| Y5 vs E83.8.2-G4 | -0.1667 | -0.2005 | +0.0179 | 20/21 | 1.00000 |

Each Δ column is the mean paired ten-fold difference in percentage points. `paired_contrasts.csv` contains all ten fold differences and patient OOF counts. Exact McNemar p-values use paired OOF predictions. The folds reuse overlapping training sets, so these patient comparisons do not establish independent-sample significance.

Y3 meets the written priority-candidate rule by improving both primary fold means over Y0 while changing probability AUC by only -0.00698 percentage points. Its ACC and AUC_original fold wins/ties/losses are 4/4/2 and 5/2/3. Patient OOF comparison has 16 corrections and 14 harms with exact McNemar p=0.85554. This supports retaining Y3 for follow-up, not claiming a stable gain. Y0/E83.8.2-N3 remains the incumbent pending independent confirmation. The strict automated screen requiring probability AUC to be non-decreasing selected no new group. The requested no-obvious-deterioration rule is recorded separately in `audits/analysis_summary.json`.

## Mechanism

| Group | Native ACC / AUC_original / probability AUC | Full ACC / AUC_original / probability AUC | Native→Full corrected / damaged | Protection-vs-unprotected corrected / damaged |
|---|---:|---:|---:|---:|
| Y0 | 93.1412% / 94.0871% / 95.8653% | 93.3079% / 94.1819% / 95.8360% | 1 / 0 | 0 / 0 |
| Y1 | 92.9689% / 94.0296% / 95.8122% | 92.9689% / 94.0296% / 95.7871% | 0 / 0 | 0 / 0 |
| Y2 | 92.3023% / 93.8283% / 94.8247% | 92.8079% / 94.1127% / 94.8626% | 3 / 0 | 0 / 0 |
| Y3 | 93.4746% / 94.3406% / 95.8294% | 93.6441% / 94.4265% / 95.8290% | 1 / 0 | 1 / 0 |
| Y4 | 92.2994% / 94.0473% / 95.1322% | 92.4661% / 94.1331% / 95.1359% | 1 / 0 | 1 / 0 |
| Y5 | 92.8079% / 93.8962% / 96.2697% | 92.8079% / 93.8962% / 96.2889% | 0 / 0 | 0 / 0 |

The groupwise Native results measure changes caused by joint training. Native→Full counts measure direct DPSL inference changes. The protection comparison changes only the deterministic inference gate at the same Best checkpoint.

Per-fold, per-class corrections and harms, six-modality edge/gate/message diagnostics, Direct-Off and Disease-Neutral interventions, relation gradients, confidence-gate distributions, residual ratios, and class-pair participation are in `audits/mechanism_summary.json` and `results/patient_mechanism.csv`.

All six groups retain the N3 squared Frobenius ALoss, lambda 0.010 from epoch 11, with training-fold labels only. Per-modality graph gradients and M1 structural-zero status are included in the mechanism audit.

## Method Interpretation

DARG gate effects are interpreted from six per-modality graph and signed-edge diagnostics, direct-message norms, and Direct-Off/Disease-Neutral interventions. Paired group-level differences retain the factorial comparison labels and are not attributed to an isolated module when another module also differs.

The data set has already been used across multiple rounds of method selection. Small mean differences are not treated as stable gains. All groups retain full G4, the fixed TADPOLE protocol, N3 ALoss, the original prompts, and the N3 DPSL optimization limits.

Paper reference reached on both requested primary means: **False**.
