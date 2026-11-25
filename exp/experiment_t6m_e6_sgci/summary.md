# E6-SGCI summary

STATUS: COMPLETE_AUDITED. All 8 groups completed 10/10 folds, 300 epochs per fold, with independent checkpoint reload and OOF checks.

| Group | ACC | Hard-AUC | Prob-AUC | OOF |
|---|---:|---:|---:|---:|
| G0_FULL_LEGACY | 91.811 ± 1.876% | 93.559 ± 2.451% | 96.585 ± 1.888% | 549 |
| G1_RES_ONLY_LEGACY | 91.969 ± 2.888% | 93.438 ± 2.880% | 96.776 ± 1.812% | 550 |
| G2_RES_ONLY_R04 | 92.305 ± 2.144% | 93.306 ± 2.330% | 96.851 ± 2.359% | 552 |
| G3_REAL_SPGS_R04 | 91.972 ± 2.342% | 93.043 ± 2.462% | 95.914 ± 2.032% | 550 |
| G4_NULL_SPGS_R04 | 91.802 ± 2.948% | 93.609 ± 3.116% | 95.762 ± 2.308% | 549 |
| G5_RES_ONLY_R20 | 92.138 ± 1.847% | 93.649 ± 2.350% | 96.170 ± 1.738% | 551 |
| G6_REAL_SPGS_R20 | 92.475 ± 2.912% | 93.098 ± 2.776% | 96.243 ± 1.753% | 553 |
| G7_NULL_SPGS_R20 | 92.305 ± 2.816% | 92.489 ± 3.017% | 95.933 ± 2.118% | 552 |

## Paired comparisons

- G1_RES_ONLY_LEGACY: net=1, wrong-to-correct=19, correct-to-wrong=18, McNemar p=1, bootstrap 95% CI=[-0.01839464882943144, 0.021739130434782608]
- G2_RES_ONLY_R04: net=3, wrong-to-correct=20, correct-to-wrong=17, McNemar p=0.742829, bootstrap 95% CI=[-0.015050167224080268, 0.02508361204013378]
- G3_REAL_SPGS_R04: net=1, wrong-to-correct=21, correct-to-wrong=20, McNemar p=1, bootstrap 95% CI=[-0.01839464882943144, 0.023411371237458192]
- G4_NULL_SPGS_R04: net=0, wrong-to-correct=25, correct-to-wrong=25, McNemar p=1, bootstrap 95% CI=[-0.023411371237458192, 0.023411371237458192]
- G5_RES_ONLY_R20: net=2, wrong-to-correct=20, correct-to-wrong=18, McNemar p=0.871415, bootstrap 95% CI=[-0.016722408026755852, 0.023411371237458192]
- G6_REAL_SPGS_R20: net=4, wrong-to-correct=23, correct-to-wrong=19, McNemar p=0.643969, bootstrap 95% CI=[-0.015050167224080268, 0.028428093645484948]
- G7_NULL_SPGS_R20: net=3, wrong-to-correct=23, correct-to-wrong=20, McNemar p=0.760792, bootstrap 95% CI=[-0.016722408026755852, 0.026755852842809364]

Residual calibration checks passed for all six calibrated groups. REAL semantic MLP gradients were nonzero. NULL semantic vectors were identical and softmax-max checks were 1/3.

The paper target gap is computed against 94.81% ACC and 95.59% AUC. AUC algorithm equivalence remains unconfirmed.

## Requested contrasts

- G3 vs G2: net -2 OOF correct, 18/20 directional transitions, McNemar p=0.8714, bootstrap 95% CI [-2.34%, 1.67%].
- G3 vs G4: net +1, 24/23 transitions, McNemar p=1.0000, bootstrap 95% CI [-2.17%, 2.34%].
- G6 vs G5: net +2, 22/20 transitions, McNemar p=0.8776, bootstrap 95% CI [-1.84%, 2.51%].
- G6 vs G7: net +1, 19/18 transitions, McNemar p=1.0000, bootstrap 95% CI [-1.84%, 2.17%].
- G6 vs G3: net +3, 22/19 transitions, McNemar p=0.7552, bootstrap 95% CI [-1.67%, 2.51%].
- G5 vs G2: net -1, 24/25 transitions, McNemar p=1.0000, bootstrap 95% CI [-2.51%, 2.17%].
