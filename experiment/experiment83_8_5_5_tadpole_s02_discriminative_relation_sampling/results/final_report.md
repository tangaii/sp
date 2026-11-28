# Experiment 83.8.5.5 Results

The protocol uses the historical transductive TADPOLE evaluation. Validation and test share the outer fold, and full-cohort RFE and scaling precede outer CV. These results are exploratory and are not an unbiased independent-generalization estimate.

## Full-model results

Fold means use population SD. AUC_original is hard-prediction macro OVR and remains the paper-comparison metric. AUC_probability and pooled OOF AUCs are reported separately.

| Group | Model | ACC fold mean +/- SD | AUC_original fold mean +/- SD | AUC_probability fold mean +/- SD | OOF correct |
|---|---|---:|---:|---:|---:|
| H0 | Native | 93.4746 +/- 2.1801% | 94.3406 +/- 2.1498% | 95.8294 +/- 2.3250% | 559/598 |
| H0 | Full | 93.6441 +/- 2.2192% | 94.4265 +/- 2.1648% | 95.8290 +/- 2.3408% | 560/598 |
| H1 | Native | 93.1384 +/- 2.8488% | 94.2624 +/- 3.0378% | 96.0132 +/- 2.6051% | 557/598 |
| H1 | Full | 93.4746 +/- 3.0327% | 94.4340 +/- 3.1326% | 96.0281 +/- 2.6419% | 559/598 |
| U1 | Native | 92.9746 +/- 2.4562% | 94.0948 +/- 2.7991% | 96.5995 +/- 1.5393% | 556/598 |
| U1 | Full | 92.9746 +/- 2.4562% | 94.0948 +/- 2.7991% | 96.6255 +/- 1.5429% | 556/598 |
| U2 | Native | 92.3023 +/- 2.6229% | 93.4962 +/- 2.6572% | 95.3453 +/- 2.5785% | 552/598 |
| U2 | Full | 92.3023 +/- 2.6229% | 93.4962 +/- 2.6572% | 95.3320 +/- 2.5600% | 552/598 |
| U3 | Native | 92.6356 +/- 3.2049% | 93.7604 +/- 2.8073% | 95.9018 +/- 2.1185% | 554/598 |
| U3 | Full | 92.6356 +/- 3.2049% | 93.7604 +/- 2.8073% | 95.9147 +/- 2.1195% | 554/598 |
| U4 | Native | 92.6441 +/- 3.5862% | 93.8165 +/- 3.1070% | 95.8802 +/- 2.0558% | 554/598 |
| U4 | Full | 92.6441 +/- 3.5862% | 93.8165 +/- 3.1070% | 95.8808 +/- 2.0563% | 554/598 |
| U5 | Native | 91.6356 +/- 3.0891% | 92.5128 +/- 2.9905% | 96.1031 +/- 1.8416% | 548/598 |
| U5 | Full | 92.1356 +/- 3.1870% | 92.8741 +/- 3.2359% | 96.1108 +/- 1.8392% | 551/598 |
| U6 | Native | 92.6328 +/- 4.0600% | 93.7548 +/- 3.5628% | 95.1395 +/- 2.9200% | 554/598 |
| U6 | Full | 92.8023 +/- 3.8266% | 93.8496 +/- 3.4158% | 95.1596 +/- 2.9079% | 555/598 |

## Pooled OOF results

Pooled OOF metrics aggregate all 598 one-time held-out predictions and are distinct from the unweighted mean of ten fold metrics.

| Group | Model | OOF correct | OOF ACC | OOF AUC_original | OOF AUC_probability |
|---|---|---:|---:|---:|---:|
| H0 | Native | 559/598 | 93.4783% | 94.3208% | 95.5207% |
| H0 | Full | 560/598 | 93.6455% | 94.4055% | 95.5023% |
| H1 | Native | 557/598 | 93.1438% | 94.2208% | 95.2913% |
| H1 | Full | 559/598 | 93.4783% | 94.3903% | 95.3016% |
| U1 | Native | 556/598 | 92.9766% | 93.9964% | 96.1463% |
| U1 | Full | 556/598 | 92.9766% | 93.9964% | 96.1353% |
| U2 | Native | 552/598 | 92.3077% | 93.4790% | 95.3903% |
| U2 | Full | 552/598 | 92.3077% | 93.4790% | 95.3979% |
| U3 | Native | 554/598 | 92.6421% | 93.7510% | 95.7677% |
| U3 | Full | 554/598 | 92.6421% | 93.7510% | 95.7543% |
| U4 | Native | 554/598 | 92.6421% | 93.7342% | 95.3312% |
| U4 | Full | 554/598 | 92.6421% | 93.7342% | 95.3379% |
| U5 | Native | 548/598 | 91.6388% | 92.4009% | 96.0479% |
| U5 | Full | 551/598 | 92.1405% | 92.7629% | 96.0473% |
| U6 | Native | 554/598 | 92.6421% | 93.7244% | 94.5739% |
| U6 | Full | 555/598 | 92.8094% | 93.8201% | 94.5906% |

## Class results

Confusion-matrix rows are true NC/MCI/AD and columns are predicted NC/MCI/AD.

| Group | Model | Correct NC / MCI / AD | Confusion matrix |
|---|---|---:|---|
| H0 | Native | 205/290/64 | `[[205,4,0],[15,290,10],[0,10,64]]` |
| H0 | Full | 205/291/64 | `[[205,4,0],[15,291,9],[0,10,64]]` |
| H1 | Native | 203/289/65 | `[[203,6,0],[17,289,9],[0,9,65]]` |
| H1 | Full | 203/291/65 | `[[203,6,0],[17,291,7],[0,9,65]]` |
| U1 | Native | 200/291/65 | `[[200,9,0],[15,291,9],[0,9,65]]` |
| U1 | Full | 200/291/65 | `[[200,9,0],[15,291,9],[0,9,65]]` |
| U2 | Native | 201/287/64 | `[[201,8,0],[18,287,10],[0,10,64]]` |
| U2 | Full | 201/287/64 | `[[201,8,0],[18,287,10],[0,10,64]]` |
| U3 | Native | 199/290/65 | `[[199,10,0],[17,290,8],[0,9,65]]` |
| U3 | Full | 199/290/65 | `[[199,10,0],[17,290,8],[0,9,65]]` |
| U4 | Native | 203/287/64 | `[[203,6,0],[20,287,8],[0,10,64]]` |
| U4 | Full | 203/287/64 | `[[203,6,0],[20,287,8],[0,10,64]]` |
| U5 | Native | 199/288/61 | `[[199,10,0],[21,288,6],[0,13,61]]` |
| U5 | Full | 201/289/61 | `[[201,8,0],[21,289,5],[0,13,61]]` |
| U6 | Native | 202/288/64 | `[[202,7,0],[16,288,11],[0,10,64]]` |
| U6 | Full | 202/289/64 | `[[202,7,0],[15,289,11],[0,10,64]]` |

## Per-fold metrics

| Group | Fold | Native ACC | Native AUC_original | Native AUC_probability | Full ACC | Full AUC_original | Full AUC_probability |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 1 | 0.966667 | 0.982165 | 0.980348 | 0.966667 | 0.982165 | 0.980720 |
| H0 | 2 | 0.933333 | 0.962073 | 0.948492 | 0.933333 | 0.962073 | 0.948492 |
| H0 | 3 | 0.950000 | 0.951274 | 0.989184 | 0.950000 | 0.951274 | 0.989184 |
| H0 | 4 | 0.900000 | 0.922937 | 0.952839 | 0.900000 | 0.922937 | 0.952839 |
| H0 | 5 | 0.950000 | 0.956257 | 0.979572 | 0.950000 | 0.956257 | 0.979572 |
| H0 | 6 | 0.950000 | 0.954120 | 0.987619 | 0.950000 | 0.954120 | 0.987619 |
| H0 | 7 | 0.900000 | 0.917103 | 0.921445 | 0.900000 | 0.917103 | 0.920644 |
| H0 | 8 | 0.950000 | 0.937189 | 0.932588 | 0.950000 | 0.937189 | 0.932588 |
| H0 | 9 | 0.932203 | 0.943313 | 0.954826 | 0.949153 | 0.951894 | 0.955210 |
| H0 | 10 | 0.915254 | 0.907634 | 0.936031 | 0.915254 | 0.907634 | 0.936031 |
| H1 | 1 | 0.966667 | 0.982165 | 0.985547 | 0.966667 | 0.982165 | 0.985919 |
| H1 | 2 | 0.933333 | 0.962073 | 0.948492 | 0.933333 | 0.962073 | 0.948492 |
| H1 | 3 | 0.950000 | 0.967147 | 0.972257 | 0.950000 | 0.967147 | 0.972257 |
| H1 | 4 | 0.916667 | 0.930161 | 0.966500 | 0.916667 | 0.930161 | 0.966500 |
| H1 | 5 | 0.950000 | 0.955188 | 0.971709 | 0.950000 | 0.955188 | 0.972451 |
| H1 | 6 | 0.966667 | 0.977735 | 0.998517 | 0.983333 | 0.986316 | 0.999629 |
| H1 | 7 | 0.866667 | 0.880872 | 0.913159 | 0.866667 | 0.880872 | 0.912418 |
| H1 | 8 | 0.933333 | 0.923506 | 0.915962 | 0.933333 | 0.923506 | 0.915962 |
| H1 | 9 | 0.915254 | 0.934731 | 0.967117 | 0.932203 | 0.943313 | 0.967117 |
| H1 | 10 | 0.915254 | 0.912658 | 0.962062 | 0.915254 | 0.912658 | 0.962062 |
| U1 | 1 | 0.950000 | 0.972683 | 0.982059 | 0.950000 | 0.972683 | 0.982059 |
| U1 | 2 | 0.933333 | 0.962073 | 0.948492 | 0.933333 | 0.962073 | 0.948492 |
| U1 | 3 | 0.950000 | 0.967147 | 0.988013 | 0.950000 | 0.967147 | 0.988013 |
| U1 | 4 | 0.933333 | 0.961052 | 0.975995 | 0.933333 | 0.961052 | 0.977483 |
| U1 | 5 | 0.916667 | 0.915993 | 0.974334 | 0.916667 | 0.915993 | 0.974334 |
| U1 | 6 | 0.950000 | 0.954120 | 0.979293 | 0.950000 | 0.954120 | 0.979293 |
| U1 | 7 | 0.866667 | 0.876839 | 0.943183 | 0.866667 | 0.876839 | 0.943147 |
| U1 | 8 | 0.950000 | 0.933155 | 0.956932 | 0.950000 | 0.933155 | 0.957303 |
| U1 | 9 | 0.932203 | 0.943313 | 0.946753 | 0.932203 | 0.943313 | 0.947137 |
| U1 | 10 | 0.915254 | 0.923110 | 0.964900 | 0.915254 | 0.923110 | 0.965286 |
| U2 | 1 | 0.950000 | 0.972683 | 0.975252 | 0.950000 | 0.972683 | 0.975252 |
| U2 | 2 | 0.933333 | 0.962073 | 0.948492 | 0.933333 | 0.962073 | 0.948492 |
| U2 | 3 | 0.933333 | 0.921513 | 0.981568 | 0.933333 | 0.921513 | 0.980262 |
| U2 | 4 | 0.916667 | 0.930161 | 0.955970 | 0.916667 | 0.930161 | 0.955970 |
| U2 | 5 | 0.933333 | 0.927539 | 0.942075 | 0.933333 | 0.927539 | 0.942075 |
| U2 | 6 | 0.966667 | 0.981769 | 0.988827 | 0.966667 | 0.981769 | 0.988791 |
| U2 | 7 | 0.866667 | 0.894838 | 0.933293 | 0.866667 | 0.894838 | 0.933687 |
| U2 | 8 | 0.916667 | 0.914924 | 0.894101 | 0.916667 | 0.914924 | 0.894101 |
| U2 | 9 | 0.915254 | 0.929424 | 0.955257 | 0.915254 | 0.929424 | 0.955257 |
| U2 | 10 | 0.898305 | 0.914697 | 0.959694 | 0.898305 | 0.914697 | 0.959308 |
| U3 | 1 | 0.966667 | 0.982165 | 0.978164 | 0.966667 | 0.982165 | 0.977827 |
| U3 | 2 | 0.933333 | 0.962073 | 0.948492 | 0.933333 | 0.962073 | 0.948492 |
| U3 | 3 | 0.950000 | 0.946867 | 0.981358 | 0.950000 | 0.946867 | 0.982256 |
| U3 | 4 | 0.883333 | 0.906791 | 0.946289 | 0.883333 | 0.906791 | 0.946289 |
| U3 | 5 | 0.950000 | 0.951154 | 0.954131 | 0.950000 | 0.951154 | 0.954873 |
| U3 | 6 | 0.966667 | 0.963770 | 0.990741 | 0.966667 | 0.963770 | 0.990741 |
| U3 | 7 | 0.883333 | 0.920350 | 0.931393 | 0.883333 | 0.920350 | 0.931800 |
| U3 | 8 | 0.916667 | 0.909822 | 0.926333 | 0.916667 | 0.909822 | 0.926333 |
| U3 | 9 | 0.932203 | 0.943313 | 0.980272 | 0.932203 | 0.943313 | 0.980272 |
| U3 | 10 | 0.881356 | 0.889739 | 0.953011 | 0.881356 | 0.889739 | 0.952584 |
| U4 | 1 | 0.966667 | 0.982165 | 0.978045 | 0.966667 | 0.982165 | 0.978045 |
| U4 | 2 | 0.933333 | 0.962073 | 0.948492 | 0.933333 | 0.962073 | 0.948492 |
| U4 | 3 | 0.950000 | 0.946867 | 0.967572 | 0.950000 | 0.946867 | 0.968470 |
| U4 | 4 | 0.900000 | 0.921808 | 0.964646 | 0.900000 | 0.921808 | 0.964239 |
| U4 | 5 | 0.950000 | 0.937189 | 0.957124 | 0.950000 | 0.937189 | 0.957124 |
| U4 | 6 | 0.950000 | 0.954120 | 0.977810 | 0.950000 | 0.954120 | 0.977810 |
| U4 | 7 | 0.833333 | 0.866675 | 0.938267 | 0.833333 | 0.866675 | 0.938267 |
| U4 | 8 | 0.916667 | 0.909822 | 0.909590 | 0.916667 | 0.909822 | 0.909590 |
| U4 | 9 | 0.932203 | 0.963312 | 0.975909 | 0.932203 | 0.963312 | 0.975909 |
| U4 | 10 | 0.932203 | 0.937616 | 0.970563 | 0.932203 | 0.937616 | 0.970135 |
| U5 | 1 | 0.933333 | 0.938514 | 0.980172 | 0.950000 | 0.952403 | 0.980172 |
| U5 | 2 | 0.916667 | 0.948184 | 0.948492 | 0.933333 | 0.962073 | 0.948492 |
| U5 | 3 | 0.933333 | 0.957666 | 0.989676 | 0.933333 | 0.957666 | 0.989676 |
| U5 | 4 | 0.900000 | 0.936553 | 0.961649 | 0.916667 | 0.944906 | 0.962835 |
| U5 | 5 | 0.933333 | 0.906575 | 0.975984 | 0.933333 | 0.906575 | 0.975984 |
| U5 | 6 | 0.966667 | 0.963770 | 0.979899 | 0.966667 | 0.963770 | 0.979899 |
| U5 | 7 | 0.850000 | 0.872291 | 0.944419 | 0.850000 | 0.872291 | 0.944419 |
| U5 | 8 | 0.916667 | 0.896925 | 0.941338 | 0.916667 | 0.896925 | 0.941338 |
| U5 | 9 | 0.932203 | 0.942132 | 0.955442 | 0.932203 | 0.942132 | 0.954640 |
| U5 | 10 | 0.881356 | 0.888670 | 0.933241 | 0.881356 | 0.888670 | 0.933627 |
| U6 | 1 | 0.950000 | 0.972683 | 0.982698 | 0.950000 | 0.972683 | 0.982698 |
| U6 | 2 | 0.933333 | 0.962073 | 0.948492 | 0.933333 | 0.962073 | 0.948492 |
| U6 | 3 | 0.933333 | 0.938514 | 0.973834 | 0.933333 | 0.938514 | 0.974613 |
| U6 | 4 | 0.883333 | 0.913455 | 0.922257 | 0.883333 | 0.913455 | 0.922257 |
| U6 | 5 | 0.983333 | 0.973420 | 0.971930 | 0.983333 | 0.973420 | 0.971930 |
| U6 | 6 | 0.983333 | 0.986316 | 0.998851 | 0.983333 | 0.986316 | 0.998851 |
| U6 | 7 | 0.866667 | 0.883009 | 0.917624 | 0.866667 | 0.883009 | 0.918055 |
| U6 | 8 | 0.933333 | 0.923506 | 0.903060 | 0.933333 | 0.923506 | 0.903861 |
| U6 | 9 | 0.932203 | 0.943313 | 0.949344 | 0.932203 | 0.943313 | 0.949344 |
| U6 | 10 | 0.864407 | 0.879188 | 0.945863 | 0.881356 | 0.888670 | 0.945863 |

## Paired comparisons

Bootstrap intervals are stratified by diagnosis with 2,000 resamples. Holm correction is across the 20 pre-registered contrasts for the paired-fold tests and exact McNemar tests.

| Contrast | corrected / damaged / net | McNemar p / Holm p | ACC delta OOF | ACC bootstrap 95% CI | AUC_original delta OOF | AUC_original bootstrap 95% CI | ACC fold W/T/L | AUC_original fold W/T/L | paired-fold Holm p ACC / AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U1 - H1 | 14 / 17 / -3 | 0.7201 / 1 | -0.502 pp | [-2.34, +1.17] pp | -0.394 pp | [-2.14, +1.24] pp | [2, 5, 3] | [3, 3, 4] | 1 / 1 |
| U2 - U1 | 17 / 21 / -4 | 0.6271 / 1 | -0.669 pp | [-2.68, +1.34] pp | -0.517 pp | [-2.51, +1.43] pp | [2, 3, 5] | [3, 2, 5] | 1 / 1 |
| U3 - U1 | 14 / 16 / -2 | 0.8555 / 1 | -0.334 pp | [-2.17, +1.51] pp | -0.245 pp | [-2.03, +1.68] pp | [4, 3, 3] | [4, 2, 4] | 1 / 1 |
| U4 - U1 | 13 / 15 / -2 | 0.8506 / 1 | -0.334 pp | [-2.01, +1.34] pp | -0.262 pp | [-1.84, +1.28] pp | [3, 4, 3] | [4, 2, 4] | 1 / 1 |
| U5 - U3 | 17 / 20 / -3 | 0.7428 / 1 | -0.502 pp | [-2.51, +1.51] pp | -0.988 pp | [-3.12, +1.03] pp | [1, 5, 4] | [2, 2, 6] | 1 / 1 |
| U5 - U4 | 17 / 20 / -3 | 0.7428 / 1 | -0.502 pp | [-2.34, +1.51] pp | -0.971 pp | [-3.00, +1.08] pp | [3, 3, 4] | [4, 1, 5] | 1 / 1 |
| U5 - U1 | 17 / 22 / -5 | 0.5224 / 1 | -0.836 pp | [-2.84, +1.17] pp | -1.233 pp | [-3.24, +0.72] pp | [2, 3, 5] | [1, 1, 8] | 1 / 0.5403 |
| U6 - U1 | 15 / 16 / -1 | 1 / 1 | -0.167 pp | [-2.01, +1.51] pp | -0.176 pp | [-1.87, +1.48] pp | [2, 4, 4] | [3, 3, 4] | 1 / 1 |
| U1 - H1 | 14 / 17 / -3 | 0.7201 / 1 | -0.502 pp | [-2.51, +1.17] pp | -0.394 pp | [-2.24, +1.28] pp | [2, 5, 3] | [3, 3, 4] | 1 / 1 |
| U2 - H1 | 12 / 19 / -7 | 0.281 / 1 | -1.171 pp | [-3.01, +0.67] pp | -0.911 pp | [-2.53, +0.75] pp | [0, 3, 7] | [2, 2, 6] | 0.02648 / 1 |
| U3 - H1 | 10 / 15 / -5 | 0.4244 / 1 | -0.836 pp | [-2.51, +0.84] pp | -0.639 pp | [-2.07, +0.85] pp | [1, 5, 4] | [1, 3, 6] | 1 / 1 |
| U4 - H1 | 14 / 19 / -5 | 0.4869 / 1 | -0.836 pp | [-2.68, +1.00] pp | -0.656 pp | [-2.33, +1.01] pp | [1, 5, 4] | [2, 2, 6] | 1 / 1 |
| U5 - H1 | 12 / 20 / -8 | 0.2153 / 1 | -1.338 pp | [-3.18, +0.50] pp | -1.627 pp | [-3.36, +0.07] pp | [0, 3, 7] | [1, 1, 8] | 0.06167 / 0.4822 |
| U6 - H1 | 9 / 13 / -4 | 0.5235 / 1 | -0.669 pp | [-2.34, +0.84] pp | -0.570 pp | [-1.75, +0.46] pp | [1, 5, 4] | [2, 4, 4] | 1 / 1 |
| U1 - H0 | 11 / 15 / -4 | 0.5572 / 1 | -0.669 pp | [-2.34, +1.00] pp | -0.409 pp | [-2.09, +1.33] pp | [1, 5, 4] | [3, 2, 5] | 1 / 1 |
| U2 - H0 | 12 / 20 / -8 | 0.2153 / 1 | -1.338 pp | [-3.18, +0.50] pp | -0.927 pp | [-2.64, +0.71] pp | [2, 1, 7] | [3, 1, 6] | 0.9425 / 1 |
| U3 - H0 | 11 / 17 / -6 | 0.3449 / 1 | -1.003 pp | [-2.84, +0.67] pp | -0.655 pp | [-2.08, +0.67] pp | [1, 4, 5] | [2, 2, 6] | 1 / 1 |
| U4 - H0 | 12 / 18 / -6 | 0.3616 / 1 | -1.003 pp | [-2.84, +0.84] pp | -0.671 pp | [-2.12, +0.79] pp | [1, 6, 3] | [2, 3, 5] | 1 / 1 |
| U5 - H0 | 14 / 23 / -9 | 0.1877 / 1 | -1.505 pp | [-3.51, +0.50] pp | -1.643 pp | [-3.80, +0.43] pp | [2, 1, 7] | [3, 1, 6] | 0.9425 / 1 |
| U6 - H0 | 11 / 16 / -5 | 0.4421 / 1 | -0.836 pp | [-2.51, +0.84] pp | -0.585 pp | [-2.10, +0.92] pp | [2, 1, 7] | [2, 1, 7] | 1 / 1 |

## Held-out DARG relation ranking

Relation ranking uses outer test-test pairs. The existing DARG RelationScorer is binary same-versus-different, so six relation OVR AUC uses its sigmoid score to rank each observed relation type against the other five. It is not a six-way classifier. Ten-fold means are descriptive because test-test pairs share patients; no pair-independent confidence interval is used.

| Group | COGNITIVE_TEST SAME/DIFFERENT AUC | Mean six-relation OVR AUC | Cognitive support relation AUC | Non-support relation AUC |
|---|---:|---:|---:|---:|
| H0 | 0.7986 | 0.5032 | N/A | 0.5032 |
| H1 | 0.8354 | 0.5062 | N/A | 0.5062 |
| U1 | 0.8154 | 0.5090 | N/A | 0.5090 |
| U2 | 0.8001 | 0.5076 | N/A | 0.5076 |
| U3 | 0.8004 | 0.5113 | N/A | 0.5113 |
| U4 | 0.7787 | 0.5010 | N/A | 0.5010 |
| U5 | 0.8010 | 0.5031 | N/A | 0.5031 |
| U6 | 0.7929 | 0.4987 | N/A | 0.4987 |

Six held-out disease-relation OVR AUCs for the COGNITIVE_TEST DARG scorer

| Group | NC-NC | MCI-MCI | AD-AD | NC-MCI | NC-AD | MCI-AD |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 0.7886 | 0.6960 | 0.7182 | 0.4261 | 0.0605 | 0.3296 |
| H1 | 0.7831 | 0.7470 | 0.6829 | 0.3559 | 0.1017 | 0.3667 |
| U1 | 0.8012 | 0.7103 | 0.7186 | 0.3925 | 0.0817 | 0.3496 |
| U2 | 0.7853 | 0.6996 | 0.7314 | 0.4142 | 0.0714 | 0.3435 |
| U3 | 0.7913 | 0.6967 | 0.7519 | 0.4117 | 0.0583 | 0.3578 |
| U4 | 0.8010 | 0.6653 | 0.7173 | 0.4602 | 0.0574 | 0.3052 |
| U5 | 0.7836 | 0.7032 | 0.7039 | 0.4174 | 0.0844 | 0.3264 |
| U6 | 0.7865 | 0.6916 | 0.7029 | 0.4407 | 0.0540 | 0.3166 |

Cognitive-support held-out AUC is not estimable under the patient-disjoint rule because valid support edges are restricted to outer-training patients. The test-test support-pair count is zero. Non-support values use eligible held-out test-test pairs.

## Cognitive support coverage

Actual support is the count of selected formal relation pairs that overlap the frozen epoch-10 original COGNITIVE_TEST GraphLearn support. It is reported separately from held-out relation AUC because support pairs are train-only.

Explicit support slots use randomized epoch-10 COGNITIVE_TEST edges. A shortfall leaves the matching common U1 placebo pair in that registered slot. Total overlap also counts graph edges that occurred naturally in the shared random base.

| Group | Requested explicit slots | Explicit graph edges selected | Deterministic shortfall | Total graph-edge overlap mean +/- min-max | Total overlap of 4,096 |
|---|---:|---:|---:|---:|---:|
| U1 | 0.0 | 0.0 | 0 | 93.2 (79-103) | 2.28% |
| U2 | 0.0 | 0.0 | 0 | 93.1 (79-103) | 2.27% |
| U3 | 0.0 | 0.0 | 0 | 91.3 (79-103) | 2.23% |
| U4 | 0.0 | 0.0 | 0 | 92.9 (79-103) | 2.27% |
| U5 | 0.0 | 0.0 | 0 | 91.0 (79-103) | 2.22% |
| U6 | 205.0 | 192.9 | 121 | 280.8 (267-289) | 6.86% |

## Patient exposure

| Group | NC mean exposure / Gini | MCI mean exposure / Gini | AD mean exposure / Gini |
|---|---:|---:|---:|
| H0 | 13.12 / 0.2819 | 15.67 / 0.2757 | 19.25 / 0.1862 |
| H1 | 13.12 / 0.2819 | 15.67 / 0.2757 | 19.25 / 0.1862 |
| U1 | 13.12 / 0.1509 | 15.67 / 0.1396 | 19.25 / 0.1158 |
| U2 | 13.12 / 0.1509 | 15.67 / 0.1407 | 19.25 / 0.1158 |
| U3 | 13.12 / 0.1500 | 15.67 / 0.1406 | 19.25 / 0.1158 |
| U4 | 13.12 / 0.1509 | 15.67 / 0.1403 | 19.25 / 0.1161 |
| U5 | 13.12 / 0.1500 | 15.67 / 0.1406 | 19.25 / 0.1161 |
| U6 | 13.12 / 0.1556 | 15.67 / 0.1518 | 19.25 / 0.1166 |

## DARG graph mechanism

| Group | Anchor effective gate | Positive / negative relation edges | Mean adjacency relative change | Mean GCN message relative norm |
|---|---:|---:|---:|---:|
| H0 | 0.001003 | 1913.70 / 220.40 | 0.000148 | 0.000701 |
| H1 | 0.009900 | 9682.20 / 2710.10 | 0.004614 | 0.002650 |
| U1 | 0.009899 | 12981.40 / 3021.10 | 0.004050 | 0.002061 |
| U2 | 0.009899 | 10643.90 / 1976.10 | 0.003731 | 0.001410 |
| U3 | 0.009899 | 10492.90 / 2360.20 | 0.003629 | 0.002537 |
| U4 | 0.009899 | 11485.10 / 1588.70 | 0.002911 | 0.001139 |
| U5 | 0.009899 | 10548.00 / 2348.60 | 0.003542 | 0.000824 |
| U6 | 0.009899 | 12049.20 / 1896.00 | 0.003187 | 0.001237 |

## DARG and DPSL inference effects

| Group | Native to Full corrected / damaged | DPSL Full-Native mean absolute logit delta | Direct-Off logit delta / changed | Disease-Neutral logit delta / changed | Cognitive graph off logit delta / changed |
|---|---:|---:|---:|---:|---:|
| H0 | 1 / 0 | 0.010743 | 0.000000 / 0.00 | 0.000000 / 0.00 | 0.000000 / 0.00 |
| H1 | 2 / 0 | 0.020860 | 0.000000 / 0.00 | 0.000006 / 0.00 | 0.000004 / 0.00 |
| U1 | 0 / 0 | 0.018898 | 0.000000 / 0.00 | 0.000006 / 0.00 | 0.000005 / 0.00 |
| U2 | 0 / 0 | 0.014941 | 0.000000 / 0.00 | 0.000003 / 0.00 | 0.000001 / 0.00 |
| U3 | 0 / 0 | 0.014919 | 0.000000 / 0.00 | 0.000006 / 0.00 | 0.000006 / 0.00 |
| U4 | 0 / 0 | 0.009469 | 0.000000 / 0.00 | 0.000002 / 0.00 | 0.000001 / 0.00 |
| U5 | 3 / 0 | 0.012625 | 0.000000 / 0.00 | 0.000001 / 0.00 | 0.000001 / 0.00 |
| U6 | 1 / 0 | 0.012460 | 0.000000 / 0.00 | 0.000003 / 0.00 | 0.000002 / 0.00 |

## Mechanism

The signed graph, gate, message, and intervention aggregates are recorded in results/mechanism_diagnostics.csv. DARG training, direct graph propagation, and DPSL residual effects are reported separately there.

The paper reference is 94.81 +/- 2.95% ACC and 95.59 +/- 2.57% AUC. AUC_original remains the paper-comparison metric. Repeated tuning on TADPOLE makes all paired tests exploratory.
H0 is the read-only Y3/Z0 R0 history. H1 is the read-only S02 history. U1-U6 use the S02 base and alter only formal relation-pair sampling.
ACC and AUC_original comparisons with H0/H1 use the prescribed descriptive thresholds. They do not establish an unbiased generalization estimate under the retained val=test and pre-CV preprocessing protocol.
## Final decision gates

- EXECUTION_GO: **PASS**. Sixty new folds, twenty historical folds, eighty Best/Latest replays, sixty deterministic Latest next-step replays, eight logs, and 4,784 OOF rows were required.
- METHOD_FIDELITY_GO: **PASS**. The sealed S02 DARG/DPSL core and the ABIDE model identity audit passed.
- DESCRIPTIVE_PERFORMANCE_GO: **NO**. At least one U group must exceed H0 ACC (93.6441%) and H1 AUC_original (94.4340%). Matching groups: none.
- STATISTICAL_SUPPORT: **NO** under patient-paired bootstrap and Holm-adjusted tests. Matching groups: none.
- FINAL_REPLACEMENT: **NO**. Retain H0 for the Y3/Z0 ACC reference and H1/S02 as the current TADPOLE base.

## Paired Baseline Results

OOF deltas below use each patient's single held-out prediction. The bootstrap is stratified by diagnosis and resamples patients, while the paired-fold tests use ten folds.

| Group vs baseline | Delta ACC | Delta AUC_original | Corrected / damaged / net | McNemar Holm p | ACC bootstrap 95% CI | AUC_original bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| U1 - H0 | -0.669 pp | -0.409 pp | 11 / 15 / -4 | 1 | [-2.34, +1.00] pp | [-2.09, +1.33] pp |
| U1 - H1 | -0.502 pp | -0.394 pp | 14 / 17 / -3 | 1 | [-2.34, +1.17] pp | [-2.14, +1.24] pp |
| U2 - H0 | -1.338 pp | -0.927 pp | 12 / 20 / -8 | 1 | [-3.18, +0.50] pp | [-2.64, +0.71] pp |
| U2 - H1 | -1.171 pp | -0.911 pp | 12 / 19 / -7 | 1 | [-3.01, +0.67] pp | [-2.53, +0.75] pp |
| U3 - H0 | -1.003 pp | -0.655 pp | 11 / 17 / -6 | 1 | [-2.84, +0.67] pp | [-2.08, +0.67] pp |
| U3 - H1 | -0.836 pp | -0.639 pp | 10 / 15 / -5 | 1 | [-2.51, +0.84] pp | [-2.07, +0.85] pp |
| U4 - H0 | -1.003 pp | -0.671 pp | 12 / 18 / -6 | 1 | [-2.84, +0.84] pp | [-2.12, +0.79] pp |
| U4 - H1 | -0.836 pp | -0.656 pp | 14 / 19 / -5 | 1 | [-2.68, +1.00] pp | [-2.33, +1.01] pp |
| U5 - H0 | -1.505 pp | -1.643 pp | 14 / 23 / -9 | 1 | [-3.51, +0.50] pp | [-3.80, +0.43] pp |
| U5 - H1 | -1.338 pp | -1.627 pp | 12 / 20 / -8 | 1 | [-3.18, +0.50] pp | [-3.36, +0.07] pp |
| U6 - H0 | -0.836 pp | -0.585 pp | 11 / 16 / -5 | 1 | [-2.51, +0.84] pp | [-2.10, +0.92] pp |
| U6 - H1 | -0.669 pp | -0.570 pp | 9 / 13 / -4 | 1 | [-2.34, +0.84] pp | [-1.75, +0.46] pp |

## Gradient Direction

The read-only 80-fold gradient replay records the shared DARG parameter intersection, gradient norms, cosine direction, a replayed full optimizer update, and a counterfactual relation-only update. These diagnostics do not alter training.

## Research Conclusion

No new relation-sampling strategy simultaneously exceeded the Y3/Z0 ACC reference and the S02 AUC_original reference. The randomized S02-distribution control was already below both historical references, and the targeted MCI strategies did not recover that gap. The gradient audit found a shared path in every fold, but the shared parameter fraction was about one percent and relation/classification gradients were near-orthogonal or negative. A relation-only update changed DARG parameters while changing train NLL and classification margins by only about 1e-8 on average. The evidence therefore points to weak alignment and weak transmission of relation supervision into the final classifier, alongside no descriptive benefit from the tested sampling rules. It does not support changing the preserved DARG/DPSL model.

The paper reports 94.81 +/- 2.95% ACC and 95.59 +/- 2.57% AUC. All paper comparisons use AUC_original. The preserved val=test, full-data RFE, and pre-CV scaling make the results historical-protocol estimates rather than an independent external validation. Repeated prior TADPOLE exploration also limits confirmatory interpretation.
