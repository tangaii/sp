# Experiment 83.8.5.3 - S02-Based TADPOLE Cognitive-Anchored DARG+DPSL Optimization

## A. STATUS

Execution is complete. All eight registered new groups completed 10/10 folds. The performance decision is NO-GO for replacing the Y3/Z0/B0 complete model because no new group improves both primary metrics over both H1/S02 and H0/B0.

| Audit | Result |
|---|---|
| EXECUTION_AUDIT | PASS; historical replay 20/20, new training 80/80 |
| METHOD_FIDELITY | PASS |
| CHECKPOINT_REPLAY | PASS 100/100; deterministic Latest recovery PASS 80/80 |
| WARMUP_PAIRING | PASS; 8 groups x 10 folds paired to H1 |
| RELATION_SAMPLING_AUDIT | PASS 80/80 |
| LOG_FORMAT_AUDIT | PASS 10/10 groups |
| PERFORMANCE_GO_NO_GO | NO-GO for model replacement |
| FINAL_CROSS_FILE_CONSISTENCY | PASS after generation; see `audits/final_consistency_audit.json` |

## B. Environment and GPU

- Authoritative directory: `/model/spromptgl/experiment/experiment83_8_5_3_tadpole_s02_cognitive_fidelity_optimization`.
- Device: PPU-ZW810E, CUDA-visible GPU 0, 96 GiB, 8 spawn workers.
- Root-repository writes were not used. The preferred `/data/adodas/SPromptGL-main/experiment/` location returned OSSFS `EIO` / errno 5 during creation, so this run switched once to `/model/spromptgl/experiment/`. The error and chosen path are recorded in `audits/path_selection.json`.

## C. Paper and source review

- SPromptGL MICCAI 2025 PDF SHA-256: `219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89`. Eq. (1) uses GraphConstructLoss with a squared Frobenius reconstruction term. Eq. (7) is `L = lambda * L_G + eta * L_Label`, with `lambda = eta = 1` in the paper's notation. GFR, GraphLearn, GCN, CAM, PEN/SPGS, RES, losses, TADPOLE protocol, Table 2, checkpoint selection, and the author log format were reviewed.
- Author repository anchor: `ba3b9446fa451fde6864d6555bdf324de0c4b64a`. Its public code is primarily the ABIDE implementation. No complete author TADPOLE training source was assumed. The author log template SHA-256 is `5c80d16204e0882e5e264e42952c714397c2d31f07835b9e60d6dafd9488e3b7`.
- SABER source repository snapshot: `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4`. ABIDE E74.4 and E82.4.6.10 source contracts were checked, including Positive-DCR, signed support-constrained double Gate, BCE/GRL/Preservation, SCMR-DA-SRPE, and the four-modality CORR anchor regression.
- TADPOLE parent commit: `21e53cf1e003cc23eacdb8e09bbe471f9fa08bd8`. The training implementation is adapted from E83.8.5.2 commit `21e53cf` and its S02 implementation.
- Paper Table 2 reports TADPOLE ACC 94.81 +/- 2.95 and AUC 95.59 +/- 2.57. This is reference context only, not strict module-equivalent comparison.

## D. Data and split identity

- Patients: 598; NC/MCI/AD = [209, 315, 74].
- Modalities and dimensions: `[('UCSFFSX', 145), ('UCBERKELEYAV45', 148), ('UPENNBIOMK9', 3), ('RISK_FACTOR', 36), ('COGNITIVE_TEST', 24), ('ROI_AVERAGE', 9)]`; total 365 dimensions.
- Data SHA-256: `0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc`. Split SHA-256: `7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5`.
- Split protocol: seed 0, 10-fold StratifiedKFold, shuffle enabled, random_state 10. Each OOF patient appears exactly once in every group.
- Preserved historical limitations: full-cohort RFE before CV, mean imputation and StandardScaler before CV, transductive patient graph, `val_idx == test_idx`, and test-fold ACC used for first-strict-improvement Best selection. These choices limit the independence of the reported generalization estimate.

## E. Historical benchmark verification

H0 is archived Y3/Z0/B0 (B0 is the E83.8.5.2 archive alias, sourced from the original Y3 through Z0/R0 lineage). H1 is E83.8.5.2-S02. Both used real archived checkpoints. Best and Latest checkpoints and predictions were independently replayed.

| Group | ACC fold mean +/- population SD (%) | AUC_original fold mean +/- population SD (%) | AUC_probability fold mean +/- population SD (%) | OOF correct / 598 | OOF ACC (%) | OOF AUC_original (%) | OOF AUC_probability (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 93.6441 +/- 2.2192 | 94.4265 +/- 2.1648 | 95.8290 +/- 2.3408 | 560/598 | 93.6455 | 94.4055 | 95.5023 |
| H1 | 93.4746 +/- 3.0327 | 94.4340 +/- 3.1326 | 96.0281 +/- 2.6419 | 559/598 | 93.4783 | 94.3903 | 95.3016 |

## F. Registered groups

All new groups inherit the complete S02 model and the common H1 fold-specific initialization plus 10-epoch warmup. Their registered difference is limited to the listed factor.

| Group | Registered factor | Initial cognitive probability | Scale | Effective Gate |
|---|---|---:|---:|---:|
| H0 | Verified historical checkpoint | 0.1 | 0.01 | 0.001 |
| H1 | Verified historical checkpoint | 0.99 | 0.01 | 0.0099 |
| A1 | Lower Cognitive Gate | 0.99 | 0.0075 | 0.007424999999999999 |
| A2 | Higher Cognitive Gate | 0.99 | 0.0125 | 0.012375 |
| P1 | Equivalent Gate p=0.90 | 0.9 | 0.011 | 0.009899999999999999 |
| P2 | Equivalent Gate p=0.50 | 0.5 | 0.0198 | 0.0099 |
| P3 | Equivalent Gate p=0.10 | 0.1 | 0.099 | 0.0099 |
| R1 | 25% Anchor-Support Relation Pairs | 0.99 | 0.01 | 0.0099 |
| R2 | 50% Anchor-Support Relation Pairs | 0.99 | 0.01 | 0.0099 |
| D1 | Frozen Adaptive Gate Diagnostic | 0.99 | 0.01 | 0.0099 |

R1 and R2 change only the formal relation-pair sampling distribution from epoch 11. This affects supervision received by all DARG relation encoders and is not a cognitive-branch-only BCE intervention. D1 freezes the existing adaptive Gate after the normal S02 reset.

## G. Method fidelity

- All complete-model groups use the six-modality E82.4.6.10-G1 DARG+DPSL core. DARG retains RelationEncoder/RelationScorer, signed positive and negative relations, original-edge support restriction, Positive-DCR, the double Gate, existing reweighting/message propagation, relation BCE 0.05, conditional adversarial loss 0.01 with GRL 0.03, preservation loss 0.01, and direct scale 0.02 with `direct_single_gate=false`.
- Cognitive alignment and cognitive edge anchors are both modality index 4. The Gate formula remains `scale * sigmoid(theta)`. S02 effective Gates are 0.0099 on COGNITIVE_TEST and 0.001 on each other modality.
- DPSL retains SCMR-DA-SRPE, frozen TADPOLE MMGPL-12 RN50 prompts, NC/MCI/AD class-pair routing, confidence protection, Native/Full paths, residual cap 0.08, and shared gradient budget 0.001. Residual-Off exactly equals Native.
- GFR/CAM, classification head, optimizers, weight decay, Paper-ALoss coefficient 0.01, epoch-11 Paper-ALoss start, 10-epoch historical warmup, early-stop condition `wait > 200`, and 450-epoch cap were kept from S02.
- No test labels entered relation-pair sampling or model objectives. R1/R2 pairs and targets were constructed from outer-training patients and labels only, using a fold-local RNG. Checkpoint replay verified train-only pairs, canonical unordered pairs, no duplicates, formal-pair identity, and warmup-pair identity.
- Structural zero-edge cases are retained as structural observations. A modality with no off-diagonal original support cannot receive an edge update through that branch; other supported graph branches and the full training gradients remained active.

## H. Warmup and RNG pairing

The eight new groups passed H1 pairing for all 90/80 group-fold comparisons. The audit covers common initial parameters and optimizer state, Python/NumPy/PyTorch/CUDA RNG, training labels, warmup relation pairs/targets, per-epoch loss and common parameter updates, and post-warmup shared state. P1/P2/P3 differ only in their registered Gate reparameterization. R1/R2 switch relation pairs only after epoch 10.

## I. Formal training completion

Eight new groups completed 80/80 folds on GPU 0 with 8 workers. H0/H1 historical verification covered 20 folds. Each new fold has Best and Latest checkpoint, fold audit, prediction, and author epoch records. The training lock and resumable checkpoints remain in the authoritative experiment directory.

## J. ACC and AUC results

AUC_original is the historical three-class hard-prediction one-hot macro OVR metric. AUC_probability is the probability macro OVR metric. The first three metric columns are 10-fold mean +/- population SD. The OOF columns aggregate 598 held-out patient predictions and are not differences of fold means.

| Group | ACC fold mean +/- SD (%) | AUC_original fold mean +/- SD (%) | AUC_probability fold mean +/- SD (%) | OOF correct / 598 | OOF correct NC / MCI / AD | OOF ACC (%) | OOF AUC_original (%) | OOF AUC_probability (%) | Native -> Full corrected / damaged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 93.6441 +/- 2.2192 | 94.4265 +/- 2.1648 | 95.8290 +/- 2.3408 | 560/598 | 205 / 291 / 64 | 93.6455 | 94.4055 | 95.5023 | 1/0 |
| H1 | 93.4746 +/- 3.0327 | 94.4340 +/- 3.1326 | 96.0281 +/- 2.6419 | 559/598 | 203 / 291 / 65 | 93.4783 | 94.3903 | 95.3016 | 2/0 |
| A1 | 93.1441 +/- 2.0377 | 94.2876 +/- 1.3025 | 96.2323 +/- 1.8858 | 557/598 | 203 / 289 / 65 | 93.1438 | 94.2539 | 95.9664 | 1/0 |
| A2 | 92.3051 +/- 2.9162 | 93.8511 +/- 2.7864 | 96.2011 +/- 1.8977 | 552/598 | 203 / 284 / 65 | 92.3077 | 93.7752 | 96.1806 | 0/0 |
| P1 | 92.9633 +/- 3.0179 | 93.7564 +/- 3.0546 | 95.4350 +/- 3.0648 | 556/598 | 202 / 291 / 63 | 92.9766 | 93.6834 | 95.3972 | 0/0 |
| P2 | 92.6384 +/- 2.7225 | 93.7545 +/- 2.8123 | 95.8763 +/- 1.9194 | 554/598 | 202 / 288 / 64 | 92.6421 | 93.6913 | 95.5764 | 2/0 |
| P3 | 92.1384 +/- 3.8183 | 93.2091 +/- 3.2034 | 95.0773 +/- 2.9544 | 551/598 | 201 / 287 / 63 | 92.1405 | 93.1948 | 94.9016 | 1/0 |
| R1 | 92.4689 +/- 3.0295 | 94.2148 +/- 3.2161 | 95.5614 +/- 3.5062 | 553/598 | 198 / 287 / 68 | 92.4749 | 94.2216 | 95.4958 | 2/0 |
| R2 | 92.6412 +/- 2.8115 | 94.5087 +/- 2.0696 | 95.8629 +/- 1.7270 | 554/598 | 201 / 285 / 68 | 92.6421 | 94.4681 | 95.8808 | 2/0 |
| D1 | 92.8023 +/- 2.8243 | 93.2911 +/- 2.7996 | 95.5905 +/- 2.2579 | 555/598 | 202 / 292 / 61 | 92.8094 | 93.2109 | 95.1166 | 2/0 |

## K. Paired comparisons with H1/S02

Corrected means the left group's prediction is correct and the right group's is wrong. Damaged means the reverse. Bootstrap intervals use 2,000 paired patient resamples within held-out fold and diagnosis strata. Holm correction is applied to the 16 planned group-vs-baseline contrasts separately for each metric family and McNemar tests.

| Contrast | Delta fold ACC / AUC_original / AUC_probability (pp) | Delta OOF ACC / AUC_original / AUC_probability (pp) | Corrected / damaged / net | Fold ACC W-T-L | Fold AUC_original W-T-L | McNemar p / Holm | Paired-fold Holm p ACC / AUC_original | Bootstrap 95% CI ACC (pp) | Bootstrap 95% CI AUC_original (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 - H1 | -0.3305 / -0.1464 / +0.2042 | -0.3344 / -0.1363 / +0.6648 | 16 / 18 / -2 | 2-6-2 | 5-1-4 | 0.8642 / 1 | 1 / 1 | [-2.3411, +1.5050] | [-1.8516, +1.4225] |
| A2 - H1 | -1.1695 / -0.5829 / +0.1730 | -1.1706 / -0.6151 / +0.8790 | 11 / 18 / -7 | 0-4-6 | 2-1-7 | 0.2649 / 1 | 0.1516 / 1 | [-2.8428, +0.5017] | [-2.0363, +0.7035] |
| P1 - H1 | -0.5113 / -0.6776 / -0.5931 | -0.5017 / -0.7069 / +0.0956 | 9 / 12 / -3 | 3-2-5 | 3-2-5 | 0.6636 / 1 | 1 / 1 | [-2.0067, +1.0033] | [-1.9939, +0.5364] |
| P2 - H1 | -0.8362 / -0.6795 / -0.1518 | -0.8361 / -0.6990 / +0.2748 | 11 / 16 / -5 | 0-6-4 | 1-2-7 | 0.4421 / 1 | 0.6228 / 1 | [-2.5084, +0.6689] | [-1.9976, +0.5162] |
| P3 - H1 | -1.3362 / -1.2249 / -0.9508 | -1.3378 / -1.1954 / -0.4000 | 11 / 19 / -8 | 1-2-7 | 0-1-9 | 0.2005 / 1 | 0.3466 / 0.0417 | [-3.1773, +0.3344] | [-2.7042, +0.1994] |
| R1 - H1 | -1.0056 / -0.2192 / -0.4667 | -1.0033 / -0.1687 / +0.1942 | 16 / 22 / -6 | 1-4-5 | 3-1-6 | 0.4177 / 1 | 0.7255 / 1 | [-3.0100, +0.8361] | [-1.8504, +1.3832] |
| R2 - H1 | -0.8333 / +0.0747 / -0.1652 | -0.8361 / +0.0778 / +0.5792 | 11 / 16 / -5 | 0-6-4 | 3-4-3 | 0.4421 / 1 | 0.6228 / 1 | [-2.5084, +0.8361] | [-1.3176, +1.4539] |
| D1 - H1 | -0.6723 / -1.1429 / -0.4376 | -0.6689 / -1.1794 / -0.1850 | 10 / 14 / -4 | 1-5-4 | 2-1-7 | 0.5413 / 1 | 0.8393 / 0.634 | [-2.1739, +0.8361] | [-2.7258, +0.2125] |

## L. Paired comparisons with H0/Y3-Z0-B0

| Contrast | Delta fold ACC / AUC_original / AUC_probability (pp) | Delta OOF ACC / AUC_original / AUC_probability (pp) | Corrected / damaged / net | Fold ACC W-T-L | Fold AUC_original W-T-L | McNemar p / Holm | Paired-fold Holm p ACC / AUC_original | Bootstrap 95% CI ACC (pp) | Bootstrap 95% CI AUC_original (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 - H0 | -0.5000 / -0.1389 / +0.4034 | -0.5017 / -0.1516 / +0.4641 | 13 / 16 / -3 | 1-6-3 | 3-3-4 | 0.7111 / 1 | 1 / 1 | [-2.3411, +1.1706] | [-1.5157, +1.1489] |
| A2 - H0 | -1.3390 / -0.5754 / +0.3721 | -1.3378 / -0.6303 / +0.6783 | 10 / 18 / -8 | 1-3-6 | 3-2-5 | 0.1849 / 1 | 0.6228 / 1 | [-3.0100, +0.3344] | [-2.1093, +0.8814] |
| P1 - H0 | -0.6808 / -0.6701 / -0.3940 | -0.6689 / -0.7221 / -0.1051 | 12 / 16 / -4 | 1-5-4 | 1-4-5 | 0.5716 / 1 | 1 / 1 | [-2.3411, +1.0033] | [-2.1509, +0.8361] |
| P2 - H0 | -1.0056 / -0.6720 / +0.0473 | -1.0033 / -0.7142 / +0.0741 | 11 / 17 / -6 | 0-5-5 | 2-2-6 | 0.3449 / 1 | 0.3466 / 1 | [-2.6756, +0.6689] | [-2.1758, +0.7162] |
| P3 - H0 | -1.5056 / -1.2174 / -0.7517 | -1.5050 / -1.2107 / -0.6007 | 13 / 22 / -9 | 1-4-5 | 2-1-7 | 0.1755 / 1 | 0.7255 / 1 | [-3.3445, +0.5017] | [-2.8167, +0.4599] |
| R1 - H0 | -1.1751 / -0.2116 / -0.2676 | -1.1706 / -0.1839 / -0.0065 | 16 / 23 / -7 | 1-3-6 | 4-1-5 | 0.3368 / 1 | 0.576 / 1 | [-3.1773, +0.8361] | [-1.9422, +1.5051] |
| R2 - H0 | -1.0028 / +0.0822 / +0.0339 | -1.0033 / +0.0626 / +0.3785 | 8 / 14 / -6 | 1-4-5 | 4-2-4 | 0.2863 / 1 | 0.7255 / 1 | [-2.5084, +0.5017] | [-1.1990, +1.3398] |
| D1 - H0 | -0.8418 / -1.1353 / -0.2385 | -0.8361 / -1.1946 / -0.3857 | 10 / 15 / -5 | 1-4-5 | 2-1-7 | 0.4244 / 1 | 0.7255 / 0.634 | [-2.3411, +0.6689] | [-2.6962, +0.2003] |

## M. Paper reference and planned factor contrasts

The paper reports 94.81 +/- 2.95 ACC and 95.59 +/- 2.57 AUC. The paper's reported result is included as context only because the present fidelity model and evaluation history are not strict end-to-end module equivalence.

| Contrast | Delta fold ACC / AUC_original / AUC_probability (pp) | Delta OOF ACC / AUC_original / AUC_probability (pp) | Corrected / damaged / net | Fold ACC W-T-L | Fold AUC_original W-T-L | McNemar p / Holm | Paired-fold Holm p ACC / AUC_original | Bootstrap 95% CI ACC (pp) | Bootstrap 95% CI AUC_original (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 - A1 | -0.8390 / -0.4365 / -0.0313 | -0.8361 / -0.4788 / +0.2142 | 18 / 23 / -5 | 1-4-5 | 4-2-4 | 0.5327 / 1 | 0.4728 / 1 | [-2.8428, +1.1706] | [-2.2144, +1.2738] |
| P1 - P2 | +0.3249 / +0.0019 / -0.4413 | +0.3344 / -0.0079 / -0.1792 | 15 / 13 / 2 | 4-2-4 | 3-2-5 | 0.8506 / 1 | 1 / 1 | [-1.3378, +2.0067] | [-1.5548, +1.5777] |
| P1 - P3 | +0.8249 / +0.5473 / +0.3577 | +0.8361 / +0.4885 / +0.4957 | 22 / 17 / 5 | 4-2-4 | 4-1-5 | 0.5224 / 1 | 1 / 1 | [-1.1706, +2.8428] | [-1.3856, +2.3540] |
| P2 - P3 | +0.5000 / +0.5454 / +0.7990 | +0.5017 / +0.4964 / +0.6749 | 17 / 14 / 3 | 4-4-2 | 5-2-3 | 0.7201 / 1 | 1 / 1 | [-1.1706, +2.3411] | [-1.1299, +2.1459] |
| R2 - R1 | +0.1723 / +0.2939 / +0.3015 | +0.1672 / +0.2465 / +0.3850 | 21 / 20 / 1 | 4-3-3 | 6-1-3 | 1 / 1 | 1 / 1 | [-1.8395, +2.1739] | [-1.2875, +1.8044] |

## N. Cognitive relation discrimination and graph support

Relation AUC is a mechanism diagnostic and does not substitute for classification ACC/AUC. Each AUC is the mean across ten fold-level pair evaluations. Pair totals, Same/Different score distributions, q mean/SD, and uncertain fraction below describe the COGNITIVE_TEST scorer over all patients or the indicated candidate/support subset. Held-out labels are used for post-hoc mechanism scoring only.

| Group | Pair support | Same / different pairs | Relation AUC | Same score mean +/- SD | Different score mean +/- SD | q mean +/- SD | Uncertain fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | all_patient_pairs | 738920 / 1046110 | 82.1982 | 0.6387 +/- 0.1348 | 0.3213 +/- 0.1656 | 0.4527 +/- 0.2255 | 0.2472 |
| A1 | candidate_edges_before_support_intersection | 15361 / 8161 | 73.3475 | 0.7016 +/- 0.0765 | 0.4935 +/- 0.1481 | 0.6289 +/- 0.1507 | 0.2167 |
| A1 | graph_valid_edges | 85190 / 53813 | 76.1656 | 0.6993 +/- 0.0821 | 0.4557 +/- 0.1425 | 0.6027 +/- 0.1671 | 0.2121 |
| A1 | actual_scored_candidates | 85190 / 53813 | 76.1656 | 0.6993 +/- 0.0821 | 0.4557 +/- 0.1425 | 0.6027 +/- 0.1671 | 0.2121 |
| A2 | all_patient_pairs | 738920 / 1046110 | 83.2242 | 0.6528 +/- 0.1429 | 0.3110 +/- 0.1747 | 0.4525 +/- 0.2391 | 0.2348 |
| A2 | candidate_edges_before_support_intersection | 16179 / 8357 | 73.9223 | 0.7142 +/- 0.0858 | 0.4965 +/- 0.1647 | 0.6398 +/- 0.1610 | 0.2095 |
| A2 | graph_valid_edges | 85262 / 51532 | 76.6134 | 0.7131 +/- 0.0887 | 0.4548 +/- 0.1610 | 0.6147 +/- 0.1791 | 0.2096 |
| A2 | actual_scored_candidates | 85262 / 51532 | 76.6134 | 0.7131 +/- 0.0887 | 0.4548 +/- 0.1610 | 0.6147 +/- 0.1791 | 0.2096 |
| P1 | all_patient_pairs | 738920 / 1046110 | 79.8918 | 0.5947 +/- 0.1160 | 0.3614 +/- 0.1527 | 0.4580 +/- 0.1843 | 0.2998 |
| P1 | candidate_edges_before_support_intersection | 15689 / 8197 | 69.2777 | 0.6560 +/- 0.0628 | 0.5362 +/- 0.1082 | 0.6134 +/- 0.1031 | 0.2903 |
| P1 | graph_valid_edges | 82818 / 50041 | 71.2066 | 0.6557 +/- 0.0630 | 0.5151 +/- 0.1092 | 0.6017 +/- 0.1120 | 0.2946 |
| P1 | actual_scored_candidates | 82818 / 50041 | 71.2066 | 0.6557 +/- 0.0630 | 0.5151 +/- 0.1092 | 0.6017 +/- 0.1120 | 0.2946 |
| P2 | all_patient_pairs | 738920 / 1046110 | 84.0377 | 0.6600 +/- 0.1481 | 0.3041 +/- 0.1795 | 0.4514 +/- 0.2465 | 0.2439 |
| P2 | candidate_edges_before_support_intersection | 16148 / 8562 | 75.8206 | 0.7230 +/- 0.0902 | 0.4900 +/- 0.1771 | 0.6409 +/- 0.1718 | 0.2376 |
| P2 | graph_valid_edges | 81226 / 50109 | 78.4192 | 0.7196 +/- 0.0950 | 0.4443 +/- 0.1739 | 0.6131 +/- 0.1904 | 0.2438 |
| P2 | actual_scored_candidates | 81226 / 50109 | 78.4192 | 0.7196 +/- 0.0950 | 0.4443 +/- 0.1739 | 0.6131 +/- 0.1904 | 0.2438 |
| P3 | all_patient_pairs | 738920 / 1046110 | 84.5111 | 0.6660 +/- 0.1528 | 0.2961 +/- 0.1871 | 0.4492 +/- 0.2573 | 0.1864 |
| P3 | candidate_edges_before_support_intersection | 16016 / 8476 | 76.0091 | 0.7317 +/- 0.0911 | 0.4933 +/- 0.1762 | 0.6489 +/- 0.1740 | 0.1614 |
| P3 | graph_valid_edges | 82124 / 51465 | 78.5702 | 0.7301 +/- 0.0945 | 0.4506 +/- 0.1755 | 0.6222 +/- 0.1934 | 0.1709 |
| P3 | actual_scored_candidates | 82124 / 51465 | 78.5702 | 0.7301 +/- 0.0945 | 0.4506 +/- 0.1755 | 0.6222 +/- 0.1934 | 0.1709 |
| R1 | all_patient_pairs | 738920 / 1046110 | 85.6962 | 0.6509 +/- 0.1674 | 0.2771 +/- 0.1994 | 0.4318 +/- 0.2657 | 0.1776 |
| R1 | candidate_edges_before_support_intersection | 15384 / 7851 | 78.2103 | 0.7197 +/- 0.1113 | 0.4726 +/- 0.1987 | 0.6365 +/- 0.1903 | 0.1952 |
| R1 | graph_valid_edges | 75250 / 44156 | 82.9584 | 0.7127 +/- 0.1198 | 0.3911 +/- 0.1913 | 0.5937 +/- 0.2203 | 0.2023 |
| R1 | actual_scored_candidates | 75250 / 44156 | 82.9584 | 0.7127 +/- 0.1198 | 0.3911 +/- 0.1913 | 0.5937 +/- 0.2203 | 0.2023 |
| R2 | all_patient_pairs | 738920 / 1046110 | 85.0178 | 0.6494 +/- 0.1841 | 0.2642 +/- 0.2027 | 0.4237 +/- 0.2749 | 0.1827 |
| R2 | candidate_edges_before_support_intersection | 15787 / 8516 | 78.5393 | 0.7186 +/- 0.1379 | 0.4275 +/- 0.2293 | 0.6166 +/- 0.2260 | 0.2305 |
| R2 | graph_valid_edges | 78638 / 49514 | 83.3951 | 0.7243 +/- 0.1434 | 0.3424 +/- 0.2123 | 0.5761 +/- 0.2580 | 0.2247 |
| R2 | actual_scored_candidates | 78638 / 49514 | 83.3951 | 0.7243 +/- 0.1434 | 0.3424 +/- 0.2123 | 0.5761 +/- 0.2580 | 0.2247 |
| D1 | all_patient_pairs | 738920 / 1046110 | 85.4840 | 0.6807 +/- 0.1578 | 0.2907 +/- 0.1923 | 0.4522 +/- 0.2660 | 0.2166 |
| D1 | candidate_edges_before_support_intersection | 15367 / 7802 | 76.8651 | 0.7462 +/- 0.0954 | 0.4929 +/- 0.1962 | 0.6595 +/- 0.1858 | 0.1985 |
| D1 | graph_valid_edges | 92238 / 56743 | 79.9939 | 0.7399 +/- 0.1034 | 0.4359 +/- 0.1930 | 0.6232 +/- 0.2094 | 0.2017 |
| D1 | actual_scored_candidates | 92238 / 56743 | 79.9939 | 0.7399 +/- 0.1034 | 0.4359 +/- 0.1930 | 0.6232 +/- 0.2094 | 0.2017 |

Six training-label relation types are represented in the formal relation samples. Counts below aggregate the ten outer-training folds and are reported separately from held-out relation AUC.

| Group | AD-AD | MCI-AD | MCI-MCI | NC-AD | NC-MCI | NC-NC |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 1777 | 6844 | 13182 | 2424 | 11212 | 5521 |
| A2 | 1777 | 6844 | 13182 | 2424 | 11212 | 5521 |
| P1 | 1777 | 6844 | 13182 | 2424 | 11212 | 5521 |
| P2 | 1777 | 6844 | 13182 | 2424 | 11212 | 5521 |
| P3 | 1777 | 6844 | 13182 | 2424 | 11212 | 5521 |
| R1 | 1599 | 5584 | 10396 | 1821 | 13075 | 8485 |
| R2 | 1490 | 4447 | 7484 | 1291 | 14742 | 11506 |
| D1 | 1777 | 6844 | 13182 | 2424 | 11212 | 5521 |

The S02 folds 1, 2, and 8 were specifically inspected. Their fold ACC/AUC and cognitive-anchor relation AUCs are listed here.

| H1 fold | ACC | AUC_original | All-patient pair AUC | Candidate AUC | Graph-valid AUC | Actual-scored AUC | Actual pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 96.6667 | 98.2165 | 75.5646 | 61.0421 | 66.0957 | 66.0957 | 14283 |
| 2 | 93.3333 | 96.2073 | 68.2983 | 57.8640 | 59.5877 | 59.5877 | 11726 |
| 8 | 93.3333 | 92.3506 | 80.1138 | 68.7891 | 73.2002 | 73.2002 | 11973 |

R1/R2 support quota execution is summarized below. R2 uses the registered deterministic fallback where training-only class-specific cognitive edges are fewer than the requested 1,024. The remainder then follows the historical sampling policy. The realized total support fraction includes incidental support edges in the historical remainder.

| Group | Fold | Forced same / different | Available same / different | Shortfall same / different | Actual supported / 4096 | Actual support fraction |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 1 | 512 / 512 | 2120 / 904 | 0 / 0 | 1113 / 4096 | 27.1729 |
| R1 | 2 | 512 / 512 | 2271 / 1016 | 0 / 0 | 1118 / 4096 | 27.2949 |
| R1 | 3 | 512 / 512 | 2210 / 909 | 0 / 0 | 1106 / 4096 | 27.0020 |
| R1 | 4 | 512 / 512 | 2365 / 1045 | 0 / 0 | 1148 / 4096 | 28.0273 |
| R1 | 5 | 512 / 512 | 2229 / 894 | 0 / 0 | 1128 / 4096 | 27.5391 |
| R1 | 6 | 512 / 512 | 2404 / 1077 | 0 / 0 | 1134 / 4096 | 27.6855 |
| R1 | 7 | 512 / 512 | 2485 / 1058 | 0 / 0 | 1168 / 4096 | 28.5156 |
| R1 | 8 | 512 / 512 | 2309 / 1017 | 0 / 0 | 1128 / 4096 | 27.5391 |
| R1 | 9 | 512 / 512 | 2399 / 962 | 0 / 0 | 1128 / 4096 | 27.5391 |
| R1 | 10 | 512 / 512 | 2495 / 992 | 0 / 0 | 1129 / 4096 | 27.5635 |
| R2 | 1 | 1024 / 904 | 2120 / 904 | 0 / 120 | 1965 / 4096 | 47.9736 |
| R2 | 2 | 1024 / 1016 | 2271 / 1016 | 0 / 8 | 2083 / 4096 | 50.8545 |
| R2 | 3 | 1024 / 909 | 2210 / 909 | 0 / 115 | 1970 / 4096 | 48.0957 |
| R2 | 4 | 1024 / 1024 | 2365 / 1045 | 0 / 0 | 2092 / 4096 | 51.0742 |
| R2 | 5 | 1024 / 894 | 2229 / 894 | 0 / 130 | 1963 / 4096 | 47.9248 |
| R2 | 6 | 1024 / 1024 | 2404 / 1077 | 0 / 0 | 2094 / 4096 | 51.1230 |
| R2 | 7 | 1024 / 1024 | 2485 / 1058 | 0 / 0 | 2106 / 4096 | 51.4160 |
| R2 | 8 | 1024 / 1017 | 2309 / 1017 | 0 / 7 | 2087 / 4096 | 50.9521 |
| R2 | 9 | 1024 / 962 | 2399 / 962 | 0 / 62 | 2029 / 4096 | 49.5361 |
| R2 | 10 | 1024 / 992 | 2495 / 992 | 0 / 32 | 2053 / 4096 | 50.1221 |

## O. Gate parameterization

| Group | Initial p4 | Scale4 | Effective Gate at epoch 11 | Best effective Gate mean | Gate trace entries | Finite gradients | requires_grad entries | Parameter updates | Latest optimizer step | Frozen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | 0.9900 | 0.00750 | 0.00742500 | 0.00742453 | 3134 | 3134 | 3134 | 3134 | 223,238,257,261,269,274,362,375,435,440 | False |
| A2 | 0.9900 | 0.01250 | 0.01237500 | 0.01237410 | 3170 | 3170 | 3170 | 3170 | 223,243,244,264,271,307,368,370,440 | False |
| P1 | 0.9000 | 0.01100 | 0.00990000 | 0.00988810 | 2872 | 2872 | 2872 | 2872 | 223,224,243,265,270,271,286,301,391,398 | False |
| P2 | 0.5000 | 0.01980 | 0.00990000 | 0.00980919 | 3191 | 3191 | 3191 | 3191 | 223,239,243,257,289,352,372,377,399,440 | False |
| P3 | 0.1000 | 0.09900 | 0.00990000 | 0.00978433 | 3280 | 3280 | 3280 | 3280 | 223,262,267,270,322,395,412,422,440 | False |
| R1 | 0.9900 | 0.01000 | 0.00990000 | 0.00989878 | 3314 | 3314 | 3314 | 3314 | 223,270,293,299,310,316,332,413,418,440 | False |
| R2 | 0.9900 | 0.01000 | 0.00990000 | 0.00989794 | 3473 | 3473 | 3473 | 3473 | 223,265,281,292,353,368,378,433,440 | False |
| D1 | 0.9900 | 0.01000 | 0.00990000 | 0.00990000 | 3303 | 0 | 0 | 0 | none | True |

P1/P2/P3 began with the same cognitive effective Gate 0.0099 and the same five non-anchor effective Gates 0.001. The paired warmup audit passed, yet their trained paths diverged. Their ACC/AUC_original fold means were P1 92.9633/93.7564, P2 92.6384/93.7545, and P3 92.1384/93.2091 percent. No planned pairwise contrast provides corrected evidence of a benefit from the alternate parameterization.

## P. DARG graph updates and messages

Counts are summed over ten folds. Adjacency-change and message ratios are means of fold values. Signed edge additions remain within original support. Direct-Off, Disease-Neutral, Cognitive-Graph-Off, and Score-Permuted are post-hoc interventions on the saved Best model.

| Group | Modality | Original edges | Positive enhancements | Negative suppressions | Relative adjacency change | Fusion-weighted GCN message ratio | Fusion-weighted Direct-DCR message ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | COGNITIVE_TEST | 139003 | 113765 | 25238 | 0.00243725 | 0.00071103911 | 2.0075979e-08 |
| A1 | RISK_FACTOR | 71238 | 10812 | 5006 | 0.00011337 | 0.00094770648 | 1.8165394e-10 |
| A1 | ROI_AVERAGE | 264377 | 13441 | 1324 | 0.00004760 | 0.0011500233 | 1.950604e-11 |
| A1 | UCBERKELEYAV45 | 45379 | 7948 | 4432 | 0.00022438 | 0.0077148312 | 5.8997628e-10 |
| A1 | UCSFFSX | 18979 | 2072 | 1948 | 0.00010953 | 0.0097569654 | 3.1724927e-10 |
| A1 | UPENNBIOMK9 | 743106 | 3818 | 849 | 0.00000847 | 0.0006800719 | 1.3091755e-12 |
| A2 | COGNITIVE_TEST | 136794 | 114973 | 21821 | 0.00433194 | 0.00096477655 | 5.9954342e-08 |
| A2 | RISK_FACTOR | 61263 | 11185 | 5401 | 0.00012859 | 0.00098562631 | 1.9522193e-10 |
| A2 | ROI_AVERAGE | 233362 | 13605 | 1608 | 0.00005254 | 0.0014613879 | 2.2501647e-11 |
| A2 | UCBERKELEYAV45 | 45530 | 8139 | 5191 | 0.00022833 | 0.0051583862 | 6.0482707e-10 |
| A2 | UCSFFSX | 22586 | 2710 | 2537 | 0.00014602 | 0.0068762767 | 4.955439e-10 |
| A2 | UPENNBIOMK9 | 681267 | 4254 | 1161 | 0.00000909 | 0.00088641512 | 7.9066194e-13 |
| P1 | COGNITIVE_TEST | 132859 | 118807 | 14052 | 0.00249421 | 0.0010804217 | 2.9110396e-08 |
| P1 | RISK_FACTOR | 57589 | 9403 | 5153 | 0.00009449 | 0.0011719167 | 1.1902713e-10 |
| P1 | ROI_AVERAGE | 248262 | 13166 | 1117 | 0.00003875 | 0.0012152528 | 1.0920052e-11 |
| P1 | UCBERKELEYAV45 | 60929 | 7332 | 4044 | 0.00015255 | 0.0037288978 | 4.6204268e-10 |
| P1 | UCSFFSX | 2404 | 910 | 703 | 0.00009076 | 0.0078006963 | 3.0910411e-10 |
| P1 | UPENNBIOMK9 | 758876 | 2232 | 868 | 0.00000446 | 0.00098765961 | 3.5799293e-13 |
| P2 | COGNITIVE_TEST | 131335 | 108285 | 23050 | 0.00352813 | 0.00082845778 | 3.9717794e-08 |
| P2 | RISK_FACTOR | 67059 | 9930 | 5999 | 0.00012649 | 0.00095581924 | 1.8927633e-10 |
| P2 | ROI_AVERAGE | 260064 | 12704 | 1642 | 0.00004587 | 0.00096455417 | 1.0593137e-11 |
| P2 | UCBERKELEYAV45 | 52552 | 8313 | 5280 | 0.00025773 | 0.0035929829 | 6.9185635e-10 |
| P2 | UCSFFSX | 35328 | 2623 | 2367 | 0.00011937 | 0.0078218151 | 3.960792e-10 |
| P2 | UPENNBIOMK9 | 685835 | 5063 | 1315 | 0.00000916 | 0.00087457214 | 7.9377973e-13 |
| P3 | COGNITIVE_TEST | 133589 | 111022 | 22567 | 0.00368052 | 0.00046553226 | 3.9915452e-08 |
| P3 | RISK_FACTOR | 62744 | 11183 | 6287 | 0.00013481 | 0.00056501268 | 2.1523272e-10 |
| P3 | ROI_AVERAGE | 215114 | 15120 | 1845 | 0.00005821 | 0.0010113613 | 2.0654036e-11 |
| P3 | UCBERKELEYAV45 | 65877 | 9092 | 5775 | 0.00022119 | 0.0026224609 | 5.9803576e-10 |
| P3 | UCSFFSX | 4045 | 1438 | 1183 | 0.00017820 | 0.0060248678 | 6.1080578e-10 |
| P3 | UPENNBIOMK9 | 687867 | 4332 | 1716 | 0.00001109 | 0.00050229052 | 1.6633533e-12 |
| R1 | COGNITIVE_TEST | 119406 | 88251 | 31155 | 0.00377754 | 0.0018414302 | 3.8906366e-08 |
| R1 | RISK_FACTOR | 65128 | 13614 | 4297 | 0.00014528 | 0.00058200123 | 2.2754564e-10 |
| R1 | ROI_AVERAGE | 247955 | 13684 | 1928 | 0.00005007 | 0.0038672614 | 1.4260637e-11 |
| R1 | UCBERKELEYAV45 | 47998 | 9241 | 7046 | 0.00026713 | 0.0056991409 | 6.2112518e-10 |
| R1 | UCSFFSX | 2895 | 723 | 690 | 0.00011824 | 0.0063752751 | 3.4728812e-10 |
| R1 | UPENNBIOMK9 | 687711 | 4055 | 1786 | 0.00001010 | 0.0020136659 | 8.0705686e-13 |
| R2 | COGNITIVE_TEST | 128152 | 87800 | 40352 | 0.00422108 | 0.0018160205 | 3.8751679e-08 |
| R2 | RISK_FACTOR | 73239 | 15138 | 3714 | 0.00015007 | 0.00060498492 | 2.4274187e-10 |
| R2 | ROI_AVERAGE | 255833 | 13860 | 2541 | 0.00006225 | 0.0018163964 | 1.7268887e-11 |
| R2 | UCBERKELEYAV45 | 46223 | 8832 | 6387 | 0.00026159 | 0.0086220038 | 7.3185125e-10 |
| R2 | UCSFFSX | 16186 | 2906 | 2640 | 0.00020488 | 0.014299528 | 6.6963314e-10 |
| R2 | UPENNBIOMK9 | 665147 | 3480 | 2800 | 0.00001170 | 0.00089625392 | 1.2527009e-12 |
| D1 | COGNITIVE_TEST | 148981 | 119214 | 29767 | 0.00390637 | 0.0008466687 | 4.3822819e-08 |
| D1 | RISK_FACTOR | 67457 | 10560 | 6113 | 0.00014571 | 0.00083734211 | 2.0502893e-10 |
| D1 | ROI_AVERAGE | 245301 | 12942 | 1572 | 0.00005108 | 0.001644867 | 1.7731554e-11 |
| D1 | UCBERKELEYAV45 | 57408 | 8645 | 5676 | 0.00024551 | 0.0081564311 | 5.9611399e-10 |
| D1 | UCSFFSX | 18255 | 2232 | 1948 | 0.00015081 | 0.0084706343 | 5.2009765e-10 |
| D1 | UPENNBIOMK9 | 669404 | 4554 | 1315 | 0.00000962 | 0.0010873387 | 1.1521228e-12 |

| Group | Intervention -> Full | Corrected / damaged | Changed argmax | Mean absolute logit delta per fold | Maximum absolute logit delta | Mean top-2 margin change per fold |
|---|---|---:|---:|---:|---:|---:|
| A1 | Direct-Off | 0 / 0 | 0 | 1.2703459e-07 | 6.981194e-06 | +2.7725421e-08 |
| A1 | Disease-Neutral | 0 / 0 | 0 | 3.1401059e-06 | 0.00048285723 | +2.0454962e-07 |
| A1 | Cognitive-Graph-Off | 0 / 0 | 0 | 5.3311428e-07 | 6.3598156e-05 | -3.5510099e-07 |
| A1 | Score-Permuted | 0 / 0 | 0 | 4.7009069e-06 | 0.00094741583 | -4.3382089e-06 |
| A2 | Direct-Off | 0 / 0 | 0 | 2.269751e-07 | 1.1265278e-05 | +1.2189529e-07 |
| A2 | Disease-Neutral | 0 / 0 | 0 | 3.9390242e-06 | 0.00026893616 | -5.0584476e-07 |
| A2 | Cognitive-Graph-Off | 0 / 0 | 0 | 3.4516074e-06 | 0.00027704239 | -4.6509556e-07 |
| A2 | Score-Permuted | 0 / 0 | 0 | 4.7386997e-06 | 0.00035047531 | +7.463689e-07 |
| P1 | Direct-Off | 0 / 0 | 0 | 1.430961e-07 | 2.6226044e-06 | +4.1581817e-08 |
| P1 | Disease-Neutral | 0 / 0 | 0 | 1.474016e-06 | 0.00013063848 | -1.3665636e-06 |
| P1 | Cognitive-Graph-Off | 0 / 0 | 0 | 1.1974056e-06 | 0.00014412403 | -2.2725888e-06 |
| P1 | Score-Permuted | 0 / 0 | 0 | 1.7773755e-06 | 0.00012097508 | -1.3199377e-06 |
| P2 | Direct-Off | 0 / 0 | 0 | 1.5119148e-07 | 4.2170286e-06 | +4.6474787e-08 |
| P2 | Disease-Neutral | 0 / 0 | 0 | 2.112265e-06 | 0.00018958747 | +8.2911156e-07 |
| P2 | Cognitive-Graph-Off | 0 / 0 | 0 | 5.2408427e-07 | 3.8087368e-05 | +1.9131406e-07 |
| P2 | Score-Permuted | 0 / 0 | 0 | 2.3446773e-06 | 0.00026336312 | +1.1770006e-06 |
| P3 | Direct-Off | 0 / 0 | 0 | 1.7510901e-07 | 4.2915344e-06 | +5.4372402e-08 |
| P3 | Disease-Neutral | 0 / 0 | 0 | 1.5178115e-06 | 0.00012770295 | -9.8902544e-07 |
| P3 | Cognitive-Graph-Off | 0 / 0 | 0 | 4.3911748e-07 | 4.3094158e-05 | -6.8907335e-09 |
| P3 | Score-Permuted | 0 / 0 | 0 | 1.5524467e-06 | 0.00014400482 | -1.5813079e-06 |
| R1 | Direct-Off | 0 / 0 | 0 | 1.6244825e-07 | 7.3611736e-06 | -3.1511314e-09 |
| R1 | Disease-Neutral | 0 / 0 | 0 | 5.5874967e-06 | 0.0011111498 | -5.1896441e-08 |
| R1 | Cognitive-Graph-Off | 0 / 0 | 0 | 5.1465295e-06 | 0.0011161566 | +8.4913044e-08 |
| R1 | Score-Permuted | 0 / 0 | 0 | 6.9955674e-06 | 0.00097966194 | +2.4520458e-06 |
| R2 | Direct-Off | 0 / 0 | 0 | 1.5616345e-07 | 4.1127205e-06 | +5.8388138e-08 |
| R2 | Disease-Neutral | 0 / 0 | 0 | 1.6644254e-05 | 0.0045352578 | +2.5033937e-05 |
| R2 | Cognitive-Graph-Off | 0 / 0 | 0 | 1.5117358e-05 | 0.0047982335 | +2.5754435e-05 |
| R2 | Score-Permuted | 0 / 0 | 0 | 1.5582757e-05 | 0.0033099651 | +2.2923474e-05 |
| D1 | Direct-Off | 0 / 0 | 0 | 1.8731121e-07 | 5.0067902e-06 | +2.4343614e-08 |
| D1 | Disease-Neutral | 0 / 0 | 0 | 3.4093229e-06 | 0.00053331256 | -2.2672379e-06 |
| D1 | Cognitive-Graph-Off | 0 / 0 | 0 | 8.9371974e-07 | 8.3595514e-05 | -7.9074317e-07 |
| D1 | Score-Permuted | 0 / 0 | 0 | 5.2403598e-06 | 0.0011126176 | -5.2369001e-06 |

Supported graph branches changed adjacency and message values, but Direct-Off caused no classification argmax changes in the new groups. Cognitive graph and score interventions are reported by group above. Numeric graph changes alone do not demonstrate a classification improvement.

## Q. DPSL Native and Full paths

| Group | Native ACC mean +/- SD (%) | Full ACC mean +/- SD (%) | Native AUC_original mean +/- SD (%) | Full AUC_original mean +/- SD (%) | Native -> Full corrected / damaged / net | Corrected NC / MCI / AD | Residual-Off exact | Residual max ratio | Shared-gradient max ratio |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| A1 | 92.9774 +/- 1.9452 | 93.1441 +/- 2.0377 | 94.1927 +/- 1.3045 | 94.2876 +/- 1.3025 | 1 / 0 / 1 | 0 / 1 / 0 | PASS 80/80 | 0.042444415 | 0.00099989121 |
| A2 | 92.3051 +/- 2.9162 | 92.3051 +/- 2.9162 | 93.8511 +/- 2.7864 | 93.8511 +/- 2.7864 | 0 / 0 / 0 | 0 / 0 / 0 | PASS 80/80 | 0.038135927 | 0.00099989699 |
| P1 | 92.9633 +/- 3.0179 | 92.9633 +/- 3.0179 | 93.7564 +/- 3.0546 | 93.7564 +/- 3.0546 | 0 / 0 / 0 | 0 / 0 / 0 | PASS 80/80 | 0.031623714 | 0.00099989442 |
| P2 | 92.3051 +/- 2.6149 | 92.6384 +/- 2.7225 | 93.3710 +/- 2.6746 | 93.7545 +/- 2.8123 | 2 / 0 / 2 | 0 / 1 / 1 | PASS 80/80 | 0.035411812 | 0.00099989369 |
| P3 | 91.9718 +/- 3.6497 | 92.1384 +/- 3.8183 | 93.0723 +/- 3.0300 | 93.2091 +/- 3.2034 | 1 / 0 / 1 | 1 / 0 / 0 | PASS 80/80 | 0.038457867 | 0.00099989353 |
| R1 | 92.1356 +/- 3.5965 | 92.4689 +/- 3.0295 | 93.8632 +/- 3.7101 | 94.2148 +/- 3.2161 | 2 / 0 / 2 | 0 / 1 / 1 | PASS 80/80 | 0.055862177 | 0.00099989664 |
| R2 | 92.3051 +/- 3.1009 | 92.6412 +/- 2.8115 | 94.3410 +/- 2.1817 | 94.5087 +/- 2.0696 | 2 / 0 / 2 | 0 / 2 / 0 | PASS 80/80 | 0.042677499 | 0.00099989202 |
| D1 | 92.4689 +/- 2.9364 | 92.8023 +/- 2.8243 | 93.0685 +/- 2.8287 | 93.2911 +/- 2.7996 | 2 / 0 / 2 | 1 / 1 / 0 | PASS 80/80 | 0.043239132 | 0.00099989639 |

Across the eight new groups, Native -> Full corrected 10 patient-fold predictions and damaged 0; these are counterfactual patient-fold transitions across separately trained groups, not additional unique patients. Full -> Native and Residual-Off -> Full are separately present in the mechanism audit with the same source-to-target direction convention. Residual-Off matched Native exactly in checkpoint replay.
- Historical direction correction: E83.8.5.2 `corrected_vs_full` and `damaged_vs_full` represented intervention -> Full. The corrected Native -> Full aggregate for its eight new groups is 5 corrected / 2 damaged. S09 is 1 / 0. This clarification is recorded here and does not modify E83.8.5.2 results.
- DPSL counterfactual Native -> Full effects are distinct from joint training effects, which are the full-model group metrics compared with H1/H0.

## R. Best and Latest checkpoint replay

- Historical H0/H1 Best and Latest: 20/20 independently replayed, including checkpoint SHA, predictions, probabilities, and metrics.
- All registered Best/Latest: 100/100 passed. All 80 new Latest recovery checks reproduced the same deterministic next optimizer/model/RNG state.
- The replay record labels H0/H1 as `server historical checkpoint archive` and new groups as `formal training`.
- R1/R2 Latest resume restored formal relation pairs, targets, Gate freeze/state, optimizer and RNG state, DPSL spectral state, and early-stop counters from Latest.

## S. Author-format training logs

The original `result_Tadpole.txt` template SHA-256 is `5c80d16204e0882e5e264e42952c714397c2d31f07835b9e60d6dafd9488e3b7`. All ten group logs passed the strict parser and field-order, field-name, precision, fold-boundary, Counter, epoch-origin, and source-data checks. Logs retain the author labels and three `trn-loss-MF` fields. The denominator is 450 and display epochs use the original zero-origin convention. Every epoch value came from the real saved training record; log generation did not execute model forward passes.

## T. Training seal and source identity

- Training seal status: `SEALED_BEFORE_TRAINING`. Sealed implementation SHA-256: `ad7cdc0737a720e96d239667fb7d474be73b7ca05fc25806922680a782bc6d85`.
- Parent commit: `21e53cf1e003cc23eacdb8e09bbe471f9fa08bd8`. Source manifest SHA-256: `585d94aba65f11f83b3aeb2ff5398b55a4618453eeb101896138fd0eda1a059e`.
- Formal training code and group configurations were sealed before training. Post-training changes are confined to independent audit/report scripts. The sealed training hash still matches the current sealed files.
- Checkpoint files are retained locally for recovery and reproducibility. No Git push was performed.

## U. Scientific conclusion

NO-GO for replacing Y3/Z0/B0. No new group simultaneously exceeds H1/S02 on ACC and AUC_original, nor does any simultaneously exceed H0/B0. R2 improves fold-mean AUC_original to 94.5087% versus H0 94.4265%, but its ACC is 92.6412%, below H0 93.6441% and H1 93.4746%. No paired comparison has Holm-adjusted evidence supporting a replacement. Results use one seed only and do not establish cross-seed stability.
- R1/R2 relation-pair support sampling is an explicit training-sampling change that affects all DARG relation encoders. It requires separate cross-dataset validation before any claim of ABIDE/TADPOLE generality.
- All method, warmup, checkpoint, relation, log, and patient-coverage audits passed. The main requested performance gate did not pass.
