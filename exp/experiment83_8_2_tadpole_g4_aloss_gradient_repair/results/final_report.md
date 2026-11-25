# Experiment 83.8.2 Final Report

Status: `PASS`. Formal execution completed 70/70 folds.
METHOD_FIDELITY: `PASS`. EXECUTION_AUDIT: `PASS`. PERFORMANCE_GO_NO_GO: `PERFORMANCE_GO`.

## Protocol and Fidelity

The data and split SHA-256 values are `0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc` and `7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5`. The first-choice OSSFS directory failed with EIO, so the run used `/model/spromptgl/experiment/experiment83_8_2_tadpole_g4_aloss_gradient_repair`.
The source matrix distinguishes the squared paper Frobenius term, the author's unsquared network calculation, the author's simple-2 backward path that excludes ALoss, the E83.8.1 six-modality train×train ALoss, and E83.6 continuous PairMean. The seven 83.8.2 groups use no PairMean.
The historical CV protocol still uses full-data RFE, imputation and scaling before outer CV, transductive patient graphs, `val_idx == test_idx`, and outer-test ACC checkpoint selection. Results are therefore protocol reproductions, not independent generalization estimates.

## Groups and Main Metrics

ACC and AUC_original are mean ± population SD over ten folds. AUC_original uses hard one-hot macro OVR. AUC_probability is auxiliary.

| Group | ACC | AUC_original | AUC_probability mean | OOF correct | NC/MCI/AD correct |
|---|---:|---:|---:|---:|---:|
| N0 | 92.9746% ± 2.7816 pp | 94.0967% ± 2.6088 pp | 96.2710% | 556/598 | 205/287/64 |
| N1 | 92.6384% ± 3.5311 pp | 94.1339% ± 2.8838 pp | 96.2357% | 554/598 | 202/286/66 |
| N2 | 91.9661% ± 3.1788 pp | 93.6364% ± 2.3223 pp | 95.3965% | 550/598 | 200/284/66 |
| N3 | 93.3079% ± 2.9019 pp | 94.1819% ± 2.6392 pp | 95.8360% | 558/598 | 202/292/64 |
| N4 | 92.4718% ± 3.0256 pp | 93.4839% ± 2.7288 pp | 95.5411% | 553/598 | 199/290/64 |
| N5 | 92.4718% ± 3.3729 pp | 93.9814% ± 2.8982 pp | 96.2853% | 553/598 | 202/285/66 |
| N6 | 91.4689% ± 2.9447 pp | 92.8779% ± 2.9563 pp | 95.9116% | 547/598 | 201/283/63 |
| H1 | 87.1130% ± 3.9345 pp | 90.6485% ± 3.0545 pp | 93.8880% | 521/598 | 184/269/68 |
| H2 | 88.7797% ± 3.9571 pp | 90.7319% ± 2.8838 pp | 95.3951% | 531/598 | 185/283/63 |
| H3 | 92.4746% ± 3.1913 pp | 93.9800% ± 2.7433 pp | 95.5545% | 553/598 | 205/283/65 |
| H4 | 92.9746% ± 2.7816 pp | 94.0967% ± 2.6088 pp | 96.2710% | 556/598 | 205/287/64 |
| H5 | 85.4407% ± 6.4461 pp | 88.3072% ± 4.7604 pp | 90.7601% | 511/598 | 188/263/60 |
| X3 | 92.1412% ± 3.7292 pp | 93.2876% ± 2.8881 pp | 96.4810% | 551/598 | 197/290/64 |
| X4 | 91.8023% ± 2.5436 pp | 92.7822% ± 2.4730 pp | 95.6699% | 549/598 | 200/287/62 |
| X5 | 91.7966% ± 3.8748 pp | 93.3526% ± 3.7211 pp | 95.4142% | 549/598 | 199/285/65 |
| X6 | 92.6384% ± 3.2867 pp | 93.3559% ± 2.9560 pp | 95.2975% | 554/598 | 201/291/62 |
| Paper | 94.8100% ± 2.9500 pp | 95.5900% ± 2.5700 pp | N/A | N/A | N/A |

## Paired Contrasts

Paired differences below are descriptive across the ten fixed folds. Positive patient net means candidate corrects more patients than it damages.

| Comparison | ΔACC | ΔAUC_original | Corrected | Damaged | Net |
|---|---:|---:|---:|---:|---:|
| N0_vs_H4 | +0.0000 pp | +0.0000 pp | 0 | 0 | +0 |
| N1_vs_N0 | -0.3362 pp | +0.0372 pp | 23 | 25 | -2 |
| N1_vs_H4 | -0.3362 pp | +0.0372 pp | 23 | 25 | -2 |
| N2_vs_N0 | -1.0085 pp | -0.4603 pp | 24 | 30 | -6 |
| N2_vs_H4 | -1.0085 pp | -0.4603 pp | 24 | 30 | -6 |
| N3_vs_N0 | +0.3333 pp | +0.0853 pp | 25 | 23 | +2 |
| N3_vs_H4 | +0.3333 pp | +0.0853 pp | 25 | 23 | +2 |
| N4_vs_N0 | -0.5028 pp | -0.6127 pp | 25 | 28 | -3 |
| N4_vs_H4 | -0.5028 pp | -0.6127 pp | 25 | 28 | -3 |
| N5_vs_N0 | -0.5028 pp | -0.1153 pp | 21 | 24 | -3 |
| N5_vs_H4 | -0.5028 pp | -0.1153 pp | 21 | 24 | -3 |
| N2_vs_N4_warmup_timing | -0.5056 pp | +0.1524 pp | 21 | 24 | -3 |
| N2_vs_N6_DPSL | +0.4972 pp | +0.7585 pp | 22 | 19 | +3 |
| N1_vs_N2_ALoss_strength | +0.6723 pp | +0.4975 pp | 26 | 22 | +4 |
| N2_vs_N3_ALoss_strength | -1.3418 pp | -0.5456 pp | 18 | 26 | -8 |

## Initialization and Warmup Pairing

N0, N1, N2, N3, and N5 share the same per-fold initial parameter, optimizer, RNG, DARG relation pairs, ten warmup losses, post-warmup model/optimizer/RNG, spectral statistics, and DPSL prompt hashes. N4 shares the same initial hashes, relation pairs, training labels, and post-warmup RNG. Its warmup losses and post-warmup model/optimizer differ because ALoss is active from epoch 1. N6 passes the per-fold common GFR/CAM/DARG initialization and warmup comparison against N0.

| Group | Fold pairing status | Folds |
|---|---|---:|
| N0 | BASELINE | 10 |
| N1 | PASS | 10 |
| N2 | PASS | 10 |
| N3 | PASS | 10 |
| N4 | ALOSS_WARMUP_ONLY_DIFFERENCE | 10 |
| N5 | PASS | 10 |
| N6 | PASS | 10 |

## Gradient and Optimizer Audit

The ALoss gradient probe uses the train-only historical loss and six GraphLearn `p.weight` tensors. Each row specifies the measured epoch. Structural zero is classified when a modality has no off-diagonal edges and its ALoss gradient is at numerical-zero scale. N5 computes one fixed coefficient per fold at epoch 11.
The Adam audit compares full optimizer update vectors from independent model and optimizer clones using the real checkpointed Adam state. It reports vector difference norms, relative difference, and cosine for the entire trainable model and the six GraphLearn `p.weight` tensors.

| Group | Epoch | ALoss/base gradient ratio mean | Range | λ mean or fixed |
|---|---:|---:|---:|---:|
| N1 | 11 | 0.132064 | [0.114836, 0.138730] | 0.001 |
| N2 | 11 | 0.396193 | [0.344508, 0.416190] | 0.003 |
| N3 | 11 | 1.320645 | [1.148360, 1.387299] | 0.010 |
| N4 | 1 | 0.249974 | [0.243264, 0.254896] | 0.003 |
| N4 | 11 | 0.384597 | [0.358224, 0.415872] | 0.003 |
| N5 | 11 | 0.100000 | [0.100000, 0.100000] | 0.0007594112 [0.0007208253, 0.0008708068] |
| N6 | 11 | 0.396193 | [0.344508, 0.416190] | 0.003 |

`audits/gradient_mechanism.json` contains 70 fold/epoch probes, with six modality gradient norms, edge counts and graph densities per probe. Each ALoss value matched its recorded formal update. Changing held-out labels preserved the loss and the target exactly in all probes. The N4 gradient ratio in the handoff is the epoch 11 measurement. Its Adam vector audit uses epoch 1.

### Adam Update-Vector Results

Each row summarizes ten independent clone-based counterfactuals at the registered audit epoch. The full vector covers all trainable model parameters. The subset covers the six GraphLearn `p.weight` tensors.

| Group | Epoch | Full `||u_base||` | Full `||u_supervised||` | Full `||difference||` | Full relative difference | Full cosine | Graph `||difference||` | Graph relative difference | Graph cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N1 | 11 | 6.64424 | 6.64424 | 0.00250271 | 0.000376676 | 0.999999940 | 0.00248968 | 0.00371408 | 0.999993122 |
| N2 | 11 | 6.64424 | 6.64425 | 0.0070078 | 0.00105472 | 0.999999452 | 0.00696527 | 0.0103907 | 0.999946016 |
| N3 | 11 | 6.64424 | 6.64428 | 0.0191294 | 0.00287909 | 0.999995828 | 0.0189522 | 0.0282728 | 0.999600631 |
| N4 | 1 | 15.7358 | 15.7358 | 0.0754757 | 0.00479644 | 0.999988449 | 0.0754757 | 0.029686 | 0.999557984 |
| N5 | 11 | 6.64424 | 6.64424 | 0.00191254 | 0.00028785 | 0.999999964 | 0.00190274 | 0.00283849 | 0.999995959 |

One full-vector cosine rounds to `1.000000119` in FP32. The audit retains the raw measurement and the nonzero difference vector. N3's average full-vector relative difference is 0.2879%, and its GraphLearn relative difference is 2.8273%.

## DARG and DPSL Mechanisms

DARG positive-DCR and signed positive/negative paths were checked at each Best checkpoint. Direct-Off and Disease-Neutral intervention counts are in `results/patient_mechanism.csv`. DPSL Residual-Off matched same-state Native logits exactly in each G4 fold; all three class-pair evidence paths were present. The Native-versus-Full counts and class-specific accuracy are in the same file.
The maximum shared gradient budget ratio was 0.000999897. The largest measured DPSL residual ratio was 0.0800000057, within FP32 rounding of the 0.08 cap.
The table separates the trained model's Native-to-Full inference changes from cross-model paired changes, which include joint training effects.

N2 Native reaches 91.7994% ACC and 93.3706% AUC_original. Against N6 this is +0.3305 and +0.4927 pp, with 22 corrections and 20 regressions. Enabling the N2 inference residual adds +0.1667 and +0.2658 pp, correcting one AD patient with no regressions. N3 Native versus N0 Native is +0.1667 pp ACC and −0.0095 pp AUC_original, with 24 corrections and 23 regressions. Enabling the N3 inference residual adds +0.1667 pp ACC and +0.0948 pp AUC_original and corrects one MCI patient.

| Group | Positive signed edges mean | Negative signed edges mean | Both signs every fold | DARG relation grad norm mean | Historical loss mean | ALoss mean | DPSL grad norm mean |
|---|---:|---:|:---:|---:|---:|---:|---:|
| N0 | 6431.70 | 1834.50 | True | 0.00247721 | -0.878684 | 0 | 0.000961687 |
| N1 | 6328.60 | 1889.30 | True | 0.00220015 | -0.713952 | 132.104 | 0.00102094 |
| N2 | 6697.90 | 1863.40 | True | 0.00241466 | -0.501143 | 124.186 | 0.000788893 |
| N3 | 6544.20 | 2166.60 | True | 0.00216988 | 0.261127 | 116.426 | 0.00112807 |
| N4 | 7069.00 | 2489.40 | True | 0.00238165 | -0.495667 | 101.757 | 0.00129937 |
| N5 | 6433.20 | 1705.90 | True | 0.00216831 | -0.772881 | 130.245 | 0.000895401 |
| N6 | 6122.60 | 1788.40 | True | 0.00225978 | -0.409224 | 130.989 | N/A |

## Identity and Decision

N0 versus historical H4 is `PARTIAL_MATCH`. The ten warmup histories, post-warmup model/optimizer/RNG hashes, relation pairs, best epoch, hard predictions, and fold metrics match. H4 has no recorded initial model, optimizer, or RNG hashes, so its initial-state identity remains partial.
N3 is the only variant that passes the registered `PERFORMANCE_GO` threshold of improving both main means against both clean references. The complete model research line remains G4 E82.4.6.10-G1 (DARG + DPSL).

## Checkpoint Continuation

Best preserved the selected inference state before the epoch's DPSL EMA advancement. `audits/verify_continuation_e8382.py` restores its RNG sidecar, replays the selected forward, and applies the original EMA advancement for G4 before testing the next step. All 70 Best checkpoints matched the recorded next epoch's loss components, ALoss coefficient and ACC. All 70 Latest checkpoints passed a deterministic isolated next step from the stored post-epoch state. The Latest test runs past the stopping point only in memory and creates no formal training output. See `audits/checkpoint_continuation.json`.

To verify one checkpoint directly from the experiment directory, run `PYTHONDONTWRITEBYTECODE=1 python -B audits/verify_continuation_e8382.py --groups N3 --folds 1 --checkpoint Best --workers 1`. To resume an unfinished fold from Latest, run `PYTHONDONTWRITEBYTECODE=1 python -B code/run_e8382.py --groups N3 --folds 1 --workers 1` with the relevant group and fold.

## Audit Artifacts

See `audits/final_audit.json`, `audits/gradient_mechanism.json`, `audits/checkpoint_recovery.json`, `audits/warmup_pairing.json`, `audits/report_repair.json`, `audits/training_seal.json`, and `results/darg_dpsl_fold_mechanism.csv` for fold-level identities, gradient measurements, recovery witnesses, corrected mechanism counts, and the frozen source seal.

## Audit Repair Scope

The report aggregator failed on a missing ACC field after its first complete 70-fold replay. Independent audit code repaired the ACC contract, restored all ten Adam vector records per group and counted CORR signed relations using the implemented q=0.5 split. A supplemental audit verified Best continuation with the EMA step and published the separate epoch 1 and epoch 11 gradient probes for N4. The seal and checkpoint hashes still match all formal folds. See `audits/report_repair.json`.
