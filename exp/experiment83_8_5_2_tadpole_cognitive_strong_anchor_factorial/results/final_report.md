# Experiment 83.8.5.2 - Y3/Z0 Cognitive Strong-Anchor DARG+DPSL Factorial

## A. Status

The registered experiment completed with 16 groups and 160/160 folds. B0-B3 are historical checkpoint replays from the server archives. S01-S09 and C01-C03 are 120 newly trained folds. The authoritative directory is `/model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial`.

The final scientific decision is `NO-GO_PERFORMANCE`. Y3/Z0 remains the incumbent complete model. The formal method, checkpoint, OOF, log, and mechanism audits pass.

## B. Execution and locked inputs

- GPU: one PPU-ZW810E with 98,304 MiB.
- Workers: 8, selected by the resource probe.
- Seed: 0.
- Cohort: 598 patients, NC/MCI/AD = 209/315/74.
- Modalities: UCSFFSX 145, UCBERKELEYAV45 148, UPENNBIOMK9 3, RISK_FACTOR 36, COGNITIVE_TEST 24, ROI_AVERAGE 9.
- Processed data SHA-256: `0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc`.
- Split SHA-256: `7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5`.
- Split rule: `StratifiedKFold(n_splits=10, shuffle=True, random_state=10)`.

## C. Source and paper review

The MICCAI paper, author repository, SABER repository, and TADPOLE history were read before training. The review covers Eq.(1), Eq.(7), GFR, GraphLearn, GCN, CAM, PEN, RES, SPGS, GraphConstructLoss, ALoss, six-modality TADPOLE routing, three-class routing, Table 2, and author training logs.

- Author repository commit: `ba3b9446fa451fde6864d6555bdf324de0c4b64a`.
- SABER HEAD: `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4`.
- E83.8.5.1 anchor: `912a3ece628f818cec7525fe6323350f5a06e27e`.
- E83.8.5 source: `ae225e5135046d1d9622968b7d011f6aa55fa60d`.
- Read-only author log template SHA-256: `5c80d16204e0882e5e264e42952c714397c2d31f07835b9e60d6dafd9488e3b7`.

The archived ABIDE E74.1 runner was checked at the real server path. Its regression is PASS 20/20 checkpoint audits. It uses ten warmup epochs, `g_m = scale_m * sigmoid(theta_m)`, probability 0.99 for the anchor, probability 0.10 for non-anchors, scale `[0.01, 0.01, 0.01, 1.0]`, in-place modification of the existing Parameter, removal of only that Parameter's Adam state, and unchanged RNG and other optimizer state.

## D. Method fidelity

The frozen model is SPromptGL GFR -> DARG -> CAM/RES -> DPSL -> three-class output. DARG follows E74.4-G2 Positive-DCR with relation BCE 0.05, conditional adversarial loss 0.01, preservation loss 0.01, GRL 0.03, 4096 training-only relation pairs, signed graph support, original edge selection, direct scale 0.02, and double gating. DPSL follows E82.4.6.10-G1 SCMR-DA-SRPE with frozen TADPOLE MMGPL-12 prompts, three classes, residual cap 0.08, and shared gradient budget 0.001. Paper-ALoss uses the squared Frobenius form with coefficient 0.01 from epoch 11 and outer-train labels only.

The preflight method-fidelity audit is PASS. The M1 graph structural-zero gradient is recorded as an expected structural property. It was not treated as a full-modality nonzero-gradient failure.

## E. Registered groups

| Group | Definition |
| --- | --- |
| B0 | Y3/Z0/R0, no anchor, ordinary edges, p4 0.10, scale4 0.01, effective 0.001 |
| B1 | R1, alignment anchor 4, ordinary edges, p4 0.10, scale4 0.01, effective 0.001 |
| B2 | E83.8.5.1 N1, alignment and cognitive edge anchor 4, p4 0.10, scale4 0.01, effective 0.001 |
| B3 | R4, alignment and cognitive edge anchor 4, p4 0.10, scale4 0.10, effective 0.010 |
| S01 | alignment anchor 4, p4 0.99, scale4 0.01, edge anchor null, effective 0.0099 |
| S02 | S01 with cognitive edge anchor 4 |
| S03 | p4 0.10, scale4 0.10, edge anchor null, effective 0.0100 |
| S04 | p4 0.99, scale4 0.10, edge anchor null, effective 0.0990 |
| S05 | S04 with cognitive edge anchor 4 |
| S06 | p4 0.10, scale4 1.00, edge anchor null, effective 0.1000 |
| S07 | S06 with cognitive edge anchor 4 |
| S08 | p4 0.99, scale4 1.00, edge anchor null, effective 0.9900 |
| S09 | S08 with cognitive edge anchor 4 |
| C01 | S08 gate, alignment anchor null, edge anchor null, full DARG+DPSL |
| C02 | S09 gate, alignment and edge anchor 4, DPSL disabled |
| C03 | S09 configuration with constructor-time 0.99 gate initialization |

## F. Complete performance table

Values are ten-fold mean +/- population standard deviation in percent. AUC_original is the historical hard one-hot macro OVR metric. The final column gives Full OOF correct counts for NC/MCI/AD.

| Group | ACC | AUC_original | AUC_probability | Full OOF | NC/MCI/AD correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 93.6441 +/- 2.2192 | 94.4265 +/- 2.1648 | 95.8290 +/- 2.3408 | 560/598 | 205/291/64 |
| B1 | 92.6384 +/- 2.2781 | 94.1104 +/- 2.1786 | 96.4483 +/- 1.9012 | 554/598 | 201/287/66 |
| B2 | 92.4689 +/- 2.4097 | 93.7285 +/- 2.6081 | 95.6975 +/- 2.4111 | 553/598 | 200/288/65 |
| B3 | 92.6328 +/- 3.3887 | 93.7182 +/- 3.1780 | 96.7077 +/- 1.6548 | 554/598 | 202/288/64 |
| C01 | 92.1356 +/- 3.0987 | 93.5802 +/- 3.0293 | 95.8870 +/- 2.0592 | 551/598 | 201/285/65 |
| C02 | 92.4718 +/- 2.5138 | 94.2587 +/- 2.3004 | 96.4291 +/- 1.9034 | 553/598 | 202/284/67 |
| C03 | 92.8079 +/- 3.1765 | 93.8574 +/- 2.7031 | 96.3238 +/- 2.1779 | 555/598 | 202/289/64 |
| S01 | 92.9746 +/- 2.6799 | 93.7420 +/- 2.8880 | 94.9304 +/- 3.1337 | 556/598 | 202/291/63 |
| S02 | 93.4746 +/- 3.0327 | 94.4340 +/- 3.1326 | 96.0281 +/- 2.6419 | 559/598 | 203/291/65 |
| S03 | 92.3051 +/- 2.3930 | 92.8730 +/- 2.8347 | 94.9672 +/- 2.9506 | 552/598 | 200/291/61 |
| S04 | 91.9689 +/- 2.8881 | 93.1643 +/- 2.7286 | 95.4784 +/- 1.7619 | 550/598 | 202/285/63 |
| S05 | 92.1384 +/- 2.8029 | 92.8673 +/- 2.8582 | 95.6261 +/- 2.4686 | 551/598 | 201/289/61 |
| S06 | 92.4746 +/- 3.4425 | 93.4378 +/- 3.2683 | 95.7793 +/- 2.4445 | 553/598 | 205/286/62 |
| S07 | 92.6384 +/- 2.2781 | 93.5495 +/- 2.0179 | 95.7236 +/- 3.2803 | 554/598 | 201/290/63 |
| S08 | 92.3051 +/- 2.6149 | 93.4159 +/- 3.1492 | 94.9847 +/- 3.1391 | 552/598 | 199/289/64 |
| S09 | 92.8079 +/- 3.1765 | 93.8574 +/- 2.7031 | 96.3238 +/- 2.1779 | 555/598 | 202/289/64 |

## G. Historical and paper comparison

Y3/Z0 is B0 at ACC 93.6441 +/- 2.2192 percent, AUC_original 94.4265 +/- 2.1648 percent, and 560/598 correct with NC/MCI/AD = 205/291/64. The paper Table 2 reference is ACC 94.81 +/- 2.95 percent and AUC 95.59 +/- 2.57 percent.

No new group exceeds B0 on both primary metrics. S02 is the closest new performance point, with ACC 93.4746 percent and AUC_original 94.4340 percent. No group reaches either paper reference mean on both reported primary metrics. Several groups improve probability AUC, but that does not satisfy the AUC_original target.

## H. Preregistered paired contrasts

The correction column is `corrected/damaged/net` relative to the right-hand group. Fold records are wins/ties/losses. Confidence intervals are paired stratified bootstrap 95 percent intervals in percentage points. Holm values are for the eight preregistered primary comparisons.

| Contrast | Delta ACC | Delta AUC_original | Delta AUC_probability | Correct/damaged/net | ACC folds | AUC folds | McNemar p / Holm p | ACC bootstrap CI | AUC bootstrap CI |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| S09 vs B0 | -0.8361 | -0.6024 | +1.0099 | 11/16/-5 | 2/3/5 | 4/3/3 | 0.4421 / 1.0000 | [-2.508,+0.836] | [-2.193,+0.969] |
| S09 vs B2 | +0.3344 | +0.0718 | +1.2523 | 19/17/+2 | 2/5/3 | 4/1/5 | 0.8679 / 1.0000 | [-1.672,+2.341] | [-1.814,+1.914] |
| S09 vs S08 | +0.5017 | +0.3988 | +2.0834 | 19/16/+3 | 4/4/2 | 5/1/4 | 0.7359 / 1.0000 | [-1.342,+2.508] | [-1.445,+2.305] |
| S09 vs S05 | +0.6689 | +1.0070 | +1.5166 | 16/12/+4 | 3/7/0 | 7/2/1 | 0.5716 / 1.0000 | [-1.171,+2.341] | [-0.500,+2.671] |
| S02 vs B3 | +0.8361 | +0.6990 | -1.2566 | 17/12/+5 | 5/3/2 | 6/2/2 | 0.4583 / 1.0000 | [-0.836,+2.676] | [-0.909,+2.384] |
| S01 vs S03 | +0.6689 | +0.8676 | +0.0143 | 17/13/+4 | 4/5/1 | 5/3/2 | 0.5847 / 1.0000 | [-1.003,+2.341] | [-0.873,+2.648] |
| S04 vs S06 | -0.5017 | -0.2496 | -0.0079 | 16/19/-3 | 2/2/6 | 3/1/6 | 0.7359 / 1.0000 | [-2.341,+1.338] | [-2.071,+1.568] |
| S05 vs S07 | -0.5017 | -0.6971 | -0.8141 | 11/14/-3 | 1/6/3 | 3/2/5 | 0.6900 / 1.0000 | [-2.174,+1.171] | [-2.052,+0.739] |

Diagnostic contrasts are also present in `results/paired_contrasts.csv`. S09 versus C02 is +0.3344 ACC points and -0.4238 AUC_original points. C03 versus S09 is identical at the recorded OOF precision for all three metrics and all hard predictions.

## I. Gate initialization and learning

The gate formula is `g_m = scale_m * sigmoid(theta_m)`. S09 starts at `[0.001, 0.001, 0.001, 0.001, 0.99, 0.001]` within floating point tolerance. The nine S groups reproduce their declared probabilities and scales. The 120 new folds show finite gate gradients and optimizer updates. The gate Parameter identity is preserved, only its optimizer state is removed at the warmup transition, and RNG is unchanged.

For S09, the mean cognitive effective gate is 0.990000 at initialization and 0.989569 at Best. The other five mean Best effective gates remain approximately 0.001006, 0.001021, 0.001001, 0.001001, and 0.001002. S09 has 2,996 finite gate-gradient records and 2,996 positive optimizer updates across its ten folds. The corresponding C03 gate has the same aggregate values because its constructor-time initialization converged to the same Best states in this run.

Warmup shared states, optimizer states, relation pairs, RNG, loss history, and frozen spectral statistics match across the registered S-group pairing audit. C03 is recorded separately as the declared constructor-initialization diagnostic.

## J. DARG graph and relation mechanism

The strong gate changes graph values, but the effect is small after fusion into the classifier. For S09, the COGNITIVE_TEST graph has 159,399 original effective edges, 159,399 scored candidates, and 159,399 modified edges. It contains 132,679 positive enhancements and 26,720 negative suppressions. Mean relative adjacency change is 0.229742. The fusion-weighted GCN message ratio is 0.000692 and the DCR message ratio is 0.000247.

COGNITIVE_TEST relation AUC for S09 is 0.80147 over all patient pairs, 0.73089 over graph-valid edges, 0.70616 over candidate edges before support intersection, and 0.73089 over actual scored candidates. The all-pair value is therefore not used as evidence that every graph edge is a reliable disease relation.

Across all new groups, the same-checkpoint Direct-Off intervention changes one argmax in aggregate and damages one correct prediction. Disease-Neutral has the same aggregate result. Cognitive-Graph-Off changes one argmax and damages one. Cognitive-Score-Permuted changes two argmax values and damages two. The numerical logit changes are finite, but the net classification contribution is not positive or stable. S09 itself has one Cognitive-Score-Permuted argmax change, with one damaged prediction and no correction.

## K. DPSL mechanism

Residual-Off equals Native exactly in every audited new fold. This verifies the residual intervention and preserves the native route. Across the 120 new folds, Native-to-Full changes seven argmax values, corrects two, and damages five. For S09, Native-to-Full changes one argmax and damages one. S09 versus C02 shows a mixed training contribution, with higher ACC and lower AUC_original for the full DPSL run. DPSL is retained faithfully, but the data do not establish a consistent positive increment.

## L. Checkpoint replay

- Historical B0-B3: 40/40 Best and Latest replays pass.
- New S01-S09 and C01-C03: 120/120 Best and Latest replays pass.
- All groups: 160/160 independent Best/Latest replays pass.
- New groups: 120/120 Latest next-step recovery replays pass after isolated deep-copy restoration of optimizer state.
- Relation pairs: 4096 per fold, with 2048 same and 2048 different pairs. Every relation-pair patient is in the outer training set and no test patient occurs in a pair.
- DPSL Residual-Off: exact Native recovery in the independent replay.

The initial recovery audit used a shallow checkpoint optimizer-state object and was corrected in the audit script. The corrected replay starts each repetition from an immutable deep copy. Training checkpoints and the sealed training implementation were not changed.

## M. OOF and log audits

The aggregate contains 9,568 rows, exactly 598 rows for each of the 16 groups. Each group contains every patient exactly once. The author log audit is PASS 16/16. Logs preserve the author field order, three `trn-loss-MF` fields, four-decimal values, `/450` denominator, fold records, and summary format. Historical logs are copied from the real archive. New logs are reconstructed from saved real epoch records and predictions. The original `/data/adodas/SPromptGL-main/result_Tadpole.txt` is unchanged.

## N. Method limitations

This is a protocol-matched exploratory comparison. Full-data RFE, mean imputation, and StandardScaler precede outer CV. The patient graph is transductive. `val_idx == test_idx`, and outer test-fold ACC selects Best checkpoints. These choices introduce optimistic selection and transductive bias, so the results are not unbiased independent generalization estimates. Test labels enter only post-training diagnostics and paired evaluation, never relation-pair supervision, ALoss targets, gate fitting, or training loss.

## O. GO/NO-GO decision

| Criterion | Result |
| --- | --- |
| Full registered execution | GO, 16/16 groups and 160/160 folds |
| Method fidelity preflight | GO, PASS |
| Checkpoint Best/Latest replay | GO, 160/160 |
| Latest deterministic next-step replay | GO, 120/120 |
| Both primary metrics above Y3/Z0 | NO-GO, no new group |
| Both paper reference means reached | NO-GO |
| Holm-adjusted paired evidence | NO-GO, no primary comparison significant |
| DARG observable classification contribution | NO-GO, numerical changes without stable positive net |
| DPSL retained and positive increment | NO-GO, exact residual audit but mixed classification effect |
| Follow-up complete-model candidate | B0 remains incumbent. S02 is the closest new descriptive point |

## P. Reproducibility and source seal

The training seal status is `SEALED_BEFORE_TRAINING`. Sealed training implementation SHA-256 is `7ed31c60cecb6845eb829127d8d7268cc6456064319d527d72a386fada25bc47`. The independent audit-only JSON serializer adapter is separately recorded at `audits/independent_audit_wrapper.json`. It converts optional diagnostic tensors after training and does not change training compute, optimizer state, RNG, checkpoint bytes, or prediction values.

## Q. Deliverables

- `results/group_summary.csv`
- `results/fold_metrics.csv`
- `results/oof_predictions.csv`
- `results/paired_contrasts.csv`
- `audits/preflight.json`
- `audits/method_fidelity.json`
- `audits/warmup_pairing_actual.json`
- `audits/gate_initialization.json`
- `audits/checkpoint_replay.json`
- `audits/mechanism_summary.json`
- `audits/log_format_audit.json`
- `audits/aggregate_results.json`
- `audits/training_seal.json`
- `results/resume_commands.md`

## R. Recovery command

```bash
cd /model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
python3 -u code/run_e8385.py --groups S01 S02 S03 S04 S05 S06 S07 S08 S09 C01 C02 C03 --workers 8
```

The runner validates and skips completed audited folds, and resumes incomplete folds from Latest. The independent replay, mechanism, log, aggregate, and final-audit commands are recorded in `results/resume_commands.md` and the audit scripts.

## S. Final interpretation

Strong COGNITIVE_TEST gating produces the requested shared DARG+DPSL implementation and clear internal graph changes. In the frozen same-checkpoint interventions, those changes rarely reach the hard classifier decision. S09 improves over several weaker new configurations, but falls below B0 on both primary metrics. C03's constructor timing does not change this run's OOF outputs. The evidence supports retaining the ABIDE-derived implementation as method-faithful infrastructure while rejecting it as a TADPOLE performance replacement for Y3/Z0.

## T. Audit references

The complete per-fold values, confusion matrices, patient-level predictions, checkpoint hashes, gate traces, relation-support strata, and intervention logits are in the CSV and JSON files listed above. `audits/final_audit.py` is the final integrity checker and records the final report hash after this report is written.

## U. Report provenance

This report was generated only from the locked data, real server checkpoint archives, formal training artifacts, independent GPU replays, and post-training diagnostic outputs. No GitHub CSV was used as a checkpoint substitute. No root model source was modified.
