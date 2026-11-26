# Experiment 83.8.5 source review

## Paper and reference implementations

The MICCAI 2025 SPromptGL paper reports 598 TADPOLE patients, ten-fold CV, and Table 2 ACC 94.81 ± 2.95% and AUC 95.59 ± 2.57%. Its GFR constructs modality-wise cosine graphs and applies GCN. CAM uses attention with prompt enhancement and the reported PEN, SPGS, and RES components. The paper combines classification and graph learning loss and describes its graph-supervision term.
[MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)

The author repository at `ba3b9446fa451fde6864d6555bdf324de0c4b64a` supplies the reference graph, GCN, attention, losses, split code, and TADPOLE log. The checked author training code hard-codes four modality blocks and demonstrates ABIDE. The six-modality implementation is the SABER TADPOLE adaptation. The author log matches the workspace template byte for byte. The two class-loss display slots repeat the same classification loss. The archived ALoss slot uses the unsquared display calculation, while backpropagation in the formal Y3 protocol uses the historical squared Frobenius loss with coefficient 0.01 starting at epoch 11.
[Author repository](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a) · [Author TADPOLE log](https://github.com/wanxixi11/SPromptGL/blob/ba3b9446fa451fde6864d6555bdf324de0c4b64a/result_Tadpole.txt)

The SABER checkout is pinned separately from the frozen experimental sources in the source manifest. The TADPOLE adaptation supplies full-cohort preprocessing, RFE, mean filling, StandardScaler, modality order and the fixed split convention. E83.8.2-N3 at `252fa96b5ff2ac89250cda378c777fc36a99d316` supplies the historical ALoss and E61 warmup lineage. The archived local Y3 and Z0 source seals and 10/10 matching checkpoint bytes identify the formal model base. The abbreviated Y3 Git revision `cbaaf1c` in the older manifest could not be resolved with local Git or the repository connector. The local sealed runner, configuration and independently reproduced checkpoint outputs establish the executed lineage.
[SABER](https://github.com/tangaii/SPromptGL/tree/c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4) · [TADPOLE adaptation and N3 snapshot](https://github.com/tangaii/sp/tree/252fa96b5ff2ac89250cda378c777fc36a99d316)

Y3/Z0 preserves six-modality GFR, GCN, CAM and RES, plus E74.4-G2 DARG and the E82.4.6.10-G1 DPSL SCMR adaptation with cached frozen RN50 MMGPL-12 prompts. It uses 450 epochs and strict Best ACC on the same validation and test fold. It retains transductive feature graphs and CV-before-split full-cohort feature selection and preprocessing. These choices constrain generalization claims and differ from an independently held-out protocol. The paper reference means are descriptive comparisons across methods.

## DARG and DPSL execution

| Control | Historical ABIDE | Y3/Z0 TADPOLE | E83.8.5 |
|---|---|---|---|
| Conditional alignment | CORR pair representation detaches in domain loss. The other encoders receive GRL with multiplier −0.03 | No CORR, all six receive GRL | The separate alignment index detaches only the domain-loss input at index 4 for R1, R3, R4 |
| Adaptive gate | CORR probability 0.99, scale 1; others 0.1 and 0.01 | All probabilities 0.1 and scales 0.01 | All probabilities remain 0.1; only the index 4 scale increases to 0.1 in R2–R4 |
| Candidate support | CORR scores all effective off-diagonal edges. Other modalities score mutual-topK ∩ original support | Six ordinary mutual-topK (12) branches | Only R4 scores all effective COGNITIVE_TEST edges |
| Edge relation | CORR uses continuous 2q−1. Ordinary modalities apply 0.55/0.45 cutoffs | Ordinary cutoff in every modality | Only R4 uses continuous 2q−1 for COGNITIVE_TEST |
| Score and propagation | Sigmoid of the historical scorer logit. Temperature 1 in Positive-DCR. Signed adjustment 0.8g(2q−1) times original edge where enabled | Same | Same |
| Direct-DCR | Positive portion of signed edge adjustment; bounded modality-local feature residual; second gate multiplies the residual at scale 0.02 | Same double gate | Same double gate |

The archive's relation encoder and scorer provide BCE and preservation gradients in all modalities. A conditional discriminator learns from detached reference relations. The legacy calibration arguments are identity buffers in this path. The scoring path uses the archived raw sigmoid logit with graph temperature 1. The train-only deterministic sampler creates 4,096 pairs per fold, with 2,048 in each relation class. Relation BCE, adversarial and preservation coefficients are 0.05, 0.01 and 0.01. Pair sampling computes supports from initial feature graphs before using labels from the outer training index. GraphLearn threshold 0.9, the historical second-modality overwrite, graph losses and mode-specific GCN calls are inherited.

The runtime verifies archive SHAs and binds only the anchor comparisons in named definitions using AST inspections. Modality names and feature order stay intact. Disease-Neutral reverts all DARG adjacency adjustments. Direct-Off disables the direct residual while using the full unified graphs. Audit-only controls neutralize the COGNITIVE_TEST edge branch or disable its gate without changing GraphLearn.

Warmup epochs 1–10 run the historical original graph and all historical relation losses. The treatment alignment starts after warmup. A shared post-warmup baseline DARG forward produces the historical DPSL spectral statistics before treatment starts. The train-only spectral fit, reference logits, cached RN50 prompts, head and optimizer initialization, and all RNG states are compared per fold before epoch 11. Frozen RN50, existing confidence protection and historical DPSL paths preserve the 0.08 residual cap and 0.001 shared gradient budget. Formal execution seeds each fold once.

## Pre-experiment anchor evidence

E83.8.5-P0.1 used nested patient-disjoint relation modeling over the locked outer cohort. COGNITIVE_TEST ranked first 10/10 for V–V six-relation Macro-AUC 0.8325, V–T 0.8501, V–V same/different 0.7153 and V–T 0.7238. It identified the anchor; no P0.1 learned parameter or prediction enters formal training. Its Z0 graph audit reports 13,123.9 effective cognitive edges and 2,491.6 candidates and scored edges on average. The 0.0140 coverage denominator is 178,503 potential unordered pairs. All cognitive candidates passed the original support intersection in that audit.
