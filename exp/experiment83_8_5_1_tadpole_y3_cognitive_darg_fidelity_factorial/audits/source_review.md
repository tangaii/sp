# Source Review

## Source identity

The MICCAI 2025 SPromptGL paper PDF was read from the publisher URL recorded in `source_manifest.json`. Its local SHA-256 is `219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89`. The review covered GFR, CAM, RES, PEN, SPGS, GraphLearn, GraphConstructLoss, Eq. (1), Eq. (7), the TADPOLE protocol, Table 2, and the paper's evaluation definitions. The paper reports TADPOLE ACC 94.81 ± 2.95% and AUC 95.59 ± 2.57%.

The original author source is anchored to `wanxixi11/SPromptGL` commit `ba3b9446fa451fde6864d6555bdf324de0c4b64a`. The local source snapshot includes `main.py`, `model.py`, `network.py`, `layers.py`, `utils.py`, and `result_Tadpole.txt`. The author log template SHA-256 is `5c80d16204e0882e5e264e42952c714397c2d31f07835b9e60d6dafd9488e3b7`, and its verified Git blob is `0446670764286afbeb86573dadfdbb50ae6af15e`.

The SABER checkout at `/data/adodas/SPromptGL-main` was inspected at HEAD `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4`. Its worktree was dirty before this experiment and was not edited. Historical method lineage was checked against the archived E74.4-G2, E82.4.6.10-G1, E83.8.2, E83.8.3, E83.8.4, and E83.8.5 records in the server experiment archive.

The requested TADPOLE reproduction reference was `tangaii/sp` commit `ae225e5`. The available verified source archive records commit `252fa96b5ff2ac89250cda378c777fc36a99d316` and config blob `73d91d16f3510df1338fafafa31e2fd284527011`. This identity difference is retained as a provenance limitation. The server E83.8.5 runtime source hashes match its local archived manifest. GitHub does not contain the excluded `.pt` files, so all historical checkpoints were read from the server archive.

## Method lineage

The DARG relation encoder, relation scorer, conditional modality discriminator, conditional adversarial objective, gradient reversal, relation BCE, preservation loss, signed Positive-DCR edge update, and original graph-support constraint trace to the ABIDE E74.4-G2 and E82.4.6.10-G1 source archive. The frozen source hashes are recorded in `source_manifest.json` and `audits/method_fidelity.json`. The CADR-GFR default GRL coefficient remains 0.03. The local adapter exposes it as a scalar and preserves the historical forward pass, loss, discriminator gradients, detached alignment reference, optimizer parameter set, and non-anchor gradient path.

The ABIDE adapter uses four modalities and the historical CORR reference at index 3. The TADPOLE adapter uses six modalities and selects COGNITIVE_TEST at index 4 through the same generic anchor parameters. Positive-DCR retains signed positive and negative edges, its historical reweight strength 0.8, direct scale 0.02, and the second Gate. It changes only weights on existing graph support.

DPSL traces to E82.4.6.10-G1 SCMR-DA-SRPE and remains jointly trained with DARG. TADPOLE uses the already verified frozen MMGPL-12 RN50/SCMR prompt vectors. Its three diagnosis pairs, confidence mode, residual cap 0.08, shared gradient budget 0.001, and AdamW settings are bound to the historical implementation. The ABIDE binary source path retains eight prompts.

The TADPOLE historical source and configuration map Y3 to E83.8.3-Y3, E83.8.4-Z0, and E83.8.5-R0. H0/R0 is the incumbent. H1/R1 isolates cognitive alignment. H2/R4 adds the cognitive edge rule and 0.10 cognitive Gate. The new groups branch from the same verified R0 warmup state and use the registered 2 × 2 Gate/GRL factors.

## Protocol and execution

The preserved training contract uses the 598-patient TADPOLE cohort, the locked ten folds, seed 0, 10-epoch DARG warmup, 450 epoch cap, `wait > 200` early stopping, and first strict test-ACC improvement checkpoint selection. The historical Paper-ALoss objective is `paper_aloss` with coefficient 0.01 from epoch 11 onward. The three displayed `trn-loss-MF` fields keep the previously audited display definitions and do not describe three independent loss terms.

The protocol retains full-data RFE, mean imputation, StandardScaler before CV, a transductive patient graph, and `val_idx == test_idx`. These choices can make the estimate optimistic. This is an exploratory comparison under the historical protocol and does not estimate unbiased independent generalization.

The paper's full SPromptGL result is a reference only. DARG+DPSL additions, available historical code, and adaptation details differ, so numeric proximity does not establish an exact reproduction.
