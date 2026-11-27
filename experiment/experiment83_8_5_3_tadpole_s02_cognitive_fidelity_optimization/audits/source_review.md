# Source and paper review

The MICCAI 2025 paper was read from the open paper PDF. Eq.(1) uses the graph construction objective with the squared Frobenius reconstruction term. Eq.(7) combines the classification, graph, prompt and auxiliary objectives in the paper's total training objective. The implementation sections describe GFR, GraphLearn and GCN, followed by CAM, PEN, RES and SPGS. The TADPOLE setup uses six modality blocks and three diagnosis classes with ten-fold cross validation.

The paper Table 2 reference for TADPOLE is ACC 94.81 +/- 2.95 percent and AUC 95.59 +/- 2.57 percent. The author implementation uses the TADPOLE result_Tadpole.txt epoch record with three trn-loss-MF fields and the original fold and summary format. That file is retained as a read-only template and its SHA is recorded in source_manifest.json.

The author repository was cloned at ba3b9446fa451fde6864d6555bdf324de0c4b64a. The SABER repository was cloned in the experiment dependency area. The E83.8.5.1 anchor resolves to 912a3ece628f818cec7525fe6323350f5a06e27e. The E83.8.5 source reference resolves to ae225e5135046d1d9622968b7d011f6aa55fa60d.

The ABIDE E74.1 contract was checked against the archived runner. Its ten-epoch warmup keeps shared model, optimizer and RNG order. The gate is reparameterized as g_m=s_m*sigmoid(theta_m) using the existing adaptive_gate_logits Parameter. Non-anchor probability is 0.10, anchor probability is 0.99, and scale is [0.01,0.01,0.01,1.0]. Adam state for that Parameter is cleared while other optimizer state and RNG are preserved. The TADPOLE adapter generalizes the same mapping to six modalities and index 4, without renaming COGNITIVE_TEST.

The historical groups are mapped to real server checkpoint archives. B0 is R0, B1 is R1, B2 is E83.8.5.1 N1, and B3 is R4. GitHub CSV or summary files are not used as checkpoint substitutes.

