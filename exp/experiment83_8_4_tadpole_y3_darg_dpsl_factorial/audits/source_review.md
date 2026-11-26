# Experiment 83.8.4 source review

## Review scope and provenance

The review was performed before formal training. The experiment uses the Y3 formal artifacts from `/model/spromptgl/experiment/experiment83_8_3_tadpole_n3_darg_dpsl_optimization`, including the ten Best and ten Latest checkpoints, fold predictions, epoch records, and audit files. The requested source anchors are recorded in `protocol.json` and `source_manifest.json`.

| Source | Commit or digest | Review conclusion |
|---|---|---|
| MICCAI paper | `c56762c64025bb9222d872459dea9f4dfe158db1e470fba89e7e7c822743e355` in the historical manifest | Paper and TADPOLE log convention reviewed |
| Original author repository | `ba3b9446fa451fde6864d6555bdf324de0c4b64a` | `main.py`, `model.py`, `network.py`, and `result_Tadpole.txt` reviewed |
| SABER repository | requested E83.8.3 anchor `cbaaf1c`, local historical manifest commit `5ad91878764863b04089a94c497707cc7459168b` | Local source snapshot and handoff reviewed. The requested short anchor is retained because the public page was unavailable during local verification |
| Experiment repository | `252fa96b5ff2ac89250cda378c777fc36a99d316` | E83.8.2-N3 source, protocol, checkpoint, and audit snapshot reviewed |

The local source snapshot hashes are immutable entries in `source_manifest.json`. The repository pages and paper were also checked directly at their recorded URLs.

## DARG function-level review

The archived Positive-DCR implementation is `sources/e83_5_sources/history_darg/soft_positive_dcr.py`, SHA-256 `3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718`.

- `_edge_delta` at lines 81–89 computes the signed relation from `q`, keeps positive and negative relations, and applies `delta = old * (REWEIGHT_STRENGTH * gate * relation)`. The edge-weight gate is therefore applied inside the adjacency change.
- `_bounded_disease_residual` at lines 92–106 computes the residual from `delta_adjacency`, then applies the bounded block transform and modality localization.
- `rudag_forward` at lines 194–210 computes `gate = GATE_SCALES[modality] * sigmoid(gate_logits[modality])`, constructs the unified adjacency and `delta_adjacency`, selects the positive residual, and computes the historical direct path as `direct = direct_scale * gate * bounded`.
- Because `bounded` is derived from the already gated `delta_adjacency`, the historical direct branch contains two uses of the same gate. This is a confirmed property of the implementation. It is retained as Z0 and Z3 for the registered comparison.
- The new single-gate branch changes only the final direct expression to `direct = direct_scale * bounded`. It leaves the unified adjacency, candidate support, signed relation, GCN message, fusion weight, and auxiliary losses unchanged. Z1, Z2, Z4, and Z5 use this branch.
- `code/common/source_runtime.py` creates an isolated archived namespace per modality, gate tuple, direct scale, and single-gate flag. This prevents construction order from mutating another group’s archived functions.
- `code/common/fidelity_model.py`, `SignedDARG.__init__` at lines 455–480, binds the six TADPOLE modalities, gate scales, direct scale, and direct-gate flag. `FidelityModel.__init__` at lines 684–723 passes the registered method fields without changing the GFR or CAM state layout.
- `code/run_e8384.py` uses the six modality widths `(145, 148, 3, 36, 24, 9)` and preserves the historical candidate top-k of 12, positive threshold 0.55, negative threshold 0.45, negative scale 0.25, and GFR threshold 0.9.

The DARG review will report edge-weight changes, signed edge counts, graph messages, direct residuals, fusion outputs, logits, and gradients separately. A nonzero gradient alone will not be treated as proof of useful disease propagation.

## DPSL function-level review

The copied SCMR runtime is `code/common/scmr_runtime.py`. `_all_pair_forward` at lines 197–203 applies confidence protection after the Native-to-Full residual is computed. `confidence_gate` at lines 336–346 uses detached Native logits and the top two native probabilities.

- The historical mode computes `g_old = clamp(4 * p1 * p2, 0, 1)`.
- The registered multiclass mode computes `g_new = clamp(4 * p1 * p2 / ((p1 + p2)^2 + 1e-12), 0, 1)`.
- The binary branch keeps the old formula regardless of mode. No labels are read by the gate.
- `apply_confidence_protection` at lines 349–377 multiplies the residual and both residual components by the detached gate, records the mode, and preserves the residual norm cap.
- The three class pairs are retained as `(0,1)`, `(0,2)`, and `(1,2)`. MMGPL-12 prompt vectors, frozen RN50 text features, spectral statistics, EMA behavior, shared gradient budget `0.001`, and CAM residual cap `0.08` remain bound to the Y3 implementation.
- Gate override tests provide exact Residual-Off and gate-one controls. These are inference probes at one checkpoint and will be reported separately from changes learned during training.

## Loss, training, and logging review

The copied runner preserves the Y3/N3 squared Frobenius ALoss path, coefficient `0.01`, activation from epoch 11, ten-epoch DARG warmup, Adam/AdamW settings, relation-pair construction, transductive graph, test-as-validation protocol, first strict test-ACC Best rule, maximum 450 epochs, and `wait > 200` early stopping. The third author-format `trn-loss-MF` field is generated from the unsquared six-modality observation used for historical display. The actual training objective is the squared Frobenius term recorded separately in checkpoint history and audits.

The original author source uses the epoch format `%3d/%d`, prints three consecutive `trn-loss-MF` fields, then `trn-auc`, `acc`, `val-auc`, and `acc`. Its archived log has denominator 300. E83.8.4 logs retain the same field order, spacing, precision, fold boundaries, and summary layout while using the actual registered denominator 450 and only epochs executed by the runner.

## Pre-training conclusions

The historical Y3 checkpoint and model state are suitable for Z0 reuse. The preflight must still verify exact fold replay, OOF identity, Latest restoration, and source identity. Z1–Z5 differ from Y3 only through the two registered DARG direct-path fields and the registered DPSL confidence-gate mode. No root model code or historical artifact is modified by this experiment.
