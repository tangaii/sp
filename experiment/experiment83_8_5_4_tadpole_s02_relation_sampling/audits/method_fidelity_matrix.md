# E83.8.5.4 Method Fidelity Matrix

| Component | Source used by T1-T4 | Experiment change |
|---|---|---|
| Six-modality GFR/CAM/RES | Copied E83.8.5.2 `code/common/fidelity_model.py` and archived source snapshots | None |
| DARG RelationEncoder/RelationScorer | Archived `history_darg/cadr_gfr.py` bound by `code/common/source_runtime.py` | Pair index and target tensors at epoch 11 only |
| Positive-DCR and signed graph | Archived `soft_positive_dcr.py` | None |
| DPSL SCMR-DA-SRPE | Copied E83.8.5.2 `code/common/scmr_runtime.py` and archived source | None |
| Warmup, Adam/AdamW, GRL, loss coefficients | Copied E83.8.5.2 `code/run_e83854.py` | Formal pair source switches after warmup |
| Patient relation sampling | `code/relation_sampling.py` | T1, T2, T3, T4 registered variables |
| ABIDE | No ABIDE source or checkpoint is imported at runtime | No change |

The server E83.8.5.3 directory is not imported or read by the training process. The new directory contains its own copied source snapshot and its own sampler.
