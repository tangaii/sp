# Experiment 83.8.5.5 Method Fidelity Matrix

| Component | S02 source retained | Experiment change |
|---|---|---|
| Six-modality GFR/CAM/RES | Copied E83.8.5.2 `code/common/fidelity_model.py` and archived source snapshots | None |
| DARG RelationEncoder/RelationScorer | Archived `history_darg/cadr_gfr.py` bound by `code/common/source_runtime.py` | Pair index and target tensors at epoch 11 only |
| Positive-DCR and signed graph | Archived `soft_positive_dcr.py` | None |
| DPSL SCMR-DA-SRPE | Copied E83.8.5.2 `code/common/scmr_runtime.py` and archived source | None |
| Warmup, Adam/AdamW, GRL, loss coefficients | Copied E83.8.5.2 training implementation | Formal pair source switches after warmup |
| Patient relation sampling | This experiment `code/relation_sampling.py` | U1-U6 common-random-base and registered cognitive interventions |
| ABIDE | No ABIDE source or checkpoint is imported at runtime | No change |

S02 historical six-relation quotas are read from the genuine per-fold S02 Best checkpoint. The server E83.8.5.3 directory is not imported or read by the training process.
