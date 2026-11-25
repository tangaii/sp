# E6-SGCI

Semantic Guidance Causal Isolation for the frozen six-modality TADPOLE cohort.

This directory is self-contained. It copies the E5 frozen input snapshot and shared ten-fold initialization states, then freezes one E6 source tree for all eight groups. G0 and G1 use the released-code-compatible legacy paths without residual calibration. G2-G7 use the same detached, label-free 240-dimensional residual norm calibration with target ratio 0.04 or 0.20.

Run one group with `python run.py GROUP --parallel`. The queue launches at most ten fold workers for that group and refuses to overwrite an accepted fold. Run `python audit.py` only after all eight groups have ten complete folds.

Static limitations are recorded in `static_audit.json`. Raw visit-level TADPOLE data are not present in the permitted repository, so visit-order, duplicate-RID, label-conflict, and Table 1 demographic checks are marked `COHORT_IDENTITY_NOT_ESTABLISHED`.
