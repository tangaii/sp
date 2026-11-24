# SPromptGL TADPOLE six-modal GAR experiment

This directory is an isolated reproduction and adaptation experiment for `tangaii/sp`. It uses the frozen 598-subject, six-modal input snapshot with dimensions `[145,148,3,36,24,9]`, labels NC/MCI/AD with counts `[209,315,74]`, and the pre-existing stratified ten-fold split (`seed=0`, `split_seed=10`). Preprocessing and RFE are inherited unchanged from the historical TADPOLE input.

The five formal groups share the copied model implementation and the MMGPL class-name prompt configuration used by the historical best group. G0 is the strict base rerun. G1 adds fold-training-node 0.97 similarity quantiles per modality. G2 adds the specified straight-through threshold mask at temperature 0.1. G3 replaces pair supervision with equal class-relation weighting. G4 combines G1-G3. The legacy GFR fusion path is retained in every group, including its first-message overwrite behavior.

All groups use 300 epochs, Adam, learning rate 0.012, weight decay 0.0036, six-modality graph-structure loss divided by six, full-cohort transductive graph construction, and first-maximal outer-test-fold ACC checkpoint selection (`val=test`). This checkpoint policy is retained for protocol compatibility and is recorded as an evaluation limitation.

`results/<group>/` contains fold checkpoints, histories, predictions, metrics, split manifests, and aggregate summaries. `audit/audit.json` independently recomputes fold and out-of-fold metrics. `summary.md` and `summary.csv` contain the final comparison.
