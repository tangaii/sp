# E5-PAR

Paper-aligned ablation and reproduction for six-modality TADPOLE in `tangaii/sp`.

## Status

`COMPLETE_AUDITED`. Seven groups completed 70/70 formal folds, each for 300 epochs. Independent checkpoint reload, prediction, split, input, initialization, RNG, structure, mechanism, and OOF checks are recorded in `audit_results.json`.

G0 has a separate `EXACT_BASELINE_REPRODUCTION` identity result. The historical probability CSV was rounded, with maximum absolute difference 3.0e-8.

## Protocol

Each group records its actual frozen source hashes. G4-G6 use the patched source that initializes `avg_pool` for `res_only`; G0-G3 retain the earlier hash because their valid paths completed before that no-effect-for-their-path correction.

The frozen six-modality input snapshot, ten-fold split, RFE-before-outer-CV workflow, val=test Best-checkpoint rule, historical weighted loss, legacy GFR fusion, prompts, and frozen RN50 cache are recorded in the group configurations.

G4 is a released-code-compatible RES-only decomposition. It uses `legacy_recalibrate` with explicit `part_strength=1`. The unreleased `equations_5_6` branch is not used because its projection modules are undefined in the released code.

See `summary.md` and `summary.csv` for results.
