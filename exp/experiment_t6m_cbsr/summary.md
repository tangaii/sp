# CBSR TADPOLE Results

**Status:** `COMPLETE_AUDITED`. Four groups completed 10/10 folds, for 40/40 formal folds. Best checkpoints were reloaded and their probabilities matched the saved predictions. Independent OOF recomputation and fold/input/split assertions passed.

The local checkout does not contain commit `0e67225cbac6ef5c8e365fe7e864c970dd3bd768`. Source provenance was instead checked against the frozen historical experiment files, input hashes, and identical ten-fold split. G0 is not an exact historical reproduction. It reached 92.31% mean ACC and 552/598 OOF correct versus historical 91.81% and 549/598. Fold-level best epochs differed in 9/10 folds, and several AUC values differed. All repaired groups share identical per-fold initial GFR/PLN state hashes with this run's G0.

## Main Results

Mean and population standard deviation are across ten folds. OOF counts and confusion matrices are independently recalculated over all 598 held-out predictions.

| Group | beta | ACC % | Hard-AUC % | Prob-AUC % | NC SEN % | MCI SEN % | AD SEN % | OOF correct | delta vs G0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0_BASELINE | 0.00 | 92.31 ± 1.84 | 93.74 ± 2.40 | 96.09 ± 2.30 | 93.78 | 92.06 | 89.19 | 552 | 0 |
| G1_CB_FULL_FIXED | 1.00 | 90.13 ± 3.20 | 92.65 ± 2.31 | 96.35 ± 1.68 | 89.95 | 89.52 | 93.24 | 539 | -13 |
| G2_CB_HALF_FIXED | 0.50 | 91.30 ± 2.09 | 93.19 ± 2.02 | 96.46 ± 1.39 | 94.26 | 90.48 | 87.84 | 546 | -6 |
| G3_CB_QUARTER_FIXED | 0.25 | 92.30 ± 3.22 | 93.74 ± 2.29 | 96.54 ± 2.09 | 91.87 | 93.02 | 90.54 | 552 | 0 |

OOF confusion matrices use rows as true NC/MCI/AD and columns as predicted NC/MCI/AD.

| Group | OOF confusion matrix |
|---|---|
| G0 | `[[196,13,0],[16,290,9],[0,8,66]]` |
| G1 | `[[188,21,0],[19,282,14],[0,5,69]]` |
| G2 | `[[197,12,0],[17,285,13],[0,9,65]]` |
| G3 | `[[192,17,0],[12,293,10],[0,7,67]]` |

Specificity by class, in NC/MCI/AD order, was G0 `95.89/92.58/98.28%`, G1 `95.12/90.81/97.33%`, G2 `96.14/91.52/97.71%`, and G3 `96.92/91.52/98.09%`.

## Repair and Audit

The prior implementation generated masks in the order NC-NC, NC-MCI, NC-AD, MCI-MCI, MCI-AD, AD-AD, then paired them with a differently ordered target list. CBSR binds each explicit relation key to its own mask and target. A six-subject audit confirmed pair counts `2,2,2,8,8,8`, no diagonal pairs or mask overlap, and correct hand-computed MSE and gradient direction for all six relations. The saved audit passed.

All four groups used the same frozen input hashes and split file. Per-fold initial GFR/PLN state hashes match across groups. The supervision uses only labels indexed by that fold's training indices. No test labels enter the graph supervision. Checkpoint selection remains post-update outer-test ACC with first-max tie handling over 300 epochs.

## Mechanism and Interpretation

G1 improved AD sensitivity by 4.05 percentage points over this-run G0, while reducing NC and MCI sensitivity by 3.83 and 2.54 points and losing 13 OOF correct predictions. Full balancing is too strong overall. G2 had no net accuracy benefit and reduced MCI and AD sensitivity. G3 retained the same 552 correct OOF predictions as G0, improved MCI sensitivity by 0.95 points, and increased fold ACC standard deviation from 1.84 to 3.22 points. Its result is a tie in OOF accuracy, not evidence of a robust gain.

The graph audit shows the largest persistent edge density in modality 3, where thresholded edges include substantial different-label connectivity. Per-fold `metrics.json` records all six modalities' density, same/different-label edge rates, relation MSE, and first-epoch GraphLearn gradient magnitudes. The audit found finite gradients. The full CB objective shifts the optimization away from pair-mean weighting, but does not remove heterogeneous graph quality or establish a consistent accuracy benefit.

The best repaired group, G3, reached 92.30% mean ACC, 2.51 percentage points below the paper's 94.81%. Corrected class balancing is **NO-GO** as a performance improvement under this protocol. G3 may be retained only as a neutral MCI-sensitivity tradeoff, while G1 and G2 should not replace the baseline.

The previous erroneous G3/G4 results are recorded only as implementation-failure cases and are not used as valid method comparisons. The original paper result and historical paper-config result are not recomputed by this run. Historical best/G0 reference was 91.81 ± 1.88% ACC, 93.56 ± 2.45% Hard-AUC, 96.59 ± 1.89% Prob-AUC, and 549/598 correct.

## Artifacts

`summary.csv` contains machine-readable group metrics. `audit_results.json` records independent audit status and OOF totals. Each group has fold histories, best checkpoints, predictions, and metrics. Per-fold histories contain classification, graph structure, pair, class-balanced, mixed supervision, total losses, six relation MSEs, train ACC, and test ACC.
