# Experiment 83.8.2 method fidelity matrix

| Object | Frozen identity | This experiment |
|---|---|---|
| Paper Eq.(1) | `(1/M) sum_m [L_S(A_m) + ||Ahat-A_m||_F^2]`, squared Frobenius | Recorded as the paper object only. N1-N6 use the six-modality squared train-by-train ALoss defined in the protocol |
| Paper Eq.(7) | `L = L_G + L_Label`, lambda=eta=1 | Recorded as paper formulation. Historical source behavior remains the runtime authority |
| Author `network.py` | Four hard-coded branches, threshold 0.9, unsquared `torch.linalg.norm`, second branch overwrites previous combined message | Preserved in source snapshot and not restored as the new training loss |
| Author `model.py` simple-2 | Backward uses `cls_loss + all_graph_loss`; printed ALoss is excluded | Preserved as reviewed historical behavior |
| E83.8.1 ALoss | Six actual GraphLearn graphs, squared Frobenius, train labels, train-by-train restriction, coefficient 1, active in warmup | Formula retained. Only coefficient and activation schedule are changed per N0-N6 registration |
| E83.6 PairMean | Continuous cosine score matrices, class-pair mean, a distinct object | Not used in this experiment |
| GFR | Six GraphLearn branches, threshold 0.9, transductive graph, original message fusion and second-modality overwrite behavior | Retained |
| DARG | Positive-DCR, signed positive and negative relation paths, ten-step historical warmup | Retained |
| DPSL | MMGPL-12 prompt, three class pairs, spectral statistics, residual cap 0.08, shared gradient cap 0.001 | Retained for N0-N5 |
| Optimizers | Existing Adam/AdamW groups, learning rates, decay and state | Retained. N5 calibration is gradient-only and does not update state |
| Checkpoint protocol | First strict outer-test ACC improvement, latest recovery state, wait > 200, max 450 | Retained |

N0-N5 share the G4 architecture. N6 shares the G3 architecture and the N2 ALoss schedule and coefficient. No historical checkpoint is used as an initializer.
