# Experiment 83.8.5.2 method fidelity matrix

| contract | implementation evidence | status |
|---|---|---|
| TADPOLE order and dimensions | source_runtime TADPOLE_MODALITIES and MODAL_DIMS in runner | pending preflight |
| E74.4-G2 Positive-DCR | archived cadr_gfr.py and soft_positive_dcr.py SHA-verified before binding | PASS |
| six modality adaptation | only cardinality, named anchor index, block slices and gate scale bindings are adapted | PASS |
| COGNITIVE_TEST strong anchor | zero-based index 4, no CORR rename, target p and scale are config-locked | pending preflight |
| gate math | g=scale*sigmoid(theta), no new Parameter, Adam state clearing after warmup | pending preflight |
| train-only relation labels | outer train_idx is used for pair construction and labels are detached | pending preflight |
| original support and signed edges | unified_adjacency reweights original positive support and retains signed relation values | pending preflight |
| direct DCR | direct_scale=0.02 and direct_single_gate=false | PASS |
| E82.4.6.10-G1 SCMR-DA-SRPE | archived SCMR implementation and frozen 12-vector prompt input | pending preflight |
| Paper-ALoss | squared Frobenius, 0.01, epoch 11 onward, train block only | PASS |
| Best/Latest | atomic latest and first strict outer test ACC improvement | PASS |
| author log | result_Tadpole.txt field sequence and precision | pending log audit |
| known structural zero | M1 graph may have structural zero gradient and is recorded, not used as all-modality gate | PASS |
| protocol limits | val=test and transductive graph disclosed in final report | PASS |

