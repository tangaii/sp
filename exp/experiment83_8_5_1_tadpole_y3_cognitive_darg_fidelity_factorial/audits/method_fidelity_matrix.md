# Method Fidelity Matrix

| Component | Historical source | Preserved implementation | TADPOLE adaptation | Evidence |
|---|---|---|---|---|
| GFR and GraphLearn | Original SPromptGL author source | Historical graph construction, adjacency normalization, GraphConstructLoss, and GFR message path | Six locked modality widths | Source review and fixed-tensor regression |
| CAM and prompts | Original SPromptGL author source | CAM, PEN, RES, and frozen prompt semantics | Three-class MMGPL-12 with twelve fixed queries | Prompt manifest and 12-query preflight |
| DARG relation encoding | E74.4-G2 and E82.4.6.10-G1 | RelationEncoder, RelationScorer, balanced relation pairs, BCE coefficient 0.05, preservation coefficient 0.01 | Six modality blocks and registered COGNITIVE_TEST anchor index 4 | Source hashes, gradient checks, label permutation check |
| Conditional adversarial alignment | E74.4-G2 lineage | Conditional discriminator, same/different condition, detached reference branch, GRL, adversarial coefficient 0.01 | CORR index 3 for ABIDE and COGNITIVE_TEST index 4 for TADPOLE | Default 0.03 forward/gradient regression and 0.015 half-gradient regression |
| Signed Positive-DCR | E74.4-G2 | Signed positive and negative relation edges, reweight strength 0.8, support-only updates, direct scale 0.02, double Gate | Six TADPOLE modalities and registered edge anchor index 4 | Edge-support, symmetry, signed-edge and direct-positive checks |
| DPSL | E82.4.6.10-G1 | SCMR-DA-SRPE, joint DARG+DPSL training, frozen RN50 prompts, old confidence mode, three diagnosis pairs | NC/MCI/AD class routing | Residual-Off exactness, residual cap, shared gradient budget, prompt identity |
| Warmup and optimization | E83.8.3-Y3 / E83.8.4-Z0 / E83.8.5-R0 | Shared initialization, optimizer, relation pairs, RNG, spectral statistics, and Paper-ALoss schedule | Same four new groups paired by fold | `audits/warmup_pairing.json` |
| Evaluation | E83.8.5 | First strict test-ACC Best, Latest state, hard one-hot macro OVR `AUC_original` | Registered factorial groups | Locked fold identity, independent checkpoint replay, patient-level predictions |

No additional network module, relation objective, disease prototype bank, gradient projection, or TADPOLE-only propagation module is registered. GRL 0.015 changes only the non-anchor gradient reversal scalar. The original default remains 0.03.
