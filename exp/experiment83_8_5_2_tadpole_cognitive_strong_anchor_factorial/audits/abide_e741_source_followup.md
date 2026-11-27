# ABIDE E74.1 source follow-up audit

The archived E74.1 source used for the four-modality regression is /model/spromptgl/experiment/experiment_abide_authentic_logs_v2/inputs/source_repo/experiment/experiment74_1_rudag/run.py, SHA-256 a487c87d52a0faeded5ff6406ff9908290fc2fdfd9d5b66e6378a5b2aad6ab5c. Its independent audit source SHA-256 is dc7e6e5850d94faa41a8572c43016bb4c54aaee418528d994dc39454bcd08f43.

The archived source's prepare(fold) loads the ten-epoch E61 warmup state, confirms the existing adaptive gate Parameter is in the inherited optimizer, writes logits for probabilities [0.10, 0.10, 0.10, 0.99] in place under torch.no_grad(), freezes edge_gate_logits, clears only that gate's optimizer state, and verifies RNG identity before and after against the warmup RNG. The effective gate uses [0.01, 0.01, 0.01, 1.0], yielding [0.001, 0.001, 0.001, 0.99].

The archived E74.1 manifest reports COMPLETE_AUDITED, 20/20 checkpoint audits, and a passing preflight. Its ten-fold evidence records identical G1/G2 shared model, optimizer, relation-pair, and warmup RNG identities. A read-only verifier independently cross-checks both task records per fold and the four-mode initialization in its mechanism table. The result is saved in audits/abide_4modal_regression.json.

This TADPOLE experiment does not edit or change the ABIDE code path. Its six-modality adapter and cognitive anchor index 4 are local to this sealed experiment.
