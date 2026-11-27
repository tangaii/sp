"""Apply the registered training delta to the copied E83.8.5.2 runner."""
from pathlib import Path

path = Path("code/run_e83853.py")
source = path.read_text()


def swap(before, after):
    global source
    if source.count(before) != 1:
        raise RuntimeError("Expected one source fragment: " + repr(before[:100]))
    source = source.replace(before, after)


def replace_definition(first, last, code):
    global source
    start = source.index(first)
    finish = source.index(last, start)
    source = source[:start] + code + "\n\n" + source[finish:]


swap('"""Train and resume Experiment 83.8.5.2 TADPOLE groups."""',
     '"""Train and resume Experiment 83.8.5.3 TADPOLE groups."""')
swap("import hashlib\n", "import hashlib\nfrom functools import lru_cache\n")
swap('import run_e833 as historical\n',
     'import run_e833 as historical\nfrom relation_sampling import formal_relation_sample\nfrom state_digest import content_sha\n')
replace_definition('HISTORICAL_ROOTS = ', 'OLD_PROMPT_RN50 = ', '''
ARCHIVE = ROOT.parent / "experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial"
HISTORICAL_GROUPS = {"H0": "B0", "H1": "S02"}
''')
replace_definition('EXPERIMENT = ', 'PAPER_ALOSS_MATHEMATICAL_COEFFICIENT = ', '''
EXPERIMENT = "Experiment 83.8.5.3"
TRAIN_GROUPS = ("A1", "A2", "P1", "P2", "P3", "R1", "R2", "D1")
FORMAL_GROUPS = (*HISTORICAL_GROUPS, *TRAIN_GROUPS)
''')
replace_definition('def require_training_gate(', 'def configure_worker()', '''
def require_training_gate(groups):
    seal_path = ROOT / "audits/training_seal.json"
    seal = json.loads(seal_path.read_text())
    if (seal["status"] != "SEALED_BEFORE_TRAINING" or seal["experiment"] != EXPERIMENT
            or seal["files"] != current_implementation_hashes()
            or seal["implementation_sha256"] != implementation_fingerprint(seal["files"])):
        raise RuntimeError("Sealed training source, input or configuration differs")
    for name, expected in seal["preflight_files"].items():
        if sha_file(ROOT / name) != expected:
            raise RuntimeError(f"Preflight evidence changed: {name}")
    preflight = json.loads((ROOT / "audits/preflight.json").read_text())
    if (preflight["status"] != "PASS" or preflight["implementation_sha256"] != seal["implementation_sha256"]
            or not set(groups) <= set(preflight["groups"])):
        raise RuntimeError("Formal-training preflight did not pass for the requested groups")
    return seal
''')
swap('def load_data():', '@lru_cache(maxsize=1)\ndef load_data():')
swap('def group_config(group: str):', '@lru_cache(maxsize=None)\ndef group_config(group: str):')
replace_definition('def warmup_reference_best(', 'def build_model(', '''
@lru_cache(maxsize=1)
def warmup_references():
    return torch.load(ROOT / "inputs/h1_warmup_reference.pt", map_location="cpu", weights_only=False)
''')
swap('    update_audits = []\n    log_records = []\n',
     '    update_audits = []\n    log_records = []\n    step_witnesses = []\n')
swap('        step_optimizers(optimizers)\n        losses = {', '''        step_optimizers(optimizers)
        step_witnesses.append({
            "epoch": epoch, "model_sha256": content_sha(model.state_dict()),
            "optimizer_sha256": content_sha({n: o.state_dict() for n, o in optimizers.items()}),
            "rng_sha256_by_family": {key: content_sha(state)
                                     for key, state in historical.capture_rng_state().items()},
        })
        losses = {''')
swap('return history, historical.capture_rng_state(), update_audits, log_records',
     'return history, historical.capture_rng_state(), update_audits, log_records, step_witnesses')
replace_definition('def compare_warmup(', 'def compare_warmup_spectral_state(', '''
def compare_warmup(fold, warmup, pair, same, train_idx, test_idx):
    reference = warmup_references()[fold]
    expected = reference["warmup"]
    keys = ("initial_model_sha256", "initial_optimizer_sha256", "initial_rng_sha256",
            "base_state_sha256", "base_state_without_adaptive_gate_sha256",
            "optimizer_state_sha256", "shared_darg_cam_gfr_state_sha256",
            "shared_optimizer_state_sha256", "rng_state_sha256", "pair_sha256",
            "train_index_sha256", "train_labels_sha256", "history", "author_log_records")
    mismatches = [key for key in keys if warmup[key] != expected[key]]
    if (mismatches or train_idx.tolist() != reference["train_idx"]
            or test_idx.tolist() != reference["test_idx"]
            or not torch.equal(pair.cpu(), reference["relation_pairs"])
            or not torch.equal(same.cpu(), reference["relation_targets"])):
        raise RuntimeError(f"H1 fold {fold} warmup mismatch: {mismatches}")
    return {"status": "PASS", "reference": "H1", "history_exact": True, "relation_pairs_exact": True,
            "shared_state_exact": True, "reference_checkpoint_sha256": reference["checkpoint_sha256"]}
''')
replace_definition('def compare_warmup_spectral_state(', '    maximum_difference = 0.0', '''
def compare_warmup_spectral_state(group, fold, config, spectral_stats):
    reference = "H1"
    source_best = warmup_references()[fold]
    source_stats = source_best["spectral_stats"]
''')
swap('"reference_checkpoint_sha256": sha_file(best_path),\n    }\n    return canonical_stats, report',
     '"reference_checkpoint_sha256": source_best["checkpoint_sha256"],\n    }\n    return canonical_stats, report')
replace_definition('def prepare_warmup(', 'def train_fold(', '''
def freeze_adaptive_gate(model, optimizers, config):
    gate = model.darg.core.adaptive_gate_logits
    if config.get("freeze_adaptive_gate", False):
        if gate in optimizers["darg"].state or gate.grad is not None:
            raise RuntimeError("Frozen adaptive Gate has an optimizer moment or gradient")
        gate.requires_grad_(False)
    return gate


def prepare_warmup(group, fold, config, x, labels, train_t, test_idx, y_np):
    model = build_model(config, fold, x.device)
    optimizers, optimizer_config = make_optimizers(model, config)
    train_idx = train_t.cpu().numpy()
    supports = historical.anchor_pair_support(model, x)
    pair_np, same_np = historical.make_pairs(train_idx, y_np, fold, supports)
    warm_pair = torch.from_numpy(pair_np).long().to(x.device)
    warm_same = torch.from_numpy(same_np).float().to(x.device)
    initial = initial_warmup_identity(model, optimizers)
    initial["rng_sha256_by_family"] = {key: content_sha(state)
                                     for key, state in historical.capture_rng_state().items()}
    history, rng_after, updates, logs, step_witnesses = run_warmup(
        model, optimizers, x, labels, train_t, warm_pair, warm_same, config,
        torch.as_tensor(test_idx, device=x.device))
    warmup = make_warmup_payload(fold, model, optimizers, history, rng_after, warm_pair,
                                 warm_same, initial, config, train_idx, y_np, updates, logs)
    warmup["steps"] = step_witnesses
    pairing = compare_warmup(fold, warmup, warm_pair, warm_same, train_idx, test_idx)
    model.eval()
    with torch.no_grad():
        _, warm_aux = model(x, labels, train_t, warm_pair, warm_same, dpsl_enabled=False,
                            darg_reference_controls=True)
    stats = fit_spectral_statistics(warm_aux["cam"], labels, train_t, fold=fold,
                                    warm_logits=warm_aux["native_logits"])
    stats["warmup_cam"] = warm_aux["cam"].detach().cpu()
    stats["warmup_logits"] = warm_aux["native_logits"].detach().cpu()
    stats, spectral_audit = compare_warmup_spectral_state(group, fold, config, stats)
    pairing.update(spectral_audit)
    model.attach_dpsl(stats, str(prompt_path_for(config, BASE_GROUPS[config["base_reference"]])),
                      824602 * 100 + fold)
    optimizers["dpsl"] = torch.optim.AdamW(model.dpsl.parameters(), lr=0.0005, weight_decay=0.01)
    optimizer_config["dpsl"] = [0.0005, 0.01]
    # Restore the same Gate convention for this identity check; scales are constants
    # in the graph source and do not enter the parameter or optimizer state.
    h1_probabilities = group_config("H1")["target_gate_probabilities"]
    reparameterize_adaptive_gate(model, optimizers, h1_probabilities, "after_warmup")
    baseline_state = initial_warmup_identity(model, optimizers)
    if baseline_state != warmup_references()[fold]["warmup_pair_audit"]["pre_epoch_11_state"]:
        raise RuntimeError(f"H1 fold {fold} pre-11 model/optimizer/RNG differs")
    gate_audit = reparameterize_adaptive_gate(
        model, optimizers, config["target_gate_probabilities"], "after_warmup")
    expected_gate = np.asarray(config["darg_gate_scales"]) * config["target_gate_probabilities"]
    np.testing.assert_allclose(gate_audit["effective_gate_after"], expected_gate, atol=2e-9, rtol=0)
    gate = freeze_adaptive_gate(model, optimizers, config)
    gate_audit["frozen"] = not gate.requires_grad
    warmup["gate_initialization"] = gate_audit
    warmup["gate_probabilities_after_warmup"] = gate_probabilities(model)
    warmup["effective_gate_after_warmup"] = effective_gate_values(model)
    pair, same, forced, relation_audit = formal_relation_sample(
        warm_aux["adjacencies"][config["edge_anchor_index"]], train_t, labels[train_t],
        warm_pair, warm_same, config.get("anchor_support_quota_per_target", 0), fold)
    if historical.sha_object(historical.capture_rng_state()) != warmup["rng_state_sha256"]:
        raise RuntimeError(f"RNG changed between warmup and epoch 11: {group} fold {fold}")
    pairing["pre_epoch_11_state"] = initial_warmup_identity(model, optimizers)
    pairing["pre_epoch_11_non_gate_sha256"] = content_sha(
        _state_without_adaptive_gate(model.state_dict()))
    pairing["pre_epoch_11_optimizer_non_gate_sha256"] = content_sha(
        _optimizer_state_without_adaptive_gate(model, optimizers))
    pairing["pre_epoch_11_s02_reference_reproduced"] = True
    witness_path = ROOT / "audits/warmup_pairing.json"
    if witness_path.is_file():
        reference_steps = json.loads(witness_path.read_text())["reference_steps_by_fold"][str(fold)]
        if step_witnesses != reference_steps:
            raise RuntimeError(f"Preflight step-by-step warmup parity differs for {group} fold {fold}")
    relation_state = {
        "warmup_relation_pairs": warm_pair, "warmup_relation_targets": warm_same,
        "formal_relation_pairs": pair, "formal_relation_targets": same,
        "formal_forced_support_mask": forced, "relation_sampling_audit": relation_audit,
    }
    return model, optimizers, optimizer_config, pair, same, warmup, pairing, stats, relation_state


def primitive(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [primitive(item) for item in value]
    return value


def terminal_witness(model, optimizers, config, x, labels, train, pair, same, epoch):
    """Observe the saved state and one uncommitted successor on the same device."""
    rng = historical.capture_rng_state()
    model.eval()
    with torch.no_grad():
        latest, _ = model(x, labels, train, pair, same)
    model.train()
    historical.restore_rng(rng)
    zero_optimizers(optimizers)
    _, aux = model(x, labels, train, pair, same)
    _, base, _, _ = historical_loss(aux, labels, train, model, "paper_aloss", 0.01)
    total = base + aux["dpsl_loss"]
    dpsl_optimizer_step(model, optimizers, base, aux["scmr_objectives"], 0.001)
    gate = model.darg.core.adaptive_gate_logits
    if config.get("freeze_adaptive_gate", False):
        if gate.grad is not None or gate in optimizers["darg"].state:
            raise RuntimeError("D1 resumed with an active adaptive Gate")
    model.eval()
    with torch.no_grad():
        full, values = model(x, labels, train, pair, same)
    model.dpsl.advance_ema(values["cam"])
    return {
        "epoch": epoch, "next_epoch": epoch + 1,
        "latest_log_probability_sha256": content_sha(latest),
        "next_eval_log_probability_sha256": content_sha(full),
        "next_loss": float(total.detach()),
        "next_model_sha256": content_sha(model.state_dict()),
        "next_optimizer_sha256": content_sha({n: o.state_dict() for n, o in optimizers.items()}),
        "next_rng_sha256": content_sha(historical.capture_rng_state()),
    }
''')
swap('        pair, same = saved["relation_pairs"], saved["relation_targets"]\n        if pair is not None:\n            pair, same = pair.to(device), same.to(device)', '''
        relation_state = {name: saved[name] for name in (
            "warmup_relation_pairs", "warmup_relation_targets", "formal_relation_pairs",
            "formal_relation_targets", "formal_forced_support_mask", "relation_sampling_audit")}
        pair = relation_state["formal_relation_pairs"].to(device)
        same = relation_state["formal_relation_targets"].to(device)
        if not torch.equal(saved["relation_pairs"], pair.cpu()) or not torch.equal(
                saved["relation_targets"], same.cpu()):
            raise RuntimeError("Latest relation-pair aliases differ")
        freeze_adaptive_gate(model, optimizers, config)''')
swap('model, optimizers, optimizer_config, pair, same, warmup, warmup_pair_audit, stats = prepare_warmup(',
     'model, optimizers, optimizer_config, pair, same, warmup, warmup_pair_audit, stats, relation_state = prepare_warmup(')
swap('            "relation_pairs": pair, "relation_targets": same,\n            "author_epoch_log_records":',
     '            "relation_pairs": pair, "relation_targets": same, **relation_state,\n            "author_epoch_log_records":')
swap('                "relation_pairs": pair, "relation_targets": same,\n                "dpsl_ema_updates":',
     '                "relation_pairs": pair, "relation_targets": same, **relation_state,\n                "dpsl_ema_updates":')
swap('                "parameter_update_l2": float((gate_after_step - gate_before_step).norm()),',
     '''                "parameter_update_l2": float((gate_after_step - gate_before_step).norm()),
                "theta": gate_after_step.cpu().tolist(),
                "gradient_by_modality": gate_gradient.detach().cpu().tolist() if gate_gradient is not None else None,
                "requires_grad": bool(gate_parameter.requires_grad),''')
swap('        current_graph_gradient_audit = graph_gradient_audit(model, aux)', '''
        if config["freeze_adaptive_gate"] and (gate_parameter.grad is not None
                or gate_parameter in optimizers["darg"].state
                or not torch.equal(gate_before_step, gate_parameter.detach())):
            raise RuntimeError(f"D1 Gate received an update in fold {fold} epoch {epoch}")
        current_graph_gradient_audit = graph_gradient_audit(model, aux)''')
swap('    best = torch.load(best_path, map_location="cpu", weights_only=False)\n    differences =',
     '''    elapsed = elapsed_before + time.monotonic() - started
    terminal = terminal_witness(model, optimizers, config, x, labels, train_t, pair, same,
                                history[-1]["epoch"])
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    differences =''')
swap('    write_json(audit_path, audit)',
     '''    audit["terminal_witness"] = terminal
    audit["relation_sampling"] = relation_state["relation_sampling_audit"]
    audit["train_seconds"] = elapsed
    write_json(audit_path, primitive(audit))''')
swap('    return audit\n', '    return {"group": group, "fold": fold, "status": audit["status"]}\n')
swap('Formal training accepts the registered new groups N1–N4', 'Formal training accepts A1 A2 P1 P2 P3 R1 R2 D1')
swap('    if not 1 <= args.workers <= 8:\n        parser.error("The global GPU worker limit is eight")',
     '    if args.workers != 8:\n        parser.error("The locked worker count is eight")')
swap('Experiment 83.8.5 is registered for exactly one physical GPU',
     'Experiment 83.8.5.3 is registered for exactly one physical GPU')
path.write_text(source)
