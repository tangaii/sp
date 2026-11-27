"""Read-only verification of the archived four-modality E74.1 regression."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABIDE = Path(
    "/model/spromptgl/experiment/experiment_abide_authentic_logs_v2/"
    "inputs/source_repo/experiment/experiment74_1_rudag"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    run_path = ABIDE / "run.py"
    audit_path = ABIDE / "audit.py"
    report_path = ABIDE / "results/report.md"
    manifest_path = ABIDE / "manifest.json"
    mechanism_path = ABIDE / "results/mechanism.csv"
    manifest = json.loads(manifest_path.read_text())
    preflight = manifest["preflight"]
    if manifest.get("status") != "COMPLETE_AUDITED":
        raise RuntimeError("Archived E74.1 manifest is not complete and audited")
    if manifest.get("checkpoint_audit") != "PASS 20/20":
        raise RuntimeError("Archived E74.1 checkpoint audit did not pass 20/20")
    if preflight.get("status") != "PASS":
        raise RuntimeError("Archived E74.1 preflight did not pass")
    if len(preflight.get("all_fold_inheritance", [])) != 10:
        raise RuntimeError("Archived E74.1 lacks ten-fold initialization evidence")
    run_text = run_path.read_text()
    for source_clause in (
        "with torch.no_grad():",
        "model.adaptive_gate_logits.copy_(theta)",
        'state["opt_added"].state.pop(model.adaptive_gate_logits, None)',
        'if rng_before != rng_after or rng_after != warmup["rng_sha"]:',
    ):
        if source_clause not in run_text:
            raise RuntimeError(f"Archived E74.1 source clause is missing: {source_clause}")

    paired_folds = []
    expected_gate = [0.001, 0.001, 0.001, 0.99]
    task_hashes = {}
    for fold in range(1, 11):
        paths = {
            group: ABIDE / ".runtime/tasks" / f"{group}_f{fold:02d}.json"
            for group in ("G1-RUDAG-U", "G2-RUDAG-Full")
        }
        tasks = {group: json.loads(path.read_text()) for group, path in paths.items()}
        for group, task in tasks.items():
            if task.get("status") != "COMPLETE" or int(task.get("fold", -1)) != fold:
                raise RuntimeError(f"Archived E74.1 task incomplete: {group} fold {fold}")
            task_hashes[f"{group}_fold{fold:02d}"] = sha(paths[group])
        left, right = (tasks["G1-RUDAG-U"]["identity"],
                       tasks["G2-RUDAG-Full"]["identity"])
        shared_keys = (
            "model_cam_sha", "optimizer_sha", "warmup_sha256",
            "warmup_rng_sha", "initial_rng_sha", "pair_sha", "effective_train_sha",
        )
        differences = [key for key in shared_keys if left.get(key) != right.get(key)]
        if differences:
            raise RuntimeError(f"Archived E74.1 G1/G2 initial state differs: {differences}")
        evidence = next(
            item for item in preflight["all_fold_inheritance"]
            if int(item["fold"]) == fold)
        if (not evidence.get("group_initial_model_sha_equal")
                or not evidence.get("optimizer_sha_equal")
                or not evidence.get("edge_gate_frozen")
                or not evidence.get("adaptive_gate_reparameterization_recorded")
                or not evidence.get("gate_values")):
            raise RuntimeError(f"Archived E74.1 four-mode preflight evidence failed fold {fold}")
        if not all(abs(float(a) - b) <= 2e-7
                   for a, b in zip(evidence["gate_values"], expected_gate)):
            raise RuntimeError(f"Archived E74.1 effective gate mapping differs fold {fold}")
        paired_folds.append({
            "fold": fold, "shared_identity_keys_equal": shared_keys,
            "model_cam_sha": left["model_cam_sha"],
            "optimizer_sha": left["optimizer_sha"],
            "warmup_sha256": left["warmup_sha256"],
            "warmup_rng_sha256": left["warmup_rng_sha"],
            "relation_pair_sha256": left["pair_sha"],
            "effective_gate": evidence["gate_values"],
        })

    with mechanism_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for group in ("G1-RUDAG-U", "G2-RUDAG-Full"):
        group_rows = [row for row in rows if row.get("group") == group]
        if len(group_rows) != 40:
            raise RuntimeError(f"Archived E74.1 four-mode graph records incomplete for {group}")
        for fold in range(1, 11):
            fold_rows = [row for row in group_rows if int(row["fold"]) == fold]
            if len(fold_rows) != 4:
                raise RuntimeError(f"Archived E74.1 modality record incomplete: {group} fold {fold}")
            by_modality = {row["modality"]: float(row["adaptive_gate_initial"])
                           for row in fold_rows}
            values = [by_modality[name] for name in ("PHENO", "ANAT", "FUNC", "CORR")]
            if not all(abs(value - target) <= 2e-7
                       for value, target in zip(values, expected_gate)):
                raise RuntimeError(
                    f"Archived E74.1 measured initialization differs: {group} fold {fold}")

    result = {
        "experiment": "Experiment 83.8.5.2",
        "status": "PASS",
        "regression_source": str(run_path),
        "regression_source_sha256": sha(run_path),
        "regression_audit_source_sha256": sha(audit_path),
        "archived_manifest_sha256": sha(manifest_path),
        "archived_report_sha256": sha(report_path),
        "archived_mechanism_sha256": sha(mechanism_path),
        "archived_experiment_status": manifest["status"],
        "archived_checkpoint_audit": manifest["checkpoint_audit"],
        "archived_preflight_status": preflight["status"],
        "folds_verified": len(paired_folds),
        "groups_verified": ["G1-RUDAG-U", "G2-RUDAG-Full"],
        "four_modality_effective_gate": expected_gate,
        "shared_initial_state_and_optimizer_rng_pairing": "PASS 10/10",
        "graph_gate_initialization_records": "PASS 20/20 folds",
        "checkpoint_tasks": "PASS 20/20",
        "paired_folds": paired_folds,
        "task_file_hashes": task_hashes,
        "data_used_for_this_check": "read-only archived experiment evidence",
        "verification_script_sha256": sha(Path(__file__)),
    }
    (ROOT / "audits/abide_4modal_regression.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "folds_verified": result["folds_verified"],
        "checkpoint_tasks": result["checkpoint_tasks"],
        "four_modality_effective_gate": result["four_modality_effective_gate"],
        "regression_source_sha256": result["regression_source_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
