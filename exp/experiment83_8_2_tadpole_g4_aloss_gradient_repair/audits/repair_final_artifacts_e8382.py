"""Repair aggregate audit artifacts from frozen 83.8.2 checkpoints."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import traceback
from collections import Counter

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "audits"
RESULTS = ROOT / "results"
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "common"))
import audit_finalize_e8382 as audit
import run_e8382 as runner

GROUPS = [f"N{index}" for index in range(7)]
UPDATE_GROUPS = ("N1", "N2", "N3", "N4", "N5")
VECTOR_FIELDS = (
    "u_base_norm", "u_supervised_norm", "u_difference_norm",
    "u_difference_relative", "u_cosine", "graph_p_weight_u_base_norm",
    "graph_p_weight_u_supervised_norm", "graph_p_weight_u_difference_norm",
    "graph_p_weight_u_difference_relative", "graph_p_weight_u_cosine",
)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correction_counts(before, after, truth):
    before_correct = before == truth
    after_correct = after == truth
    return {
        "changed": int((before != after).sum()),
        "corrected": int((~before_correct & after_correct).sum()),
        "damaged": int((before_correct & ~after_correct).sum()),
    }


def intervention_record(full_logp, intervention_logp, test_idx, truth):
    full_prediction = full_logp.argmax(1).detach().cpu().numpy()[test_idx]
    intervention_prediction = intervention_logp.argmax(1).detach().cpu().numpy()[test_idx]
    return {
        **correction_counts(full_prediction, intervention_prediction, truth),
        "max_abs_logit_delta": float(
            (full_logp[test_idx] - intervention_logp[test_idx]).abs().max().detach()
        ),
    }


def modality_signed_counts(edge_scores):
    counts = []
    for modality, values in enumerate(edge_scores):
        values = values.detach()
        if modality == 3:
            positive = int((values > 0.5).sum().item())
            negative = int((values < 0.5).sum().item())
        else:
            positive = int((values >= 0.55).sum().item())
            negative = int((values <= 0.45).sum().item())
        counts.append({"positive": positive, "negative": negative,
                       "scored_edges": int(values.numel())})
    return counts


def mechanism_task(payload):
    group, fold, x_np, y_np, split = payload
    runner.configure_worker()
    device = torch.device("cuda:0")
    best_path = ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    training_audit = json.loads(
        (ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json").read_text())
    shared_ratio = training_audit.get("dpsl_shared_gradient_ratio_max")
    if shared_ratio is not None and float(shared_ratio) > 0.001 + 1e-9:
        raise RuntimeError(f"DPSL shared gradient cap exceeded in {group} fold {fold}")
    config, x, labels, train, pair, same, model = audit.load_checkpoint_model(
        group, checkpoint, x_np, y_np, device)
    test_idx = np.asarray(checkpoint["test_idx"], dtype=np.int64)
    test = torch.as_tensor(test_idx, dtype=torch.long, device=device)
    truth = y_np[test_idx]
    inference_labels = torch.zeros_like(labels)
    model.eval()

    with torch.no_grad():
        full_logp, full_aux = model(
            x, inference_labels, train, pair, same, dpsl_enabled=True)
        direct_off_logp = model(
            x, inference_labels, train, pair, same, dpsl_enabled=True,
            darg_intervention="direct_off")[0]
        disease_neutral_logp = model(
            x, inference_labels, train, pair, same, dpsl_enabled=True,
            darg_intervention="disease_neutral")[0]
        native_logp = None
        residual_off_gap = None
        class_pairs = []
        residual_max = 0.0
        if model.dpsl is not None:
            native_logp = model(
                x, inference_labels, train, pair, same, dpsl_enabled=False)[0]
            residual_off_logp = model(
                x, inference_labels, train, pair, same, dpsl_enabled=True,
                dpsl_intervention="residual_off")[0]
            residual_off_gap = float((residual_off_logp - native_logp).abs().max())
            class_pair_tensor = full_aux["dpsl"].get("class_pairs")
            class_pairs = (class_pair_tensor.detach().cpu().tolist()
                           if class_pair_tensor is not None else [])
            residual_max = float(full_aux["dpsl"]["ratio"].max())
            if residual_off_gap != 0.0:
                raise RuntimeError(f"Residual-Off differs from Native in {group} fold {fold}")
            if residual_max > 0.08 + 1e-7:
                raise RuntimeError(f"DPSL residual cap exceeded in {group} fold {fold}")

    signed_by_modality = modality_signed_counts(full_aux["edge_q_used"])
    positive_edges = sum(item["positive"] for item in signed_by_modality)
    negative_edges = sum(item["negative"] for item in signed_by_modality)
    full_prediction = full_logp.argmax(1).detach().cpu().numpy()[test_idx]
    native_prediction = (native_logp.argmax(1).detach().cpu().numpy()[test_idx]
                         if native_logp is not None else None)
    darg_state = checkpoint.get("optimizers", {}).get("darg", {}).get("state", {})
    relation_optimizer_steps = int(sum(
        float(value.get("step", 0)) > 0 for value in darg_state.values()))
    modality_names = ("PHENO", "ANAT", "FUNC", "CORR")
    row = {
        "group": group, "fold": fold, "best_epoch": int(checkpoint["epoch"]),
        "best_sha256": sha_file(best_path),
        "darg_positive_dcr_callable": bool(callable(model.darg.positive.rudag_forward)),
        "darg_relation_optimizer_state_steps": relation_optimizer_steps,
        "positive_signed_edges": positive_edges,
        "negative_signed_edges": negative_edges,
        "signed_relation_both_present": bool(positive_edges and negative_edges),
        "per_modality_signed_edges": json.dumps(signed_by_modality, separators=(",", ":")),
        "dpsl_active": model.dpsl is not None,
        "dpsl_residual_max": residual_max,
        "dpsl_residual_off_native_max_abs": residual_off_gap,
        "dpsl_shared_gradient_ratio_max": shared_ratio,
        "dpsl_class_pair_count": len(class_pairs),
        "dpsl_class_pairs": json.dumps(class_pairs, separators=(",", ":")),
        "full_correct": int((full_prediction == truth).sum()),
        "native_correct": (int((native_prediction == truth).sum())
                           if native_prediction is not None else None),
        "native_to_full_corrected": (
            int(((native_prediction != truth) & (full_prediction == truth)).sum())
            if native_prediction is not None else None),
        "native_to_full_damaged": (
            int(((native_prediction == truth) & (full_prediction != truth)).sum())
            if native_prediction is not None else None),
        "NC_correct_full": int(((truth == 0) & (full_prediction == 0)).sum()),
        "MCI_correct_full": int(((truth == 1) & (full_prediction == 1)).sum()),
        "AD_correct_full": int(((truth == 2) & (full_prediction == 2)).sum()),
    }
    for index, modality in enumerate(modality_names):
        row[f"{modality}_positive_edges"] = signed_by_modality[index]["positive"]
        row[f"{modality}_negative_edges"] = signed_by_modality[index]["negative"]
        row[f"{modality}_scored_edges"] = signed_by_modality[index]["scored_edges"]
    row.update({f"direct_off_{key}": value for key, value in intervention_record(
        full_logp, direct_off_logp, test_idx, truth).items()})
    row.update({f"disease_neutral_{key}": value for key, value in intervention_record(
        full_logp, disease_neutral_logp, test_idx, truth).items()})
    return row


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def numeric_summary(rows):
    result = {}
    for group in GROUPS:
        selected = [row for row in rows if row["group"] == group]
        result[group] = {
            "folds": len(selected),
            "positive_signed_edges_mean": float(np.mean([row["positive_signed_edges"] for row in selected])),
            "negative_signed_edges_mean": float(np.mean([row["negative_signed_edges"] for row in selected])),
            "positive_signed_edges_min": min(row["positive_signed_edges"] for row in selected),
            "negative_signed_edges_min": min(row["negative_signed_edges"] for row in selected),
            "positive_and_negative_paths_all_folds": all(
                row["signed_relation_both_present"] for row in selected),
        }
    return result


def build_warmup_pairing():
    g4_groups = ("N0", "N1", "N2", "N3", "N5")
    g4_prompt_groups = ("N0", "N1", "N2", "N3", "N4", "N5")
    per_fold = []
    group_checks = {group: [] for group in GROUPS}
    for fold in range(1, 11):
        payloads = {}
        audits = {}
        spectral_hashes = {}
        for group in GROUPS:
            best_path = ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"
            checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            audit_path = ROOT / "groups" / group / "audits" / f"fold{fold:02d}.json"
            record = json.loads(audit_path.read_text())
            payloads[group] = checkpoint["warmup"]
            audits[group] = record
            spectral_hashes[group] = runner.sha_object(checkpoint.get("spectral_stats"))
        reference = payloads["N0"]
        for group in GROUPS:
            current = payloads[group]
            pair_audit = audits[group].get("warmup_pair_audit", {})
            check = {
                "group": group, "fold": fold,
                "pair_audit_status": pair_audit.get("status"),
                "initial_model_equal_to_N0": current.get("initial_model_sha256") == reference.get("initial_model_sha256"),
                "initial_optimizer_equal_to_N0": current.get("initial_optimizer_sha256") == reference.get("initial_optimizer_sha256"),
                "initial_rng_equal_to_N0": current.get("initial_rng_sha256") == reference.get("initial_rng_sha256"),
                "relation_pairs_equal_to_N0": current.get("pair_sha256") == reference.get("pair_sha256"),
                "training_labels_equal_to_N0": current.get("train_labels_sha256") == reference.get("train_labels_sha256"),
                "post_warmup_model_equal_to_N0": current.get("base_state_sha256") == reference.get("base_state_sha256"),
                "post_warmup_optimizer_equal_to_N0": current.get("optimizer_state_sha256") == reference.get("optimizer_state_sha256"),
                "post_warmup_rng_equal_to_N0": current.get("rng_state_sha256") == reference.get("rng_state_sha256"),
                "warmup_history_equal_to_N0": [
                    {key: item[key] for key in ("epoch", "loss")}
                    for item in current.get("history", [])
                ] == [
                    {key: item[key] for key in ("epoch", "loss")}
                    for item in reference.get("history", [])
                ],
                "dpsl_spectral_stats_equal_to_N0": (
                    spectral_hashes[group] == spectral_hashes["N0"]
                    if group in g4_groups else None),
                "dpsl_prompt_equal_to_N0": (
                    audits[group].get("prompt_sha256") == audits["N0"].get("prompt_sha256")
                    if group in g4_prompt_groups else None),
                "initial_model_sha256": current.get("initial_model_sha256"),
                "initial_optimizer_sha256": current.get("initial_optimizer_sha256"),
                "initial_rng_sha256": current.get("initial_rng_sha256"),
                "post_warmup_model_sha256": current.get("base_state_sha256"),
                "post_warmup_optimizer_sha256": current.get("optimizer_state_sha256"),
                "post_warmup_rng_sha256": current.get("rng_state_sha256"),
                "dpsl_spectral_stats_sha256": spectral_hashes[group],
                "dpsl_prompt_sha256": audits[group].get("prompt_sha256"),
            }
            per_fold.append(check)
            group_checks[group].append(check)

    summaries = {}
    for group, checks in group_checks.items():
        if group == "N0":
            status = "BASELINE"
        elif group == "N4":
            expected = all(
                item["pair_audit_status"] == "ALOSS_WARMUP_ONLY_DIFFERENCE"
                and item["initial_model_equal_to_N0"]
                and item["initial_optimizer_equal_to_N0"]
                and item["initial_rng_equal_to_N0"]
                and item["relation_pairs_equal_to_N0"]
                and item["training_labels_equal_to_N0"]
                and item["post_warmup_rng_equal_to_N0"]
                and item["dpsl_prompt_equal_to_N0"]
                and not item["post_warmup_model_equal_to_N0"]
                and not item["post_warmup_optimizer_equal_to_N0"]
                and not item["warmup_history_equal_to_N0"]
                for item in checks)
            status = "ALOSS_WARMUP_ONLY_DIFFERENCE" if expected else "FAIL"
        else:
            expected_status = "PASS"
            expected = all(
                item["pair_audit_status"] == expected_status
                and item["initial_model_equal_to_N0"]
                and item["initial_optimizer_equal_to_N0"]
                and item["initial_rng_equal_to_N0"]
                and item["relation_pairs_equal_to_N0"]
                and item["training_labels_equal_to_N0"]
                and item["post_warmup_rng_equal_to_N0"]
                and item["warmup_history_equal_to_N0"]
                and item["post_warmup_model_equal_to_N0"]
                and item["post_warmup_optimizer_equal_to_N0"]
                and item["post_warmup_rng_equal_to_N0"]
                and (item["dpsl_spectral_stats_equal_to_N0"] is not False)
                and (item["dpsl_prompt_equal_to_N0"] is not False)
                for item in checks)
            status = "PASS" if expected else "FAIL"
        summaries[group] = {
            "status": status, "folds": len(checks),
            "pair_audit_statuses": dict(Counter(item["pair_audit_status"] for item in checks)),
            "initial_model_match_folds": sum(item["initial_model_equal_to_N0"] for item in checks),
            "initial_optimizer_match_folds": sum(item["initial_optimizer_equal_to_N0"] for item in checks),
            "initial_rng_match_folds": sum(item["initial_rng_equal_to_N0"] for item in checks),
            "post_warmup_model_match_folds": sum(item["post_warmup_model_equal_to_N0"] for item in checks),
            "post_warmup_optimizer_match_folds": sum(item["post_warmup_optimizer_equal_to_N0"] for item in checks),
            "post_warmup_rng_match_folds": sum(item["post_warmup_rng_equal_to_N0"] for item in checks),
            "warmup_history_match_folds": sum(item["warmup_history_equal_to_N0"] for item in checks),
            "spectral_stats_match_folds": sum(item["dpsl_spectral_stats_equal_to_N0"] is True for item in checks),
            "prompt_match_folds": sum(item["dpsl_prompt_equal_to_N0"] is True for item in checks),
        }
    if any(item["status"] == "FAIL" for item in summaries.values()):
        raise RuntimeError(f"Warmup pairing audit failed: {summaries}")
    return {"experiment": "Experiment 83.8.2", "status": "PASS",
            "groups": summaries, "fold_checks": per_fold}


def load_summary():
    summary = {}
    numeric = {"ACC_mean", "ACC_std_population", "AUC_original_mean",
               "AUC_original_std_population", "AUC_probability_mean"}
    for row in read_csv(RESULTS / "group_summary.csv"):
        summary[row["group"]] = {
            key: (float(value) if key in numeric and value else value)
            for key, value in row.items()
        }
    return summary


def load_paired():
    rows = read_csv(RESULTS / "paired_contrasts.csv")
    float_fields = ("ACC_candidate", "ACC_reference", "ACC_delta",
                    "AUC_original_candidate", "AUC_original_reference",
                    "AUC_original_delta")
    int_fields = ("patient_corrected", "patient_damaged", "patient_net")
    for row in rows:
        if row["fold"] != "MEAN":
            row["fold"] = int(row["fold"])
        for field in float_fields:
            row[field] = float(row[field])
        for field in int_fields:
            row[field] = int(row[field])
    return rows


def main():
    runner.configure_worker()
    seal = json.loads((AUDITS / "training_seal.json").read_text())
    if seal["files"] != runner.current_implementation_hashes():
        raise RuntimeError("The sealed training source, inputs, or configs changed")
    x_np, y_np, _, splits = runner.load_data()
    split_by_fold = {int(row["fold"]): row for row in splits}
    tasks = [(group, fold, x_np, y_np, split_by_fold[fold])
             for group in GROUPS for fold in range(1, 11)]
    results = []
    failures = []
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(mechanism_task, task): (task[0], task[1]) for task in tasks}
        for future in as_completed(futures):
            group, fold = futures[future]
            try:
                results.append(future.result())
                print(f"MECHANISM_PASS {group} fold {fold:02d}", flush=True)
            except Exception as exc:
                failures.append({"group": group, "fold": fold, "error": repr(exc),
                                 "traceback": traceback.format_exc()})
                print(f"MECHANISM_FAIL {group} fold {fold:02d}: {exc!r}",
                      file=sys.stderr, flush=True)
    if failures:
        (AUDITS / "mechanism_repair_failures.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"Mechanism audit failed for {len(failures)} folds")
    results.sort(key=lambda row: (row["group"], row["fold"]))
    if len(results) != 70:
        raise RuntimeError(f"Expected 70 mechanism records, received {len(results)}")

    mechanism_path = RESULTS / "darg_dpsl_fold_mechanism.csv"
    write_csv(mechanism_path, results)
    mechanism_summary = numeric_summary(results)
    gradient_path = AUDITS / "gradient_mechanism.json"
    gradient_payload = json.loads(gradient_path.read_text())
    update_vectors = audit.collect_update_vector_audits()
    update_summary = audit.summarize_update_vectors(update_vectors)
    for group in GROUPS:
        gradient_payload["darg_dpsl"]["groups"][group].update(mechanism_summary[group])
    gradient_payload["adam_update_vectors"] = update_vectors
    gradient_payload["adam_update_vector_summary"] = update_summary
    gradient_payload["darg_dpsl"]["signed_relation_count_definition"] = (
        "PHENO/ANAT/FUNC use q>=0.55 and q<=0.45 per _edge_delta; CORR uses relation=2*q-1, "
        "so q>0.5 is positive and q<0.5 is negative")
    gradient_path.write_text(json.dumps(gradient_payload, indent=2, sort_keys=True) + "\n")

    patient_path = RESULTS / "patient_mechanism.csv"
    patient_rows = read_csv(patient_path)
    for patient_row in patient_rows:
        group = patient_row.get("group")
        if group not in GROUPS or patient_row.get("relative_to"):
            continue
        selected = [row for row in results if row["group"] == group]
        patient_row["direct_off_max_abs_logit_delta"] = max(
            row["direct_off_max_abs_logit_delta"] for row in selected)
        patient_row["disease_neutral_max_abs_logit_delta"] = max(
            row["disease_neutral_max_abs_logit_delta"] for row in selected)
        patient_row["dpsl_residual_off_max_abs"] = max(
            [row["dpsl_residual_off_native_max_abs"] or 0.0 for row in selected])
    write_csv(patient_path, patient_rows)

    final_audit_path = AUDITS / "final_audit.json"
    final_audit_payload = json.loads(final_audit_path.read_text())
    historical_summary, _, _ = audit.historical_records()
    fidelity = {
        "status": "PASS", "n0_h4_status": final_audit_payload["n0_h4_status"],
        "method_fidelity_matrix_sha256": sha_file(AUDITS / "method_fidelity_matrix.md"),
        "training_seal_sha256": sha_file(AUDITS / "training_seal.json"),
        "training_sources_unmodified": True,
    }
    warmup_pairing = build_warmup_pairing()
    warmup_pairing_path = AUDITS / "warmup_pairing.json"
    warmup_pairing_path.write_text(json.dumps(warmup_pairing, indent=2, sort_keys=True) + "\n")
    report = audit.report_text(
        load_summary(), load_paired(), fidelity, final_audit_payload,
        gradient_payload["aloss_gradients"], gradient_payload["darg_dpsl"],
        historical_summary, update_summary, warmup_pairing)
    report += (
        "\n## Audit Repair Scope\n\n"
        "The first complete replay reached 70/70 PASS, then the report aggregator stopped on a missing ACC field. "
        "The independent audit scripts were corrected after training. This repair restored all ten Adam update-vector "
        "records per group and counted CORR signed relations using its implemented q=0.5 split. The sealed training "
        "implementation, configs, data, Best, Latest, and predictions were not changed. See `audits/report_repair.json`.\n"
    )
    (RESULTS / "final_report.md").write_text(report, encoding="utf-8")

    final_audit_payload["post_training_audit_repair"] = {
        "status": "PASS", "mechanism_folds": len(results),
        "adam_update_vector_folds": {group: len(rows) for group, rows in update_vectors.items()},
        "warmup_pairing_status": warmup_pairing["status"],
        "training_sources_unmodified": True,
    }
    final_audit_path.write_text(json.dumps(final_audit_payload, indent=2, sort_keys=True) + "\n")
    repair_record = {
        "experiment": "Experiment 83.8.2",
        "status": "PASS",
        "training_seal_sha256": sha_file(AUDITS / "training_seal.json"),
        "training_implementation_sha256": seal["implementation_sha256"],
        "training_sources_unmodified": runner.current_implementation_hashes() == seal["files"],
        "modified_audit_sources": [
            {"path": str(AUDITS / "audit_finalize_e8382.py"),
             "sha256": sha_file(AUDITS / "audit_finalize_e8382.py"),
             "scope": "report aggregation, complete Adam-vector collection, CORR sign-count semantics"},
            {"path": str(Path(__file__).resolve()), "sha256": sha_file(Path(__file__).resolve()),
             "scope": "regenerate independent mechanism and result artifacts from frozen checkpoints"},
        ],
        "artifacts_regenerated": [
            str(mechanism_path), str(gradient_path), str(patient_path),
            str(RESULTS / "final_report.md"), str(final_audit_path), str(warmup_pairing_path),
        ],
        "mechanism_folds": len(results),
        "adam_update_vector_folds": {group: len(rows) for group, rows in update_vectors.items()},
        "warmup_pairing_status": warmup_pairing["status"],
        "warmup_pairing_groups": {group: item["status"] for group, item in warmup_pairing["groups"].items()},
    }
    (AUDITS / "report_repair.json").write_text(
        json.dumps(repair_record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS", "mechanism_folds": len(results),
        "adam_update_vector_folds": {group: len(rows) for group, rows in update_vectors.items()},
        "both_signed_paths_all_folds": {
            group: mechanism_summary[group]["positive_and_negative_paths_all_folds"]
            for group in GROUPS},
    }, indent=2))


if __name__ == "__main__":
    main()
