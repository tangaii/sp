"""Verify patient, summary, mechanism, and final-report consistency."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["H0", "H1", "A1", "A2", "P1", "P2", "P3", "R1", "R2", "D1"]
NEW_GROUPS = GROUPS[2:]


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def read_csv(relative):
    with (ROOT / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report_table_rows(report, heading, next_heading):
    section = report.split(heading, 1)[1].split(next_heading, 1)[0]
    rows = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] in GROUPS:
            rows[cells[0]] = cells
    return rows


def close(left, right, tolerance=5e-9):
    return abs(float(left) - float(right)) <= tolerance


def main():
    report_path = ROOT / "results/final_report.md"
    report = report_path.read_text()
    summaries = {row["group"]: row for row in read_csv("results/group_summary.csv")}
    predictions = read_csv("results/oof_predictions.csv")
    mechanism = read_json("audits/mechanism_summary.json")
    replay = read_json("audits/checkpoint_replay.json")
    logs = read_json("audits/log_format_audit.json")
    seal = read_json("audits/training_seal.json")

    if set(summaries) != set(GROUPS):
        raise RuntimeError("Group summary does not contain exactly the ten registered groups")
    grouped = {group: [] for group in GROUPS}
    for row in predictions:
        grouped[row["group"]].append(row)

    patient_checks = {}
    for group in GROUPS:
        rows = grouped[group]
        ids = [int(row["sample_index"]) for row in rows]
        if len(rows) != 598 or len(set(ids)) != 598:
            raise RuntimeError(f"{group}: OOF coverage is not 598 unique patients")
        corrected = sum(
            int(row["native_prediction"]) != int(row["true_label"])
            and int(row["full_prediction"]) == int(row["true_label"])
            for row in rows
        )
        damaged = sum(
            int(row["native_prediction"]) == int(row["true_label"])
            and int(row["full_prediction"]) != int(row["true_label"])
            for row in rows
        )
        summary = summaries[group]
        if (corrected != int(summary["native_to_full_corrected"])
                or damaged != int(summary["native_to_full_damaged"])
                or corrected - damaged != int(summary["native_to_full_net_correct"])):
            raise RuntimeError(f"{group}: patient predictions disagree with group summary")
        patient_checks[group] = {
            "oof_patients": len(rows), "corrected": corrected,
            "damaged": damaged, "net": corrected - damaged,
        }

    if set(mechanism["groups"]) != set(NEW_GROUPS):
        raise RuntimeError("Mechanism summary must cover all eight new groups")
    for group in NEW_GROUPS:
        item = mechanism["groups"][group]["interventions"]["native_to_full"]
        expected = patient_checks[group]
        if (item["corrected"] != expected["corrected"]
                or item["damaged"] != expected["damaged"]
                or item["net"] != expected["net"]):
            raise RuntimeError(f"{group}: patient predictions disagree with mechanism summary")

    report_rows = report_table_rows(
        report, "## J. ACC and AUC results", "## K. Paired comparisons with H1/S02")
    if set(report_rows) != set(GROUPS):
        raise RuntimeError("Final report results table does not contain exactly ten groups")
    metric_columns = {
        "ACC_mean": (1, 0),
        "AUC_original_mean": (2, 0),
        "AUC_probability_mean": (3, 0),
        "full_OOF_correct": (4, None),
        "full_OOF_accuracy": (6, None),
        "full_AUC_original": (7, None),
        "full_AUC_probability": (8, None),
    }
    for group, cells in report_rows.items():
        summary = summaries[group]
        for key, (index, _) in metric_columns.items():
            cell = cells[index]
            if key.endswith("_mean"):
                cell = float(cell.split("+/-", 1)[0].strip()) / 100
            elif key == "full_OOF_correct":
                cell = cell.split("/", 1)[0].strip()
            elif key in ("full_OOF_accuracy", "full_AUC_original", "full_AUC_probability"):
                cell = float(cell) / 100
            if not close(cell, summary[key], tolerance=5.1e-7):
                raise RuntimeError(f"{group}: final report {key} differs from group summary")
        expected_class_correct = " / ".join(
            str(int(summaries[group][f"full_correct_{label}"]))
            for label in ("NC", "MCI", "AD"))
        if cells[5] != expected_class_correct:
            raise RuntimeError(f"{group}: final report class counts differ from group summary")
        if cells[9] != f"{patient_checks[group]['corrected']}/{patient_checks[group]['damaged']}":
            raise RuntimeError(f"{group}: final report Native-to-Full direction differs")

    if replay["best_latest_replay_status"] != "PASS" or replay["best_latest_folds_passed"] != 100:
        raise RuntimeError("Best/Latest checkpoint replay is incomplete")
    if replay["latest_next_step_recovery_status"] != "PASS" or replay["latest_next_step_passed"] != 80:
        raise RuntimeError("Deterministic Latest recovery replay is incomplete")
    if logs["status"] != "PASS" or len(logs["groups"]) != 10:
        raise RuntimeError("Author-format log audit is incomplete")

    current_sealed = {}
    for relative in seal["files"]:
        current_sealed[relative] = sha256(ROOT / relative)
    if current_sealed != seal["files"]:
        raise RuntimeError("A sealed training source, configuration, or input changed")

    result = {
        "experiment": "Experiment 83.8.5.3",
        "status": "PASS",
        "registered_groups": GROUPS,
        "oof_coverage": patient_checks,
        "mechanism_direction_checks": len(NEW_GROUPS),
        "report_metric_rows_checked": len(report_rows),
        "best_latest_replay": f"PASS {replay['best_latest_folds_passed']}/100",
        "latest_deterministic_recovery": f"PASS {replay['latest_next_step_passed']}/80",
        "author_format_logs": "PASS 10/10",
        "training_seal": "PASS",
        "training_implementation_sha256": seal["implementation_sha256"],
        "source_sha256": {
            "group_summary.csv": sha256(ROOT / "results/group_summary.csv"),
            "oof_predictions.csv": sha256(ROOT / "results/oof_predictions.csv"),
            "mechanism_summary.json": sha256(ROOT / "audits/mechanism_summary.json"),
            "final_report.md": sha256(report_path),
        },
    }
    destination = ROOT / "audits/final_consistency_audit.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "groups": len(GROUPS),
        "report_metric_rows_checked": result["report_metric_rows_checked"],
        "mechanism_direction_checks": result["mechanism_direction_checks"],
        "latest_recovery": result["latest_deterministic_recovery"],
    }, indent=2))


if __name__ == "__main__":
    main()
