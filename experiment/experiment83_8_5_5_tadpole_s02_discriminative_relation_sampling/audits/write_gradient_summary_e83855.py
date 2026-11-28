"""Summarize the read-only fold-level DARG gradient diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "audits/gradient_replay.json").read_text())
    if report.get("status") != "PASS" or report.get("folds_passed") != 80:
        raise RuntimeError("The 80-fold gradient replay has not passed")
    frame = pd.DataFrame(report["fold_results"])
    rows = []
    for group, selected in frame.groupby("group", sort=False):
        shared_cos = selected["shared_gradient_cosine"].dropna()
        update_cls = selected["actual_update_cosine_with_classification_gradient"].dropna()
        update_rel = selected["actual_update_cosine_with_relation_gradient"].dropna()
        rows.append({
            "group": group, "folds": len(selected),
            "no_shared_grad_path_folds": int((selected["shared_path_status"] == "NO_SHARED_GRAD_PATH").sum()),
            "shared_parameter_tensor_count_mean": selected["shared_parameter_tensor_count"].mean(),
            "effective_shared_gradient_ratio_mean": selected["effective_shared_gradient_ratio"].mean(),
            "classification_gradient_norm_mean": selected["shared_classification_gradient_norm"].mean(),
            "relation_gradient_norm_mean": selected["shared_relation_gradient_norm"].mean(),
            "relation_to_classification_norm_ratio_mean": selected[
                "shared_relation_to_classification_norm_ratio"].mean(),
            "shared_gradient_cosine_mean": shared_cos.mean(),
            "shared_gradient_cosine_negative_folds": int((shared_cos < 0).sum()),
            "actual_update_cosine_classification_mean": update_cls.mean(),
            "actual_update_cosine_relation_mean": update_rel.mean(),
            "actual_update_cosine_relation_negative_folds": int((update_rel < 0).sum()),
            "actual_optimizer_update_norm_mean": selected["actual_optimizer_update_norm"].mean(),
            "relation_only_parameter_update_norm_mean": selected["relation_only_parameter_update_norm"].mean(),
            "relation_update_native_logit_max_abs_mean": selected[
                "relation_update_native_logit_max_abs"].mean(),
            "relation_update_train_nll_delta_mean": selected["relation_update_train_nll_delta"].mean(),
            "relation_update_train_margin_delta_mean": selected["relation_update_train_margin_delta"].mean(),
            "relation_update_test_margin_delta_mean": selected["relation_update_test_margin_delta"].mean(),
            "relation_update_test_margin_delta_std": selected["relation_update_test_margin_delta"].std(ddof=0),
        })
    output = ROOT / "results/gradient_alignment_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    report["summary_csv"] = str(output)
    report["summary_csv_sha256"] = __import__("hashlib").sha256(output.read_bytes()).hexdigest()
    (ROOT / "audits/gradient_replay.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    mechanism_path = ROOT / "audits/mechanism_summary.json"
    mechanism = json.loads(mechanism_path.read_text()) if mechanism_path.is_file() else {
        "experiment": "Experiment 83.8.5.5", "status": "PASS", "groups": report["groups"], "folds": 80}
    mechanism["gradient_alignment"] = {
        "summary_csv": str(output), "summary_csv_sha256": report["summary_csv_sha256"],
        "fold_diagnostics": "audits/gradient_replay.json",
        "diagnostic_only": True, "optimizer_changed": False, "new_loss_added": False,
    }
    mechanism_path.write_text(json.dumps(mechanism, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "summary_csv": str(output), "groups": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
