"""Calculate paired formal results from independently reproduced fold outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e8385 as run
from metrics import acc_auc
from preflight_e8385 import require

LABELS = ("NC", "MCI", "AD")
MODES = ("full", "native")
METRICS = ("ACC", "AUC_original", "AUC_probability")
CONTRASTS = (("R1", "R0"), ("R2", "R0"), ("R3", "R0"), ("R3", "R1"),
             ("R3", "R2"), ("R4", "R3"), ("R4", "R0"))
BOOTSTRAP_SEED = 8385
BOOTSTRAP_DRAWS = 10000


def document(name):
    return json.loads((ROOT / name).read_text())


def summarize_metrics(rows):
    return {f"{name}_{stat}": float(getattr(values, stat)())
            for name, values in rows.items() for stat in ("mean", "std")}


def metric_values(truth, prediction, probabilities):
    result = acc_auc(truth, prediction, 3)
    return {"ACC": result["ACC"], "AUC_original": result["AUC"],
            "AUC_probability": float(roc_auc_score(np.eye(3)[truth], probabilities,
                                                   average="macro", multi_class="ovr"))}


def scores(patients, mode):
    truth = patients["true_label"].to_numpy(dtype=np.int64)
    pred = patients[f"{mode}_pred"].to_numpy(dtype=np.int64)
    prob = patients[[f"{mode}_prob_{label}" for label in LABELS]].to_numpy(dtype=np.float32)
    require(np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=2e-6)
            and np.array_equal(pred, prob.argmax(1)), f"Invalid {mode} OOF class probabilities")
    return truth, pred, prob


def folds_and_groups(patients, replay, protocol, locked_split, y, ids):
    folds, groups = [], []
    split_by_fold = {split["fold"]: split for split in locked_split}
    for group in run.FORMAL_GROUPS:
        oof = patients.loc[patients["group"] == group]
        require(np.array_equal(oof["sample_index"], np.arange(598))
                and np.array_equal(oof["sample_id"], ids["RID"])
                and np.array_equal(oof["true_label"], y), f"{group} OOF identity differs")
        fold_rows = []
        for fold, group_patients in oof.groupby("fold"):
            require(group_patients["sample_index"].tolist() == split_by_fold[fold]["test_idx"],
                    f"{group}:{fold} OOF split differs")
            fold_replay = replay["groups"][group][str(fold)]
            fold_audit = document(f"groups/{group}/audits/fold{fold:02d}.json")
            row = {
                "group": group, "fold": fold, "seed": 0,
                "status": "REUSED_VERIFIED_FORMAL" if group == "R0" else "TRAINED_AUDITED_FORMAL",
                "train_count": len(split_by_fold[fold]["train_idx"]), "test_count": len(group_patients),
                "best_epoch": fold_replay["best"]["best_epoch"],
                "last_epoch": fold_replay["latest"]["latest_epoch"],
                "best_sha256": fold_replay["best_sha256"], "latest_sha256": fold_replay["latest_sha256"],
                "prediction_sha256": fold_replay["prediction_csv_sha256"],
                "model_source_sha256": fold_replay["identity"]["implementation_sha256"],
                "config_sha256": fold_replay["identity"]["group_config_sha256"],
                "data_sha256": run.DATA_SHA, "split_sha256": run.SPLIT_SHA,
                "warmup_exact": fold_replay["warmup_exact"],
                "train_seconds": fold_audit["train_seconds"],
            }
            for mode in MODES:
                truth, pred, prob = scores(group_patients, mode)
                prefix = "" if mode == "full" else "Native_"
                metrics = metric_values(truth, pred, prob)
                if mode == "full":
                    verified = fold_replay["best"]["metrics"]
                    require(metrics == {"ACC": verified["ACC"], "AUC_original": verified["AUC"],
                                        "AUC_probability": verified["AUC_probability"]},
                            f"{group}:{fold} the independent Best replay and OOF metric differ")
                row.update({f"{prefix}{key}": val for key, val in metrics.items()})
                for class_id, label in enumerate(LABELS):
                    row[f"{prefix}{label}_correct"] = int(((truth == class_id) & (pred == truth)).sum())
                row[f"{prefix}correct"] = int((truth == pred).sum())
            fold_rows.append(row)
        require(len(fold_rows) == 10, f"{group} must have ten formal folds")
        group_row = {"group": group, "status": fold_rows[0]["status"], "folds": 10, "patients": 598}
        for mode in MODES:
            prefix = "" if mode == "full" else "Native_"
            truth, pred, prob = scores(oof, mode)
            metric_matrix = {f"{prefix}{name}": np.array([item[f"{prefix}{name}"] for item in fold_rows])
                             for name in METRICS}
            group_row.update(summarize_metrics(metric_matrix))
            group_row.update({f"{prefix}{name}_pooled": value
                              for name, value in metric_values(truth, pred, prob).items()})
            group_row[f"{prefix}correct"] = int((truth == pred).sum())
            for class_id, label in enumerate(LABELS):
                group_row[f"{prefix}{label}_correct"] = int(((truth == class_id) & (pred == truth)).sum())
        paper = protocol["paper_reference"]
        group_row["paper_ACC_delta_pp"] = 100 * (group_row["ACC_mean"] - paper["ACC"])
        group_row["paper_AUC_original_delta_pp"] = 100 * (group_row["AUC_original_mean"] - paper["AUC"])
        group_row["PAPER_REFERENCE_MEAN_REACHED"] = group_row["paper_ACC_delta_pp"] >= 0 and (
            group_row["paper_AUC_original_delta_pp"] >= 0)
        folds.extend(fold_rows)
        groups.append(group_row)
    for group_row in groups:
        for metric in METRICS:
            group_row[f"Z0_{metric}_delta_pp"] = 100 * (group_row[f"{metric}_mean"] - groups[0][f"{metric}_mean"])
        group_row["dual_primary_improvement"] = (
            group_row["Z0_ACC_delta_pp"] > 0 and group_row["Z0_AUC_original_delta_pp"] > 0)
    baseline = groups[0]
    require(baseline["correct"] == 560 and round(100 * baseline["ACC_mean"], 4) == 93.6441
            and round(100 * baseline["AUC_original_mean"], 4) == 94.4265
            and round(100 * baseline["AUC_probability_mean"], 4) == 95.8290,
            "Reused metrics disagree with verified Y3/Z0")
    return pd.DataFrame(folds), pd.DataFrame(groups)


def intervals(truth, fold, before, after, fold_deltas):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    difference = after.astype(np.int16) - before
    total = np.zeros(BOOTSTRAP_DRAWS, dtype=np.int16)
    for f in np.unique(fold):
        for label in range(3):
            stratum = difference[(fold == f) & (truth == label)]
            draws = rng.choice(stratum, (BOOTSTRAP_DRAWS, len(stratum)), replace=True)
            total += draws.sum(1).astype(np.int16)
    p_lo, p_hi = 100 * np.quantile(total / len(truth), [.025, .975])
    sampled_folds = rng.integers(0, len(fold_deltas), (BOOTSTRAP_DRAWS, len(fold_deltas)))
    fold_bounds = np.quantile(fold_deltas[sampled_folds].mean(1), [.025, .975], axis=0)
    return p_lo, p_hi, fold_bounds


def compare_groups(oof, folds):
    panels = {group: oof.loc[oof["group"] == group] for group in run.FORMAL_GROUPS}
    rows = []
    for after, before in CONTRASTS:
        pa, pb = panels[after], panels[before]
        require(np.array_equal(pa[["sample_index", "fold", "true_label"]],
                               pb[["sample_index", "fold", "true_label"]]),
                f"Unpaired OOF comparison {after}-{before}")
        truth = pa["true_label"].to_numpy()
        fold_id = pa["fold"].to_numpy()
        for mode in MODES:
            full = mode == "full"
            a, b = pa[f"{mode}_pred"].to_numpy(), pb[f"{mode}_pred"].to_numpy()
            after_ok, before_ok = a == truth, b == truth
            corrected, damaged = after_ok & ~before_ok, before_ok & ~after_ok
            c, d = int(corrected.sum()), int(damaged.sum())
            prefix = "" if full else "Native_"
            names = [f"{prefix}{name}" for name in METRICS]
            a_metric = folds.loc[folds["group"] == after].sort_values("fold")[names].to_numpy()
            b_metric = folds.loc[folds["group"] == before].sort_values("fold")[names].to_numpy()
            fold_deltas = 100 * (a_metric - b_metric)
            ties = np.isclose(fold_deltas, 0.0, rtol=0, atol=1e-12)
            lo, hi, fold_bounds = intervals(truth, fold_id, before_ok, after_ok, fold_deltas)
            row = {"contrast": f"{after}-{before}", "mode": mode, "patient_count": len(truth),
                   "corrected": c, "damaged": d, "net_correct": c - d, "argmax_flips": int((a != b).sum()),
                   "patient_ACC_delta_pp": 100 * (c - d) / len(truth),
                   "patient_ACC_CI95_low_pp": float(lo), "patient_ACC_CI95_high_pp": float(hi),
                   "McNemar_exact_p": float(binomtest(c, c + d, .5).pvalue) if c + d else 1.0}
            for class_id, label in enumerate(LABELS):
                selected = truth == class_id
                row[f"{label}_corrected"] = int((selected & corrected).sum())
                row[f"{label}_damaged"] = int((selected & damaged).sum())
                row[f"{label}_net_correct"] = row[f"{label}_corrected"] - row[f"{label}_damaged"]
            for i, name in enumerate(METRICS):
                row[f"delta_{name}_pp"] = float(fold_deltas[:, i].mean())
                row[f"{name}_fold_W"] = int((~ties[:, i] & (fold_deltas[:, i] > 0)).sum())
                row[f"{name}_fold_T"] = int(ties[:, i].sum())
                row[f"{name}_fold_L"] = int((~ties[:, i] & (fold_deltas[:, i] < 0)).sum())
                row[f"{name}_fold_CI95_low_pp"] = float(fold_bounds[0, i])
                row[f"{name}_fold_CI95_high_pp"] = float(fold_bounds[1, i])
            rows.append(row)
    for mode in MODES:
        adjusted = 0.0
        family = sorted([row for row in rows if row["mode"] == mode], key=lambda row: row["McNemar_exact_p"])
        for i, row in enumerate(family):
            adjusted = max(adjusted, (len(family) - i) * row["McNemar_exact_p"])
            row["McNemar_Holm_p"] = min(1.0, adjusted)
    return pd.DataFrame(rows)


def write_csv(path, frame):
    run.atomic_write(path, lambda stream: stream.write(frame.to_csv(index=False).encode()))


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run.require_training_gate(run.TRAIN_GROUPS)
    _, y, ids, splits = run.load_data()
    protocol = document("protocol.json")
    replay = document("audits/checkpoint_replay.json")
    require(replay["status"] == "PASS" and replay["best_replayed"] == replay["latest_restored"]
            == replay["deterministic_next_steps"] == 50, "Formal aggregation requires all checkpoint replays")
    mechanism = document("audits/mechanism_summary.json")
    oof_path = ROOT / "results/oof_predictions.csv"
    require(mechanism["status"] == "PASS" and mechanism["folds"] == 50
            and mechanism["patient_audit_sha256"] == run.sha_file(oof_path),
            "Formal aggregation requires all observed mechanism runs")
    oof = pd.read_csv(oof_path, float_precision="round_trip").sort_values(["group", "sample_index"])
    require(len(oof) == 2990 and tuple(oof["group"].unique()) == run.FORMAL_GROUPS,
            "Expected 598 independent OOF patients for each of five groups")
    folds, groups = folds_and_groups(oof, replay, protocol, splits, y, ids)
    pairs = compare_groups(oof, folds)
    write_csv(ROOT / "results/fold_metrics.csv", folds)
    write_csv(ROOT / "results/group_summary.csv", groups)
    write_csv(ROOT / "results/paired_contrasts.csv", pairs)
    print(groups[["group", "ACC_mean", "AUC_original_mean", "AUC_probability_mean", "correct",
                  "dual_primary_improvement", "PAPER_REFERENCE_MEAN_REACHED"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
