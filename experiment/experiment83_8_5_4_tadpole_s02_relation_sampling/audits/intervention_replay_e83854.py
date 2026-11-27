"""Replay graph and DPSL inference interventions from the Best checkpoints."""
from __future__ import annotations

import json
import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83854 as runner

SOURCE = {
    "H0": (Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl"), "R0"),
    "H1": (Path("/model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial"), "S02"),
}


def checkpoint_path(group, fold):
    if group in SOURCE:
        root, source_group = SOURCE[group]
        return root / "groups" / source_group / "best_checkpoints" / f"fold{fold:02d}.pt"
    return ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"


def count_changes(reference, alternative, truth):
    ref_correct, alt_correct = reference == truth, alternative == truth
    return {
        "changed": int((reference != alternative).sum()),
        "corrected": int((~ref_correct & alt_correct).sum()),
        "damaged": int((ref_correct & ~alt_correct).sum()),
    }


def json_default(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    runner.configure_worker()
    device = torch.device("cuda:0")
    x, y, _, raw_splits = runner.load_data()
    splits = {int(item["fold"]): item for item in raw_splits}
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    rows = []
    for group in runner.FORMAL_GROUPS:
        for fold in range(1, 11):
            payload = torch.load(checkpoint_path(group, fold), map_location="cpu", weights_only=False)
            split = splits[fold]
            train = torch.as_tensor(split["train_idx"], dtype=torch.long, device=device)
            test = np.asarray(split["test_idx"], dtype=np.int64)
            pair = payload["relation_pairs"].to(device)
            same = payload["relation_targets"].to(device)
            config = runner.group_config(group)
            model = runner.build_model(config, fold, device, spectral_stats=payload["spectral_stats"])
            model.load_state_dict(payload["model"], strict=True)
            model.darg.enter_positive_dcr()
            model.eval()
            parts, start = [], 0
            for width in runner.MODAL_DIMS:
                parts.append(x_t[:, start:start + width])
                start += width
            with torch.no_grad():
                _, _, graph_info = model.darg.positive.rudag_forward(
                    model.darg.core, parts, pair, "E74.4-G2-Positive-DCR",
                    intervention="full", fold=fold, collect_diagnostics=True)
                graph_summary = {
                    "edge_diagnostics": graph_info["edge_diagnostics"],
                    "direct_scale": graph_info.get("direct_scale"),
                    "positive_only_direct": graph_info.get("positive_only_direct"),
                    "intervention": graph_info.get("intervention"),
                    "fused_delta_relative_l2": graph_info.get("fused_delta_relative_l2"),
                    "gates": [float(value.detach().cpu()) for value in graph_info.get("gates", [])],
                }
                full, _ = model(x_t, y_t, train, pair, same, dpsl_enabled=True)
                native, _ = model(x_t, y_t, train, pair, same, dpsl_enabled=False)
                residual_off, _ = model(x_t, y_t, train, pair, same,
                                        dpsl_enabled=True, dpsl_intervention="residual_off")
                direct_off, _ = model(x_t, y_t, train, pair, same, dpsl_enabled=True,
                                      darg_intervention="direct_off")
                disease_neutral, _ = model(x_t, y_t, train, pair, same, dpsl_enabled=True,
                                           darg_intervention="disease_neutral")
                cognitive_off, _ = model(x_t, y_t, train, pair, same, dpsl_enabled=True,
                                         darg_intervention="cognitive_graph_off")
            if not torch.equal(residual_off, native):
                raise RuntimeError(f"{group} fold {fold}: Residual-Off is not exactly Native")
            full_test = full[test].cpu().numpy()
            truth = y[test]
            full_pred = full.argmax(1)[test].cpu().numpy()
            variants = {
                "native": native,
                "direct_off": direct_off,
                "disease_neutral": disease_neutral,
                "cognitive_graph_off": cognitive_off,
            }
            row = {
                "group": group, "fold": fold,
                "residual_off_exact_native": True,
                "residual_off_native_max_abs": float((residual_off[test] - native[test]).abs().max()),
                "dpsl_full_native_logit_max_abs": float((full[test] - native[test]).abs().max()),
                "dpsl_full_native_logit_mean_abs": float((full[test] - native[test]).abs().mean()),
                "dpsl_native_to_full": count_changes(
                    native.argmax(1)[test].cpu().numpy(), full_pred, truth),
                "gate_probabilities": model.darg.core.adaptive_gate_logits.sigmoid().detach().cpu().tolist(),
                "gate_scales": [float(value) for value in model.darg.gate_scales],
                "effective_gates": [float(value) for value in (
                    model.darg.core.adaptive_gate_logits.sigmoid().detach().cpu()
                    * torch.as_tensor(model.darg.gate_scales))],
                "graph_mechanism": json.dumps(graph_summary, default=json_default),
            }
            for name, logp in variants.items():
                alternative = logp.argmax(1)[test].cpu().numpy()
                row[f"full_vs_{name}_logit_max_abs"] = float(np.abs(
                    full_test - logp[test].cpu().numpy()).max(initial=0))
                row[f"full_vs_{name}_logit_mean_abs"] = float(np.abs(
                    full_test - logp[test].cpu().numpy()).mean())
                row[f"full_vs_{name}_prediction_changes"] = count_changes(full_pred, alternative, truth)
            rows.append(row)
            print(f"AUDITED {group} fold {fold:02d}", flush=True)
            del model, payload, pair, same, graph_info, graph_summary
            del full, native, residual_off, direct_off, disease_neutral, cognitive_off
            del parts, variants, row
            gc.collect()
    output = ROOT / "results/intervention_diagnostics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    report = {
        "experiment": runner.EXPERIMENT, "status": "PASS", "folds": len(rows),
        "residual_off_exact_native_all_folds": all(row["residual_off_exact_native"] for row in rows),
        "DARG_interventions": ["direct_off", "disease_neutral", "cognitive_graph_off"],
        "DPSL_intervention": "Residual-Off exactly reproduces Native, then Full-vs-Native deltas are measured",
        "output": str(output), "sha256": runner.sha_file(output),
    }
    (ROOT / "audits/intervention_replay.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
