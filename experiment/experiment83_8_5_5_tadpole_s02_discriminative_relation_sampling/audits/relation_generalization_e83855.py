"""Held-out DARG relation ranking on patient-disjoint test-test pairs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83855 as runner

GROUPS = runner.FORMAL_GROUPS
CLASS_NAMES = ("NC", "MCI", "AD")
RELATIONS = (("NC-NC", 0, 0), ("MCI-MCI", 1, 1), ("AD-AD", 2, 2),
             ("NC-MCI", 0, 1), ("NC-AD", 0, 2), ("MCI-AD", 1, 2))
SOURCE = {
    "H0": (Path("/model/spromptgl/experiment/experiment83_8_5_tadpole_cognitive_anchored_darg_dpsl"), "R0"),
    "H1": (Path("/model/spromptgl/experiment/experiment83_8_5_2_tadpole_cognitive_strong_anchor_factorial"), "S02"),
}


def checkpoint_path(group: str, fold: int) -> Path:
    if group in SOURCE:
        root, source_group = SOURCE[group]
        return root / "groups" / source_group / "best_checkpoints" / f"fold{fold:02d}.pt"
    return ROOT / "groups" / group / "best_checkpoints" / f"fold{fold:02d}.pt"


def auc_or_none(truth: np.ndarray, score: np.ndarray):
    if len(np.unique(truth)) != 2:
        return None
    return float(roc_auc_score(truth, score))


def main() -> None:
    runner.configure_worker()
    device = torch.device("cuda:0")
    x, y, _, raw_splits = runner.load_data()
    splits = {int(item["fold"]): item for item in raw_splits}
    rows = []
    for group in GROUPS:
        for fold in range(1, 11):
            split = splits[fold]
            test = np.asarray(split["test_idx"], dtype=np.int64)
            left, right = np.triu_indices(len(test), k=1)
            pair_np = np.column_stack((test[left], test[right]))
            label_left, label_right = y[pair_np[:, 0]], y[pair_np[:, 1]]
            relation_id = np.full(len(pair_np), -1, dtype=np.int64)
            for index, (_, class_left, class_right) in enumerate(RELATIONS):
                pair_labels = np.sort(np.column_stack((label_left, label_right)), axis=1)
                mask = np.all(pair_labels == np.asarray((class_left, class_right)), axis=1)
                relation_id[mask] = index
            if np.any(relation_id < 0):
                raise RuntimeError(f"{group} fold {fold}: held-out relation pair is unclassified")

            payload = torch.load(checkpoint_path(group, fold), map_location="cpu", weights_only=False)
            config = runner.group_config(group)
            model = runner.build_model(config, fold, device, spectral_stats=payload["spectral_stats"])
            model.load_state_dict(payload["model"], strict=True)
            model.darg.enter_positive_dcr()
            model.eval()
            x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            parts, start = [], 0
            for width in runner.MODAL_DIMS:
                parts.append(x_tensor[:, start:start + width])
                start += width
            pair_tensor = torch.as_tensor(pair_np, dtype=torch.long, device=device)
            with torch.no_grad():
                _, _, logits = model.darg.core._encode_pairs(parts, pair_tensor)
            probabilities = [torch.sigmoid(value.reshape(-1)).cpu().numpy() for value in logits]

            support_payload = torch.load(checkpoint_path("U2", fold), map_location="cpu", weights_only=False)
            support_record = support_payload.get("cognitive_support")
            if not support_record or support_record.get("source") != "epoch_10_frozen_original_graphlearn_COGNITIVE_TEST":
                raise RuntimeError(f"fold {fold}: shared epoch-10 cognitive graph support is missing")
            support_pairs = np.asarray(support_record["pairs"], dtype=np.int64)
            support_keys = set((support_pairs[:, 0] * len(y) + support_pairs[:, 1]).tolist())
            support_keys.update((support_pairs[:, 1] * len(y) + support_pairs[:, 0]).tolist())
            pair_keys = pair_np[:, 0] * len(y) + pair_np[:, 1]
            supported = np.isin(pair_keys, np.fromiter(support_keys, dtype=np.int64))

            for modality, score in zip(runner.MODAL_NAMES, probabilities):
                same_target = (label_left == label_right).astype(np.int8)
                rows.append({
                    "group": group, "fold": fold, "modality": modality,
                    "test_test_pair_count": int(len(pair_np)),
                    "same_different_auc": auc_or_none(same_target, score),
                    "cognitive_support_pair_count": int(supported.sum()),
                    "non_support_pair_count": int((~supported).sum()),
                    "support_same_different_auc": auc_or_none(same_target[supported], score[supported]),
                    "non_support_same_different_auc": auc_or_none(same_target[~supported], score[~supported]),
                    "support_sha256": support_record["edge_sha256"],
                    "evaluation_patient_set": "outer_test_test_only",
                })
                for relation_index, (relation_name, _, _) in enumerate(RELATIONS):
                    one_vs_rest = (relation_id == relation_index).astype(np.int8)
                    rows.append({
                        "group": group, "fold": fold, "modality": modality,
                        "relation_name": relation_name,
                        "test_test_pair_count": int(len(pair_np)),
                        "relation_ovr_auc": auc_or_none(one_vs_rest, score),
                        "evaluation_patient_set": "outer_test_test_only",
                        "ranking_score": "DARG sigmoid same-vs-different RelationScorer output",
                    })
                    rows.append({
                        "group": group, "fold": fold, "modality": modality,
                        "relation_name": relation_name,
                        "test_test_pair_count": int(len(pair_np)),
                        "support_relation_ovr_auc": auc_or_none(
                            one_vs_rest[supported], score[supported]),
                        "non_support_relation_ovr_auc": auc_or_none(
                            one_vs_rest[~supported], score[~supported]),
                        "support_relation_pair_count": int(supported.sum()),
                        "non_support_relation_pair_count": int((~supported).sum()),
                        "evaluation_patient_set": "outer_test_test_only",
                        "ranking_score": "DARG sigmoid same-vs-different RelationScorer output",
                    })
    result = pd.DataFrame(rows)
    if not set(result["evaluation_patient_set"].dropna()) == {"outer_test_test_only"}:
        raise RuntimeError("Relation evaluation includes a training patient pair")
    output = ROOT / "results/relation_generalization.csv"
    result.to_csv(output, index=False)
    summary = {
        "experiment": runner.EXPERIMENT,
        "status": "PASS",
        "groups": list(GROUPS),
        "folds": 80,
        "modalities": list(runner.MODAL_NAMES),
        "held_out_pair_definition": "both endpoints are in the outer test fold, so neither patient occurs in this fold's DARG relation training pairs",
        "relation_ovr_score_definition": "the existing binary DARG RelationScorer sigmoid output ranks each of six observed diagnosis-pair types one-vs-rest; the model has no six-way relation head",
        "support_definition": "the shared epoch-10 frozen original COGNITIVE_TEST GraphLearn non-diagonal edges, restricted to outer test-test pairs for the support-stratified diagnostic",
        "rows": int(len(result)),
        "csv_sha256": runner.sha_file(output),
    }
    (ROOT / "audits/relation_generalization.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
