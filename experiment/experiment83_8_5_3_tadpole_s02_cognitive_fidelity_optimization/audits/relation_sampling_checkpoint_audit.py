"""Verify saved warmup and formal relation-pair identities after training."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import run_e83853 as runner


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    _, labels, _, splits = runner.load_data()
    split_by_fold = {int(item["fold"]): item for item in splits}
    references = torch.load(ROOT / "inputs/h1_warmup_reference.pt",
                            map_location="cpu", weights_only=False)
    preflight = json.loads((ROOT / "audits/relation_sampling_preflight.json").read_text())
    records = []
    for group in runner.TRAIN_GROUPS:
        for fold in range(1, 11):
            split = split_by_fold[fold]
            train = np.asarray(split["train_idx"], dtype=np.int64)
            test = np.asarray(split["test_idx"], dtype=np.int64)
            expected_warm_pairs = references[fold]["relation_pairs"]
            expected_warm_targets = references[fold]["relation_targets"]
            expected_sampling = preflight["groups"][group][fold - 1]
            paths = [ROOT / "groups" / group / f"{kind}_checkpoints" / f"fold{fold:02d}.pt"
                     for kind in ("best", "latest")]
            checkpoint_sha = {}
            formal_pairs = None
            formal_targets = None
            for kind, path in zip(("best", "latest"), paths):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                warm_pairs = payload["warmup_relation_pairs"]
                warm_targets = payload["warmup_relation_targets"]
                pairs = payload["formal_relation_pairs"]
                targets = payload["formal_relation_targets"]
                forced = payload["formal_forced_support_mask"].bool()
                require(torch.equal(warm_pairs, expected_warm_pairs),
                        f"{group} fold {fold}: {kind} warmup pairs differ from H1")
                require(torch.equal(warm_targets, expected_warm_targets),
                        f"{group} fold {fold}: {kind} warmup targets differ from H1")
                require(torch.equal(pairs, payload["relation_pairs"])
                        and torch.equal(targets, payload["relation_targets"]),
                        f"{group} fold {fold}: {kind} formal pair aliases differ")
                pair_np = pairs.cpu().numpy()
                target_np = targets.cpu().numpy().astype(np.int64)
                require(pair_np.shape == (4096, 2) and int(target_np.sum()) == 2048,
                        f"{group} fold {fold}: {kind} formal cardinality differs")
                require(np.all(pair_np[:, 0] < pair_np[:, 1])
                        and len(np.unique(pair_np[:, 0] * len(labels) + pair_np[:, 1])) == 4096,
                        f"{group} fold {fold}: {kind} has duplicate or reversed pairs")
                require(np.isin(pair_np, train).all() and not np.isin(pair_np, test).any(),
                        f"{group} fold {fold}: {kind} relation supervision crosses the train boundary")
                expected_targets = (labels[pair_np[:, 0]] == labels[pair_np[:, 1]]).astype(np.int64)
                require(np.array_equal(target_np, expected_targets),
                        f"{group} fold {fold}: {kind} targets differ from train labels")
                sampling = payload["relation_sampling_audit"]
                require(sampling == expected_sampling,
                        f"{group} fold {fold}: {kind} relation-sampling certificate differs from preflight")
                require(int(forced.sum()) == int(sampling["forced_count"]),
                        f"{group} fold {fold}: {kind} forced-support mask count differs")
                quota = runner.group_config(group).get("anchor_support_quota_per_target", 0)
                if quota:
                    forced_targets = target_np[forced.numpy()]
                    same_forced = int(forced_targets.sum())
                    different_forced = int((~forced_targets.astype(bool)).sum())
                    registered = sampling["quota"]
                    require(same_forced == registered["same"]["forced"]
                            and different_forced == registered["different"]["forced"],
                            f"{group} fold {fold}: forced Same/Different quota differs")
                checkpoint_sha[kind] = runner.sha_file(path)
                if formal_pairs is None:
                    formal_pairs, formal_targets = pairs.clone(), targets.clone()
                else:
                    require(torch.equal(pairs, formal_pairs)
                            and torch.equal(targets, formal_targets),
                            f"{group} fold {fold}: Best and Latest relation pairs differ")
                del payload
            records.append({
                "group": group, "fold": fold, "status": "PASS",
                "warmup_pair_sha256": runner.sha_object((expected_warm_pairs, expected_warm_targets)),
                "formal_pair_sha256": runner.sha_object((formal_pairs, formal_targets)),
                "formal_sampling": expected_sampling,
                "best_checkpoint_sha256": checkpoint_sha["best"],
                "latest_checkpoint_sha256": checkpoint_sha["latest"],
            })
            print(f"PASS relation checkpoint {group} fold {fold:02d}", flush=True)
    result = {
        "experiment": runner.EXPERIMENT,
        "status": "PASS" if len(records) == 80 else "FAIL",
        "folds_expected": 80, "folds_passed": len(records), "records": records,
        "warmup_pairs_match_h1": True, "formal_pairs_restored_exactly": True,
        "train_only_labels_and_pairs": True, "best_latest_relation_state_identical": True,
        "preflight_sha256": runner.sha_file(ROOT / "audits/relation_sampling_preflight.json"),
        "script_sha256": runner.sha_file(Path(__file__)),
    }
    (ROOT / "audits/relation_sampling_checkpoint_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "folds_passed": len(records)}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
