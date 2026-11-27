"""Fold-local cognitive-support quotas for the formal DARG relation loss."""
import hashlib
from itertools import combinations_with_replacement

import numpy as np
import torch

CLASS_NAMES = ("NC", "MCI", "AD")


def coverage(pairs, support, local_target, classes, train_np):
    n = len(local_target)
    core_keys = support[:, 0] * n + support[:, 1]
    keys = pairs[:, 0] * n + pairs[:, 1]
    supported = np.isin(keys, core_keys)
    truth = local_target[pairs[:, 0]] == local_target[pairs[:, 1]]
    orderless_classes = np.sort(local_target[pairs], axis=1)
    composition = {}
    for a, b in combinations_with_replacement(range(3), 2):
        members = (orderless_classes == (a, b)).all(axis=1)
        label = f"{classes[a]}-{classes[b]}"
        composition[label] = {"count": int(members.sum()), "supported": int((members & supported).sum())}
    if (pairs.shape != (4096, 2) or int(truth.sum()) != 2048
            or not np.isin(pairs, train_np).all() or np.any(pairs[:, 0] >= pairs[:, 1])
            or len(np.unique(keys)) != len(keys)):
        raise RuntimeError("Invalid relation pair size, balance, patient boundary, or order")
    return {
        "pair_count": int(len(pairs)), "same": int(truth.sum()), "different": int((~truth).sum()),
        "actual_supported": int(supported.sum()),
        "actual_support_fraction": float(supported.mean()),
        "actual_supported_by_target": {
            "same": int((supported & truth).sum()), "different": int((supported & ~truth).sum()),
        },
        "same_different_composition": composition,
        "exact_duplicates": 0, "reverse_duplicates": 0,
        "unordered_pairs_canonical": True, "train_only": True,
        "pair_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
    }


@torch.no_grad()
def formal_relation_sample(adjacency, train, train_labels, warmup_pairs, warmup_targets,
                           quota, fold):
    train_np = train.detach().cpu().numpy()
    train_y = train_labels.detach().cpu().numpy()
    local_target = np.full(len(adjacency), -1, dtype=np.int64)
    if train_y.shape[0] == len(adjacency):
        local_target[train_np] = train_y[train_np]
    elif train_y.shape[0] == train_np.shape[0]:
        local_target[train_np] = train_y
    else:
        raise ValueError("train_labels must contain all patients or the train fold")
    graph = adjacency[train[:, None], train[None, :]] > 0
    if not torch.equal(graph, graph.T):
        raise RuntimeError("Asymmetric original cognitive support at epoch 10")
    src, dst = torch.triu(graph, diagonal=1).nonzero(as_tuple=True)
    support = torch.stack((train[src], train[dst]), dim=1).cpu().numpy()
    historical = warmup_pairs.cpu().numpy()
    historical_targets = warmup_targets.cpu().numpy().astype(bool)
    warm_audit = coverage(historical, support, local_target, CLASS_NAMES, train_np)
    np.testing.assert_array_equal(historical_targets,
                                  local_target[historical[:, 0]] == local_target[historical[:, 1]])

    rngs = [np.random.default_rng(seed) for seed in np.random.SeedSequence(
        [838503, fold]).spawn(3)]
    support_targets = local_target[support[:, 0]] == local_target[support[:, 1]]
    forced = np.zeros(len(historical), dtype=bool)
    final = historical
    targets = historical_targets
    quota_record = {}
    if quota:
        part_pairs, part_targets, part_forced = [], [], []
        for same, rng in zip((True, False), rngs[:2]):
            pool = support[support_targets == same]
            key = len(adjacency)
            force = pool[rng.permutation(len(pool))[:quota]]
            force_keys = force[:, 0] * key + force[:, 1]
            historical_class = historical[historical_targets == same]
            old_keys = historical_class[:, 0] * key + historical_class[:, 1]
            chosen = historical_class[~np.isin(old_keys, force_keys)][:2048 - len(force)]
            part_pairs.extend((force, chosen))
            part_targets.append(np.full(2048, same))
            part_forced.append(np.arange(2048) < len(force))
            quota_record["same" if same else "different"] = {
                "requested": quota, "available": int(len(pool)), "forced": int(len(force)),
                "shortfall": quota - int(len(force)), "historical_remainder": len(chosen),
            }
        permutation = rngs[2].permutation(4096)
        final = np.concatenate(part_pairs)[permutation]
        targets = np.concatenate(part_targets)[permutation]
        forced = np.concatenate(part_forced)[permutation]

    formal_audit = coverage(final, support, local_target, CLASS_NAMES, train_np)
    np.testing.assert_array_equal(targets, local_target[final[:, 0]] == local_target[final[:, 1]])
    audit = {
        "status": "PASS", "epoch_first_used": 11, "seed": [838503, fold],
        "support_count": int(len(support)), "support_same": int(support_targets.sum()),
        "support_different": int((~support_targets).sum()),
        "support_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
        "support_symmetric": True, "forced_count": int(forced.sum()), "quota": quota_record,
        "warmup": warm_audit, "formal": formal_audit,
    }
    return (torch.as_tensor(final, device=train.device),
            torch.as_tensor(targets.astype(np.float32), device=train.device),
            torch.as_tensor(forced, device=train.device), audit)
