"""Deterministic six-disease relation-pair samplers for E83.8.5.4.

The sampler operates only on the outer-training patients.  It returns canonical
unordered pairs and keeps the registered 2048/2048 and six-category quotas.
All randomness is local to this module and never touches the model RNG streams.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


CLASS_NAMES = ("NC", "MCI", "AD")
RELATION_NAMES = ("NC-NC", "MCI-MCI", "AD-AD", "NC-MCI", "NC-AD", "MCI-AD")
RELATION_CLASSES = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
TOTAL_PAIRS = 4096
SAME_PAIRS = 2048
DIFFERENT_PAIRS = 2048


@dataclass(frozen=True)
class PairPool:
    name: str
    left_class: int
    right_class: int
    pairs: np.ndarray


def _canonical(pairs: np.ndarray) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not len(pairs):
        return np.empty((0, 2), dtype=np.int64)
    return np.sort(pairs, axis=1)


def _keys(pairs: np.ndarray, n_patients: int) -> np.ndarray:
    pairs = _canonical(pairs)
    return pairs[:, 0] * int(n_patients) + pairs[:, 1]


def _unique_sorted(pairs: np.ndarray, n_patients: int) -> np.ndarray:
    pairs = _canonical(pairs)
    if not len(pairs):
        return pairs
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    if not len(pairs):
        return pairs
    order = np.argsort(_keys(pairs, n_patients), kind="mergesort")
    pairs = pairs[order]
    keep = np.ones(len(pairs), dtype=bool)
    keep[1:] = _keys(pairs, n_patients)[1:] != _keys(pairs, n_patients)[:-1]
    return pairs[keep]


def _quota_split(total: int, weights: Sequence[int]) -> np.ndarray:
    """Hamilton allocation with fixed left-to-right remainder resolution."""
    weights = np.asarray(weights, dtype=np.int64)
    if total < 0 or np.any(weights < 0) or int(weights.sum()) <= 0:
        raise ValueError("Invalid quota weights")
    exact = total * weights.astype(np.float64) / float(weights.sum())
    result = np.floor(exact).astype(np.int64)
    remaining = int(total - int(result.sum()))
    order = sorted(range(len(weights)), key=lambda i: (-float(exact[i] - result[i]), i))
    for index in order[:remaining]:
        result[index] += 1
    return result


def _make_pools(train_ids: np.ndarray, labels: np.ndarray) -> Tuple[PairPool, ...]:
    train_ids = np.asarray(train_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    pools = []
    for name, (left_class, right_class) in zip(RELATION_NAMES, RELATION_CLASSES):
        left = train_ids[labels[train_ids] == left_class]
        right = train_ids[labels[train_ids] == right_class]
        if left_class == right_class:
            if len(left) < 2:
                pairs = np.empty((0, 2), dtype=np.int64)
            else:
                pairs = np.asarray(list(combinations(left.tolist(), 2)), dtype=np.int64)
        else:
            pairs = np.asarray([(a, b) for a in left for b in right], dtype=np.int64)
        pools.append(PairPool(name, left_class, right_class,
                              _unique_sorted(pairs, len(labels))))
    return tuple(pools)


def _category_targets() -> np.ndarray:
    # Fixed 683/683/682 order for same and different relation blocks.
    return np.asarray([683, 683, 682, 683, 683, 682], dtype=np.int64)


def _candidate_support(support: Optional[np.ndarray], n_patients: int,
                       train_ids: np.ndarray) -> np.ndarray:
    if support is None:
        return np.empty((0, 2), dtype=np.int64)
    support = _unique_sorted(np.asarray(support, dtype=np.int64), n_patients)
    allowed = np.zeros(n_patients, dtype=bool)
    allowed[np.asarray(train_ids, dtype=np.int64)] = True
    if len(support):
        support = support[allowed[support[:, 0]] & allowed[support[:, 1]]]
    return support


def _support_by_pool(pools: Sequence[PairPool], support: np.ndarray,
                     n_patients: int) -> Tuple[np.ndarray, ...]:
    support_keys = set(_keys(support, n_patients).tolist())
    return tuple(pool.pairs[np.fromiter(
        (int(key) in support_keys for key in _keys(pool.pairs, n_patients)),
        dtype=bool, count=len(pool.pairs))] for pool in pools)


def _choose_random(pool: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count < 0 or count > len(pool):
        raise ValueError(f"Requested {count} pairs from a pool of {len(pool)}")
    if count == 0:
        return np.empty((0, 2), dtype=np.int64)
    selected = rng.choice(len(pool), size=count, replace=False)
    return pool[np.sort(selected)]


def _choose_with_prefix(pool: np.ndarray, support: np.ndarray, count: int,
                        rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
    """Take a deterministic support prefix, then a local-RNG fallback prefix."""
    if count > len(pool):
        raise ValueError(f"Requested {count} pairs from a pool of {len(pool)}")
    support = _unique_sorted(support, int(max(pool.max(initial=0), 0) + 1))
    support_keys = set(map(tuple, support.tolist()))
    supported = np.asarray([tuple(pair) in support_keys for pair in pool], dtype=bool)
    supported_pool = pool[supported]
    fallback_pool = pool[~supported]
    forced = min(count, len(supported_pool))
    selected_support = supported_pool[:forced]
    remaining = count - forced
    selected_fallback = _choose_random(fallback_pool, remaining, rng)
    return np.concatenate((selected_support, selected_fallback), axis=0), forced, count - forced


def _relation_type_counts(pairs: np.ndarray, labels: np.ndarray) -> Dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)
    values = np.sort(labels[pairs], axis=1)
    return {name: int(np.all(values == np.asarray(classes), axis=1).sum())
            for name, classes in zip(RELATION_NAMES, RELATION_CLASSES)}


def _exposure_stats(pairs: np.ndarray, train_ids: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    train_ids = np.asarray(train_ids, dtype=np.int64)
    counts = np.bincount(pairs.reshape(-1), minlength=len(labels)).astype(np.int64)
    result = {}
    for class_id, name in enumerate(CLASS_NAMES):
        values = counts[train_ids[labels[train_ids] == class_id]].astype(np.float64)
        if len(values):
            sorted_values = np.sort(values)
            p95 = float(np.percentile(values, 95, method="linear"))
            mean = float(values.mean())
            if mean > 0:
                gini = float(np.abs(values[:, None] - values[None, :]).mean() / (2.0 * mean))
            else:
                gini = 0.0
            result[name] = {
                "count": int(len(values)), "mean": mean, "std": float(values.std(ddof=0)),
                "max": int(values.max(initial=0)), "p95": p95, "gini": gini,
                "min": int(sorted_values.min()),
                "total_endpoint_exposure": int(values.sum()),
            }
        else:
            result[name] = {"count": 0, "mean": 0.0, "std": 0.0, "max": 0,
                            "p95": 0.0, "gini": 0.0, "min": 0,
                            "total_endpoint_exposure": 0}
    return result


def _coverage(pairs: np.ndarray, pools: Sequence[PairPool], support: np.ndarray,
              train_ids: np.ndarray, labels: np.ndarray, target_support: int,
              forced_by_relation: Mapping[str, int], fallback_by_relation: Mapping[str, int],
              sampler: str, fold: int, seed: Sequence[int]) -> Dict[str, object]:
    n_patients = len(labels)
    support_keys = set(_keys(support, n_patients).tolist())
    pair_keys = _keys(pairs, n_patients)
    if len(pair_keys) != len(set(pair_keys.tolist())):
        raise RuntimeError("Duplicate unordered relation pair")
    if np.any(pairs[:, 0] >= pairs[:, 1]):
        raise RuntimeError("Relation pairs must be canonical and non-self")
    train_set = set(np.asarray(train_ids, dtype=np.int64).tolist())
    if any(int(value) not in train_set for value in pairs.reshape(-1)):
        raise RuntimeError("Relation pair escaped the outer-training patients")
    relation_counts = _relation_type_counts(pairs, labels)
    expected = dict(zip(RELATION_NAMES, _category_targets().tolist()))
    if relation_counts != expected:
        raise RuntimeError(f"Six relation quotas differ: {relation_counts} != {expected}")
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    if len(pairs) != TOTAL_PAIRS or int(same.sum()) != SAME_PAIRS:
        raise RuntimeError("Relation total or same/different quota differs")
    support_mask = np.fromiter((int(key) in support_keys for key in pair_keys),
                               dtype=bool, count=len(pair_keys))
    return {
        "sampler": sampler, "fold": int(fold), "seed": [int(value) for value in seed],
        "pair_count": int(len(pairs)), "same_pairs": int(same.sum()),
        "different_pairs": int((~same).sum()), "relation_counts": relation_counts,
        "target_relation_counts": expected, "target_support_count": int(target_support),
        "actual_support_count": int(support_mask.sum()),
        "actual_support_fraction": float(support_mask.mean()),
        "forced_support_by_relation": dict(forced_by_relation),
        "fallback_by_relation": dict(fallback_by_relation),
        "support_available": int(len(support)),
        "support_sha256": hashlib.sha256(np.asarray(support, dtype=np.int64).tobytes()).hexdigest(),
        "pair_sha256": hashlib.sha256(np.asarray(pairs, dtype=np.int64).tobytes()).hexdigest(),
        "target_sha256": hashlib.sha256(np.asarray(same, dtype=np.float32).tobytes()).hexdigest(),
        "canonical_unordered_unique": True, "non_self": True, "train_only": True,
        "patient_exposure": _exposure_stats(pairs, train_ids, labels),
    }


def _sample_t1(train_ids: np.ndarray, labels: np.ndarray, fold: int,
               rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, object]]:
    pools = _make_pools(train_ids, labels)
    targets = _category_targets()
    parts = [_choose_random(pool.pairs, int(target), rng)
             for pool, target in zip(pools, targets)]
    order = rng.permutation(TOTAL_PAIRS)
    pairs = np.concatenate(parts, axis=0)[order]
    audit = _coverage(pairs, pools, np.empty((0, 2), dtype=np.int64), train_ids, labels,
                      0, {name: 0 for name in RELATION_NAMES},
                      {name: int(target) for name, target in zip(RELATION_NAMES, targets)},
                      "T1", fold, (83854, fold, 1))
    return pairs, audit


def _sample_supported(train_ids: np.ndarray, labels: np.ndarray, support: np.ndarray,
                      fold: int, fraction: float, sampler: str) -> Tuple[np.ndarray, Dict[str, object]]:
    pools = _make_pools(train_ids, labels)
    support = _candidate_support(support, len(labels), train_ids)
    support_pools = _support_by_pool(pools, support, len(labels))
    targets = _category_targets()
    support_target = int(round(TOTAL_PAIRS * float(fraction)))
    support_targets = _quota_split(support_target, targets)
    parts, forced_by_relation, fallback_by_relation = [], {}, {}
    for index, (pool, support_pool, target, support_quota) in enumerate(zip(
            pools, support_pools, targets, support_targets)):
        forced = min(int(support_quota), len(support_pool))
        selected_support = support_pool[:forced]
        fallback_pool = pool.pairs[
            ~np.isin(_keys(pool.pairs, len(labels)), _keys(support_pool, len(labels)))
        ]
        fallback = int(target) - forced
        category_rng = np.random.default_rng(np.random.SeedSequence([83854, fold, index, 23]))
        fallback_order = category_rng.permutation(len(fallback_pool))
        selected_fallback = fallback_pool[fallback_order[:fallback]]
        selected = np.concatenate((selected_support, selected_fallback), axis=0)
        parts.append(selected)
        forced_by_relation[pool.name] = int(forced)
        fallback_by_relation[pool.name] = int(fallback)
    # The same deterministic final permutation is used for T2 and T3.  Their
    # pair contents differ only through the requested support prefix length.
    order_rng = np.random.default_rng(np.random.SeedSequence([83854, fold, 99]))
    pairs = np.concatenate(parts, axis=0)[order_rng.permutation(TOTAL_PAIRS)]
    audit = _coverage(pairs, pools, support, train_ids, labels, support_target,
                      forced_by_relation, fallback_by_relation, sampler, fold,
                      (83854, fold, 2))
    audit["requested_support_by_relation"] = {
        name: int(value) for name, value in zip(RELATION_NAMES, support_targets)
    }
    audit["support_shortfall_by_relation"] = {
        name: int(max(0, int(value) - forced_by_relation[name]))
        for name, value in zip(RELATION_NAMES, support_targets)
    }
    return pairs, audit


def _sample_t4(train_ids: np.ndarray, labels: np.ndarray, fold: int,
               rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, object]]:
    pools = _make_pools(train_ids, labels)
    targets = _category_targets()
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.zeros(len(labels), dtype=np.int64)
    parts = []
    chosen_keys = set()
    for pool, target in zip(pools, targets):
        candidates = pool.pairs
        if len(candidates) < int(target):
            raise RuntimeError(f"Insufficient legal candidates for {pool.name}")
        class_sizes = {class_id: int(np.sum(labels[np.asarray(train_ids)] == class_id))
                       for class_id in range(3)}
        class_demand = {class_id: 0.0 for class_id in range(3)}
        if pool.left_class == pool.right_class:
            class_demand[pool.left_class] = 2.0 * int(target)
        else:
            class_demand[pool.left_class] = float(target)
            class_demand[pool.right_class] = float(target)
        target_exposure = {class_id: class_demand[class_id] / max(class_sizes[class_id], 1)
                           for class_id in range(3)}
        tie = rng.random(len(candidates))
        available = np.ones(len(candidates), dtype=bool)
        selected = []
        for _ in range(int(target)):
            indices = np.flatnonzero(available)
            pair_values = candidates[indices]
            left, right = pair_values[:, 0], pair_values[:, 1]
            before_left = counts[left] / np.asarray(
                [target_exposure[int(value)] for value in labels[left]])
            before_right = counts[right] / np.asarray(
                [target_exposure[int(value)] for value in labels[right]])
            # Max-normalized exposure is the primary criterion.  The sum breaks
            # ties toward pairs that improve the lower-exposure endpoint.
            primary = np.maximum(before_left, before_right)
            secondary = before_left + before_right
            minimum_primary = primary.min()
            best = indices[primary == minimum_primary]
            if len(best) > 1:
                best_secondary = secondary[primary == minimum_primary]
                minimum_secondary = best_secondary.min()
                best = best[best_secondary == minimum_secondary]
            chosen_index = best[np.argmin(tie[best])]
            pair = candidates[chosen_index]
            key = (int(pair[0]), int(pair[1]))
            if key in chosen_keys:
                available[chosen_index] = False
                continue
            selected.append(pair)
            chosen_keys.add(key)
            counts[pair] += 1
            available[chosen_index] = False
        if len(selected) != int(target):
            raise RuntimeError(f"T4 could not select {target} unique pairs for {pool.name}")
        parts.append(np.asarray(selected, dtype=np.int64))
    pairs = np.concatenate(parts, axis=0)[rng.permutation(TOTAL_PAIRS)]
    audit = _coverage(pairs, pools, np.empty((0, 2), dtype=np.int64), train_ids, labels,
                      0, {name: 0 for name in RELATION_NAMES},
                      {name: int(target) for name, target in zip(RELATION_NAMES, targets)},
                      "T4", fold, (83854, fold, 4))
    audit["coverage_strategy"] = "diagnosis_internal_standardized_exposure_greedy"
    audit["uniform_global_hard_cap"] = False
    audit["soft_fallback"] = "none_required"
    return pairs, audit


def sample_formal_pairs(sampler: str, train_ids: np.ndarray, labels: np.ndarray,
                        fold: int, cognitive_support: Optional[np.ndarray] = None):
    """Return formal pairs, binary targets, and a complete audit record."""
    sampler = str(sampler).upper()
    train_ids = np.asarray(train_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if sampler not in {"T1", "T2", "T3", "T4"}:
        raise ValueError(f"Unknown E83.8.5.4 sampler {sampler}")
    if len(np.unique(train_ids)) != len(train_ids):
        raise ValueError("Outer-training patient indices must be unique")
    if not np.all((labels[train_ids] >= 0) & (labels[train_ids] <= 2)):
        raise ValueError("Outer-training labels must be NC/MCI/AD encoded as 0/1/2")
    rng = np.random.default_rng(np.random.SeedSequence([83854, int(fold), {"T1": 1, "T2": 2, "T3": 3, "T4": 4}[sampler]]))
    if sampler == "T1":
        pairs, audit = _sample_t1(train_ids, labels, fold, rng)
    elif sampler in {"T2", "T3"}:
        if cognitive_support is None:
            raise ValueError(f"{sampler} requires frozen epoch-10 COGNITIVE_TEST support")
        fraction = 0.10 if sampler == "T2" else 0.25
        pairs, audit = _sample_supported(train_ids, labels, cognitive_support, fold,
                                          fraction, sampler)
    else:
        pairs, audit = _sample_t4(train_ids, labels, fold, rng)
    targets = (labels[pairs[:, 0]] == labels[pairs[:, 1]]).astype(np.float32)
    if int(targets.sum()) != SAME_PAIRS:
        raise RuntimeError("Formal sampler did not produce 2048 same pairs")
    audit["target_matches_labels"] = True
    audit["target_pair_sha256"] = hashlib.sha256(targets.tobytes()).hexdigest()
    return pairs.astype(np.int64), targets, audit


def audit_pair_array(pairs: np.ndarray, labels: np.ndarray, train_ids: np.ndarray,
                     sampler: str = "unknown") -> Dict[str, object]:
    """Independent structural check used by preflight and post-training audits."""
    pools = _make_pools(np.asarray(train_ids, dtype=np.int64), np.asarray(labels, dtype=np.int64))
    return _coverage(np.asarray(pairs, dtype=np.int64), pools, np.empty((0, 2), dtype=np.int64),
                     np.asarray(train_ids, dtype=np.int64), np.asarray(labels, dtype=np.int64),
                     0, {}, {}, sampler, 0, (0,))
