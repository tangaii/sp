"""Common-random-base, fold-local cognitive patient-pair sampling."""
from __future__ import annotations

from itertools import combinations
import hashlib
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


CLASS_NAMES = ("NC", "MCI", "AD")
RELATION_NAMES = ("NC-NC", "MCI-MCI", "AD-AD", "NC-MCI", "NC-AD", "MCI-AD")
RELATION_CLASSES = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
TOTAL_PAIRS = 4096
SAME_PAIRS = 2048
DIFFERENT_PAIRS = 2048
HARD_FRACTION = 0.10
GRAPH_FRACTION = 0.05
HARD_RELATIONS = {"U2": "MCI-MCI", "U3": "NC-MCI", "U4": "MCI-AD"}
DIFFICULTY_BANDS = {
    "MCI-MCI": (0.10, 0.30),
    "NC-MCI": (0.70, 0.90),
    "MCI-AD": (0.70, 0.90),
}


def _canonical(pairs: np.ndarray) -> np.ndarray:
    values = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return np.sort(values, axis=1) if len(values) else np.empty((0, 2), dtype=np.int64)


def _keys(pairs: np.ndarray, n_patients: int) -> np.ndarray:
    values = _canonical(pairs)
    return values[:, 0] * int(n_patients) + values[:, 1]


def _sha(pairs: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pairs, dtype=np.int64).tobytes()).hexdigest()


def _quota_split(total: int, weights: Sequence[int]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.int64)
    exact = total * weights.astype(np.float64) / float(weights.sum())
    allocation = np.floor(exact).astype(np.int64)
    remainder = int(total - allocation.sum())
    order = sorted(range(len(weights)), key=lambda index: (-(exact[index] - allocation[index]), index))
    allocation[order[:remainder]] += 1
    return allocation


def relation_counts(pairs: np.ndarray, labels: np.ndarray) -> Dict[str, int]:
    values = np.sort(np.asarray(labels)[np.asarray(pairs, dtype=np.int64)], axis=1)
    return {
        name: int(np.all(values == np.asarray(classes), axis=1).sum())
        for name, classes in zip(RELATION_NAMES, RELATION_CLASSES)
    }


def _make_pools(train_ids: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, ...]:
    pools = []
    for left_class, right_class in RELATION_CLASSES:
        left = train_ids[labels[train_ids] == left_class]
        right = train_ids[labels[train_ids] == right_class]
        if left_class == right_class:
            values = np.asarray(list(combinations(left.tolist(), 2)), dtype=np.int64)
        else:
            values = np.asarray([(a, b) for a in left for b in right], dtype=np.int64)
        pools.append(_canonical(values))
    return tuple(pools)


def _sample_excluding(pool: np.ndarray, count: int, excluded: set[int],
                      rng: np.random.Generator, n_patients: int) -> np.ndarray:
    if count == 0:
        return np.empty((0, 2), dtype=np.int64)
    keys = _keys(pool, n_patients)
    available = pool[~np.isin(keys, np.fromiter(excluded, dtype=np.int64))] if excluded else pool
    if count > len(available):
        raise RuntimeError(f"Only {len(available)} eligible pairs remain for a quota of {count}")
    chosen = rng.choice(len(available), size=count, replace=False)
    return available[np.sort(chosen)]


def _exposure_stats(pairs: np.ndarray, train_ids: np.ndarray,
                    labels: np.ndarray) -> Dict[str, object]:
    endpoint_counts = np.bincount(pairs.reshape(-1), minlength=len(labels))
    result = {}
    for class_id, name in enumerate(CLASS_NAMES):
        values = endpoint_counts[train_ids[labels[train_ids] == class_id]].astype(np.float64)
        ordered = np.sort(values)
        mean = float(values.mean()) if len(values) else 0.0
        if len(values) and mean > 0:
            weights = 2 * np.arange(1, len(values) + 1) - len(values) - 1
            gini = float(np.dot(weights, ordered) / (len(values) * values.sum()))
        else:
            gini = 0.0
        result[name] = {
            "count": int(len(values)), "mean": mean,
            "std": float(values.std(ddof=0)) if len(values) else 0.0,
            "max": int(values.max(initial=0)), "p95": float(np.percentile(values, 95)) if len(values) else 0.0,
            "gini": gini, "min": int(values.min()) if len(values) else 0,
            "total_endpoint_exposure": int(values.sum()),
        }
    return result


def _build_shared_base(train_ids: np.ndarray, labels: np.ndarray, quotas: np.ndarray,
                       fold: int):
    rng = np.random.default_rng(np.random.SeedSequence([83855, int(fold), 1]))
    pools = _make_pools(train_ids, labels)
    parts = []
    for name, pool, quota in zip(RELATION_NAMES, pools, quotas):
        if len(pool) < int(quota):
            raise RuntimeError(f"{name} has too few legal training pairs for the S02 quota")
        parts.append(pool[np.sort(rng.choice(len(pool), size=int(quota), replace=False))])
    category_ids = np.concatenate([
        np.full(int(quota), index, dtype=np.int64)
        for index, quota in enumerate(quotas)
    ])
    raw_pairs = np.concatenate(parts, axis=0)
    order = rng.permutation(len(raw_pairs))
    return raw_pairs[order], category_ids[order], pools, rng


def _eligible_quantile(pairs: np.ndarray, category_pool: np.ndarray, scores: np.ndarray,
                        band: tuple[float, float], retained_keys: set[int],
                        placebo_keys: set[int], rng: np.random.Generator,
                        n_patients: int, count: int):
    values = scores[category_pool[:, 0], category_pool[:, 1]].astype(np.float64)
    lower, upper = np.quantile(values, band)
    keys = _keys(category_pool, n_patients)
    in_band = (values >= lower) & (values <= upper)
    excluded = np.fromiter(retained_keys | placebo_keys, dtype=np.int64)
    eligible = category_pool[in_band & ~np.isin(keys, excluded)]
    fallback = "none"
    if len(eligible) < count:
        # Pre-registered shortage rule: first widen the target band by five
        # percentile points on either side, then use remaining candidates
        # nearest the original band's midpoint. Both steps remain within the
        # relation class and never consult test outcomes.
        widened = (max(0.0, band[0] - 0.05), min(1.0, band[1] + 0.05))
        widened_mask = (values >= np.quantile(values, widened[0])) & (
            values <= np.quantile(values, widened[1]))
        widened_candidates = category_pool[widened_mask & ~np.isin(keys, excluded)]
        existing = set(_keys(eligible, n_patients).tolist())
        widened_keys = _keys(widened_candidates, n_patients)
        widened_candidates = widened_candidates[~np.isin(widened_keys,
                                                            np.fromiter(existing, dtype=np.int64))]
        eligible = np.concatenate((eligible, widened_candidates), axis=0)
        fallback = "widened_by_5_percentile_points"
    if len(eligible) < count:
        remaining = category_pool[~np.isin(keys, excluded)]
        midpoint = (band[0] + band[1]) / 2.0
        ranks = np.searchsorted(np.sort(values), values, side="right") / len(values)
        remaining_ranks = ranks[~np.isin(keys, excluded)]
        order = np.argsort(np.abs(remaining_ranks - midpoint), kind="stable")
        eligible = remaining[order]
        fallback = "nearest_remaining_percentiles_to_band_midpoint"
    if len(eligible) < count:
        return None, {
            "band_percentiles": [band[0] * 100, band[1] * 100],
            "candidate_count_before_exclusions": int(np.sum(in_band)),
            "candidate_count_after_exclusions": int(len(eligible)),
            "similarity_lower": float(lower), "similarity_upper": float(upper),
            "fallback": fallback, "status": "INSUFFICIENT_CANDIDATES",
        }
    selected = _sample_excluding(eligible, count, set(), rng, n_patients)
    chosen_scores = scores[selected[:, 0], selected[:, 1]].astype(np.float64)
    pool_rank = np.searchsorted(np.sort(values), chosen_scores, side="right") / len(values)
    return selected, {
        "band_percentiles": [band[0] * 100, band[1] * 100],
        "candidate_count_before_exclusions": int(np.sum(in_band)),
        "candidate_count_after_exclusions": int(len(eligible)),
        "similarity_lower": float(lower), "similarity_upper": float(upper),
        "selected_similarity_mean": float(chosen_scores.mean()),
        "selected_similarity_min": float(chosen_scores.min()),
        "selected_similarity_max": float(chosen_scores.max()),
        "selected_percentile_mean": float(pool_rank.mean()),
        "selected_percentile_min": float(pool_rank.min()),
        "selected_percentile_max": float(pool_rank.max()),
        "fallback": fallback,
        "selected_pair_sha256": _sha(selected),
    }


def sample_formal_pairs(sampler: str, train_ids: np.ndarray, labels: np.ndarray,
                        fold: int, cognitive_support: Optional[np.ndarray] = None,
                        historical_quotas: Optional[Sequence[int]] = None,
                        cognitive_similarity: Optional[np.ndarray] = None):
    """Sample one method from a shared deterministic S02-quota base.

    All alternatives are pure functions of fold, training data, the frozen
    epoch-10 cognitive graph, and S02's actual relation quotas. Every
    replacement uses independent NumPy generators and leaves model RNG intact.
    """
    sampler = str(sampler).upper()
    if sampler not in {"U1", "U2", "U3", "U4", "U5", "U6"}:
        raise ValueError(f"Unknown E83.8.5.5 sampler {sampler}")
    train_ids = np.asarray(train_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(train_ids)) != len(train_ids):
        raise ValueError("Training patient indices must be unique")
    if len(train_ids) == 0 or np.any((labels[train_ids] < 0) | (labels[train_ids] > 2)):
        raise ValueError("Training labels must use the locked NC/MCI/AD encoding")
    if historical_quotas is None:
        raise ValueError("The S02 fold-specific relation quota is required")
    quotas = np.asarray(historical_quotas, dtype=np.int64)
    if quotas.shape != (6,) or int(quotas.sum()) != TOTAL_PAIRS:
        raise ValueError(f"Invalid S02 six-relation quota {quotas.tolist()}")
    same_target = sum(int(quotas[index]) for index in range(3))
    if same_target != SAME_PAIRS:
        raise ValueError("S02 same/different relation quota must remain 2048/2048")

    base, base_category, pools, rng = _build_shared_base(train_ids, labels, quotas, fold)
    n_patients = len(labels)
    base_keys = _keys(base, n_patients)
    if len(np.unique(base_keys)) != TOTAL_PAIRS:
        raise RuntimeError("The common random base contains duplicate unordered pairs")

    hard_positions, graph_positions = {}, {}
    for category_index, name in enumerate(RELATION_NAMES):
        category_rows = np.flatnonzero(base_category == category_index)
        difficult_count = (int(round(quotas[category_index] * HARD_FRACTION))
                           if name in DIFFICULTY_BANDS else 0)
        shuffled = rng.permutation(category_rows)
        hard_positions[name] = shuffled[:difficult_count]
        graph_quota = int(_quota_split(round(TOTAL_PAIRS * GRAPH_FRACTION), quotas)[category_index])
        graph_positions[name] = shuffled[difficult_count:difficult_count + graph_quota]

    registered_positions = np.concatenate([
        hard_positions[name] for name in RELATION_NAMES
    ] + [graph_positions[name] for name in RELATION_NAMES])
    retained_mask = np.ones(TOTAL_PAIRS, dtype=bool)
    retained_mask[registered_positions] = False
    retained = base[retained_mask]
    retained_keys = set(_keys(retained, n_patients).tolist())

    placebo_parts, placebo_keys = {}, set()
    category_rngs = {}
    for category_index, (name, pool, quota) in enumerate(zip(RELATION_NAMES, pools, quotas)):
        count = len(hard_positions[name]) + len(graph_positions[name])
        category_rng = np.random.default_rng(np.random.SeedSequence([83855, int(fold), category_index, 2]))
        category_rngs[name] = category_rng
        placebo = _sample_excluding(pool, count, retained_keys, category_rng, n_patients)
        placebo_parts[name] = placebo
        placebo_keys.update(_keys(placebo, n_patients).tolist())

    difficulty_meta = {}
    hard_pairs = {}
    if sampler in {"U2", "U3", "U4", "U5"}:
        if cognitive_similarity is None:
            raise ValueError(f"{sampler} requires the frozen epoch-10 GraphLearn similarity")
        similarity = np.asarray(cognitive_similarity, dtype=np.float32)
        if similarity.shape != (len(labels), len(labels)):
            raise ValueError("Cognitive similarity matrix has a different patient order")
        if not np.allclose(similarity, similarity.T, atol=1e-7, rtol=0.0):
            raise ValueError("Frozen cognitive similarity must be symmetric")
        hard_methods = ({"U2": ("MCI-MCI",), "U3": ("NC-MCI",),
                         "U4": ("MCI-AD",), "U5": ("NC-MCI", "MCI-AD")}[sampler])
        for category_index, name in enumerate(RELATION_NAMES):
            if name not in hard_methods:
                continue
            selected, details = _eligible_quantile(
                retained, pools[category_index], similarity, DIFFICULTY_BANDS[name],
                retained_keys, placebo_keys, category_rngs[name], n_patients,
                len(hard_positions[name]))
            if selected is None:
                raise RuntimeError(f"Even the pre-registered cognitive fallback is insufficient for {name}: {details}")
            hard_pairs[name] = selected
            difficulty_meta[name] = details
    else:
        for name in DIFFICULTY_BANDS:
            difficulty_meta[name] = {"status": "not_applied_in_group"}

    selected_support = {}
    support_capacity, support_shortfall = {}, {}
    if sampler == "U6":
        if cognitive_support is None:
            raise ValueError("U6 requires the frozen epoch-10 COGNITIVE_TEST graph support")
        support = _canonical(np.asarray(cognitive_support, dtype=np.int64))
        allowed = np.zeros(len(labels), dtype=bool)
        allowed[train_ids] = True
        if len(support):
            support = support[(support[:, 0] < support[:, 1])
                              & allowed[support[:, 0]] & allowed[support[:, 1]]]
        all_support_keys = set(_keys(support, n_patients).tolist())
        for category_index, name in enumerate(RELATION_NAMES):
            class_pair = np.sort(labels[support], axis=1) if len(support) else np.empty((0, 2), dtype=np.int64)
            classes = np.asarray(RELATION_CLASSES[category_index])
            support_in_class = support[np.all(class_pair == classes, axis=1)] if len(support) else support
            available = _keys(support_in_class, n_patients)
            available = support_in_class[~np.isin(available, np.fromiter(retained_keys | placebo_keys, dtype=np.int64))]
            requested = len(graph_positions[name])
            category_rng = np.random.default_rng(np.random.SeedSequence([83855, int(fold), category_index, 6]))
            selected = _sample_excluding(available, min(requested, len(available)), set(), category_rng,
                                         n_patients)
            selected_support[name] = selected
            support_capacity[name] = int(len(available))
            support_shortfall[name] = int(requested - len(selected))
    else:
        support = _canonical(np.asarray(cognitive_support, dtype=np.int64)) if cognitive_support is not None else np.empty((0, 2), dtype=np.int64)
        all_support_keys = set(_keys(support, n_patients).tolist()) if len(support) else set()

    pairs_by_group = {group: base.copy() for group in ("U1", "U2", "U3", "U4", "U5", "U6")}
    for category_index, name in enumerate(RELATION_NAMES):
        placebo_slots = np.concatenate((hard_positions[name], graph_positions[name]))
        pairs_by_group["U1"][placebo_slots] = placebo_parts[name]
        for group in ("U2", "U3", "U4", "U5", "U6"):
            pairs_by_group[group][placebo_slots] = placebo_parts[name]
        for group, relation in HARD_RELATIONS.items():
            if relation == name and relation in hard_pairs:
                pairs_by_group[group][hard_positions[name]] = hard_pairs[name]
        if name in hard_methods if sampler in {"U2", "U3", "U4", "U5"} else False:
            pairs_by_group[sampler][hard_positions[name]] = hard_pairs[name]
        if sampler == "U6":
            pairs_by_group["U6"][graph_positions[name][:len(selected_support[name])]] = selected_support[name]

    pairs = pairs_by_group[sampler]
    keys = _keys(pairs, n_patients)
    if len(pairs) != TOTAL_PAIRS or np.any(pairs[:, 0] >= pairs[:, 1]) or len(np.unique(keys)) != TOTAL_PAIRS:
        raise RuntimeError(f"{sampler} contains an invalid, self, or repeated unordered pair")
    training_set = set(train_ids.tolist())
    if any(int(patient) not in training_set for patient in pairs.reshape(-1)):
        raise RuntimeError(f"{sampler} contains a patient outside the outer training fold")
    counts = relation_counts(pairs, labels)
    expected = dict(zip(RELATION_NAMES, quotas.tolist()))
    if counts != expected:
        raise RuntimeError(f"{sampler} relation quotas changed: {counts} != {expected}")
    targets = (labels[pairs[:, 0]] == labels[pairs[:, 1]]).astype(np.float32)
    if int(targets.sum()) != SAME_PAIRS or int(len(targets) - targets.sum()) != DIFFERENT_PAIRS:
        raise RuntimeError(f"{sampler} same/different targets changed")

    actual_support = np.fromiter((int(key) in all_support_keys for key in keys), dtype=bool, count=len(keys))
    intervention_categories = {}
    for name in RELATION_NAMES:
        rows = np.flatnonzero(base_category == RELATION_NAMES.index(name))
        intervention_categories[name] = {
            "s02_quota": int(quotas[RELATION_NAMES.index(name)]),
            "hard_positions": len(hard_positions[name]),
            "hard_fraction": (len(hard_positions[name]) / int(quotas[RELATION_NAMES.index(name)])),
            "hard_position_ids": hard_positions[name].astype(int).tolist(),
            "graph_positions": len(graph_positions[name]),
            "graph_position_ids": graph_positions[name].astype(int).tolist(),
            "base_pair_sha256": _sha(base[rows]),
            "placebo_pair_sha256": _sha(placebo_parts[name]),
            "selected_support": int(len(selected_support.get(name, np.empty((0, 2), dtype=np.int64)))),
            "support_capacity": int(support_capacity.get(name, 0)),
            "support_shortfall": int(support_shortfall.get(name, 0)),
        }
    audit = {
        "sampler": sampler, "fold": int(fold),
        "common_base_seed": [83855, int(fold), 1],
        "common_base_pair_sha256": _sha(base),
        "base_pairs": int(len(base)), "pair_count": int(len(pairs)),
        "same_pairs": int(targets.sum()), "different_pairs": int(len(targets) - targets.sum()),
        "relation_counts": counts, "target_relation_counts": expected,
        "pair_sha256": _sha(pairs), "target_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
        "support_pair_sha256": _sha(support) if len(support) else hashlib.sha256(b"").hexdigest(),
        "support_source": "epoch_10_frozen_original_COGNITIVE_TEST_GraphLearn_mask",
        "support_pool_count": int(len(support)), "actual_support_count": int(actual_support.sum()),
        "actual_support_fraction": float(actual_support.mean()),
        "requested_support_count": int(sum(len(value) for value in graph_positions.values())),
        "deterministic_support_shortfall": int(sum(support_shortfall.values())),
        "support_shortfall_by_relation": support_shortfall,
        "support_capacity_by_relation": support_capacity,
        "unique_unordered_pairs": True, "non_self": True, "train_only": True,
        "target_matches_training_labels": True,
        "patient_exposure": _exposure_stats(pairs, train_ids, labels),
        "intervention_categories": intervention_categories,
        "difficulty_sampling": difficulty_meta,
        "difficulty_population_quantiles": {
            name: [percentile * 100 for percentile in band]
            for name, band in DIFFICULTY_BANDS.items()
        },
        "intervention_targets": {
            "mci_mm_diversity_fraction": HARD_FRACTION,
            "nc_mci_semihard_fraction": HARD_FRACTION,
            "mci_ad_semihard_fraction": HARD_FRACTION,
            "cognitive_graph_fraction": GRAPH_FRACTION,
        },
        "test_labels_read_by_sampler": False,
    }
    return pairs.astype(np.int64), targets, audit


def audit_pair_array(pairs: np.ndarray, labels: np.ndarray, train_ids: np.ndarray,
                     sampler: str = "unknown", expected_quotas: Optional[Sequence[int]] = None,
                     cognitive_support: Optional[np.ndarray] = None):
    pairs = _canonical(np.asarray(pairs, dtype=np.int64))
    train_ids = np.asarray(train_ids, dtype=np.int64)
    keys = _keys(pairs, len(labels))
    counts = relation_counts(pairs, labels)
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    allowed = set(train_ids.tolist())
    valid = (len(pairs) == TOTAL_PAIRS and len(np.unique(keys)) == TOTAL_PAIRS
             and bool(np.all(pairs[:, 0] < pairs[:, 1]))
             and all(int(patient) in allowed for patient in pairs.reshape(-1))
             and int(same.sum()) == SAME_PAIRS and int((~same).sum()) == DIFFERENT_PAIRS)
    if expected_quotas is not None:
        valid = valid and counts == dict(zip(RELATION_NAMES, np.asarray(expected_quotas).tolist()))
    support_keys = set(_keys(_canonical(np.asarray(cognitive_support, dtype=np.int64)), len(labels)).tolist()) if cognitive_support is not None and len(cognitive_support) else set()
    support_count = sum(int(key) in support_keys for key in keys)
    return {
        "status": "PASS" if valid else "FAIL", "sampler": sampler,
        "fold": 0, "pair_count": int(len(pairs)), "same_pairs": int(same.sum()),
        "different_pairs": int((~same).sum()), "relation_counts": counts,
        "pair_sha256": _sha(pairs), "actual_support_count": int(support_count),
        "actual_support_fraction": float(support_count / len(pairs)) if len(pairs) else 0.0,
        "unique_unordered_pairs": len(np.unique(keys)) == len(keys),
        "canonical_unordered_unique": len(np.unique(keys)) == len(keys),
        "non_self": bool(np.all(pairs[:, 0] < pairs[:, 1])),
        "train_only": all(int(patient) in allowed for patient in pairs.reshape(-1)),
        "target_matches_training_labels": bool(np.array_equal(same, labels[pairs[:, 0]] == labels[pairs[:, 1]])),
    }
