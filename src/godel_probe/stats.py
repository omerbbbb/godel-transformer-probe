"""Held-out signature statistic with a shuffled-label permutation null.

The original rule ("a signature fires if ANY of its factors is present")
fires on almost every example once a signature holds hundreds of factors,
so a rate of 1.0 on both classes carries no information. Here each held-out
prompt instead gets a graded score

    score(x) = frac(near_two factors present in x) - frac(near_one factors present in x)

and the statistic is the *paired held-out accuracy*: the fraction of test
chains whose two-hop prompt scores higher than its matched one-hop prompt
(ties count 0.5; chance = 0.5).

Null distribution: within each discovery chain, the one/two labels are
swapped at random (preserving the matched-pair structure), the near-GCD
signatures are rebuilt, and the same held-out statistic is recomputed on the
unchanged test set.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

import numpy as np


def _near_sets(pres_two: np.ndarray, pres_one: np.ndarray, threshold: float):
    """Boolean masks of near-GCD factors given per-class prevalence vectors."""
    near_two = (pres_two >= threshold) & (pres_one <= 1.0 - threshold)
    near_one = (pres_one >= threshold) & (pres_two <= 1.0 - threshold)
    return near_two, near_one


def _scores(test_matrix: np.ndarray, near_two: np.ndarray, near_one: np.ndarray) -> np.ndarray:
    n2, n1 = near_two.sum(), near_one.sum()
    s2 = test_matrix[:, near_two].sum(axis=1) / n2 if n2 else np.zeros(len(test_matrix))
    s1 = test_matrix[:, near_one].sum(axis=1) / n1 if n1 else np.zeros(len(test_matrix))
    return s2 - s1


def _paired_accuracy(scores_two: np.ndarray, scores_one: np.ndarray) -> float:
    diff = scores_two - scores_one
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / len(diff))


def permutation_test(
    discover_pairs: list[tuple[Counter, Counter]],
    test_pairs: list[tuple[Counter, Counter]],
    threshold: float = 0.8,
    permutations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired held-out accuracy of near-GCD signatures vs a shuffled-label null.

    ``discover_pairs`` / ``test_pairs`` are lists of (two_hop, one_hop)
    factor multisets, one tuple per chain.
    """
    m = len(discover_pairs)
    if m == 0 or not test_pairs:
        return {"status": "skipped", "reason": "empty discovery or test split"}

    # Label-invariant pre-filter: a factor can enter a near-GCD set under SOME
    # labelling only if its total presence count c over the 2m discovery
    # prompts satisfies ceil(thr*m) <= c <= m + floor((1-thr)*m).
    # Filtering on c is independent of the labels, so the null stays valid.
    presence = Counter()
    for two, one in discover_pairs:
        presence.update(two.keys())
        presence.update(one.keys())
    lo = math.ceil(threshold * m - 1e-9)
    hi = m + math.floor((1.0 - threshold) * m + 1e-9)
    candidates = sorted((f for f, c in presence.items() if lo <= c <= hi), key=repr)
    index = {f: i for i, f in enumerate(candidates)}
    k = len(candidates)

    def matrix(counters):
        mat = np.zeros((len(counters), k), dtype=np.float32)
        for row, counter in enumerate(counters):
            cols = [index[f] for f in counter.keys() if f in index]
            mat[row, cols] = 1.0
        return mat

    d_two = matrix([p[0] for p in discover_pairs])
    d_one = matrix([p[1] for p in discover_pairs])
    t_two = matrix([p[0] for p in test_pairs])
    t_one = matrix([p[1] for p in test_pairs])

    def statistic(swap: np.ndarray) -> tuple[float, int, int]:
        s = swap[:, None]
        lab_two = np.where(s, d_one, d_two)
        lab_one = np.where(s, d_two, d_one)
        near_two, near_one = _near_sets(lab_two.mean(0), lab_one.mean(0), threshold)
        acc = _paired_accuracy(
            _scores(t_two, near_two, near_one), _scores(t_one, near_two, near_one)
        )
        return acc, int(near_two.sum()), int(near_one.sum())

    observed, n_two, n_one = statistic(np.zeros(m, dtype=bool))

    rng = np.random.default_rng(seed)
    null = np.array(
        [statistic(rng.random(m) < 0.5)[0] for _ in range(permutations)]
    )
    p_value = float((1 + (null >= observed).sum()) / (1 + permutations))
    sd = float(null.std())
    return {
        "status": "ok",
        "statistic": "paired held-out accuracy (two-hop score > matched one-hop score; chance 0.5)",
        "threshold": threshold,
        "discover_chains": m,
        "test_chains": len(test_pairs),
        "candidate_factors": k,
        "near_two_factors": n_two,
        "near_one_factors": n_one,
        "observed": observed,
        "permutations": permutations,
        "null_mean": float(null.mean()),
        "null_sd": sd,
        "null_95th": float(np.quantile(null, 0.95)),
        "p_value_one_sided": p_value,
        "effect_over_null": observed - float(null.mean()),
        "effect_z": (observed - float(null.mean())) / sd if sd > 0 else None,
    }
