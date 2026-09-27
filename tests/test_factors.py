from collections import Counter

import torch

from godel_probe.factors import (
    assign_primes,
    exact_gcd_factorization,
    prevalence,
    primes,
    structural_factors,
)


def test_primes():
    assert primes(0) == []
    assert primes(6) == [2, 3, 5, 7, 11, 13]


def test_prime_assignment_is_deterministic():
    factors = [("E", 0, 0, 1, 0), ("E", 0, 0, 1, 1)]
    assert assign_primes(factors) == assign_primes(reversed(factors))


def test_exact_gcd_factorization_respects_multiplicity():
    a = Counter({"x": 3, "y": 1})
    b = Counter({"x": 2, "z": 5})
    assert exact_gcd_factorization([a, b]) == Counter({"x": 2})


def test_prevalence_counts_presence_not_multiplicity():
    counters = [Counter({"x": 10}), Counter({"x": 1, "y": 2})]
    assert prevalence(counters) == {"x": 1.0, "y": 0.5}


def test_structural_factors_synthetic_attention():
    # Two layers, one head, three tokens. Causal rows only.
    layer0 = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.1, 0.9, 0.0],
        [0.2, 0.3, 0.5],
    ]])
    layer1 = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.1, 0.7, 0.2],
    ]])

    factors = structural_factors([layer0, layer1], topk=1)

    assert ("E", 0, 0, 1, 1) in factors
    assert ("E", 1, 0, 2, 1) in factors
    # layer1 query=2 -> intermediate=1; layer0 query=1 -> source=1
    assert ("P", 1, 0, 2, 1, 0, 1) in factors
