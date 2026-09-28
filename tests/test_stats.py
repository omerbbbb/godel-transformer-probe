from collections import Counter

from godel_probe.stats import permutation_test


def _pairs(n, informative):
    pairs = []
    for i in range(n):
        shared = Counter({("E", 0, 0, 1, 0): 1, ("E", 0, 0, i + 2, 0): 1})
        two, one = Counter(shared), Counter(shared)
        if informative:
            two[("E", 1, 0, 3, 1)] = 1
            one[("E", 1, 0, 3, 2)] = 1
        pairs.append((two, one))
    return pairs


def test_informative_signature_is_significant():
    out = permutation_test(_pairs(10, True), _pairs(10, True), permutations=500, seed=0)
    assert out["observed"] == 1.0
    assert out["p_value_one_sided"] < 0.05
    assert out["near_two_factors"] == 1 and out["near_one_factors"] == 1


def test_uninformative_signature_is_not_significant():
    out = permutation_test(_pairs(10, False), _pairs(10, False), permutations=200, seed=0)
    assert out["observed"] == 0.5
    assert out["p_value_one_sided"] > 0.5
