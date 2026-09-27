from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import torch


Factor = tuple


def primes(n: int) -> list[int]:
    """Return the first n prime numbers."""
    if n < 0:
        raise ValueError("n must be non-negative")

    out: list[int] = []
    candidate = 2

    while len(out) < n:
        is_prime = True
        limit = int(math.sqrt(candidate))
        for p in out:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break

        if is_prime:
            out.append(candidate)

        candidate += 1 if candidate == 2 else 2

    return out


def structural_factors(
    attentions: list[torch.Tensor], topk: int = 1
) -> Counter[Factor]:
    """Extract structural factors from attention tensors.

    Parameters
    ----------
    attentions:
        List of tensors with shape [heads, tokens, tokens], one per layer.
    topk:
        Number of strongest legal attention sources retained per head/query.

    Returns
    -------
    Counter
        Multiset of structural factors.

    Factor definitions
    ------------------
    E:
        ("E", layer, head, query_position, key_position)
    P:
        ("P", layer, head, query_position, intermediate_position,
         previous_layer_head, previous_source_position)

    The implementation intentionally avoids task labels and token identities.
    The factor multiset is treated as the prime factorization of an implicit
    Gödel number; the astronomically large integer itself is never constructed.
    """
    if topk < 1:
        raise ValueError("topk must be >= 1")
    if not attentions:
        return Counter()

    factors: list[Factor] = []
    top_sources: dict[tuple[int, int, int], tuple[int, ...]] = {}

    for layer, attn in enumerate(attentions):
        if attn.ndim != 3:
            raise ValueError("Each attention tensor must have shape [H, T, T]")
        heads, tokens, tokens2 = attn.shape
        if tokens != tokens2:
            raise ValueError("Attention matrices must be square in token axes")

        for head in range(heads):
            for query in range(1, tokens):
                legal = attn[head, query, : query + 1]
                k = min(topk, legal.numel())
                _, indices = torch.topk(legal, k=k)
                sources = tuple(int(i) for i in indices.tolist())
                top_sources[(layer, head, query)] = sources

                for source in sources:
                    factors.append(("E", layer, head, query, source))

    # Two-layer composed ancestry:
    # query --(layer, head)--> intermediate
    # intermediate --(layer-1, previous_head)--> previous_source
    for layer in range(1, len(attentions)):
        heads = attentions[layer].shape[0]
        previous_heads = attentions[layer - 1].shape[0]
        tokens = attentions[layer].shape[1]

        for head in range(heads):
            for query in range(1, tokens):
                for intermediate in top_sources[(layer, head, query)]:
                    if intermediate == 0:
                        continue
                    for previous_head in range(previous_heads):
                        for previous_source in top_sources.get(
                            (layer - 1, previous_head, intermediate), ()
                        ):
                            factors.append(
                                (
                                    "P",
                                    layer,
                                    head,
                                    query,
                                    intermediate,
                                    previous_head,
                                    previous_source,
                                )
                            )

    return Counter(factors)


def assign_primes(all_factors: Iterable[Factor]) -> dict[Factor, int]:
    """Assign primes deterministically by lexicographic factor representation."""
    factors = sorted(set(all_factors), key=repr)
    ps = primes(len(factors))
    return {factor: ps[i] for i, factor in enumerate(factors)}


def exact_gcd_factorization(
    counters: list[Counter[Factor]],
) -> Counter[Factor]:
    """Compute the exact GCD in factorized form across several multisets."""
    if not counters:
        return Counter()

    common = set(counters[0])
    for counter in counters[1:]:
        common &= set(counter)

    return Counter(
        {factor: min(counter[factor] for counter in counters) for factor in common}
    )


def prevalence(counters: list[Counter[Factor]]) -> dict[Factor, float]:
    """Fraction of computations containing each factor at least once."""
    if not counters:
        return {}

    seen = Counter()
    for counter in counters:
        seen.update(counter.keys())

    n = len(counters)
    return {factor: count / n for factor, count in seen.items()}
