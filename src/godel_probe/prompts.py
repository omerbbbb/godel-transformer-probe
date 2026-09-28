from __future__ import annotations

import random
from typing import Any


LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")

INSTRUCTION = (
    "Follow the arrows until the chain ends. "
    "Return only the final capital letter.\n"
)


def _distractor_facts(
    rng: random.Random, used: set[str], n: int
) -> list[tuple[str, str]]:
    """Create ``n`` irrelevant arrow facts that never touch the real chain.

    Distractors are emitted as disjoint decoy chains of length two
    (``X -> Y``, ``Y -> Z``), plus a single edge when ``n`` is odd, so the
    prompt contains other chain-ends and other two-step chains with the same
    format and vocabulary. Their letters are disjoint from the real chain,
    so following the arrows from the real start never reaches them.
    """
    pool = [letter for letter in LETTERS if letter not in used]
    facts: list[tuple[str, str]] = []
    remaining = n
    while remaining > 0:
        length = 2 if remaining >= 2 else 1
        needed = length + 1
        if len(pool) < needed:
            raise ValueError("too many distractors for the letter vocabulary")
        letters = rng.sample(pool, needed)
        for letter in letters:
            pool.remove(letter)
        for i in range(length):
            facts.append((letters[i], letters[i + 1]))
        remaining -= length
    return facts


def make_pairs(
    n: int = 80,
    seed: int = 0,
    distractors: int = 2,
    shuffle: bool = True,
) -> list[dict[str, Any]]:
    """Create matched one-hop and two-hop arrow-chain prompts.

    Each generated chain A -> B -> C yields two questions that share exactly
    the same fact lines (same order, same distractors) and differ only in the
    start letter:

    * two-hop: start at A, answer C
    * one-hop: start at B, answer C

    ``distractors`` irrelevant facts are added and, when ``shuffle`` is true,
    the fact lines are put in a seeded random order per chain. This removes
    the positional shortcut of the original prompts, in which the answer was
    always the last letter of the last fact line.

    ``distractors=0, shuffle=False`` reproduces the original (confounded)
    prompts exactly.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for chain in range(n):
        a, b, c = rng.sample(LETTERS, 3)
        facts = [(a, b), (b, c)]
        if distractors:
            facts += _distractor_facts(rng, {a, b, c}, distractors)
        if shuffle:
            rng.shuffle(facts)

        fact_text = "\n".join(f"{src} -> {dst}" for src, dst in facts)
        base = INSTRUCTION + fact_text + "\n"
        for kind, start in (("two", a), ("one", b)):
            rows.append(
                {
                    "kind": kind,
                    "chain": chain,
                    "prompt": base + f"Start: {start}\nFinal:",
                    "answer": c,
                    "triple": [a, b, c],
                    "start": start,
                    "facts": [list(fact) for fact in facts],
                }
            )

    return rows


def follow_chain(facts: list[list[str]] | list[tuple[str, str]], start: str) -> str:
    """Ground-truth solver: follow arrows from ``start`` until the chain ends."""
    nxt = {src: dst for src, dst in facts}
    current, seen = start, {start}
    while current in nxt:
        current = nxt[current]
        if current in seen:
            raise ValueError("cycle in facts")
        seen.add(current)
    return current
