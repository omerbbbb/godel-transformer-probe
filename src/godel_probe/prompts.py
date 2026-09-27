from __future__ import annotations

import random
from typing import Any


LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")


def make_pairs(n: int = 80, seed: int = 0) -> list[dict[str, Any]]:
    """Create matched one-hop and two-hop arrow-chain prompts.

    Each generated chain A -> B -> C yields two questions that share the same
    facts but start at different locations:

    * two-hop: start at A, answer C
    * one-hop: start at B, answer C
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for _ in range(n):
        a, b, c = rng.sample(LETTERS, 3)
        facts = f"{a} -> {b}\n{b} -> {c}"
        base = (
            "Follow the arrows until the chain ends. "
            "Return only the final capital letter.\n"
            + facts
            + "\n"
        )
        rows.append(
            {
                "kind": "two",
                "prompt": base + f"Start: {a}\nFinal:",
                "answer": c,
                "triple": [a, b, c],
            }
        )
        rows.append(
            {
                "kind": "one",
                "prompt": base + f"Start: {b}\nFinal:",
                "answer": c,
                "triple": [a, b, c],
            }
        )

    return rows
