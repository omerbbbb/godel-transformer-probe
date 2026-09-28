"""Shortcut check: accuracy of trivial, non-reasoning heuristics.

If any of these heuristics solves the generated prompts well above its
chance level, a model can pass the correctness filter without following the
arrows, and structural comparisons between one-hop and two-hop prompts are
confounded.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from .prompts import LETTERS, make_pairs


def _last_fact_target(row):
    return row["facts"][-1][1]


def _first_fact_target(row):
    return row["facts"][0][1]


def _last_fact_source(row):
    return row["facts"][-1][0]


def _first_chain_end(row):
    """First letter (in reading order) that is a target but never a source."""
    sources = {src for src, _ in row["facts"]}
    for _, dst in row["facts"]:
        if dst not in sources:
            return dst
    return None


def _last_chain_end(row):
    sources = {src for src, _ in row["facts"]}
    ends = [dst for _, dst in row["facts"] if dst not in sources]
    return ends[-1] if ends else None


def _one_step_from_start(row):
    """Take one arrow from the start (correct for one-hop, wrong for two-hop)."""
    for src, dst in row["facts"]:
        if src == row["start"]:
            return dst
    return None


HEURISTICS: dict[str, Callable[[dict], Any]] = {
    "last_fact_target": _last_fact_target,
    "first_fact_target": _first_fact_target,
    "last_fact_source": _last_fact_source,
    "first_chain_end": _first_chain_end,
    "last_chain_end": _last_chain_end,
    "one_step_from_start": _one_step_from_start,
}


def shortcut_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy of each heuristic, overall and per kind, with chance levels."""
    n_facts = len(rows[0]["facts"]) if rows else 0
    sources = {src for src, _ in rows[0]["facts"]} if rows else set()
    n_ends = sum(1 for _, dst in rows[0]["facts"] if dst not in sources) if rows else 0
    out: dict[str, Any] = {
        "n_prompts": len(rows),
        "n_facts_per_prompt": n_facts,
        "chance_uniform_letter": round(1 / len(LETTERS), 4),
        "chance_uniform_fact_line": round(1 / n_facts, 4) if n_facts else None,
        "chance_uniform_chain_end": round(1 / n_ends, 4) if n_ends else None,
        "heuristics": {},
    }
    for name, fn in HEURISTICS.items():
        stats = {}
        for kind in ("all", "one", "two"):
            subset = [r for r in rows if kind == "all" or r["kind"] == kind]
            if subset:
                stats[kind] = round(
                    sum(fn(r) == r["answer"] for r in subset) / len(subset), 4
                )
        out["heuristics"][name] = stats
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Report trivial-heuristic accuracy on the prompt set.")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args(argv)
    rows = make_pairs(args.pairs, args.seed, args.distractors, not args.no_shuffle)
    print(json.dumps(shortcut_report(rows), indent=2))


if __name__ == "__main__":
    main()
