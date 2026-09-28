from collections import Counter

from godel_probe.prompts import follow_chain, make_pairs
from godel_probe.shortcuts import shortcut_report


def test_make_pairs_is_deterministic():
    assert make_pairs(5, seed=11) == make_pairs(5, seed=11)


def test_each_chain_has_one_and_two_hop_forms():
    rows = make_pairs(4, seed=3)
    assert len(rows) == 8

    grouped = {}
    for row in rows:
        grouped.setdefault(row["chain"], set()).add(row["kind"])

    assert all(kinds == {"one", "two"} for kinds in grouped.values())


def test_matched_prompts_differ_only_in_start():
    rows = make_pairs(20, seed=5)
    for two, one in zip(rows[::2], rows[1::2]):
        assert two["kind"] == "two" and one["kind"] == "one"
        assert two["facts"] == one["facts"]
        a, b, c = two["triple"]
        assert two["prompt"].replace(f"Start: {a}\n", f"Start: {b}\n") == one["prompt"]


def test_answer_follows_arrows_and_distractors_are_irrelevant():
    rows = make_pairs(50, seed=1, distractors=3)
    for row in rows:
        assert len(row["facts"]) == 2 + 3
        assert follow_chain(row["facts"], row["start"]) == row["answer"]
        chain_letters = set(row["triple"])
        distractors = [f for f in row["facts"]
                       if f not in ([row["triple"][0], row["triple"][1]],
                                    [row["triple"][1], row["triple"][2]])]
        assert len(distractors) == 3
        for src, dst in distractors:
            assert src not in chain_letters and dst not in chain_letters


def test_legacy_mode_reproduces_original_prompts():
    rows = make_pairs(3, seed=7, distractors=0, shuffle=False)
    a, b, c = rows[0]["triple"]
    assert rows[0]["prompt"] == (
        "Follow the arrows until the chain ends. "
        "Return only the final capital letter.\n"
        f"{a} -> {b}\n{b} -> {c}\nStart: {a}\nFinal:"
    )


def _answer_line(row):
    return next(i for i, (src, dst) in enumerate(row["facts"]) if dst == row["answer"])


def test_legacy_prompts_have_constant_answer_position():
    # Documents the confound that motivated the fix.
    rows = make_pairs(100, seed=7, distractors=0, shuffle=False)
    assert {_answer_line(r) for r in rows} == {1}
    report = shortcut_report(rows)
    assert report["heuristics"]["last_fact_target"]["all"] == 1.0


def test_answer_position_is_not_constant():
    rows = make_pairs(400, seed=7)
    positions = Counter(_answer_line(r) for r in rows[::2])
    n_facts = len(rows[0]["facts"])
    assert set(positions) == set(range(n_facts))
    # roughly uniform: every line position holds the answer 15-35% of the time
    assert all(0.15 < positions[i] / 400 < 0.35 for i in range(n_facts))


def test_positional_heuristics_near_chance_after_fix():
    rows = make_pairs(400, seed=7)
    report = shortcut_report(rows)
    chance_line = report["chance_uniform_fact_line"]
    chance_end = report["chance_uniform_chain_end"]
    h = report["heuristics"]
    for name in ("last_fact_target", "first_fact_target"):
        assert abs(h[name]["all"] - chance_line) < 0.08, (name, h[name])
    for name in ("first_chain_end", "last_chain_end"):
        assert abs(h[name]["all"] - chance_end) < 0.1, (name, h[name])
    assert h["last_fact_source"]["all"] == 0.0
    # taking a single step solves one-hop but never two-hop
    assert h["one_step_from_start"]["one"] == 1.0
    assert h["one_step_from_start"]["two"] == 0.0
