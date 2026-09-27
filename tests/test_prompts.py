from godel_probe.prompts import make_pairs


def test_make_pairs_is_deterministic():
    assert make_pairs(5, seed=11) == make_pairs(5, seed=11)


def test_each_chain_has_one_and_two_hop_forms():
    rows = make_pairs(4, seed=3)
    assert len(rows) == 8

    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row["triple"]), set()).add(row["kind"])

    assert all(kinds == {"one", "two"} for kinds in grouped.values())
