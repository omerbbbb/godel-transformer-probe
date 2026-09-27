from __future__ import annotations

import random
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .factors import (
    assign_primes,
    exact_gcd_factorization,
    prevalence,
    structural_factors,
)
from .prompts import LETTERS, make_pairs


def load_model(name: str, device: torch.device):
    """Load a frozen causal language model with eager attention when supported."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    kwargs = {"torch_dtype": (torch.float16 if device.type == "cuda" else torch.float32)}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)

    model.to(device).eval()
    return tokenizer, model


def sequence_logprob(model, tokenizer, prompt: str, answer: str, device) -> float:
    """Score the complete answer string without sampling or generation randomness."""
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    answer_tokens = tokenizer(
        " " + answer, return_tensors="pt", add_special_tokens=False
    )

    input_ids = torch.cat(
        [prompt_tokens.input_ids, answer_tokens.input_ids], dim=1
    ).to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    prompt_length = prompt_tokens.input_ids.shape[1]
    score = 0.0

    for j in range(answer_tokens.input_ids.shape[1]):
        position = prompt_length + j - 1
        target = answer_tokens.input_ids[0, j].to(device)
        score += torch.log_softmax(logits[0, position], dim=-1)[target].item()

    return score


def answer_correct(model, tokenizer, row, device, candidates=LETTERS):
    """Return whether the highest-log-probability candidate is correct."""
    scores = {
        candidate: sequence_logprob(
            model, tokenizer, row["prompt"], candidate, device
        )
        for candidate in candidates
    }
    prediction = max(scores, key=scores.get)
    return prediction == row["answer"], prediction, scores[row["answer"]]


def get_attentions(model, tokenizer, prompt: str, device):
    """Extract per-layer attention tensors for a prompt."""
    encoded = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)

    with torch.no_grad():
        output = model(
            **encoded,
            output_attentions=True,
            use_cache=False,
        )

    if output.attentions is None:
        raise RuntimeError("Model did not return attentions")

    attentions = [
        tensor[0].detach().float().cpu() for tensor in output.attentions
    ]
    return attentions, encoded.input_ids[0].cpu()


def _rate(counters, signature):
    if not counters:
        return None
    return sum(any(factor in counter for factor in signature) for counter in counters) / len(counters)


def run_model(name: str, args) -> dict[str, Any]:
    """Run the complete discovery/test structural probe for one model."""
    device = torch.device(args.device)
    print(f"\n=== {name} ===")

    tokenizer, model = load_model(name, device)
    rows = make_pairs(args.pairs, args.seed)

    # Fairness filter:
    # A chain is retained only when the frozen model solves BOTH matched forms.
    by_triple: dict[tuple[str, str, str], list[tuple[int, dict]]] = {}
    for i, row in enumerate(rows):
        by_triple.setdefault(tuple(row["triple"]), []).append((i, row))

    kept_indices: list[int] = []
    for pair in by_triple.values():
        pair_ok = []
        for _, row in pair:
            ok, _, _ = answer_correct(model, tokenizer, row, device)
            pair_ok.append(ok)
        if all(pair_ok):
            kept_indices.extend(i for i, _ in pair)

    paired_correct = len(kept_indices) // 2
    print(f"paired-correct chains: {paired_correct}/{args.pairs}")

    if paired_correct < args.min_correct_pairs:
        return {
            "model": name,
            "status": "too_weak",
            "paired_correct": paired_correct,
        }

    selected = [(i, rows[i]) for i in kept_indices]
    random.Random(args.seed + 1).shuffle(selected)

    # Split by chain, not by individual question, preventing matched-pair leakage.
    triples = []
    seen = set()
    for _, row in selected:
        triple = tuple(row["triple"])
        if triple not in seen:
            seen.add(triple)
            triples.append(triple)

    cut = max(1, len(triples) // 2)
    discover = set(triples[:cut])
    test = set(triples[cut:])

    records = []
    for j, (_, row) in enumerate(selected, 1):
        attentions, _ = get_attentions(model, tokenizer, row["prompt"], device)
        records.append(
            {
                "kind": row["kind"],
                "triple": tuple(row["triple"]),
                "factors": structural_factors(attentions, args.topk),
            }
        )
        if j % 10 == 0:
            print(f"  extracted {j}/{len(selected)}")

    all_factors = set()
    for record in records:
        all_factors.update(record["factors"])

    prime_map = assign_primes(all_factors)

    def subset(split: str, kind: str):
        pool = discover if split == "discover" else test
        return [
            record["factors"]
            for record in records
            if record["triple"] in pool and record["kind"] == kind
        ]

    discover_one = subset("discover", "one")
    discover_two = subset("discover", "two")
    test_one = subset("test", "one")
    test_two = subset("test", "two")

    gcd_one = exact_gcd_factorization(discover_one)
    gcd_two = exact_gcd_factorization(discover_two)

    unique_one = set(gcd_one) - set(gcd_two)
    unique_two = set(gcd_two) - set(gcd_one)

    exact = {
        "two_on_two": _rate(test_two, unique_two),
        "two_on_one": _rate(test_one, unique_two),
        "one_on_one": _rate(test_one, unique_one),
        "one_on_two": _rate(test_two, unique_one),
        "gcd_one_factors": len(gcd_one),
        "gcd_two_factors": len(gcd_two),
        "unique_one_factors": len(unique_one),
        "unique_two_factors": len(unique_two),
    }

    prevalence_one = prevalence(discover_one)
    prevalence_two = prevalence(discover_two)
    threshold = args.near_gcd

    near_two = {
        factor
        for factor, value in prevalence_two.items()
        if value >= threshold
        and prevalence_one.get(factor, 0.0) <= 1.0 - threshold
    }
    near_one = {
        factor
        for factor, value in prevalence_one.items()
        if value >= threshold
        and prevalence_two.get(factor, 0.0) <= 1.0 - threshold
    }

    near = {
        "threshold": threshold,
        "two_on_two": _rate(test_two, near_two),
        "two_on_one": _rate(test_one, near_two),
        "one_on_one": _rate(test_one, near_one),
        "one_on_two": _rate(test_two, near_one),
        "near_one_factors": len(near_one),
        "near_two_factors": len(near_two),
    }

    sample_two = sorted(list(near_two or unique_two), key=repr)[:8]
    prime_examples = [
        {"prime": prime_map[factor], "factor": repr(factor)}
        for factor in sample_two
    ]

    print("exact gcd test:", exact)
    print("near-gcd test:", near)

    return {
        "model": name,
        "status": "ok",
        "paired_correct": paired_correct,
        "discover_chains": len(discover),
        "test_chains": len(test),
        "exact": exact,
        "near": near,
        "prime_examples": prime_examples,
        "note": (
            "E=(local attention edge); P=(two-layer composed ancestry path). "
            "No task labels or token identities enter the factor definition."
        ),
    }
