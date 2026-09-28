from __future__ import annotations

import argparse
import json

from .experiment import run_model


DEFAULT_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "gpt2",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether one-hop and two-hop transformer computations "
            "contain reusable structural signatures."
        )
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--min-correct-pairs", type=int, default=8)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--near-gcd", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="auto = float32 on CPU, bfloat16 on CUDA if supported (else float32)",
    )
    parser.add_argument(
        "--distractors", type=int, default=2,
        help="irrelevant arrow facts added to every prompt (0 = original prompts)",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="keep facts in chain order (with --distractors 0: original confounded prompts)",
    )
    parser.add_argument(
        "--permutations", type=int, default=1000,
        help="shuffled-label permutations for the null baseline",
    )
    parser.add_argument("--out", default="godel_probe_results.json")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    results = []
    for name in args.models:
        try:
            results.append(run_model(name, args))
        except Exception as exc:
            results.append(
                {"model": name, "status": "error", "error": repr(exc)}
            )
            print("ERROR", name, repr(exc))

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
