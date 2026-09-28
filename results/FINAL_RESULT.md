# Final result (v0.2 — corrected prompts)

## Question

Can a prime-factor representation of attention structure produce reusable,
class-associated signatures for matched one-hop and two-hop computations in a
frozen transformer?

## Why v0.2 exists

The original v1 experiment was later found to contain two confounds:

1. the answer was always the target of the last fact line, so a positional
   shortcut could solve the prompt without following the arrows;
2. float16 scoring could produce non-finite values that the old code silently
   counted as wrong answers.

v0.2 adds two distractor facts, shuffles fact order, reports shortcut heuristics,
uses safer dtype selection with hard checks for non-finite values, and replaces
the permissive "any factor present" decision rule with a graded paired statistic
compared against a within-pair label-permutation null.

## Final run

Configuration:

```text
seed:          7
chains:        150
facts/prompt:  4 (2 relevant + 2 distractor)
shuffle:       yes
near-GCD:      0.80
permutations:  1000
dtype:         auto -> bfloat16
GPU:           Tesla T4
```

A chain was retained only when the same frozen model answered both its one-hop
and two-hop versions correctly. Retained chains were split by chain into
discovery and held-out test sets.

### Shortcut diagnostics

Across the 300 generated prompts:

```text
last fact target:  0.2733   (random fact-line baseline: 0.25)
first fact target: 0.2467   (random fact-line baseline: 0.25)
first chain end:    0.5133   (random chain-end baseline: 0.50)
last chain end:     0.4867   (random chain-end baseline: 0.50)
```

The positional shortcut that scored 100% in v1 is therefore absent in the
corrected prompt distribution.

## Held-out permutation result

The main statistic is paired held-out accuracy: for each test chain, the
structural score of the two-hop prompt is compared with the matched one-hop
prompt. Ties count as 0.5; chance is 0.5.

| Model | Paired-correct | Discovery | Test | Observed | Null mean | Null 95th | One-sided p |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B-Instruct` | 51 / 150 | 25 | 26 | **0.75** | 0.4997 | 0.50 | **0.000999** |
| `Qwen/Qwen2.5-3B-Instruct` | 60 / 150 | 30 | 30 | **1.00** | 0.5004 | 0.50 | **0.000999** |

With 1,000 permutations and the add-one correction used in the code,
`0.000999 = 1/1001` is the minimum attainable Monte-Carlo p-value. No sampled
null permutation matched or exceeded the observed statistic for either model.

## Legacy "any factor" rule

The old exact-GCD rule remained non-discriminative in the corrected run:

```text
Qwen2.5-1.5B: two-on-two=1.0, two-on-one=1.0, one-on-one=1.0, one-on-two=1.0
Qwen2.5-3B:   two-on-two=1.0, two-on-one=1.0, one-on-one=1.0, one-on-two=1.0
```

This confirms why that rule is retained only as a diagnostic. The reported
v0.2 result comes from the graded paired score and its permutation null.

## Interpretation

Under this corrected setup, the current structural representation — top
attention edges (`E`) plus two-layer composed ancestry paths (`P`) — contains a
held-out signal associated with one-hop versus two-hop computation among chains
that the model solved correctly in both forms. The signal appears in both tested
Qwen2.5 model sizes and is stronger in the 3B run.

This is intentionally a narrow conclusion. The experiment does **not** show
that the probe has recovered the model's full causal reasoning mechanism, and
it does not establish generalization to other tasks, seeds, model families, or
prompt distributions. Conditioning on paired-correct chains also changes the
evaluated population.

## Reproducibility

The complete corrected artifacts are in `results/final_run_v3/`:

- `summary.json`
- `shortcut_check.json`
- `environment.json`
- model-specific JSON outputs
- stdout/stderr logs

A smaller corrected pilot is preserved in `results/final_run_v2/`, and the
superseded/confounded v1 run is preserved in `results/final_run/`.
