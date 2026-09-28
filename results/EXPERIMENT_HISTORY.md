# Executed experiment history

This file documents the **actual Colab runs recovered from the original
notebooks**. It separates executed evidence from later unexecuted code.

## 1. Initial TransformerLens attempt — failed during model loading

Notebook:

`archive/godel_forward_trace_v1_failed.ipynb`

Model:

`pythia-70m`

The first implementation used `TransformerLens`. Model loading failed with:

```text
AttributeError: 'GPTNeoXForCausalLM' object has no attribute 'embed_out'
```

No forward-pass result was produced from this version.

## 2. Direct Hugging Face Pythia forward trace — executed, but numerically invalid in later layers

Notebook:

`notebooks/executed/godel_forward_trace_v2_pythia_executed.ipynb`

Model:

`EleutherAI/pythia-70m`

Prompt:

```text
The dog chased the red ball.
```

Observed model/trace output:

```text
device: cpu
tokens: The, dog, chased, the, red, ball, .
next-token prediction: <|endoftext|>
layers with attention: 6
attention shape layer 0: (1, 8, 7, 7)
hidden-state tensors: 7

Gödel nodes created: 2536
final node: G2535
top-level formal code:
G_final = 2^12 * 3^G2534
```

The notebook also verified that attention rows in layers 0 and 1 summed to 1.0.

However, the sanity check exposed `NaN` attention values:

- layer 3: several heads returned `NaN`;
- layers 4 and 5: all displayed heads returned `NaN`.

Therefore this run is preserved as an important diagnostic step, but it is **not
treated as a valid numerical attention result**.

It motivated replacing this Pythia trace with a later `distilgpt2` version using
explicit float32/eager attention checks.

## 3. Structural Gödel probe on Pythia-70M — executed and stopped by fairness filter

Notebook:

`notebooks/executed/godel_probe_pythia70m_executed.ipynb`

Command:

```bash
python godel_probe.py   --models EleutherAI/pythia-70m   --pairs 20   --out /content/godel_probe_results.json
```

Actual output:

```text
=== EleutherAI/pythia-70m ===
paired-correct chains: 0/20

Saved: /content/godel_probe_results.json
```

Saved JSON:

```json
[
  {
    "model": "EleutherAI/pythia-70m",
    "status": "too_weak",
    "paired_correct": 0
  }
]
```

Interpretation:

The experiment intentionally keeps a chain only when the frozen model solves
**both** the one-hop and two-hop versions. Pythia-70M passed this filter on
0/20 chains, so there was no defensible discovery/test set and the code
correctly stopped before computing class-specific exact-GCD or near-GCD
signatures.

This is a negative/diagnostic result, not evidence for or against the existence
of a useful structural signature.

## 4. Later distilgpt2 trace code

`notebooks/godel_forward_trace_v3.ipynb` contains the later trace design that
switches to `distilgpt2`, checks that all attention matrices are finite, and
retains the strongest attention sources per head.

The preserved copy available to this repository does **not** contain executed
outputs, so the README does not claim a successful V3 run.

## 5. Final v1 GPU run (Qwen2.5) — later found to be confounded

See `FINAL_RESULT.md` and `final_run/`. Qwen2.5-1.5B-Instruct: 0/30
paired-correct; Qwen2.5-3B-Instruct: 30/30, with exact-GCD / near-GCD
signatures that did not separate one-hop from two-hop held-out prompts.

## 6. Review: positional shortcut and float16 issue

A later review found two problems with the v1 setup:

1. **Positional shortcut.** In every v1 prompt the answer was the last letter
   of the last fact line. The heuristic "answer = target of the last fact
   line" scores 100% on v1 prompts; Pythia-70M dropped from 21/40 to 4/40
   correct when the two fact lines were swapped. Passing the paired-correct
   filter therefore does not show that a model followed the arrows.
2. **float16.** v1 loaded models in float16 on GPU (and `torch_dtype="auto"`
   in the original Colab script). NaN log-probabilities were silently counted
   as wrong answers. Re-checks on CPU with the v1 prompts, seed 7:
   Qwen2.5-1.5B-Instruct in float32 → 10/10 paired-correct;
   Pythia-70M in float16 → non-finite scores for all candidates;
   Pythia-70M in float32 → 7/20 paired-correct (vs the 0/20 recorded above).

v0.2 adds distractor facts, per-chain shuffled fact order, a shortcut check,
a permutation-null statistic, configurable dtype (default bfloat16/float32)
and hard errors on non-finite values. With the new prompts, Pythia-70M passes
only 1/20 chains (6/60 at a larger sample), consistent with the v1 passes
relying on the shortcut.

## 7. Corrected v0.2 pilot — 30 chains

The first rerun after fixing the prompt shortcut and dtype handling used the
v0.2 prompt generator with two distractor facts, shuffled fact order, and a
1,000-permutation null.

- Qwen2.5-1.5B-Instruct: 10/30 paired-correct, 5 held-out chains,
  observed paired accuracy 1.00, one-sided `p = 0.0609`.
- Qwen2.5-3B-Instruct: 13/30 paired-correct, 7 held-out chains,
  observed paired accuracy 1.00, one-sided `p = 0.0659`.

The shortcut checks were near their random baselines, but the held-out sets were
too small for a stable conclusion. The pilot artifacts are preserved in
`final_run_v2/` and motivated a larger rerun without changing the analysis.

## 8. Corrected v0.2 final rerun — 150 chains

The same corrected analysis was rerun with 150 chains on a Tesla T4 using
bfloat16, seed 7, two distractors, shuffled fact order, near-GCD threshold 0.80,
and 1,000 permutations.

Shortcut diagnostics across 300 prompts:

```text
last fact target:  0.2733  (random fact-line baseline 0.25)
first fact target: 0.2467  (random fact-line baseline 0.25)
first chain end:    0.5133  (random chain-end baseline 0.50)
last chain end:     0.4867  (random chain-end baseline 0.50)
```

Model results:

```text
Qwen2.5-1.5B-Instruct
  paired-correct: 51/150
  discovery/test: 25/26
  paired held-out accuracy: 0.75
  null mean: 0.4996923
  p(one-sided): 0.000999

Qwen2.5-3B-Instruct
  paired-correct: 60/150
  discovery/test: 30/30
  paired held-out accuracy: 1.00
  null mean: 0.5003667
  p(one-sided): 0.000999
```

The old exact-GCD "any factor" rule still fired on both classes and remains a
non-discriminator. The corrected main result therefore relies on the graded
paired score plus the within-discovery-pair label-permutation null.

With 1,000 permutations, `0.000999 = 1/1001` is the minimum p-value reportable
under the add-one correction implemented by the code; no sampled null
permutation reached the observed statistic in either model.

The conclusion is deliberately limited: the structural representation contains
a class-associated held-out signal in this correctness-conditioned setup. It is
not claimed to recover a complete causal mechanism or to generalize beyond the
tested task, seed, and Qwen2.5 model sizes.
