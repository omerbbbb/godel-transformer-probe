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

## What still needs to be run

**A rerun of the main experiment (Qwen2.5-3B-Instruct and 1.5B-Instruct)
with the de-confounded v0.2 prompts is pending.** Until then, no held-out
structural result from this repository should be treated as evidence about
one-hop vs two-hop computation.
