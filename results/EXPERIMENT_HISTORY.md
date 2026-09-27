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

## What still needs one fresh run

To turn the structural probe from a research prototype into a completed
experimental result, run the current probe on one or more models that solve
enough matched chains to pass `--min-correct-pairs`.

Only then can the repository report held-out exact-GCD / near-GCD measurements.
