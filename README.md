# Gödel Transformer Probe

A small research prototype for testing whether **frozen pretrained transformers**
show reusable structural signatures when solving matched one-hop and two-hop
reasoning tasks.

The project borrows the idea of **Gödel numbering**: instead of representing a
computation with one enormous integer, it assigns primes to discrete structural
factors and works directly with the implied prime factorization.

> **No training or fine-tuning is performed.** The probe only inspects frozen
> pretrained models.

## Status: confound found in v1, fixed in v0.2 — rerun pending

A review of the v1 result (below) found a **positional shortcut** in the
prompts. In v1 every prompt listed the facts as `A -> B` then `B -> C`, so the
correct answer was *always the last letter of the last fact line*. A model can
pass the "paired-correct" filter without following the arrows at all.

Evidence:

- the trivial heuristic "answer = target of the last fact line" scores
  **100%** on the v1 prompts (`godel-shortcut-check --distractors 0 --no-shuffle`);
- Pythia-70M answered 21/40 v1 prompts correctly, but only 4/40 when the two
  fact lines were simply swapped.

So the v1 result (Qwen2.5-3B solving 30/30 matched chains, with signatures
that did not separate one-hop from two-hop) cannot be read as evidence about
multi-hop computation: the model may have been copying the last letter in
both conditions.

Fixes in v0.2:

1. **De-confounded prompts.** Each prompt adds irrelevant distractor facts
   (default 2, forming a disjoint decoy chain with the same format and
   vocabulary) and the fact lines are shuffled in a seeded random order per
   chain. The matched one-hop / two-hop prompts still share identical fact
   lines and differ only in the start letter.
2. **Shortcut check.** Every run reports the accuracy of trivial heuristics
   (last/first fact line, first/last chain end, one step from start) next to
   their chance levels. With the new prompts, positional heuristics sit at
   chance (≈25% for a random fact line, ≈50% for a random chain end).
3. **Permutation null.** The old rule ("a signature fires if *any* of its
   factors is present") fires on nearly everything once a signature has
   hundreds of factors. It is kept for comparison, but the main statistic is
   now a graded score with a shuffled-label null (see *Statistics* below).
4. **dtype.** v1 loaded models in float16 on GPU. Qwen2.5-class models can
   overflow in float16, and NaN log-probabilities were silently counted as
   wrong answers — a likely explanation for Qwen2.5-1.5B scoring 0/30. The
   dtype is now configurable (`--dtype`, default `auto` = bfloat16 on GPUs
   that support it, float32 otherwise) and non-finite scores or attentions
   now raise an error instead of being counted as incorrect. CPU checks
   (v1 prompts, seed 7): Qwen2.5-1.5B-Instruct in float32 solved **10/10**
   matched chains, and Pythia-70M in float16 produced non-finite
   log-probabilities for every candidate, which the v1 code would have
   counted as 24 wrong answers. So the earlier `0/30` (and probably Pythia's
   `0/20`) looks like a numerical artifact, not model weakness.

**A rerun of the main experiment with the de-confounded prompts is pending.**
The v1 artifacts are kept unchanged in `results/final_run/` as a record.

## Research question

If a model solves the same underlying arrow chain in a one-hop and a two-hop
form, do the two classes of successful computations contain distinguishable,
reusable structural factors that generalize to unseen chains?

The current probe is intentionally narrow. It is a structural experiment, not a
claim that attention alone fully represents the model's causal computation.

## Method

For each random chain

```text
A -> B
B -> C
```

the code creates two matched prompts:

```text
two-hop: start at A -> answer C
one-hop: start at B -> answer C
```

Since v0.2 each prompt also contains distractor facts (a disjoint decoy
chain), and all fact lines are shuffled per chain, e.g.

```text
Follow the arrows until the chain ends. Return only the final capital letter.
Q -> M
K -> Q
T -> W
W -> F
Start: K
Final:
```

(answer `M`; `T -> W -> F` is a decoy). `--distractors 0 --no-shuffle`
reproduces the original v1 prompts exactly.

A chain is kept only when the **same frozen model answers both versions
correctly**. This avoids comparing a successful computation with a failed one.

For every retained prompt, the probe extracts attention and defines two
structural factor types:

```text
E = local attention edge
    (layer, head, query position, key position)

P = two-layer composed ancestry path
    query -> intermediate -> previous source
```

The factor definitions do **not** include task labels or token identities.

Each distinct factor receives a prime deterministically. The multiset of factors
is therefore treated as the factorization of an implicit Gödel number:

```text
computation structure
        |
        v
{factor_1, factor_2, ...}
        |
        v
{prime_1^k1, prime_2^k2, ...}
```

The astronomical integer itself is never constructed.

## Discovery / test split

Retained chains are split **by chain**, not by individual query, so the matched
one-hop and two-hop forms of a chain cannot leak across the split.

The discovery half is used to construct:

1. an **exact GCD signature**: factors shared across every computation in a
   class;
2. an exploratory **near-GCD signature**: factors appearing in at least a fixed
   fraction of one class and rarely in the other.

Those signatures are then frozen and evaluated on unseen chains.

## Repository structure

```text
.
├── src/godel_probe/
│   ├── cli.py          # command-line interface
│   ├── experiment.py   # model loading, scoring and experiment loop
│   ├── factors.py      # structural factors, primes, GCD and prevalence
│   ├── prompts.py      # matched one-hop / two-hop dataset (+ distractors, shuffling)
│   ├── shortcuts.py    # trivial-heuristic accuracy ("shortcut check")
│   └── stats.py        # graded signature score + permutation null
├── tests/              # deterministic unit tests
├── notebooks/
│   ├── godel_forward_trace_v3.ipynb
│   └── executed/
│       ├── godel_probe_pythia70m_executed.ipynb
│       └── godel_forward_trace_v2_pythia_executed.ipynb
├── results/
│   ├── FINAL_RESULT.md
│   ├── EXPERIMENT_HISTORY.md
│   ├── pythia70m_probe.json
│   ├── README.md
│   ├── example_output.json
│   └── final_run/
│       ├── summary.json
│       ├── environment.json
│       ├── Qwen__Qwen2.5-1.5B-Instruct.json
│       ├── Qwen__Qwen2.5-3B-Instruct.json
│       ├── *.log
│       └── godel_probe_used.py
└── archive/            # original / failed research attempts
```

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/omerbbbb/godel-transformer-probe.git
cd godel-transformer-probe

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
```

For development/tests:

```bash
pip install -e ".[dev]"
pytest
```

## Run the probe

Cheap first run:

```bash
godel-probe   --models EleutherAI/pythia-70m   --pairs 20   --out results/pythia70m.json
```

Cross-model run:

```bash
godel-probe   --models     EleutherAI/pythia-70m     EleutherAI/pythia-160m     gpt2   --pairs 40   --out results/cross_model.json
```

Main configuration (de-confounded prompts, GPU):

```bash
godel-probe \
  --models Qwen/Qwen2.5-3B-Instruct \
  --pairs 30 \
  --min-correct-pairs 8 \
  --near-gcd 0.80 \
  --seed 7 \
  --device cuda \
  --permutations 1000 \
  --out results/qwen25_3b.json
```

Reproduce the v1 (confounded) configuration: add `--distractors 0 --no-shuffle --dtype float16`.

Check trivial heuristics on the prompt set without loading a model:

```bash
godel-shortcut-check --pairs 200                            # v0.2 prompts
godel-shortcut-check --pairs 200 --distractors 0 --no-shuffle   # v1 prompts
```

You can also run:

```bash
python -m godel_probe --models gpt2 --pairs 20
```

## Main CLI arguments

| Argument | Meaning | Default |
|---|---|---:|
| `--models` | Hugging Face causal-LM model names | Pythia 70M, Pythia 160M, GPT-2 |
| `--pairs` | Number of random arrow chains | `40` |
| `--min-correct-pairs` | Minimum paired-correct chains required | `8` |
| `--topk` | Strongest attention sources retained per query/head | `1` |
| `--near-gcd` | Near-GCD prevalence threshold | `0.80` |
| `--seed` | Dataset/split random seed | `7` |
| `--device` | PyTorch device | `cpu` |
| `--dtype` | `auto`, `float32`, `bfloat16`, `float16` (`auto` = float32 on CPU, bfloat16 on CUDA if supported) | `auto` |
| `--distractors` | Irrelevant arrow facts per prompt (0 = v1 prompts) | `2` |
| `--no-shuffle` | Keep facts in chain order | off |
| `--permutations` | Shuffled-label permutations for the null baseline | `1000` |
| `--out` | JSON output path | `godel_probe_results.json` |

## Output

Each model returns one of three statuses:

- `ok` — enough matched chains were solved and the probe ran;
- `too_weak` — too few chains were solved in both forms;
- `error` — loading or attention extraction failed.

Every result records `config`, `environment` (versions, device, dtype, GPU)
and `shortcut_check`. For successful runs, the JSON also contains:

- number of paired-correct chains;
- discovery/test split sizes;
- `permutation_test` — the main statistic with its null distribution;
- `exact_any_factor_rule` / `near_any_factor_rule` — the v1 "any factor" rates, kept for comparison;
- number of class-specific factors;
- a few example prime assignments.

## Statistics

Signatures are built on the discovery chains as before (near-GCD: factors
present in ≥ threshold of one class and ≤ 1 − threshold of the other). Each
held-out prompt then gets a graded score

```text
score(x) = fraction of two-hop signature factors present in x
         − fraction of one-hop signature factors present in x
```

The statistic is the **paired held-out accuracy**: the fraction of test
chains whose two-hop prompt scores higher than its matched one-hop prompt
(ties count ½; chance = 0.5).

The null distribution comes from swapping the one/two labels at random
within each discovery chain, rebuilding the signatures, and recomputing the
statistic on the unchanged test set (default 1,000 permutations). The JSON
reports the observed value, null mean / SD / 95th percentile, a one-sided
p-value `(1 + #null ≥ observed) / (1 + N)`, and the effect over the null.
With only ~15 test chains, p-values are coarse; treat this as a first check,
not a definitive test.

`results/example_output.json` shows the schema only and is deliberately labeled
as illustrative rather than experimental evidence.



## v1 executed result (confounded prompts — superseded)

> This run used the v1 prompts, which contain the positional shortcut described
> at the top of this README. It is kept as a record, not as a valid result.

A final GPU run was performed with the exact probe preserved in
`results/final_run/godel_probe_used.py`.

Environment:

```text
Python:       3.13.15
PyTorch:      2.11.0+cu128
Transformers: 5.17.0
GPU:          NVIDIA RTX PRO 6000 Blackwell Server Edition
seed:         7
pairs:        30
near-GCD:     0.80
```

Two frozen instruction-tuned models were attempted:

| Model | Paired-correct chains | Result |
|---|---:|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 0 / 30 | stopped as `too_weak` |
| `Qwen/Qwen2.5-3B-Instruct` | 30 / 30 | reached held-out structural evaluation |

### Qwen2.5-3B-Instruct

The 3B model passed the fairness filter on all 30 chains, so the experiment
split the retained chains into discovery and held-out test sets and evaluated
the frozen signatures.

Exact-GCD test:

| Signature | Test on two-hop | Test on one-hop |
|---|---:|---:|
| two-hop discovery signature | 1.000 | 1.000 |
| one-hop discovery signature | 1.000 | 1.000 |

The discovery GCDs contained 1,490 factors unique to the two-hop discovery GCD
and 1,463 unique to the one-hop discovery GCD. However, those signatures fired
on **100% of both held-out classes**. In other words, being unique to one
*discovery GCD* did not make a factor signature class-specific on unseen
computations.

Near-GCD test at threshold `0.80`:

| Signature | Same-class test rate | Other-class test rate |
|---|---:|---:|
| two-hop near-GCD | 1.000 | 1.000 |
| one-hop near-GCD | 1.000 | 0.867 |

Using the same "contains any signature factor" rule, the near-GCD sets also failed to produce a useful held-out separator.
The one-hop signature showed only weak asymmetry, while the two-hop signature
appeared in every test example from both classes.

### Conclusion

**The current structural representation did not distinguish one-hop from
two-hop computations on held-out chains.**

This is a negative result, but it is informative: the experiment successfully
reached held-out evaluation after controlling for model correctness, and the
result shows that top-attention edges plus two-layer ancestry paths are too
shared across these matched prompts to act as class-specific signatures under
the present GCD rules.

Because of the prompt shortcut found later, it is also possible that the
model solved both forms the same way (copying the last letter), which would
by itself make the structures indistinguishable.

The result does **not** show that the model has no structural difference between
one-hop and two-hop reasoning. It only shows that this particular factorization
and signature test did not isolate one.

The complete JSON outputs, logs, environment metadata, and exact script used for
the run are committed under `results/final_run/`.

## Earlier executed experiments

The project history also preserves the failed attempts that led to the final
run.

### Pythia-70M structural probe

The original Colab probe returned:

```text
paired-correct chains: 0/20
status: too_weak
```

so it correctly stopped before structural evaluation.

(Later re-check: this script loaded the model with `torch_dtype="auto"`;
Pythia-70M in float16 yields non-finite scores, and the same v1 prompts in
float32 give 7/20 paired-correct. See `results/EXPERIMENT_HISTORY.md`.)

### Earlier Pythia forward trace

A direct Hugging Face forward pass produced a recursive trace with **2,536
nodes** and final node `G2535`, but later attention layers contained `NaN`
values. That run is preserved as a diagnostic artifact rather than treated as a
valid numerical result.

See `results/EXPERIMENT_HISTORY.md` for the full sequence of experiments.

## Why keep only paired-correct chains?

Suppose the model solves a one-hop question but fails the matched two-hop
question. Differences in attention could then reflect **success vs. failure**
rather than **one-hop vs. two-hop computation**.

Conditioning on both forms being correct does not solve every confound, but it
makes the comparison substantially cleaner.

## Why this is only a first probe

The current representation is deliberately simple:

- it uses attention structure, not the full residual-stream computation;
- it retains only the top-k attention sources and discards most magnitudes;
- the two-layer `P` factor captures only shallow ancestry;
- factors are architecture/sequence-position specific;
- the near-GCD threshold is a fixed heuristic, not a learned classifier;
- conditioning on paired-correct examples changes the evaluated population;
- cross-model runs should not be interpreted as identical prime identities
  across architectures.

A stronger follow-up would add MLP/residual contributions, causal ablations,
longer ancestry paths, repeated splits/seeds, and explicit null distributions.

## Notebook

`notebooks/godel_forward_trace_v3.ipynb` is the exploratory forward-trace
notebook that motivated the structural representation. It runs a real frozen
model forward pass and builds a recursive graph whose nodes record operation
type and parent computations.

The production experiment is the package under `src/`; the notebook is kept as
a transparent research artifact rather than as the main implementation.

## Project status

**Research prototype / portfolio project.**

The repository is meant to demonstrate experimental design, transformer
instrumentation, deterministic structural encoding, held-out evaluation, and
careful separation between an interesting probe and a stronger scientific
claim.
