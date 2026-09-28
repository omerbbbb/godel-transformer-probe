# Gödel Transformer Probe

A small research prototype for testing whether **frozen pretrained transformers**
show reusable structural signatures when solving matched one-hop and two-hop
reasoning tasks.

The project borrows the idea of **Gödel numbering**: instead of representing a
computation with one enormous integer, it assigns primes to discrete structural
factors and works directly with the implied prime factorization.

> **No training or fine-tuning is performed.** The probe only inspects frozen
> pretrained models.

## Status: v0.2 rerun complete — de-confounded held-out result

A review of the original v1 experiment found two important problems: a
**positional shortcut** in the prompt format and a **float16 numerical issue**.
The repository keeps those earlier artifacts for transparency, but the main
result now comes from the corrected v0.2 setup.

The v0.2 fixes are:

1. **De-confounded prompts.** Each prompt adds two irrelevant distractor facts
   and shuffles all fact lines in a seeded order. The matched one-hop / two-hop
   prompts still share the same facts and differ only in the start letter.
2. **Shortcut check.** Every run measures several trivial heuristics before any
   model analysis. In the final 150-chain run, `last_fact_target` scored
   **27.3%** and `first_fact_target` **24.7%**, close to the 25% random-fact-line
   baseline. The old v1 `last_fact_target` shortcut scored 100%.
3. **Permutation null.** The original "any signature factor is present" rule
   was too permissive. It is retained as a diagnostic, but the main statistic
   is now a graded near-GCD score evaluated on held-out matched chains against
   a within-pair label-permutation null.
4. **Safer dtype handling.** `--dtype auto` uses bfloat16 on supported CUDA
   devices and float32 otherwise, while non-finite answer scores or attentions
   raise errors instead of being silently counted as failures.

### Corrected v0.2 result

The final rerun used **150 random chains**, seed 7, two distractor facts, shuffled
fact order, `near_gcd=0.80`, and 1,000 label permutations. Only chains solved
correctly in **both** matched forms were retained before the discovery/test split.

| Model | Paired-correct | Discovery | Test | Held-out paired accuracy | Null mean | One-sided permutation p |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B-Instruct` | 51 / 150 | 25 | 26 | **0.75** | 0.4997 | **0.000999** |
| `Qwen/Qwen2.5-3B-Instruct` | 60 / 150 | 30 | 30 | **1.00** | 0.5004 | **0.000999** |

With 1,000 permutations, `0.000999 = 1/1001` is the minimum attainable
Monte-Carlo p-value under the implemented add-one correction; no sampled null
permutation matched or exceeded the observed statistic in either run.

This is evidence that the **current graded structural score** separates the
held-out one-hop and two-hop members of solved matched chains better than the
label-permutation null in this setup. It is **not** evidence that the probe has
identified the model's full causal mechanism, and it should not be generalized
beyond the tested prompts, seed, model family, and correctness-conditioned
population. The older exact/"any factor" rule remained non-discriminative and
is kept only as a comparison.

Full corrected artifacts are in `results/final_run_v3/`; the smaller corrected
pilot is in `results/final_run_v2/`; the confounded v1 run remains in
`results/final_run/`.

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
│   ├── final_run_v3/   # corrected 150-chain main result
│   ├── final_run_v2/   # corrected 30-chain pilot
│   └── final_run/      # original v1 run (superseded/confounded)
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

Main configuration used for the corrected final run (run each model separately
if GPU memory is limited):

```bash
godel-probe \
  --models Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
  --pairs 150 \
  --min-correct-pairs 8 \
  --near-gcd 0.80 \
  --seed 7 \
  --device cuda \
  --dtype auto \
  --distractors 2 \
  --permutations 1000 \
  --out results/qwen25_v02.json
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
The final run used 26 and 30 held-out matched chains. With 1,000 permutations,
the smallest reportable p-value is `1/1001 ≈ 0.000999`.

`results/example_output.json` shows the schema only and is deliberately labeled
as illustrative rather than experimental evidence.



## Corrected v0.2 executed result

The main result is the 150-chain rerun under the de-confounded prompt format.
The exact artifacts are committed under `results/final_run_v3/`.

Environment:

```text
Python:       3.13.15
PyTorch:      2.11.0+cu128
Transformers: 5.16.1
GPU:          Tesla T4
dtype:        bfloat16
seed:         7
pairs:        150
distractors:  2
near-GCD:     0.80
permutations: 1000
```

Shortcut diagnostics on the 300 generated prompts:

| Heuristic | Accuracy | Relevant random baseline |
|---|---:|---:|
| target of last fact line | 0.2733 | 0.25 |
| target of first fact line | 0.2467 | 0.25 |
| first chain end | 0.5133 | 0.50 |
| last chain end | 0.4867 | 0.50 |

The deterministic `one_step_from_start` heuristic is 1.0 on the one-hop class
and 0.0 on the two-hop class by construction; it is included as a sanity check,
not as a class-blind shortcut.

Main permutation result:

| Model | Paired-correct | Test chains | Observed | Null mean | Null 95th | p (one-sided) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 51/150 | 26 | **0.75** | 0.4997 | 0.50 | **0.000999** |
| Qwen2.5-3B-Instruct | 60/150 | 30 | **1.00** | 0.5004 | 0.50 | **0.000999** |

The old exact/"any factor" rule still fired on both classes (1.0/1.0) and is
therefore not useful as a discriminator. The corrected conclusion comes from
the graded paired score plus the permutation null, not from that legacy rule.

### Interpretation

Within the correctness-conditioned matched-chain population used here, the
graded structural signature separated held-out one-hop and two-hop prompts
above the shuffled-label null in **both tested Qwen sizes**. The effect was
partial for 1.5B (0.75 paired accuracy) and perfect on this held-out sample for
3B (1.00).

This does **not** show that attention edges and two-layer ancestry paths are a
complete causal description of reasoning, nor does it establish generalization
to other tasks, model families, seeds, or prompt distributions. It shows that
this particular structural encoding contains a reproducible class-associated
signal under the tested setup after the obvious positional shortcut and dtype
artifact were addressed.

See `results/FINAL_RESULT.md` for the concise result statement and
`results/EXPERIMENT_HISTORY.md` for the full progression from failed/confounded
runs to the corrected experiment.

## v1 executed result (superseded)

The original v1 GPU run is kept in `results/final_run/` for auditability. It
used prompts where the answer was always the target of the last fact line and
used unsafe float16 loading. Its headline numbers (1.5B: 0/30; 3B: 30/30) are
**not treated as evidence** about multi-hop computation. The review that found
those issues motivated v0.2.

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
