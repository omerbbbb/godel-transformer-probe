
# Final result

## Question

Can a prime-factor representation of attention structure produce reusable,
class-specific signatures for matched one-hop and two-hop computations in a
frozen transformer?

## Final run

The final run used 30 matched chains and a pair-level fairness filter: a chain
was retained only when the same frozen model solved both the one-hop and
two-hop versions correctly.

### Model screening

- `Qwen/Qwen2.5-1.5B-Instruct`: **0/30 paired-correct** → `too_weak`.
- `Qwen/Qwen2.5-3B-Instruct`: **30/30 paired-correct** → full evaluation.

## Held-out result on Qwen2.5-3B-Instruct

### Exact GCD

```text
two-hop signature on two-hop test: 1.000
two-hop signature on one-hop test: 1.000
one-hop signature on one-hop test: 1.000
one-hop signature on two-hop test: 1.000
```

The discovery GCDs had 1,490 factors unique to the two-hop GCD and 1,463 unique
to the one-hop GCD, but at least one factor from each signature set appeared in every held-out example from
both classes.

### Near-GCD (threshold = 0.80)

```text
two-hop signature on two-hop test: 1.000
two-hop signature on one-hop test: 1.000
one-hop signature on one-hop test: 1.000
one-hop signature on two-hop test: 0.867
```

## Interpretation

The current representation — top attention edges (`E`) plus two-layer composed
ancestry paths (`P`) — **did not yield a useful held-out discriminator** between
one-hop and two-hop computations.

This is a negative result, not a failed run. The 3B model solved all matched
chains, the discovery/test protocol executed fully, and the held-out test showed
that the candidate signatures were too broadly shared to be class-specific.

The result does not establish that the underlying computations are identical.
It only rejects this particular structural encoding + GCD signature rule as a
successful separator in the tested setup.

## Reproducibility

The exact artifacts from the run are in `results/final_run/`:

- `summary.json`
- model-specific JSON outputs
- stdout/stderr logs
- `environment.json`
- `godel_probe_used.py`

Environment:

```text
Python 3.13.15
PyTorch 2.11.0+cu128
Transformers 5.17.0
NVIDIA RTX PRO 6000 Blackwell Server Edition
```
