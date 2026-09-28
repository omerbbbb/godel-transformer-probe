
# Results

> The `final_run/` results use the v1 prompts, which contain a positional
> shortcut (see the top-level README). A rerun with the de-confounded v0.2
> prompts is pending; its outputs will go in `final_run_v2/`.

## Final run

`final_run/` contains the complete output bundle from the final GPU experiment.
The main result is summarized in `FINAL_RESULT.md`.

The key outcome is a **negative held-out result**: Qwen2.5-3B-Instruct solved
30/30 matched chains, but the exact-GCD and near-GCD signatures did not cleanly
distinguish one-hop from two-hop test computations.

## Earlier recovered runs

`pythia70m_probe.json` records the original Pythia-70M result:

```text
paired-correct chains: 0/20
status: too_weak
```

`EXPERIMENT_HISTORY.md` documents the sequence of earlier attempts, including
the Pythia attention-NaN diagnostic.

## Schema example

`example_output.json` is illustrative only. Do not cite it as an experimental
result.
