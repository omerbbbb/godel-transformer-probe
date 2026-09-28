# Results

The repository preserves all major experiment stages rather than only the
headline run.

## Main result — `final_run_v3/`

This is the corrected v0.2 experiment and the result to cite.

- 150 random chains, seed 7
- two distractor facts per prompt + shuffled fact order
- Qwen2.5-1.5B-Instruct: 51/150 paired-correct, 26 held-out chains,
  paired accuracy 0.75, permutation `p = 0.000999`
- Qwen2.5-3B-Instruct: 60/150 paired-correct, 30 held-out chains,
  paired accuracy 1.00, permutation `p = 0.000999`
- shortcut diagnostics are stored in `shortcut_check.json`

See `FINAL_RESULT.md` for interpretation and limitations.

## Corrected pilot — `final_run_v2/`

The first v0.2 rerun used only 30 chains. It produced a promising 1.00 paired
accuracy for both model sizes, but only 5 and 7 held-out chains, giving
one-sided permutation p-values around 0.06. That underpowered pilot motivated
the 150-chain run above.

## Superseded v1 — `final_run/`

The original GPU run is retained for transparency but should **not** be cited as
valid evidence about multi-hop computation. A later review found a positional
shortcut in its prompt format and an unsafe float16 scoring path. Both issues
are documented in `EXPERIMENT_HISTORY.md` and fixed in v0.2.

## Earlier recovered runs

`pythia70m_probe.json` records an original Pythia-70M result from the older
pipeline. The later dtype audit showed that float16 non-finite values affected
that stage, so it is historical/diagnostic rather than a main result.

`EXPERIMENT_HISTORY.md` documents the full progression, including the Pythia
attention-NaN diagnostic and the later prompt/dtype audit.

## Schema example

`example_output.json` is illustrative only. Do not cite it as experimental
evidence.
