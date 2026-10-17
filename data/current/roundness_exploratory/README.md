# Exploratory class decomposition

This directory holds the reported decomposition of equal-minus-unequal target scores by target and equal-operand decimal structure in GPT-2 Small.

## Main class contrast

The archived output reports 552 target summaries and 3,864 prompt-score records. Scores are mean conditional log-probabilities per answer token, in nats per token. Mean target-level advantage is +0.034669 for non-round targets (n=276), +0.327697 when the equal operand ends in 5 (n=138), and +0.180622 when it ends in 0 (n=138). The end-5 versus non-round magnitude-band-adjusted estimate is +0.293028 nats per answer token. The cluster-by-magnitude-band sensitivity (43 bands) reports a 95% interval [+0.172369, +0.413687] and two-sided p=1.4670e-05.

This is an exploratory class association, not an isolated roundness effect or a circuit result. For equal operands, T=2a ties target class to the equal operand's final digit. Unequal-control composition also differs across classes.

## Reproduction inputs

`exploratory_decomposition_reproduce_v3.py` expects these three CSV files in the same directory:

- `confirmatory_target_level_contrasts.csv`
- `confirmatory_prompt_level_sequence_scores.csv`
- `prereg_fullrange_strict_split_manifest_v2.csv`

The target contrasts, prompt-level sequence scores, and split manifest are included in this directory, so the full calculation can be rerun from the archived inputs and code. The original v3 JSON records the reported numerical checks and retains its historical key `primary_preregistered_HC3`, although the top-level status explicitly says the decomposition is post hoc and outside the preregistered analysis. The versioned v4 JSON is a schema-only migration: it renames that field to `exploratory_HC3`, adds migration metadata, and preserves all statistical values without recalculation. Use v4 for downstream consumers that need an unambiguous field name; retain v3 as the historical output. No separate registration record is archived. The manuscript treats the decomposition as exploratory and reports the same estimates.

`exploratory_decomposition_reproduce_v3.py` remains the producer for the historical v3 output. It is not silently relabeled as producing v4; the v4 file documents a schema migration of the saved result, not a new analysis run.

The three original input CSVs accompany the derived summaries. File hashes and statuses are recorded in `data/MANIFEST.csv`.
