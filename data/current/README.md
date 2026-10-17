# Current data artifacts

This directory contains the currently cited, committed data inputs:

- `addition_grid_79cell.csv` — the addition-grid measurement table.
- `doubles_scan_scores.csv` — 39 prompt-level double/control scores.
- `variance_scaling.csv` — per-target means across the three tested prompt formats; it does not contain the prompt-level controls needed to recompute pooled SDs.
- `operator_swap_results.json` — aggregate operator/connector values; it does not contain prompt-level scores.
- `strict_unused_targets_results.json` — saved logits and neighbor values for 11 targets across six prompt formats, including two unequal operand orders.
- `eleven_target_target_summary.csv` and `eleven_target_recomputed_summary.json` — per-target contrasts and exact sign-flip summaries recomputed from the saved logits by `src/analyze_unused_targets.py`. The six-format average and arithmetic-only result are reported separately.
- `eleven_target_summary.csv` — compact record of the independently recomputed 11-target summaries.

File hashes, producing-source notes, and unavailable final-notebook exports are listed in [`../MANIFEST.csv`](../MANIFEST.csv). Older/retracted artifacts were removed after their relevant historical values were captured in the [audit trail](../../docs/Audit_Trail.md).
