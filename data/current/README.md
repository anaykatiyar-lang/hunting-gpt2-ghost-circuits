# Current data artifacts

This directory contains the currently cited, committed data inputs:

- `addition_grid_79cell.csv` — the addition-grid measurement table.
- `doubles_scan_scores.csv` — 39 prompt-level double/control scores.
- `variance_scaling.csv` — per-target means across the three tested prompt formats; it does not contain the prompt-level controls needed to recompute pooled SDs.
- `operator_swap_results.json` — aggregate operator/connector values; it does not contain prompt-level scores.
- `eleven_target_summary.csv` — notebook-reported aggregate for a separate 11-target held-out set; per-target scores were not supplied, so it is not independently recomputed.

File hashes, producing-source notes, and unavailable final-notebook exports are listed in [`../MANIFEST.csv`](../MANIFEST.csv). Older/retracted artifacts were removed after their relevant historical values were captured in the [audit trail](../../docs/Audit_Trail.md).
