"""Recompute target-level contrasts for the separate 11-target score export."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/current/strict_unused_targets_results.json"
TARGET_CSV = ROOT / "data/current/eleven_target_target_summary.csv"
SUMMARY_JSON = ROOT / "data/current/eleven_target_recomputed_summary.json"
TEMPLATES = ("arithmetic", "matched_comma", "neutral_0", "neutral_1", "neutral_2", "neutral_3")


def score(record: dict, split: str, template: str) -> float:
    item = record["splits"][split][template]
    calculated = item["target_logit"] - (item["left_neighbor_logit"] + item["right_neighbor_logit"]) / 2
    if abs(calculated - item["margin_at"]) > 1e-5:
        raise ValueError(f"saved margin mismatch for target {record['target']} {split} {template}")
    return calculated


def exact_one_sided_sign_flip(values: list[float]) -> float:
    observed = sum(values)
    count = 0
    total = 1 << len(values)
    for signs in itertools.product((-1, 1), repeat=len(values)):
        if sum(value * sign for value, sign in zip(values, signs)) >= observed - 1e-12:
            count += 1
    return count / total


def main() -> None:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    targets = [int(row["target"]) for row in records]
    if len(records) != 11 or len(set(targets)) != 11:
        raise ValueError("expected exactly 11 unique target records")

    output_rows = []
    for record in records:
        row = {"target": record["target"], "stratum": record["stratum"]}
        all_deltas = []
        for template in TEMPLATES:
            equal = score(record, "equal", template)
            controls = [score(record, split, template) for split in ("unequal_order_1", "unequal_order_2")]
            delta = equal - sum(controls) / len(controls)
            row[f"{template}_delta"] = delta
            all_deltas.append(delta)
        row["all_template_mean_delta"] = sum(all_deltas) / len(all_deltas)
        output_rows.append(row)

    TARGET_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {"n_targets": len(output_rows), "targets": targets, "templates": list(TEMPLATES)}
    for name, field in [(template, f"{template}_delta") for template in TEMPLATES] + [
        ("all_templates_mean", "all_template_mean_delta")
    ]:
        values = [row[field] for row in output_rows]
        summary[name] = {
            "mean_contrast": sum(values) / len(values),
            "positive_targets": sum(value > 0 for value in values),
            "exact_one_sided_sign_flip_p": exact_one_sided_sign_flip(values),
        }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
