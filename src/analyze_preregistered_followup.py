"""Recompute the 20-target arithmetic follow-up summary from its raw JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TARGETS = [38, 40, 46, 48, 50, 54, 56, 58, 60, 68, 70, 72, 74, 78, 80, 84, 86, 90, 94, 98]
STRATA = {
    "low": (38, 58),
    "mid": (60, 78),
    "high": (80, 98),
}


def margin(record: dict) -> float:
    """Return target logit minus the mean of the two numeric-neighbor logits."""
    value = record["target_logit"] - (
        record["left_neighbor_logit"] + record["right_neighbor_logit"]
    ) / 2.0
    if not np.isclose(value, record["margin_at"], rtol=0.0, atol=1e-5):
        raise ValueError(
            f"Saved margin mismatch for prompt {record.get('prompt_text')!r}: "
            f"stored={record['margin_at']}, recomputed={value}"
        )
    return float(value)


def exact_sign_flip(values: np.ndarray) -> tuple[float, float]:
    """Enumerate all sign assignments; return one- and two-sided p-values."""
    null_sums = np.array([0.0])
    for value in values:
        null_sums = np.concatenate((null_sums + value, null_sums - value))
    observed = float(values.sum())
    one_sided = np.count_nonzero(null_sums >= observed - 1e-12) / len(null_sums)
    two_sided = np.count_nonzero(np.abs(null_sums) >= abs(observed) - 1e-12) / len(null_sums)
    return float(one_sided), float(two_sided)


def protocol_stratum(target: int) -> str:
    for name, (lower, upper) in STRATA.items():
        if lower <= target <= upper:
            return name
    raise ValueError(f"Target {target} does not belong to a defined stratum")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input",
        type=Path,
        default=repo / "data/current/preregistered_20_target_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "data/current/preregistered_20_target_summary.csv",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026101203)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("Expected a JSON list of target records")
    targets = [int(row["target"]) for row in records]
    if sorted(targets) != TARGETS:
        raise ValueError(f"Target set mismatch: found {sorted(targets)}")
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate target records found")

    rows = []
    for record in sorted(records, key=lambda item: int(item["target"])):
        target = int(record["target"])
        splits = record["splits"]
        equal = margin(splits["equal"]["arithmetic"])
        controls = [
            margin(splits[name]["arithmetic"])
            for name in ("unequal_order_1", "unequal_order_2")
        ]
        control_mean = float(np.mean(controls))
        rows.append(
            {
                "target": target,
                "stored_stratum": record["stratum"],
                "protocol_stratum": protocol_stratum(target),
                "equal_arithmetic_margin": equal,
                "unequal_arithmetic_margin_mean": control_mean,
                "advantage": equal - control_mean,
                "positive": equal - control_mean > 0,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    values = np.array([row["advantage"] for row in rows], dtype=float)
    groups = {
        name: values[
            np.array([row["protocol_stratum"] == name for row in rows], dtype=bool)
        ]
        for name in STRATA
    }
    counts = {name: len(group) for name, group in groups.items()}
    one_sided, two_sided = exact_sign_flip(values)

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_means = np.empty(args.bootstrap_resamples, dtype=float)
    for index in range(args.bootstrap_resamples):
        sample = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups.values()]
        )
        bootstrap_means[index] = sample.mean()
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

    positive = int(np.count_nonzero(values > 0))
    criteria = {
        "positive mean and 95% interval excludes zero": values.mean() > 0 and ci_low > 0,
        "exact one-sided sign-flip p < .05": one_sided < 0.05,
        "mean at least 0.10 logits": values.mean() >= 0.10,
        "at least 70% of targets positive": positive >= int(np.ceil(0.70 * len(values))),
    }

    print(f"Targets: {len(values)}; original-plan numeric strata: {counts}")
    print(f"Mean arithmetic contrast: {values.mean():+.5f}")
    print(f"Stratified bootstrap 95% interval: [{ci_low:+.5f}, {ci_high:+.5f}]")
    print(f"Exact one-sided sign-flip p: {one_sided:.6f}")
    print(f"Exact two-sided sign-flip p: {two_sided:.6f}")
    print(f"Positive targets: {positive}/{len(values)} ({positive / len(values):.0%})")
    for name, passed in criteria.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    print("All four numerical criteria met:", all(criteria.values()))
    print(f"Per-target summary: {args.output}")


if __name__ == "__main__":
    main()
