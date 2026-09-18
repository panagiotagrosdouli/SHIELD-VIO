#!/usr/bin/env python3
"""Aggregate paired paper metrics over complete experimental-run units."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from shield_vio.evaluation.statistics import paired_grouped_bootstrap

RUN_KEY_FIELDS = ("dataset", "sequence", "estimator", "degradation_condition", "seed")


def _run_key(row: dict[str, str]) -> tuple[str, str, str, str, int]:
    missing = [field for field in RUN_KEY_FIELDS if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"missing run-key fields: {', '.join(missing)}")
    return (
        row["dataset"].strip(),
        row["sequence"].strip(),
        row["estimator"].strip(),
        row["degradation_condition"].strip(),
        int(row["seed"]),
    )


def load_metric_table(
    path: Path, *, method_field: str, metric: str
) -> dict[str, dict[tuple[str, str, str, str, int], float]]:
    methods: dict[str, dict[tuple[str, str, str, str, int], float]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(RUN_KEY_FIELDS) | {method_field, metric}
        absent = required - set(reader.fieldnames or ())
        if absent:
            raise ValueError(f"missing columns: {', '.join(sorted(absent))}")
        for row in reader:
            method = str(row[method_field]).strip()
            if not method:
                raise ValueError("method must be non-empty")
            key = _run_key(row)
            bucket = methods.setdefault(method, {})
            if key in bucket:
                raise ValueError(f"duplicate experimental unit for method {method}: {key!r}")
            bucket[key] = float(row[metric])
    return methods


def aggregate(
    path: Path,
    *,
    method_a: str,
    method_b: str,
    metric: str,
    higher_is_better: bool,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    methods = load_metric_table(path, method_field="method", metric=metric)
    if method_a not in methods or method_b not in methods:
        raise ValueError("both requested methods must be present in the metric table")
    result = paired_grouped_bootstrap(
        methods[method_a],
        methods[method_b],
        metric=metric,
        higher_is_better=higher_is_better,
        n_bootstrap=n_bootstrap,
        seed=seed,
        missing="raise",
    )
    return {
        "schema_version": "SHIELD_VIO_PAIRED_AGGREGATE_V1",
        "source_table": str(path),
        "experimental_unit_fields": list(RUN_KEY_FIELDS),
        "method_a": method_a,
        "method_b": method_b,
        "comparison": asdict(result),
        "claim_boundary": (
            "Paired inference over complete experimental-run units. Frames are never "
            "treated as independent replicates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--metric", required=True)
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--higher-is-better", action="store_true")
    direction.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(
        args.metrics_csv,
        method_a=args.method_a,
        method_b=args.method_b,
        metric=args.metric,
        higher_is_better=args.higher_is_better,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
