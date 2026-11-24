#!/usr/bin/env python3
"""Evaluate a canonical held-out sequence using an independently sealed model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from shield_vio.experiments.canonical_held_out import evaluate_canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "bundle", "split-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-sha256", "dataset", "sequence", "estimator"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=("test", "shifted_test"), required=True)
    args = parser.parse_args()
    report, predictions = evaluate_canonical(
        args.root, args.bundle, args.split_config, args.expected_sha256,
        dataset=args.dataset, sequence=args.sequence, estimator=args.estimator,
        split=args.split,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(predictions)
        writer.writerows(zip(*(values.tolist() for values in predictions.values()), strict=True))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
