#!/usr/bin/env python3
"""Resolve the frozen public benchmark matrix and provenance manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

from shield_vio.experiments.public_benchmark import resolve_benchmark_matrix, write_benchmark_matrix, write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("configs/paper/public_benchmark.yaml"))
    parser.add_argument("--splits", type=Path, default=Path("configs/paper/public_dataset_splits.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/paper/public_benchmark"))
    args = parser.parse_args()
    runs = resolve_benchmark_matrix(args.matrix, args.splits)
    args.output.mkdir(parents=True, exist_ok=True)
    write_benchmark_matrix(runs, args.output / "benchmark_matrix.csv")
    write_split_manifest(runs, args.splits, args.output / "split_manifest.json")
    print(f"resolved {len(runs)} sequence-level runs into {args.output}")


if __name__ == "__main__":
    main()
