#!/usr/bin/env python3
"""Export the availability audit for SHIELD_VIO_FAILURE_V1 observables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.evaluation.primary_observables import (
    build_primary_observables,
    write_primary_observable_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-ground-truth-gap-seconds", type=float, default=0.02)
    parser.add_argument("--rpe-interval-seconds", type=float, default=1.0)
    parser.add_argument("--rpe-pair-tolerance-seconds", type=float, default=0.075)
    args = parser.parse_args()

    table = build_primary_observables(
        args.run_dir,
        args.sequence_root,
        max_ground_truth_gap_seconds=args.max_ground_truth_gap_seconds,
        rpe_interval_seconds=args.rpe_interval_seconds,
        rpe_pair_tolerance_seconds=args.rpe_pair_tolerance_seconds,
    )
    report = write_primary_observable_artifacts(table, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
