#!/usr/bin/env python3
"""Export non-confirmatory primary V2 failure events and future targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.evaluation.failure_definition import load_failure_definition
from shield_vio.evaluation.primary_failure_targets import (
    build_primary_failure_targets,
    write_primary_failure_artifacts,
)
from shield_vio.evaluation.primary_observables import build_primary_observables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--failure-config",
        type=Path,
        default=Path("configs/paper/failure_primary_v2.yaml"),
    )
    parser.add_argument(
        "--evidence-level",
        choices=("PUBLIC_DATASET_SMOKE", "PUBLIC_DATASET_DEVELOPMENT"),
        default="PUBLIC_DATASET_SMOKE",
    )
    parser.add_argument("--max-ground-truth-gap-seconds", type=float, default=0.02)
    parser.add_argument("--rpe-interval-seconds", type=float, default=1.0)
    parser.add_argument("--rpe-pair-tolerance-seconds", type=float, default=0.075)
    args = parser.parse_args()

    definition = load_failure_definition(args.failure_config)
    table = build_primary_observables(
        args.run_dir,
        args.sequence_root,
        max_ground_truth_gap_seconds=args.max_ground_truth_gap_seconds,
        rpe_interval_seconds=args.rpe_interval_seconds,
        rpe_pair_tolerance_seconds=args.rpe_pair_tolerance_seconds,
        failure_definition=definition,
    )
    result = build_primary_failure_targets(table, definition)
    manifest = write_primary_failure_artifacts(
        result,
        args.output,
        failure_config_path=args.failure_config,
        source_run_dir=args.run_dir,
        evidence_level=args.evidence_level,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
