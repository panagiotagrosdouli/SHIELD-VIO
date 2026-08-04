#!/usr/bin/env python3
"""Run the non-confirmatory EuRoC failure-prediction paper vertical slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from shield_vio.experiments.paper_vertical_slice import (
    VerticalSliceConfig,
    run_public_dataset_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--sequence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper/vertical_slice_smoke.yaml"),
    )
    parser.add_argument("--history-seconds", type=float)
    parser.add_argument("--horizon-seconds", type=float)
    parser.add_argument("--position-error-threshold-m", type=float)
    parser.add_argument("--persistence-seconds", type=float)
    parser.add_argument("--max-ground-truth-gap-seconds", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    failure = payload["failure_definition"]
    heuristics = payload["heuristics"]
    config = VerticalSliceConfig(
        history_seconds=(
            float(payload["history_seconds"])
            if args.history_seconds is None
            else args.history_seconds
        ),
        horizon_seconds=(
            float(payload["horizon_seconds"])
            if args.horizon_seconds is None
            else args.horizon_seconds
        ),
        position_error_threshold_m=(
            float(failure["threshold_m"])
            if args.position_error_threshold_m is None
            else args.position_error_threshold_m
        ),
        persistence_seconds=(
            float(failure["persistence_seconds"])
            if args.persistence_seconds is None
            else args.persistence_seconds
        ),
        max_ground_truth_gap_seconds=(
            float(failure["max_ground_truth_gap_seconds"])
            if args.max_ground_truth_gap_seconds is None
            else args.max_ground_truth_gap_seconds
        ),
        covariance_threshold=float(heuristics["covariance_trace_threshold"]),
        feature_count_threshold=float(heuristics["tracked_feature_count_threshold"]),
        nis_threshold=float(heuristics["innovation_nis_threshold"]),
        seed=int(payload["seed"] if args.seed is None else args.seed),
    )
    manifest = run_public_dataset_smoke(
        args.run_dir,
        args.sequence_root,
        args.output,
        config=config,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
