"""Compute ATE and RPE metrics for an estimated trajectory against EuRoC ground truth."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from shield_vio.evaluation.euroc import (
    associate_ground_truth,
    load_estimator_trajectory,
    load_euroc_ground_truth,
)
from shield_vio.evaluation.trajectory_metrics import summarize_trajectory_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_root", help="EuRoC sequence directory")
    parser.add_argument("estimate", help="Estimator trajectory CSV or TUM-style text")
    parser.add_argument("--timestamp-unit", default="seconds")
    parser.add_argument("--max-gap", type=float, default=0.02)
    parser.add_argument("--rpe-delta", type=int, default=1)
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Disable rigid SE(3) alignment before metric computation",
    )
    args = parser.parse_args()

    estimate = load_estimator_trajectory(args.estimate, timestamp_unit=args.timestamp_unit)
    ground_truth = load_euroc_ground_truth(args.sequence_root)
    _, estimated_positions, reference_positions = associate_ground_truth(
        estimate,
        ground_truth,
        max_gap=args.max_gap,
    )
    metrics = summarize_trajectory_metrics(
        estimated_positions,
        reference_positions,
        rpe_delta=args.rpe_delta,
        align=not args.no_align,
    )
    print(json.dumps(asdict(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
