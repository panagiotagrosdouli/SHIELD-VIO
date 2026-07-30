"""Evaluate an estimator trajectory against EuRoC ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.evaluation.euroc import (
    associate_ground_truth,
    load_estimator_trajectory,
    load_euroc_ground_truth,
)
from shield_vio.evaluation.metrics import ate, rpe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path, help="EuRoC sequence root")
    parser.add_argument("--estimate", required=True, type=Path, help="timestamp x y z trajectory")
    parser.add_argument(
        "--timestamp-unit",
        choices=("seconds", "milliseconds", "microseconds", "nanoseconds"),
        default="seconds",
    )
    parser.add_argument("--max-gap", type=float, default=0.02)
    parser.add_argument("--rpe-delta", type=int, default=1)
    parser.add_argument("--with-scale", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ground_truth = load_euroc_ground_truth(args.sequence)
    estimate = load_estimator_trajectory(
        args.estimate,
        timestamp_unit=args.timestamp_unit,
        relative_time=True,
    )
    timestamps, estimated_positions, gt_positions = associate_ground_truth(
        estimate,
        ground_truth,
        max_gap=args.max_gap,
    )
    metrics = {
        "sequence": args.sequence.name,
        "estimate": str(args.estimate),
        "associated_samples": int(len(timestamps)),
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "alignment": "sim3" if args.with_scale else "se3",
        "ate_rmse_m": ate(
            estimated_positions,
            gt_positions,
            align=True,
            with_scale=args.with_scale,
        ),
        "rpe_translation_rmse_m": rpe(
            estimated_positions,
            gt_positions,
            delta=args.rpe_delta,
            align=True,
            with_scale=args.with_scale,
        ),
        "rpe_delta_samples": args.rpe_delta,
        "max_association_gap_seconds": args.max_gap,
    }
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
