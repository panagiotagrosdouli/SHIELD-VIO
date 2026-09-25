#!/usr/bin/env python3
"""Fit the real-public train/calibration/validation development bundle without opening test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.experiments.split_safe_development import fit_development_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        required=True,
        help="Directory containing one canonical artifact directory per registered sequence.",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/paper/public_dataset_splits.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="euroc")
    parser.add_argument("--estimator", default="internal_eskf")
    parser.add_argument("--horizon-seconds", type=float, default=2.0)
    parser.add_argument("--max-false-alarms-per-minute", type=float, default=0.2)
    parser.add_argument("--logistic-iterations", type=int, default=1000)
    args = parser.parse_args()

    report = fit_development_bundle(
        args.artifacts_root,
        args.split_config,
        args.output,
        dataset=args.dataset,
        estimator=args.estimator,
        horizon_seconds=args.horizon_seconds,
        max_false_alarms_per_minute=args.max_false_alarms_per_minute,
        logistic_iterations=args.logistic_iterations,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
