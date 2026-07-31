#!/usr/bin/env python3
"""Run the built-in ESKF over a EuRoC sequence and export evaluated artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.experiments.euroc_runner import run_euroc_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_root", type=Path, help="Path containing the EuRoC mav0 directory")
    parser.add_argument("--output", type=Path, required=True, help="Artifact output directory")
    parser.add_argument("--camera", default="cam0", help="EuRoC camera stream used for frame timestamps")
    parser.add_argument("--skip-evaluation", action="store_true", help="Do not compare against EuRoC ground truth")
    parser.add_argument("--max-gap", type=float, default=0.02, help="Maximum timestamp association gap in seconds")
    parser.add_argument("--rpe-delta", type=int, default=1, help="RPE displacement interval in associated samples")
    parser.add_argument("--with-scale", action="store_true", help="Use Sim(3) instead of SE(3) alignment")
    args = parser.parse_args()

    summary = run_euroc_sequence(
        args.sequence_root,
        args.output,
        camera=args.camera,
        evaluate=not args.skip_evaluation,
        max_gap=args.max_gap,
        rpe_delta=args.rpe_delta,
        with_scale=args.with_scale,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
