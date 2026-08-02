#!/usr/bin/env python3
"""Run and summarize a real EuRoC sequence with the built-in ESKF baseline."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from shield_vio.experiments.euroc_runner import run_euroc_sequence


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = run_euroc_sequence(args.sequence_root, args.output, evaluate=True)
    metrics = json.loads((args.output / "metrics.json").read_text(encoding="utf-8"))
    report = {
        "benchmark": "EuRoC MH_01_easy real-data baseline",
        "sequence": args.sequence_root.name,
        "implementation": "built-in ESKF, IMU propagation only",
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "camera_frames": summary.camera_frames,
        "imu_samples": summary.imu_samples,
        "ate_rmse_m": metrics["ate_rmse_m"],
        "rpe_translation_rmse_m": metrics["rpe_translation_rmse_m"],
        "associated_samples": metrics["associated_samples"],
        "duration_seconds": metrics["duration_seconds"],
        "claim_boundary": (
            "Real EuRoC sensor and ground-truth execution, but an IMU-propagation baseline; "
            "not a complete visual-inertial odometry result."
        ),
    }
    (args.output / "real_baseline_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
