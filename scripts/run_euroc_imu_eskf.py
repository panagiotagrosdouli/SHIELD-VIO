"""Run inertial-only ESKF propagation on a EuRoC MAV sequence.

This command is intentionally explicit that it is an IMU propagation baseline,
not complete visual-inertial odometry. Its outputs can be inspected for filter
health and passed to the trajectory evaluation tooling where appropriate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.datasets.euroc_imu import load_euroc_imu
from shield_vio.estimation.imu_runner import run_imu_propagation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path, help="EuRoC sequence root containing mav0/imu0/data.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-stride", type=int, default=1)
    parser.add_argument("--max-dt", type=float, default=0.1, dest="max_dt_s")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    imu_path = args.sequence / "mav0" / "imu0" / "data.csv"
    samples = load_euroc_imu(imu_path)
    estimator, recorder, summary = run_imu_propagation(
        samples,
        record_stride=args.record_stride,
        max_dt_s=args.max_dt_s,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recorder.write_tum(args.output_dir / "trajectory.tum")
    recorder.write_csv(args.output_dir / "trajectory_diagnostics.csv")
    final = estimator.state
    report = {
        "mode": "imu_propagation_baseline",
        "sequence": str(args.sequence),
        "sample_count": summary.sample_count,
        "duration_s": summary.duration_s,
        "mean_dt_s": summary.mean_dt_s,
        "max_dt_s": summary.max_dt_s,
        "recorded_pose_count": len(recorder.samples),
        "final_position_m": final.position_m.tolist(),
        "final_velocity_mps": final.velocity_mps.tolist(),
        "final_accel_bias_mps2": final.accel_bias_mps2.tolist(),
        "final_gyro_bias_rps": final.gyro_bias_rps.tolist(),
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
