"""Validate and summarize a local EuRoC MAV sequence.

This command is the first stage of the public-dataset runner. It intentionally
stops before estimator execution and verifies that stream ingestion,
calibration, units, timestamps, and camera/IMU synchronization are usable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shield_vio.datasets.calibration import load_camera_calibration, load_imu_calibration
from shield_vio.datasets.euroc import (
    estimate_stream_rate_hz,
    read_camera_frames,
    read_imu_samples,
    synchronize_camera_and_imu,
)


def build_summary(sequence_root: Path) -> dict[str, object]:
    camera_yaml = sequence_root / "mav0" / "cam0" / "sensor.yaml"
    imu_yaml = sequence_root / "mav0" / "imu0" / "sensor.yaml"

    frames = read_camera_frames(sequence_root)
    imu_samples = read_imu_samples(sequence_root)
    synchronized = synchronize_camera_and_imu(frames, imu_samples)
    camera_calibration = load_camera_calibration(camera_yaml)
    imu_calibration = load_imu_calibration(imu_yaml)

    imu_counts = np.asarray([len(item.imu_samples) for item in synchronized[1:]], dtype=int)
    empty_intervals = int(np.count_nonzero(imu_counts == 0))

    return {
        "sequence": sequence_root.name,
        "sequence_root": str(sequence_root.resolve()),
        "camera": {
            "frame_count": len(frames),
            "first_timestamp_ns": frames[0].timestamp_ns,
            "last_timestamp_ns": frames[-1].timestamp_ns,
            "observed_rate_hz": estimate_stream_rate_hz(
                [frame.timestamp_ns for frame in frames]
            ),
            "declared_rate_hz": camera_calibration.rate_hz,
            "resolution": list(camera_calibration.resolution),
            "camera_model": camera_calibration.camera_model,
            "distortion_model": camera_calibration.distortion_model,
        },
        "imu": {
            "sample_count": len(imu_samples),
            "first_timestamp_ns": imu_samples[0].timestamp_ns,
            "last_timestamp_ns": imu_samples[-1].timestamp_ns,
            "observed_rate_hz": estimate_stream_rate_hz(
                [sample.timestamp_ns for sample in imu_samples]
            ),
            "declared_rate_hz": imu_calibration.rate_hz,
        },
        "synchronization": {
            "camera_intervals": max(len(synchronized) - 1, 0),
            "empty_imu_intervals": empty_intervals,
            "minimum_imu_samples_per_interval": int(imu_counts.min()) if imu_counts.size else 0,
            "median_imu_samples_per_interval": (
                float(np.median(imu_counts)) if imu_counts.size else 0.0
            ),
            "maximum_imu_samples_per_interval": int(imu_counts.max()) if imu_counts.size else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path, help="Path to a EuRoC sequence directory")
    parser.add_argument("--output", type=Path, help="Optional JSON summary path")
    args = parser.parse_args()

    summary = build_summary(args.sequence)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
