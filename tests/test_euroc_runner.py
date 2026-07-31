from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from shield_vio.datasets.euroc import CameraFrame, ImuSample, SynchronizedFrame
from shield_vio.experiments.euroc_runner import run_synchronized_frames


def _sample(timestamp_ns: int) -> ImuSample:
    return ImuSample(
        timestamp_ns=timestamp_ns,
        timestamp_s=timestamp_ns * 1e-9,
        angular_velocity_rad_s=np.zeros(3),
        linear_acceleration_m_s2=np.array([0.0, 0.0, 9.81]),
    )


def test_runner_exports_trajectory_health_and_manifest(tmp_path: Path) -> None:
    packets = [
        SynchronizedFrame(
            frame=CameraFrame(20, 20e-9, tmp_path / "0.png"),
            imu_samples=(_sample(10), _sample(20)),
        ),
        SynchronizedFrame(
            frame=CameraFrame(40, 40e-9, tmp_path / "1.png"),
            imu_samples=(_sample(30), _sample(40)),
        ),
    ]

    summary = run_synchronized_frames(packets, tmp_path / "results")

    assert summary.backend == "eskf"
    assert summary.camera_frames == 2
    assert summary.imu_samples == 3
    assert summary.trajectory_rows == 2

    output = tmp_path / "results"
    assert (output / "trajectory.csv").is_file()
    assert (output / "health.csv").is_file()
    assert (output / "experiment_manifest.json").is_file()

    with (output / "trajectory.csv").open(encoding="utf-8") as stream:
        trajectory = list(csv.DictReader(stream))
    assert len(trajectory) == 2
    assert trajectory[-1]["state_timestamp_ns"] == "40"

    with (output / "health.csv").open(encoding="utf-8") as stream:
        health = list(csv.DictReader(stream))
    assert health[-1]["tracking_status"] == "tracking"
    assert health[-1]["propagated_imu_samples"] == "3"

    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["artifacts"]["trajectory"] == "trajectory.csv"


def test_runner_rejects_stream_without_imu(tmp_path: Path) -> None:
    packet = SynchronizedFrame(
        frame=CameraFrame(20, 20e-9, tmp_path / "0.png"),
        imu_samples=(),
    )

    try:
        run_synchronized_frames([packet], tmp_path / "results")
    except ValueError as error:
        assert "no IMU samples" in str(error)
    else:
        raise AssertionError("Expected ValueError")
