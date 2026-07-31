from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from shield_vio.backends.base import EstimatorState
from shield_vio.datasets.euroc import CameraFrame, ImuSample, SynchronizedFrame
from shield_vio.experiments.euroc_runner import (
    evaluate_run_artifacts,
    load_runner_trajectory,
    run_synchronized_frames,
)
from shield_vio.experiments.visual_measurements import LinearVisualMeasurement


def _sample(timestamp_ns: int) -> ImuSample:
    return ImuSample(
        timestamp_ns=timestamp_ns,
        timestamp_s=timestamp_ns * 1e-9,
        angular_velocity_rad_s=np.zeros(3),
        linear_acceleration_m_s2=np.array([0.0, 0.0, 9.81]),
    )


class _SyntheticVisualProvider:
    @property
    def name(self) -> str:
        return "synthetic_linear"

    def measure(
        self, packet: SynchronizedFrame, state: EstimatorState
    ) -> LinearVisualMeasurement:
        matrix = np.zeros((3, state.covariance.shape[0]))
        matrix[:, :3] = np.eye(3)
        return LinearVisualMeasurement(
            residual=np.zeros(3),
            measurement_matrix=matrix,
            measurement_covariance=np.eye(3) * 0.01,
            detected_features=80,
            tracked_features=60,
            correspondence_count=50,
            inlier_count=40,
            status="accepted",
        )


def _packets(tmp_path: Path) -> list[SynchronizedFrame]:
    return [
        SynchronizedFrame(
            frame=CameraFrame(20, 20e-9, tmp_path / "0.png"),
            imu_samples=(_sample(10), _sample(20)),
        ),
        SynchronizedFrame(
            frame=CameraFrame(40, 40e-9, tmp_path / "1.png"),
            imu_samples=(_sample(30), _sample(40)),
        ),
    ]


def test_runner_exports_trajectory_health_and_manifest(tmp_path: Path) -> None:
    summary = run_synchronized_frames(_packets(tmp_path), tmp_path / "results")

    assert summary.backend == "eskf"
    assert summary.camera_frames == 2
    assert summary.imu_samples == 3
    assert summary.trajectory_rows == 2
    assert summary.metrics_path is None
    assert summary.visual_provider is None
    assert summary.visual_measurements == 0

    output = tmp_path / "results"
    assert (output / "trajectory.csv").is_file()
    assert (output / "health.csv").is_file()
    assert (output / "experiment_manifest.json").is_file()
    assert not (output / "visual_updates.csv").exists()

    with (output / "trajectory.csv").open(encoding="utf-8") as stream:
        trajectory = list(csv.DictReader(stream))
    assert len(trajectory) == 2
    assert trajectory[-1]["state_timestamp_ns"] == "40"

    with (output / "health.csv").open(encoding="utf-8") as stream:
        health = list(csv.DictReader(stream))
    assert health[-1]["tracking_status"] == "tracking"
    assert health[-1]["propagated_imu_samples"] == "3"

    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["experiment"] == "euroc_vio"
    assert manifest["artifacts"]["trajectory"] == "trajectory.csv"
    assert "metrics" not in manifest["artifacts"]
    assert "visual_updates" not in manifest["artifacts"]


def test_runner_fuses_visual_measurements_and_exports_diagnostics(tmp_path: Path) -> None:
    output = tmp_path / "visual-results"
    summary = run_synchronized_frames(
        _packets(tmp_path), output, visual_provider=_SyntheticVisualProvider()
    )

    assert summary.visual_provider == "synthetic_linear"
    assert summary.visual_measurements == 2
    with (output / "visual_updates.csv").open(encoding="utf-8") as stream:
        updates = list(csv.DictReader(stream))
    assert len(updates) == 2
    assert updates[-1]["status"] == "accepted"
    assert updates[-1]["inlier_ratio"] == "0.8"
    assert float(updates[-1]["innovation_nis"]) == 0.0

    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["visual_provider"] == "synthetic_linear"
    assert manifest["visual_measurements"] == 2
    assert manifest["artifacts"]["visual_updates"] == "visual_updates.csv"


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


def test_evaluate_run_artifacts_writes_ate_and_rpe(tmp_path: Path) -> None:
    output = tmp_path / "results"
    output.mkdir()
    trajectory_path = output / "trajectory.csv"
    with trajectory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_timestamp_ns", "state_timestamp_ns", "px", "py", "pz"])
        for index in range(4):
            timestamp_ns = 1_000_000_000 + index * 1_000_000_000
            writer.writerow([timestamp_ns, timestamp_ns, index, 0.0, 0.0])

    sequence = tmp_path / "MH_01_easy"
    ground_truth = sequence / "mav0" / "state_groundtruth_estimate0"
    ground_truth.mkdir(parents=True)
    with (ground_truth / "data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["#timestamp", "p_RS_R_x", "p_RS_R_y", "p_RS_R_z"])
        for index in range(4):
            timestamp_ns = 1_000_000_000 + index * 1_000_000_000
            writer.writerow([timestamp_ns, index, 0.0, 0.0])

    loaded = load_runner_trajectory(trajectory_path)
    assert loaded.timestamps.tolist() == [0.0, 1.0, 2.0, 3.0]

    metrics = evaluate_run_artifacts(sequence, output, max_gap=0.1)

    assert metrics["associated_samples"] == 4
    assert metrics["ate_rmse_m"] < 1e-12
    assert metrics["rpe_translation_rmse_m"] < 1e-12
    written = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert written == metrics
