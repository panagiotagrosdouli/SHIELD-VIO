"""End-to-end EuRoC IMU propagation, evaluation, and artifact export."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from shield_vio.backends.base import EstimatorBackend, EstimatorHealth, EstimatorState
from shield_vio.backends.eskf import ESKFBackend
from shield_vio.datasets.euroc import (
    SynchronizedFrame,
    read_camera_frames,
    read_imu_samples,
    synchronize_camera_and_imu,
)
from shield_vio.evaluation.euroc import (
    TimestampedTrajectory,
    associate_ground_truth,
    load_euroc_ground_truth,
)
from shield_vio.evaluation.metrics import ate, rpe


@dataclass(frozen=True)
class EuRoCRunSummary:
    backend: str
    camera_frames: int
    imu_samples: int
    trajectory_rows: int
    output_dir: str
    metrics_path: str | None = None


def _state_row(frame_timestamp_ns: int, state: EstimatorState) -> list[object]:
    return [
        frame_timestamp_ns,
        state.timestamp_ns,
        *state.position.tolist(),
        *state.velocity.tolist(),
        *state.orientation_wxyz.tolist(),
        *state.accel_bias.tolist(),
        *state.gyro_bias.tolist(),
    ]


def _health_row(frame_timestamp_ns: int, health: EstimatorHealth) -> list[object]:
    return [
        frame_timestamp_ns,
        health.timestamp_ns,
        health.initialized,
        health.tracking_status,
        health.propagated_imu_samples,
        health.covariance_trace,
        health.covariance_condition_number,
        "" if health.innovation_nis is None else health.innovation_nis,
    ]


def _write_manifest(summary: EuRoCRunSummary, destination: Path) -> None:
    artifacts = {
        "trajectory": "trajectory.csv",
        "health": "health.csv",
    }
    if summary.metrics_path is not None:
        artifacts["metrics"] = Path(summary.metrics_path).name
    manifest = {
        "schema_version": 2,
        "experiment": "euroc_imu_propagation",
        **asdict(summary),
        "artifacts": artifacts,
    }
    (destination / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_runner_trajectory(path: str | Path) -> TimestampedTrajectory:
    """Load the runner's headered trajectory CSV for position evaluation."""
    trajectory_path = Path(path)
    with trajectory_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("runner trajectory must contain at least two samples")
    try:
        timestamps_ns = np.asarray([int(row["frame_timestamp_ns"]) for row in rows])
        positions = np.asarray(
            [[float(row["px"]), float(row["py"]), float(row["pz"])] for row in rows]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid runner trajectory: {trajectory_path}") from exc
    timestamps = (timestamps_ns - timestamps_ns[0]).astype(float) * 1e-9
    return TimestampedTrajectory(timestamps, positions)


def evaluate_run_artifacts(
    sequence_root: str | Path,
    output_dir: str | Path,
    *,
    max_gap: float = 0.02,
    rpe_delta: int = 1,
    with_scale: bool = False,
) -> dict[str, object]:
    """Evaluate a completed runner trajectory against EuRoC ground truth."""
    destination = Path(output_dir)
    estimate = load_runner_trajectory(destination / "trajectory.csv")
    ground_truth = load_euroc_ground_truth(sequence_root)
    timestamps, estimated_positions, gt_positions = associate_ground_truth(
        estimate,
        ground_truth,
        max_gap=max_gap,
    )
    metrics: dict[str, object] = {
        "sequence": Path(sequence_root).name,
        "associated_samples": int(len(timestamps)),
        "duration_seconds": float(timestamps[-1] - timestamps[0]),
        "alignment": "sim3" if with_scale else "se3",
        "ate_rmse_m": ate(
            estimated_positions,
            gt_positions,
            align=True,
            with_scale=with_scale,
        ),
        "rpe_translation_rmse_m": rpe(
            estimated_positions,
            gt_positions,
            delta=rpe_delta,
            align=True,
            with_scale=with_scale,
        ),
        "rpe_delta_samples": rpe_delta,
        "max_association_gap_seconds": max_gap,
    }
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def run_synchronized_frames(
    packets: Iterable[SynchronizedFrame],
    output_dir: str | Path,
    *,
    backend: EstimatorBackend | None = None,
) -> EuRoCRunSummary:
    packets = list(packets)
    if not packets:
        raise ValueError("At least one synchronized frame is required")
    samples = [sample for packet in packets for sample in packet.imu_samples]
    if not samples:
        raise ValueError("The synchronized stream contains no IMU samples")

    estimator = backend or ESKFBackend()
    estimator.initialize(samples[0].timestamp_ns)
    propagated = 0
    trajectory_rows: list[list[object]] = []
    health_rows: list[list[object]] = []

    for packet in packets:
        for sample in packet.imu_samples:
            if sample.timestamp_ns <= estimator.snapshot().timestamp_ns:
                continue
            estimator.process_imu(
                sample.timestamp_ns,
                sample.linear_acceleration_m_s2,
                sample.angular_velocity_rad_s,
            )
            propagated += 1
        trajectory_rows.append(_state_row(packet.frame.timestamp_ns, estimator.snapshot()))
        health_rows.append(_health_row(packet.frame.timestamp_ns, estimator.health()))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trajectory_path = destination / "trajectory.csv"
    health_path = destination / "health.csv"

    with trajectory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame_timestamp_ns",
                "state_timestamp_ns",
                "px",
                "py",
                "pz",
                "vx",
                "vy",
                "vz",
                "qw",
                "qx",
                "qy",
                "qz",
                "bax",
                "bay",
                "baz",
                "bgx",
                "bgy",
                "bgz",
            ]
        )
        writer.writerows(trajectory_rows)

    with health_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame_timestamp_ns",
                "state_timestamp_ns",
                "initialized",
                "tracking_status",
                "propagated_imu_samples",
                "covariance_trace",
                "covariance_condition_number",
                "innovation_nis",
            ]
        )
        writer.writerows(health_rows)

    summary = EuRoCRunSummary(
        estimator.name,
        len(packets),
        propagated,
        len(trajectory_rows),
        str(destination),
    )
    _write_manifest(summary, destination)
    return summary


def run_euroc_sequence(
    sequence_root: str | Path,
    output_dir: str | Path,
    *,
    camera: str = "cam0",
    backend: EstimatorBackend | None = None,
    evaluate: bool = True,
    max_gap: float = 0.02,
    rpe_delta: int = 1,
    with_scale: bool = False,
) -> EuRoCRunSummary:
    frames = read_camera_frames(sequence_root, camera=camera)
    imu_samples = read_imu_samples(sequence_root)
    packets = synchronize_camera_and_imu(frames, imu_samples, include_pre_first_frame=True)
    summary = run_synchronized_frames(packets, output_dir, backend=backend)
    if not evaluate:
        return summary

    evaluate_run_artifacts(
        sequence_root,
        output_dir,
        max_gap=max_gap,
        rpe_delta=rpe_delta,
        with_scale=with_scale,
    )
    evaluated = EuRoCRunSummary(
        backend=summary.backend,
        camera_frames=summary.camera_frames,
        imu_samples=summary.imu_samples,
        trajectory_rows=summary.trajectory_rows,
        output_dir=summary.output_dir,
        metrics_path=str(Path(output_dir) / "metrics.json"),
    )
    _write_manifest(evaluated, Path(output_dir))
    return evaluated
