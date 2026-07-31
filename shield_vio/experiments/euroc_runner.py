"""End-to-end EuRoC IMU propagation and artifact export."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from shield_vio.backends.base import EstimatorBackend, EstimatorHealth, EstimatorState
from shield_vio.backends.eskf import ESKFBackend
from shield_vio.datasets.euroc import SynchronizedFrame, read_camera_frames, read_imu_samples, synchronize_camera_and_imu


@dataclass(frozen=True)
class EuRoCRunSummary:
    backend: str
    camera_frames: int
    imu_samples: int
    trajectory_rows: int
    output_dir: str


def _state_row(frame_timestamp_ns: int, state: EstimatorState) -> list[object]:
    return [frame_timestamp_ns, state.timestamp_ns, *state.position.tolist(), *state.velocity.tolist(), *state.orientation_wxyz.tolist(), *state.accel_bias.tolist(), *state.gyro_bias.tolist()]


def _health_row(frame_timestamp_ns: int, health: EstimatorHealth) -> list[object]:
    return [frame_timestamp_ns, health.timestamp_ns, health.initialized, health.tracking_status, health.propagated_imu_samples, health.covariance_trace, health.covariance_condition_number, "" if health.innovation_nis is None else health.innovation_nis]


def run_synchronized_frames(packets: Iterable[SynchronizedFrame], output_dir: str | Path, *, backend: EstimatorBackend | None = None) -> EuRoCRunSummary:
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
            estimator.process_imu(sample.timestamp_ns, sample.linear_acceleration_m_s2, sample.angular_velocity_rad_s)
            propagated += 1
        trajectory_rows.append(_state_row(packet.frame.timestamp_ns, estimator.snapshot()))
        health_rows.append(_health_row(packet.frame.timestamp_ns, estimator.health()))

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trajectory_path = destination / "trajectory.csv"
    health_path = destination / "health.csv"

    with trajectory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_timestamp_ns", "state_timestamp_ns", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "bax", "bay", "baz", "bgx", "bgy", "bgz"])
        writer.writerows(trajectory_rows)

    with health_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_timestamp_ns", "state_timestamp_ns", "initialized", "tracking_status", "propagated_imu_samples", "covariance_trace", "covariance_condition_number", "innovation_nis"])
        writer.writerows(health_rows)

    summary = EuRoCRunSummary(estimator.name, len(packets), propagated, len(trajectory_rows), str(destination))
    manifest = {"schema_version": 1, "experiment": "euroc_imu_propagation", **asdict(summary), "artifacts": {"trajectory": trajectory_path.name, "health": health_path.name}}
    (destination / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_euroc_sequence(sequence_root: str | Path, output_dir: str | Path, *, camera: str = "cam0", backend: EstimatorBackend | None = None) -> EuRoCRunSummary:
    frames = read_camera_frames(sequence_root, camera=camera)
    imu_samples = read_imu_samples(sequence_root)
    packets = synchronize_camera_and_imu(frames, imu_samples, include_pre_first_frame=True)
    return run_synchronized_frames(packets, output_dir, backend=backend)
