"""Convert pinned OpenVINS total-state exports into SHIELD-VIO run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from shield_vio.datasets.euroc import read_camera_frames
from shield_vio.experiments.euroc_runner import evaluate_run_artifacts


@dataclass(frozen=True)
class OpenVINSStateRow:
    timestamp_imu_s: float
    timestamp_camera_ns: int
    orientation_wxyz: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    cam_to_imu_offset_s: float
    variances: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _rows(path: Path) -> list[list[float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: list[list[float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                values = [float(item) for item in stripped.split()]
            except ValueError as exc:
                raise ValueError(f"invalid numeric OpenVINS row {line_number} in {path}") from exc
            if not all(np.isfinite(value) for value in values):
                raise ValueError(f"non-finite OpenVINS row {line_number} in {path}")
            result.append(values)
    if not result:
        raise ValueError(f"OpenVINS export contains no data rows: {path}")
    return result


def _parse_states(
    estimate_path: Path,
    deviation_path: Path,
) -> list[OpenVINSStateRow]:
    estimates = _rows(estimate_path)
    deviations = _rows(deviation_path)
    if len(estimates) != len(deviations):
        raise ValueError("OpenVINS estimate/deviation row counts must match")

    states: list[OpenVINSStateRow] = []
    previous_camera_ns: int | None = None
    for index, (estimate, deviation) in enumerate(
        zip(estimates, deviations, strict=True),
        start=1,
    ):
        if len(estimate) < 18:
            raise ValueError(f"OpenVINS estimate row {index} has fewer than 18 fields")
        if len(deviation) < 16:
            raise ValueError(f"OpenVINS deviation row {index} has fewer than 16 fields")
        if abs(estimate[0] - deviation[0]) > 5e-6:
            raise ValueError(f"OpenVINS estimate/deviation timestamps disagree at row {index}")

        timestamp_imu_s = float(estimate[0])
        q_xyzw_jpl = np.asarray(estimate[1:5], dtype=float)
        q_norm = float(np.linalg.norm(q_xyzw_jpl))
        if q_norm <= np.finfo(float).eps:
            raise ValueError(f"OpenVINS row {index} contains a zero quaternion")
        q_xyzw_jpl /= q_norm
        # OpenVINS documents its state as JPL G->I. The same xyzw coefficients
        # correspond to Hamilton I->G, which is SHIELD-VIO's body->world convention.
        orientation_wxyz = np.asarray(
            [q_xyzw_jpl[3], q_xyzw_jpl[0], q_xyzw_jpl[1], q_xyzw_jpl[2]],
            dtype=float,
        )

        cam_to_imu_offset_s = float(estimate[17])
        timestamp_camera_ns = int(
            round((timestamp_imu_s - cam_to_imu_offset_s) * 1_000_000_000)
        )
        if previous_camera_ns is not None and timestamp_camera_ns <= previous_camera_ns:
            raise ValueError("OpenVINS camera-clock state timestamps must increase strictly")
        previous_camera_ns = timestamp_camera_ns

        std = np.asarray(deviation[1:16], dtype=float)
        if np.any(std < 0.0):
            raise ValueError(f"OpenVINS row {index} contains negative standard deviations")
        variances = np.square(std)

        states.append(
            OpenVINSStateRow(
                timestamp_imu_s=timestamp_imu_s,
                timestamp_camera_ns=timestamp_camera_ns,
                orientation_wxyz=orientation_wxyz,
                position=np.asarray(estimate[5:8], dtype=float),
                velocity=np.asarray(estimate[8:11], dtype=float),
                gyro_bias=np.asarray(estimate[11:14], dtype=float),
                accel_bias=np.asarray(estimate[14:17], dtype=float),
                cam_to_imu_offset_s=cam_to_imu_offset_s,
                variances=variances,
            )
        )
    return states


def _covariance_stats(variances: np.ndarray) -> tuple[float, float | None, float]:
    values = np.asarray(variances, dtype=float)
    if values.shape != (15,) or np.any(values < 0) or np.any(~np.isfinite(values)):
        raise ValueError("OpenVINS diagonal covariance approximation must contain 15 variances")
    trace = float(np.sum(values))
    minimum = float(np.min(values))
    positive = values[values > np.finfo(float).eps]
    condition = None if len(positive) != len(values) else float(np.max(values) / np.min(values))
    return trace, condition, minimum


def import_openvins_total_state(
    estimate_path: str | Path,
    deviation_path: str | Path,
    sequence_root: str | Path,
    output_dir: str | Path,
    *,
    adapter_config: str | Path,
    evaluate: bool = True,
) -> dict[str, Any]:
    """Map OpenVINS total-state exports onto camera-frame SHIELD-VIO artifacts."""

    estimate = Path(estimate_path)
    deviation = Path(deviation_path)
    sequence = Path(sequence_root)
    destination = Path(output_dir)
    config_path = Path(adapter_config)
    config = _mapping(config_path)

    if config.get("schema_version") != "SHIELD_VIO_OPENVINS_ADAPTER_V1":
        raise ValueError("unsupported OpenVINS adapter schema")
    capabilities = config.get("backend_capabilities")
    if not isinstance(capabilities, dict):
        raise ValueError("OpenVINS adapter config lacks backend_capabilities")

    states = _parse_states(estimate, deviation)
    frames = read_camera_frames(sequence, camera="cam0")
    frame_ns = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
    state_ns = np.asarray([state.timestamp_camera_ns for state in states], dtype=np.int64)

    first_frame_index = int(np.searchsorted(frame_ns, state_ns[0], side="left"))
    if first_frame_index >= len(frame_ns):
        raise ValueError("OpenVINS state export begins after the EuRoC camera stream ends")
    frame_ns = frame_ns[first_frame_index:]
    latest_state = np.searchsorted(state_ns, frame_ns, side="right") - 1
    valid = latest_state >= 0
    if not np.any(valid):
        raise ValueError("no camera frame has an OpenVINS state at or before it")
    frame_ns = frame_ns[valid]
    latest_state = latest_state[valid]

    destination.mkdir(parents=True, exist_ok=True)
    trajectory_path = destination / "trajectory.csv"
    health_path = destination / "health.csv"

    trajectory_rows: list[list[object]] = []
    health_rows: list[list[object]] = []
    for timestamp_ns, state_index in zip(frame_ns, latest_state, strict=True):
        state = states[int(state_index)]
        trace, condition, minimum_variance = _covariance_stats(state.variances)
        trajectory_rows.append(
            [
                int(timestamp_ns),
                int(state.timestamp_camera_ns),
                *state.position.tolist(),
                *state.velocity.tolist(),
                *state.orientation_wxyz.tolist(),
                *state.accel_bias.tolist(),
                *state.gyro_bias.tolist(),
            ]
        )
        health_rows.append(
            [
                int(timestamp_ns),
                int(state.timestamp_camera_ns),
                True,
                "external_state",
                0,
                trace,
                "" if condition is None else condition,
                "",
                minimum_variance,
                0.0,
                0,
                int(bool(capabilities["reset_event"]["exported_value"])),
                int(bool(capabilities["relocalization_event"]["exported_value"])),
            ]
        )

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
                "covariance_min_eigenvalue",
                "covariance_symmetry_error",
                "terminal_tracking_loss_observable",
                "reset_event",
                "relocalization_event",
            ]
        )
        writer.writerows(health_rows)

    metrics_path: str | None = None
    if evaluate:
        evaluate_run_artifacts(sequence, destination)
        metrics_path = "metrics.json"

    state_ages = (frame_ns - state_ns[latest_state]).astype(float) * 1e-9
    manifest = {
        "schema_version": "SHIELD_VIO_EXTERNAL_RUN_V1",
        "experiment": "external_vio_import",
        "backend": "openvins",
        "external_revision": str(config["openvins_revision"]),
        "adapter_schema": str(config["schema_version"]),
        "adapter_config": str(config_path),
        "adapter_config_sha256": _sha256(config_path),
        "camera_frames": int(len(frame_ns)),
        "external_state_rows": int(len(states)),
        "trajectory_rows": int(len(trajectory_rows)),
        "visual_provider": None,
        "visual_measurements": 0,
        "metrics_path": metrics_path,
        "covariance_representation": str(
            config["covariance_semantics"]["representation"]
        ),
        "innovation_nis_available": False,
        "terminal_tracking_loss_observable": False,
        "reset_event_semantics": str(capabilities["reset_event"]["rationale"]),
        "relocalization_event_semantics": str(
            capabilities["relocalization_event"]["rationale"]
        ),
        "state_age_seconds": {
            "median": float(np.median(state_ages)),
            "max": float(np.max(state_ages)),
        },
        "source_artifact_sha256": {
            estimate.name: _sha256(estimate),
            deviation.name: _sha256(deviation),
        },
        "artifacts": {
            "trajectory": trajectory_path.name,
            "health": health_path.name,
            **({} if metrics_path is None else {"metrics": metrics_path}),
        },
        "claim_boundary": (
            "External OpenVINS total-state import. Covariance health is a diagonal "
            "approximation from published standard deviations; NIS and visual-tracking "
            "diagnostics are unavailable and remain missing."
        ),
    }
    (destination / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
