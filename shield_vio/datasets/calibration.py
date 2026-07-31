"""Typed parsing for EuRoC ``sensor.yaml`` calibration files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraCalibration:
    sensor_type: str
    comment: str
    transform_body_sensor: np.ndarray
    rate_hz: float
    resolution: tuple[int, int]
    camera_model: str
    intrinsics: np.ndarray
    distortion_model: str
    distortion_coefficients: np.ndarray


@dataclass(frozen=True)
class ImuCalibration:
    sensor_type: str
    comment: str
    transform_body_sensor: np.ndarray
    rate_hz: float
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    accelerometer_noise_density: float
    accelerometer_random_walk: float


def _load_yaml(path: str | Path) -> dict[str, Any]:
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    with calibration_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {calibration_path}")
    return data


def _required(data: dict[str, Any], key: str, path: Path) -> Any:
    if key not in data:
        raise ValueError(f"Missing calibration key {key!r} in {path}")
    return data[key]


def _transform(data: dict[str, Any], path: Path) -> np.ndarray:
    raw = _required(data, "T_BS", path)
    if isinstance(raw, dict):
        rows = int(_required(raw, "rows", path))
        cols = int(_required(raw, "cols", path))
        values = _required(raw, "data", path)
        matrix = np.asarray(values, dtype=float).reshape(rows, cols)
    else:
        matrix = np.asarray(raw, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"T_BS must be 4x4 in {path}; received {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"T_BS contains non-finite values in {path}")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError(f"T_BS has an invalid homogeneous final row in {path}")
    return matrix


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    calibration_path = Path(path)
    data = _load_yaml(calibration_path)
    resolution = tuple(int(value) for value in _required(data, "resolution", calibration_path))
    if len(resolution) != 2:
        raise ValueError(f"Camera resolution must contain width and height in {calibration_path}")
    intrinsics = np.asarray(_required(data, "intrinsics", calibration_path), dtype=float)
    distortion = np.asarray(
        _required(data, "distortion_coefficients", calibration_path), dtype=float
    )
    if intrinsics.shape != (4,):
        raise ValueError(f"Expected [fu, fv, cu, cv] intrinsics in {calibration_path}")
    return CameraCalibration(
        sensor_type=str(_required(data, "sensor_type", calibration_path)),
        comment=str(data.get("comment", "")),
        transform_body_sensor=_transform(data, calibration_path),
        rate_hz=float(_required(data, "rate_hz", calibration_path)),
        resolution=(resolution[0], resolution[1]),
        camera_model=str(_required(data, "camera_model", calibration_path)),
        intrinsics=intrinsics,
        distortion_model=str(_required(data, "distortion_model", calibration_path)),
        distortion_coefficients=distortion,
    )


def load_imu_calibration(path: str | Path) -> ImuCalibration:
    calibration_path = Path(path)
    data = _load_yaml(calibration_path)
    return ImuCalibration(
        sensor_type=str(_required(data, "sensor_type", calibration_path)),
        comment=str(data.get("comment", "")),
        transform_body_sensor=_transform(data, calibration_path),
        rate_hz=float(_required(data, "rate_hz", calibration_path)),
        gyroscope_noise_density=float(
            _required(data, "gyroscope_noise_density", calibration_path)
        ),
        gyroscope_random_walk=float(
            _required(data, "gyroscope_random_walk", calibration_path)
        ),
        accelerometer_noise_density=float(
            _required(data, "accelerometer_noise_density", calibration_path)
        ),
        accelerometer_random_walk=float(
            _required(data, "accelerometer_random_walk", calibration_path)
        ),
    )
