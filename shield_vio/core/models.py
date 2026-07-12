"""Typed, validated data models for SHIELD-VIO research pipelines."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _vec(x: Any, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite vector of length {n}")
    return a


def _cov(x: Any, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.shape != (n, n) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite {n}x{n} matrix")
    if not np.allclose(a, a.T, atol=1e-9):
        raise ValueError(f"{name} must be symmetric")
    if np.min(np.linalg.eigvalsh(a)) < -1e-9:
        raise ValueError(f"{name} must be positive semidefinite")
    return a


@dataclass
class IMUState:
    timestamp_s: float
    frame_id: str
    position_m: np.ndarray
    velocity_mps: np.ndarray
    quaternion_wxyz: np.ndarray
    accel_bias_mps2: np.ndarray
    gyro_bias_rps: np.ndarray

    def __post_init__(self) -> None:
        self.position_m = _vec(self.position_m, 3, "position_m")
        self.velocity_mps = _vec(self.velocity_mps, 3, "velocity_mps")
        q = _vec(self.quaternion_wxyz, 4, "quaternion_wxyz")
        n = np.linalg.norm(q)
        if n < 1e-12:
            raise ValueError("quaternion norm is zero")
        self.quaternion_wxyz = q / n
        self.accel_bias_mps2 = _vec(self.accel_bias_mps2, 3, "accel_bias_mps2")
        self.gyro_bias_rps = _vec(self.gyro_bias_rps, 3, "gyro_bias_rps")
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


@dataclass
class IMUMeasurement:
    timestamp_s: float
    frame_id: str
    acceleration_mps2: np.ndarray
    angular_velocity_rps: np.ndarray
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6))

    def __post_init__(self) -> None:
        self.acceleration_mps2 = _vec(self.acceleration_mps2, 3, "acceleration_mps2")
        self.angular_velocity_rps = _vec(self.angular_velocity_rps, 3, "angular_velocity_rps")
        self.covariance = _cov(self.covariance, 6, "covariance")


@dataclass
class FeatureObservation:
    timestamp_s: float
    camera_frame_id: str
    track_id: int
    pixel_xy: np.ndarray
    age_frames: int
    reprojection_error_px: float = 0.0

    def __post_init__(self) -> None:
        self.pixel_xy = _vec(self.pixel_xy, 2, "pixel_xy")
        if self.track_id < 0 or self.age_frames < 1:
            raise ValueError("track_id must be non-negative and age_frames >= 1")


@dataclass
class InnovationRecord:
    timestamp_s: float
    residual: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        self.residual = np.asarray(self.residual, dtype=float).reshape(-1)
        self.covariance = _cov(self.covariance, self.residual.size, "innovation covariance")

    @property
    def nis(self) -> float:
        return float(self.residual @ np.linalg.solve(self.covariance, self.residual))


@dataclass
class FailurePrediction:
    timestamp_s: float
    horizon_s: float
    diagnostic_score: float
    calibrated_probability: float | None
    confidence: float
    method: str
    dominant_factors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.diagnostic_score <= 1.0:
            raise ValueError("diagnostic_score must be in [0,1]")
        if self.calibrated_probability is not None and not 0.0 <= self.calibrated_probability <= 1.0:
            raise ValueError("calibrated_probability must be in [0,1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")


@dataclass
class ExperimentManifest:
    name: str
    seed: int
    command: str
    backend: str
    detector: str
    sequence: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
