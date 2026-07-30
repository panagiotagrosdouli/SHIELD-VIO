"""Trajectory recording and export for estimator evaluation.

The recorder captures immutable snapshots of the ESKF state and exports both a
compact evaluation CSV and the standard TUM trajectory format.  It deliberately
keeps uncertainty and bias diagnostics in the CSV while producing a minimal TUM
file that can be consumed by external evaluation tools.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from shield_vio.estimation.error_state_ekf import ESKFState


@dataclass(frozen=True)
class TrajectorySample:
    timestamp_s: float
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    velocity_mps: np.ndarray
    accel_bias_mps2: np.ndarray
    gyro_bias_rps: np.ndarray
    position_covariance_trace_m2: float
    attitude_covariance_trace_rad2: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        for name in (
            "position_m",
            "velocity_mps",
            "accel_bias_mps2",
            "gyro_bias_rps",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3-vector")
            object.__setattr__(self, name, value.copy())

        quaternion = np.asarray(self.quaternion_wxyz, dtype=float)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("quaternion_wxyz must be a finite 4-vector")
        norm = float(np.linalg.norm(quaternion))
        if norm <= np.finfo(float).eps:
            raise ValueError("quaternion_wxyz must have non-zero norm")
        object.__setattr__(self, "quaternion_wxyz", quaternion / norm)

        if self.position_covariance_trace_m2 < 0 or not np.isfinite(
            self.position_covariance_trace_m2
        ):
            raise ValueError("position covariance trace must be finite and non-negative")
        if self.attitude_covariance_trace_rad2 < 0 or not np.isfinite(
            self.attitude_covariance_trace_rad2
        ):
            raise ValueError("attitude covariance trace must be finite and non-negative")

    @classmethod
    def from_eskf_state(cls, state: ESKFState) -> "TrajectorySample":
        covariance = np.asarray(state.covariance, dtype=float)
        if covariance.shape != (15, 15) or not np.all(np.isfinite(covariance)):
            raise ValueError("ESKF covariance must be a finite 15x15 matrix")
        return cls(
            timestamp_s=float(state.timestamp_s),
            position_m=np.asarray(state.position_m, dtype=float),
            quaternion_wxyz=np.asarray(state.quaternion_wxyz, dtype=float),
            velocity_mps=np.asarray(state.velocity_mps, dtype=float),
            accel_bias_mps2=np.asarray(state.accel_bias_mps2, dtype=float),
            gyro_bias_rps=np.asarray(state.gyro_bias_rps, dtype=float),
            position_covariance_trace_m2=float(np.trace(covariance[0:3, 0:3])),
            attitude_covariance_trace_rad2=float(np.trace(covariance[6:9, 6:9])),
        )


class TrajectoryRecorder:
    """Collect strictly time-ordered estimator states."""

    def __init__(self) -> None:
        self._samples: list[TrajectorySample] = []

    @property
    def samples(self) -> tuple[TrajectorySample, ...]:
        return tuple(self._samples)

    def append(self, state: ESKFState) -> TrajectorySample:
        sample = TrajectorySample.from_eskf_state(state)
        if self._samples and sample.timestamp_s <= self._samples[-1].timestamp_s:
            raise ValueError("trajectory timestamps must be strictly increasing")
        self._samples.append(sample)
        return sample

    def extend(self, states: Iterable[ESKFState]) -> None:
        for state in states:
            self.append(state)

    def write_tum(self, path: str | Path) -> Path:
        """Write ``timestamp tx ty tz qx qy qz qw`` rows."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            for sample in self._samples:
                px, py, pz = sample.position_m
                qw, qx, qy, qz = sample.quaternion_wxyz
                handle.write(
                    f"{sample.timestamp_s:.9f} {px:.9f} {py:.9f} {pz:.9f} "
                    f"{qx:.12f} {qy:.12f} {qz:.12f} {qw:.12f}\n"
                )
        return destination

    def write_csv(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "timestamp_s",
            "position_x_m",
            "position_y_m",
            "position_z_m",
            "quaternion_w",
            "quaternion_x",
            "quaternion_y",
            "quaternion_z",
            "velocity_x_mps",
            "velocity_y_mps",
            "velocity_z_mps",
            "accel_bias_x_mps2",
            "accel_bias_y_mps2",
            "accel_bias_z_mps2",
            "gyro_bias_x_rps",
            "gyro_bias_y_rps",
            "gyro_bias_z_rps",
            "position_covariance_trace_m2",
            "attitude_covariance_trace_rad2",
        ]
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for sample in self._samples:
                writer.writerow(
                    [
                        sample.timestamp_s,
                        *sample.position_m,
                        *sample.quaternion_wxyz,
                        *sample.velocity_mps,
                        *sample.accel_bias_mps2,
                        *sample.gyro_bias_rps,
                        sample.position_covariance_trace_m2,
                        sample.attitude_covariance_trace_rad2,
                    ]
                )
        return destination
