"""EuRoC MAV IMU loading utilities.

The EuRoC IMU CSV schema is::

    #timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],
    w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],
    a_RS_S_z [m s^-2]

This module performs strict validation because timestamp or column-order errors
can silently invalidate inertial-estimation experiments.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IMUSample:
    timestamp_s: float
    angular_velocity_rps: np.ndarray
    acceleration_mps2: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        for name in ("angular_velocity_rps", "acceleration_mps2"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3-vector")
            object.__setattr__(self, name, value.copy())


def load_euroc_imu(path: str | Path) -> tuple[IMUSample, ...]:
    """Load ``mav0/imu0/data.csv`` into strictly time-ordered samples."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    samples: list[IMUSample] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 7:
                raise ValueError(f"{source}:{line_number}: expected at least 7 columns")
            try:
                values = np.asarray([float(value.strip()) for value in row[:7]], dtype=float)
            except ValueError as exc:
                raise ValueError(f"{source}:{line_number}: non-numeric IMU row") from exc
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{source}:{line_number}: non-finite IMU value")

            sample = IMUSample(
                timestamp_s=float(values[0] * 1e-9),
                angular_velocity_rps=values[1:4],
                acceleration_mps2=values[4:7],
            )
            if samples and sample.timestamp_s <= samples[-1].timestamp_s:
                raise ValueError(f"{source}:{line_number}: timestamps must be strictly increasing")
            samples.append(sample)

    if len(samples) < 2:
        raise ValueError(f"{source}: at least two IMU samples are required")
    return tuple(samples)
