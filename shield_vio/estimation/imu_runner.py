"""Execute the ESKF propagation model over a timestamped IMU sequence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from shield_vio.datasets.euroc_imu import IMUSample
from shield_vio.estimation.error_state_ekf import ErrorStateEKF
from shield_vio.estimation.trajectory import TrajectoryRecorder


@dataclass(frozen=True)
class IMURunSummary:
    sample_count: int
    duration_s: float
    mean_dt_s: float
    max_dt_s: float


def run_imu_propagation(
    samples: Iterable[IMUSample],
    *,
    estimator: ErrorStateEKF | None = None,
    record_stride: int = 1,
    max_dt_s: float = 0.1,
) -> tuple[ErrorStateEKF, TrajectoryRecorder, IMURunSummary]:
    """Propagate an ESKF through IMU samples and record its trajectory.

    The first sample establishes the absolute dataset timestamp. Each subsequent
    measurement is integrated over the interval since the previous sample.
    Large gaps are rejected rather than silently destabilising the filter.
    """

    sequence = tuple(samples)
    if len(sequence) < 2:
        raise ValueError("at least two IMU samples are required")
    if record_stride <= 0:
        raise ValueError("record_stride must be positive")
    if max_dt_s <= 0 or not np.isfinite(max_dt_s):
        raise ValueError("max_dt_s must be finite and positive")

    filter_ = estimator or ErrorStateEKF()
    recorder = TrajectoryRecorder()
    initial_time = float(sequence[0].timestamp_s)
    filter_.state.timestamp_s = initial_time
    recorder.append(filter_.state)

    dts: list[float] = []
    for index, (previous, current) in enumerate(zip(sequence[:-1], sequence[1:]), start=1):
        dt_s = float(current.timestamp_s - previous.timestamp_s)
        if dt_s <= 0:
            raise ValueError("IMU timestamps must be strictly increasing")
        if dt_s > max_dt_s:
            raise ValueError(f"IMU gap {dt_s:.6f}s exceeds max_dt_s={max_dt_s:.6f}s")

        filter_.propagate(
            acceleration_mps2=current.acceleration_mps2,
            angular_velocity_rps=current.angular_velocity_rps,
            dt_s=dt_s,
        )
        # Avoid cumulative floating-point drift in the externally visible time.
        filter_.state.timestamp_s = float(current.timestamp_s)
        dts.append(dt_s)
        if index % record_stride == 0 or index == len(sequence) - 1:
            recorder.append(filter_.state)

    summary = IMURunSummary(
        sample_count=len(sequence),
        duration_s=float(sequence[-1].timestamp_s - initial_time),
        mean_dt_s=float(np.mean(dts)),
        max_dt_s=float(np.max(dts)),
    )
    return filter_, recorder, summary
