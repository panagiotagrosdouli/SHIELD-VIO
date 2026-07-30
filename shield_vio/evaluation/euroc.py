"""EuRoC trajectory I/O and timestamp association utilities.

This module deliberately separates public-dataset parsing from a particular VIO
backend. Estimators may export timestamped positions in TUM-style text or CSV;
the resulting trajectory is associated with EuRoC ground truth by linear
interpolation on the ground-truth timeline.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TimestampedTrajectory:
    """A strictly time-ordered position trajectory."""

    timestamps: np.ndarray
    positions: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        positions = np.asarray(self.positions, dtype=float)
        if timestamps.ndim != 1 or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("trajectory must contain timestamps (N,) and positions (N, 3)")
        if len(timestamps) != len(positions) or len(timestamps) < 2:
            raise ValueError("trajectory timestamps and positions must have matching length >= 2")
        if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(positions)):
            raise ValueError("trajectory contains non-finite values")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("trajectory timestamps must be strictly increasing")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "positions", positions)


def _data_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = next(csv.reader([stripped])) if "," in stripped else stripped.split()
            rows.append([item.strip() for item in row])
    return rows


def load_euroc_ground_truth(sequence_root: str | Path) -> TimestampedTrajectory:
    """Load EuRoC body-frame ground-truth positions.

    The expected file is
    ``mav0/state_groundtruth_estimate0/data.csv``. EuRoC timestamps are stored in
    nanoseconds and are converted to seconds relative to the first sample. The
    first three state columns are the world-frame body position in metres.
    """

    root = Path(sequence_root)
    path = root / "mav0/state_groundtruth_estimate0/data.csv"
    rows = _data_rows(path)
    if not rows:
        raise ValueError(f"ground-truth file contains no samples: {path}")
    try:
        timestamps_ns = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
        positions = np.asarray([[float(row[1]), float(row[2]), float(row[3])] for row in rows])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid EuRoC ground-truth row in {path}") from exc
    timestamps = (timestamps_ns - timestamps_ns[0]).astype(float) * 1e-9
    return TimestampedTrajectory(timestamps, positions)


def load_estimator_trajectory(
    path: str | Path,
    *,
    timestamp_unit: str = "seconds",
    relative_time: bool = True,
) -> TimestampedTrajectory:
    """Load estimator output containing ``timestamp x y z`` as CSV or whitespace.

    Additional columns (for example quaternion components) are accepted and
    ignored by position-only evaluation. ``timestamp_unit`` may be ``seconds``,
    ``milliseconds``, ``microseconds``, or ``nanoseconds``.
    """

    trajectory_path = Path(path)
    rows = _data_rows(trajectory_path)
    if not rows:
        raise ValueError(f"estimator trajectory contains no samples: {trajectory_path}")
    factors = {
        "seconds": 1.0,
        "milliseconds": 1e-3,
        "microseconds": 1e-6,
        "nanoseconds": 1e-9,
    }
    if timestamp_unit not in factors:
        raise ValueError(f"unsupported timestamp unit: {timestamp_unit}")
    try:
        timestamps = np.asarray([float(row[0]) for row in rows]) * factors[timestamp_unit]
        positions = np.asarray([[float(row[1]), float(row[2]), float(row[3])] for row in rows])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid estimator trajectory row in {trajectory_path}") from exc
    if relative_time:
        timestamps = timestamps - timestamps[0]
    return TimestampedTrajectory(timestamps, positions)


def associate_ground_truth(
    estimate: TimestampedTrajectory,
    ground_truth: TimestampedTrajectory,
    *,
    max_gap: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Associate estimate samples with interpolated ground-truth positions.

    Estimate samples outside the ground-truth interval are discarded. A sample
    is also discarded when the nearest ground-truth observation is farther than
    ``max_gap`` seconds. This prevents interpolation across missing data or large
    timestamp offsets.

    Returns:
        ``(timestamps, estimated_positions, ground_truth_positions)``.
    """

    if max_gap <= 0:
        raise ValueError("max_gap must be positive")
    gt_t = ground_truth.timestamps
    est_t = estimate.timestamps
    inside = (est_t >= gt_t[0]) & (est_t <= gt_t[-1])
    candidate_indices = np.flatnonzero(inside)
    if not len(candidate_indices):
        raise ValueError("estimate and ground truth do not overlap in time")

    candidate_times = est_t[candidate_indices]
    right = np.searchsorted(gt_t, candidate_times, side="left")
    right = np.clip(right, 1, len(gt_t) - 1)
    left = right - 1
    nearest_gap = np.minimum(candidate_times - gt_t[left], gt_t[right] - candidate_times)
    valid = nearest_gap <= max_gap
    selected = candidate_indices[valid]
    if len(selected) < 3:
        raise ValueError("fewer than three trajectory samples satisfy timestamp association")

    times = est_t[selected]
    interpolated = np.column_stack(
        [np.interp(times, gt_t, ground_truth.positions[:, axis]) for axis in range(3)]
    )
    return times, estimate.positions[selected], interpolated
