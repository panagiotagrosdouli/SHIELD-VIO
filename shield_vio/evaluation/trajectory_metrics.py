"""Trajectory alignment and position-error metrics for VIO evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Summary statistics for aligned position trajectories."""

    sample_count: int
    ate_rmse_m: float
    ate_mean_m: float
    ate_median_m: float
    ate_max_m: float
    rpe_rmse_m: float
    rpe_mean_m: float
    rpe_median_m: float
    rpe_max_m: float
    rpe_delta: int


def align_positions_se3(
    estimated_positions: np.ndarray,
    reference_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly align estimated positions to a reference trajectory.

    Uses the Kabsch solution without scale estimation. This is appropriate when
    evaluating metric VIO and prevents a scale fit from hiding estimator drift.

    Returns:
        ``(aligned_positions, rotation, translation)`` where the transform is
        applied as ``aligned = estimated @ rotation.T + translation``.
    """

    estimated, reference = _validated_pair(estimated_positions, reference_positions)
    estimated_centroid = estimated.mean(axis=0)
    reference_centroid = reference.mean(axis=0)
    estimated_centered = estimated - estimated_centroid
    reference_centered = reference - reference_centroid

    covariance = estimated_centered.T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T

    translation = reference_centroid - rotation @ estimated_centroid
    aligned = estimated @ rotation.T + translation
    return aligned, rotation, translation


def absolute_trajectory_error(
    estimated_positions: np.ndarray,
    reference_positions: np.ndarray,
    *,
    align: bool = True,
) -> np.ndarray:
    """Return per-sample Euclidean absolute trajectory error in metres."""

    estimated, reference = _validated_pair(estimated_positions, reference_positions)
    if align:
        estimated, _, _ = align_positions_se3(estimated, reference)
    return np.linalg.norm(estimated - reference, axis=1)


def relative_pose_error(
    estimated_positions: np.ndarray,
    reference_positions: np.ndarray,
    *,
    delta: int = 1,
    align: bool = True,
) -> np.ndarray:
    """Return translational relative pose error over a sample interval.

    ``delta`` is expressed in samples. Each error compares the displacement over
    ``delta`` samples in the estimate with the corresponding reference
    displacement.
    """

    estimated, reference = _validated_pair(estimated_positions, reference_positions)
    if not isinstance(delta, int) or delta < 1:
        raise ValueError("delta must be a positive integer")
    if delta >= len(estimated):
        raise ValueError("delta must be smaller than the trajectory length")
    if align:
        estimated, _, _ = align_positions_se3(estimated, reference)

    estimated_displacement = estimated[delta:] - estimated[:-delta]
    reference_displacement = reference[delta:] - reference[:-delta]
    return np.linalg.norm(estimated_displacement - reference_displacement, axis=1)


def summarize_trajectory_metrics(
    estimated_positions: np.ndarray,
    reference_positions: np.ndarray,
    *,
    rpe_delta: int = 1,
    align: bool = True,
) -> TrajectoryMetrics:
    """Compute aligned ATE and translational RPE summary statistics."""

    ate = absolute_trajectory_error(estimated_positions, reference_positions, align=align)
    rpe = relative_pose_error(
        estimated_positions,
        reference_positions,
        delta=rpe_delta,
        align=align,
    )
    return TrajectoryMetrics(
        sample_count=len(ate),
        ate_rmse_m=_rmse(ate),
        ate_mean_m=float(np.mean(ate)),
        ate_median_m=float(np.median(ate)),
        ate_max_m=float(np.max(ate)),
        rpe_rmse_m=_rmse(rpe),
        rpe_mean_m=float(np.mean(rpe)),
        rpe_median_m=float(np.median(rpe)),
        rpe_max_m=float(np.max(rpe)),
        rpe_delta=rpe_delta,
    )


def _validated_pair(
    estimated_positions: np.ndarray,
    reference_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    estimated = np.asarray(estimated_positions, dtype=float)
    reference = np.asarray(reference_positions, dtype=float)
    if estimated.ndim != 2 or estimated.shape[1] != 3:
        raise ValueError("estimated_positions must have shape (N, 3)")
    if reference.shape != estimated.shape:
        raise ValueError("reference_positions must match estimated_positions shape")
    if len(estimated) < 3:
        raise ValueError("at least three trajectory samples are required")
    if not np.all(np.isfinite(estimated)) or not np.all(np.isfinite(reference)):
        raise ValueError("trajectory positions must be finite")
    return estimated, reference


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))
