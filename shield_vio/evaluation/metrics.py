"""Evaluation metrics for trajectories and failure prediction.

The functions in this module are dependency-light and deterministic so they can
be reused by synthetic experiments, public-dataset benchmarks, and CI tests.
Trajectory metrics operate on positions and support the rigid alignment that is
normally required when evaluating monocular or visual--inertial estimators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionMetrics:
    """Binary failure-prediction metrics.

    ``time_to_detection`` is measured from the first positive label to the first
    true-positive prediction. ``warning_lead_time`` is positive only when a
    prediction occurs before the first labelled failure while still inside the
    optional warning window.
    """

    precision: float
    recall: float
    false_alarm_rate: float
    time_to_detection: float | None
    f1: float = 0.0
    specificity: float = 0.0
    accuracy: float = 0.0
    warning_lead_time: float | None = None
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0


def _positions(trajectory: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(trajectory, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty Nx3+ array")
    positions = array[:, :3]
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"{name} contains non-finite positions")
    return positions


def align_positions(est: np.ndarray, gt: np.ndarray, *, with_scale: bool = False) -> np.ndarray:
    """Align estimated positions to ground truth using Umeyama alignment.

    Args:
        est: Estimated trajectory as an ``Nx3+`` array.
        gt: Ground-truth trajectory with matching shape.
        with_scale: Estimate a global scale in addition to rotation and
            translation. This should normally be ``False`` for metric VIO.

    Returns:
        Aligned estimated positions as an ``Nx3`` array.
    """

    source = _positions(est, "est")
    target = _positions(gt, "gt")
    if source.shape != target.shape:
        raise ValueError("est and gt must contain the same number of positions")
    if len(source) < 3:
        raise ValueError("at least three poses are required for alignment")

    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt

    scale = 1.0
    if with_scale:
        variance = float(np.mean(np.sum(source_centered**2, axis=1)))
        if variance <= np.finfo(float).eps:
            raise ValueError("cannot estimate scale from a degenerate trajectory")
        scale = float(np.sum(singular_values * np.diag(correction)) / variance)

    translation = target_mean - scale * (rotation @ source_mean)
    return (scale * (rotation @ source.T)).T + translation


def ate(
    est: np.ndarray,
    gt: np.ndarray,
    *,
    align: bool = True,
    with_scale: bool = False,
) -> float:
    """Return translational absolute trajectory error RMSE in metres."""

    estimated = _positions(est, "est")
    ground_truth = _positions(gt, "gt")
    if estimated.shape != ground_truth.shape:
        raise ValueError("ATE expects matching trajectories")
    if align:
        estimated = align_positions(estimated, ground_truth, with_scale=with_scale)
    errors = np.linalg.norm(estimated - ground_truth, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def rpe(
    est: np.ndarray,
    gt: np.ndarray,
    delta: int = 1,
    *,
    align: bool = False,
    with_scale: bool = False,
) -> float:
    """Return translational relative pose error RMSE in metres.

    RPE is computed from displacement differences separated by ``delta`` poses.
    Rigid alignment is optional because translation cancels in the relative
    displacement, while a frame rotation may still need to be removed.
    """

    estimated = _positions(est, "est")
    ground_truth = _positions(gt, "gt")
    if estimated.shape != ground_truth.shape or delta <= 0 or len(estimated) <= delta:
        raise ValueError("invalid trajectories or delta")
    if align:
        estimated = align_positions(estimated, ground_truth, with_scale=with_scale)
    estimated_delta = estimated[delta:] - estimated[:-delta]
    ground_truth_delta = ground_truth[delta:] - ground_truth[:-delta]
    errors = np.linalg.norm(estimated_delta - ground_truth_delta, axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def failure_detection_metrics(
    pred: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray | None = None,
    *,
    warning_window: float | None = None,
) -> DetectionMetrics:
    """Compute sample-level binary metrics and temporal warning statistics.

    ``pred`` and ``labels`` must be one-dimensional boolean-compatible arrays.
    Timestamps must be finite, strictly increasing, and expressed in a
    consistent time unit. When ``warning_window`` is provided, a prediction in
    ``[failure_time - warning_window, failure_time)`` is treated as an early
    warning for the temporal statistic only; the sample-level confusion matrix
    remains unchanged.
    """

    predictions = np.asarray(pred, dtype=bool)
    targets = np.asarray(labels, dtype=bool)
    if predictions.ndim != 1 or targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("pred and labels must be matching one-dimensional arrays")
    if len(predictions) == 0:
        raise ValueError("pred and labels must not be empty")

    tp = int(np.sum(predictions & targets))
    fp = int(np.sum(predictions & ~targets))
    fn = int(np.sum(~predictions & targets))
    tn = int(np.sum(~predictions & ~targets))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(predictions)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    time_to_detection: float | None = None
    warning_lead_time: float | None = None
    if timestamps is not None:
        times = np.asarray(timestamps, dtype=float)
        if times.ndim != 1 or times.shape != predictions.shape:
            raise ValueError("timestamps must match pred and labels")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("timestamps must be finite and strictly increasing")

        failure_indices = np.flatnonzero(targets)
        if len(failure_indices):
            first_failure = int(failure_indices[0])
            true_positive_indices = np.flatnonzero(predictions & targets)
            if len(true_positive_indices):
                first_detection = int(true_positive_indices[0])
                time_to_detection = max(0.0, float(times[first_detection] - times[first_failure]))

            if warning_window is not None:
                if warning_window < 0:
                    raise ValueError("warning_window must be non-negative")
                failure_time = float(times[first_failure])
                early = np.flatnonzero(
                    predictions
                    & (times < failure_time)
                    & (times >= failure_time - warning_window)
                )
                if len(early):
                    warning_lead_time = float(failure_time - times[int(early[0])])

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        false_alarm_rate=false_alarm_rate,
        time_to_detection=time_to_detection,
        f1=f1,
        specificity=specificity,
        accuracy=accuracy,
        warning_lead_time=warning_lead_time,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )
