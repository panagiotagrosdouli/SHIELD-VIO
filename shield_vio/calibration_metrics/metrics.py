"""Calibration and binary failure-prediction metrics.

All inputs are NumPy arrays. Probabilities are clipped only for logarithmic
metrics; Brier score and ECE use the supplied values after range validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationSummary:
    brier_score: float
    negative_log_likelihood: float
    expected_calibration_error: float
    maximum_calibration_error: float
    sample_count: int


def _validate(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities and labels must be non-empty and equally sized")
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("labels must be binary")
    return p, y


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p, y = _validate(probabilities, labels)
    return float(np.mean((p - y) ** 2))


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p, y = _validate(probabilities, labels)
    eps = np.finfo(float).eps
    q = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1.0 - q)))


def calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 10
) -> tuple[float, float]:
    p, y = _validate(probabilities, labels)
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    for index in range(num_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (p >= lower) & ((p < upper) if index < num_bins - 1 else (p <= upper))
        if not np.any(mask):
            continue
        gap = abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
        ece += float(np.mean(mask)) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def summarize_calibration(
    probabilities: np.ndarray, labels: np.ndarray, num_bins: int = 10
) -> CalibrationSummary:
    p, y = _validate(probabilities, labels)
    ece, mce = calibration_error(p, y, num_bins=num_bins)
    return CalibrationSummary(
        brier_score=brier_score(p, y),
        negative_log_likelihood=negative_log_likelihood(p, y),
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        sample_count=int(p.size),
    )
