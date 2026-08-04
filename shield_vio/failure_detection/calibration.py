"""Held-out probability calibration for failure-prediction scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlattCalibrator:
    """Fit a logistic mapping from uncalibrated probabilities on a held-out split."""

    l2: float = 1e-6
    max_iterations: int = 100
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        self.intercept_: float | None = None
        self.slope_: float | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        p, y = _validate(probabilities, labels)
        if len(np.unique(y)) != 2:
            raise ValueError("Platt calibration requires both label classes")
        logits = _logit(p)
        design = np.column_stack([np.ones(len(logits)), logits])
        weights = np.zeros(2, dtype=float)
        penalty = np.diag([0.0, self.l2])
        for _ in range(self.max_iterations):
            fitted = _sigmoid(design @ weights)
            gradient = design.T @ (fitted - y) + penalty @ weights
            variance = np.maximum(fitted * (1.0 - fitted), 1e-8)
            hessian = design.T @ (variance[:, None] * design) + penalty
            step = np.linalg.solve(hessian, gradient)
            weights -= step
            if float(np.linalg.norm(step)) <= self.tolerance:
                break
        self.intercept_, self.slope_ = float(weights[0]), float(weights[1])
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.slope_ is None:
            raise RuntimeError("fit must be called before predict_proba")
        values = np.asarray(probabilities, dtype=float).reshape(-1)
        if len(values) == 0 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError("probabilities must be finite values in [0, 1]")
        return _sigmoid(self.intercept_ + self.slope_ * _logit(values))

    def to_dict(self) -> dict[str, float | str]:
        if self.intercept_ is None or self.slope_ is None:
            raise RuntimeError("fit must be called before serialization")
        return {
            "method": "platt_logit",
            "intercept": self.intercept_,
            "slope": self.slope_,
            "l2": self.l2,
        }


def _validate(probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    if len(p) < 2 or p.shape != y.shape or np.any(~np.isfinite(p)):
        raise ValueError("calibration inputs must be finite, equal vectors")
    if np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("probabilities and binary labels must be in [0, 1]")
    return p, y


def _logit(probabilities: np.ndarray) -> np.ndarray:
    eps = 1e-6
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))
