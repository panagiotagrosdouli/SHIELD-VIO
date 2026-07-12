"""Transparent failure-detector baselines without heavyweight dependencies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogisticFailureDetector:
    learning_rate: float = 0.05
    iterations: int = 500
    l2: float = 1e-3

    def __post_init__(self) -> None:
        self.weights: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticFailureDetector":
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.size or x.shape[0] < 2:
            raise ValueError("features must be 2-D with one binary label per row")
        if np.any((y != 0.0) & (y != 1.0)):
            raise ValueError("labels must be binary")
        self.mean = np.mean(x, axis=0)
        self.scale = np.std(x, axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        z = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(z.shape[0]), z])
        self.weights = np.zeros(design.shape[1], dtype=float)
        for _ in range(self.iterations):
            prediction = self._sigmoid(design @ self.weights)
            gradient = design.T @ (prediction - y) / y.size
            gradient[1:] += self.l2 * self.weights[1:]
            self.weights -= self.learning_rate * gradient
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None or self.mean is None or self.scale is None:
            raise RuntimeError("fit must be called before predict_proba")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.mean.size:
            raise ValueError("feature dimensions do not match fitted model")
        z = (x - self.mean) / self.scale
        design = np.column_stack([np.ones(z.shape[0]), z])
        return self._sigmoid(design @ self.weights)


@dataclass(frozen=True)
class RuleBasedDetector:
    nis_threshold: float = 11.345
    covariance_threshold: float = 1.0
    visual_quality_threshold: float = 0.35
    imu_health_threshold: float = 0.4

    def score(
        self,
        nis: float,
        covariance_trace: float,
        visual_quality: float,
        imu_health: float,
    ) -> float:
        terms = np.array(
            [
                min(max(nis / self.nis_threshold, 0.0), 2.0) / 2.0,
                min(max(covariance_trace / self.covariance_threshold, 0.0), 2.0) / 2.0,
                1.0 - min(max(visual_quality, 0.0), 1.0),
                1.0 - min(max(imu_health, 0.0), 1.0),
            ]
        )
        return float(np.clip(np.mean(terms), 0.0, 1.0))

    def predict(self, **signals: float) -> bool:
        return self.score(**signals) >= 0.5
