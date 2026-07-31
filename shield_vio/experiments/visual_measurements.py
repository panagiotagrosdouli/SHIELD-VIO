"""Backend-neutral visual measurement contracts for experiment runners."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from shield_vio.backends.base import EstimatorState
from shield_vio.datasets.euroc import SynchronizedFrame


@dataclass(frozen=True)
class LinearVisualMeasurement:
    """Linearized visual constraint ready for an estimator backend update."""

    residual: np.ndarray
    measurement_matrix: np.ndarray
    measurement_covariance: np.ndarray
    detected_features: int = 0
    tracked_features: int = 0
    correspondence_count: int = 0
    inlier_count: int = 0
    status: str = "measurement"

    def __post_init__(self) -> None:
        residual = np.asarray(self.residual, dtype=float)
        matrix = np.asarray(self.measurement_matrix, dtype=float)
        covariance = np.asarray(self.measurement_covariance, dtype=float)
        if residual.ndim != 1 or not len(residual):
            raise ValueError("residual must be a non-empty vector")
        if matrix.ndim != 2 or matrix.shape[0] != len(residual):
            raise ValueError("measurement_matrix rows must match residual size")
        if covariance.shape != (len(residual), len(residual)):
            raise ValueError("measurement_covariance must match residual size")
        if not all(np.all(np.isfinite(value)) for value in (residual, matrix, covariance)):
            raise ValueError("visual measurement arrays must be finite")
        if min(
            self.detected_features,
            self.tracked_features,
            self.correspondence_count,
            self.inlier_count,
        ) < 0:
            raise ValueError("visual diagnostic counts must be non-negative")
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "measurement_matrix", matrix)
        object.__setattr__(self, "measurement_covariance", covariance)

    @property
    def inlier_ratio(self) -> float | None:
        if self.correspondence_count == 0:
            return None
        return self.inlier_count / self.correspondence_count


@runtime_checkable
class VisualMeasurementProvider(Protocol):
    """Produce an optional visual constraint at a synchronized camera frame."""

    @property
    def name(self) -> str:
        """Return a stable provider identifier for experiment manifests."""

    def measure(
        self,
        packet: SynchronizedFrame,
        state: EstimatorState,
    ) -> LinearVisualMeasurement | None:
        """Return a constraint or ``None`` when no update should be attempted."""
