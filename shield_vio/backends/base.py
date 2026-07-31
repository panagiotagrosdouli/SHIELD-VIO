"""Estimator backend contracts shared by dataset runners and safety monitors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EstimatorState:
    """Immutable estimator snapshot suitable for logging and backend interchange."""

    timestamp_ns: int
    position: np.ndarray
    velocity: np.ndarray
    orientation_wxyz: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class EstimatorHealth:
    """Backend-neutral health signals consumed by SHIELD-VIO monitors."""

    timestamp_ns: int
    initialized: bool
    tracking_status: str
    propagated_imu_samples: int
    covariance_trace: float
    covariance_condition_number: float
    innovation_nis: float | None


@runtime_checkable
class EstimatorBackend(Protocol):
    """Minimal contract for ESKF and external VIO implementations."""

    @property
    def name(self) -> str:
        """Return the stable backend identifier used in manifests."""

    def initialize(self, timestamp_ns: int) -> EstimatorState:
        """Initialize the estimator at a timestamp and return its first state."""

    def process_imu(
        self,
        timestamp_ns: int,
        linear_acceleration_m_s2: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
    ) -> EstimatorState:
        """Consume one timestamped IMU sample and return the propagated state."""

    def process_visual_update(
        self,
        timestamp_ns: int,
        residual: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> EstimatorState:
        """Apply a visual measurement update and return the corrected state."""

    def snapshot(self) -> EstimatorState:
        """Return an immutable copy of the current estimator state."""

    def health(self) -> EstimatorHealth:
        """Return backend-neutral health diagnostics for the current state."""
