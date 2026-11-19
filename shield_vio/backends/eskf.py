"""Adapter exposing the built-in ESKF through the estimator backend contract."""
from __future__ import annotations

import numpy as np

from shield_vio.backends.base import EstimatorHealth, EstimatorState
from shield_vio.estimation.eskf import ErrorStateEKF
from shield_vio.estimation.state import VIOState
from shield_vio.imu.model import IMUNoiseModel
from shield_vio.core.math import normalize_quaternion


class ESKFBackend:
    """Stateful adapter around :class:`ErrorStateEKF`."""

    def __init__(
        self,
        noise: IMUNoiseModel | None = None,
        gravity: np.ndarray | None = None,
    ) -> None:
        self._filter = ErrorStateEKF(noise=noise, gravity=gravity)
        self._state = VIOState()
        self._timestamp_ns: int | None = None
        self._propagated_imu_samples = 0

    @property
    def name(self) -> str:
        return "eskf"

    def initialize(self, timestamp_ns: int) -> EstimatorState:
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        self._state = VIOState()
        self._timestamp_ns = int(timestamp_ns)
        self._propagated_imu_samples = 0
        self._filter.last_innovation = None
        return self.snapshot()

    def initialize_from_stationary_imu(
        self,
        accelerations: np.ndarray,
        angular_velocities: np.ndarray,
        *,
        accel_tolerance_mps2: float = 1.0,
        accel_std_max_mps2: float = 0.75,
        gyro_threshold_rps: float = 0.25,
    ) -> EstimatorState:
        """Infer initial roll/pitch from stationary IMU measurements only.

        World yaw is unobservable from gravity and remains arbitrary. No ground
        truth is used. Reject moving or implausible windows instead of silently
        treating arbitrary motion acceleration as gravity.
        """
        self._require_initialized()
        accel = np.asarray(accelerations, dtype=float)
        gyro = np.asarray(angular_velocities, dtype=float)
        if (
            accel.ndim != 2 or accel.shape[1] != 3
            or gyro.shape != accel.shape or len(accel) < 10
            or not np.all(np.isfinite(accel)) or not np.all(np.isfinite(gyro))
        ):
            raise ValueError("Gravity alignment requires >=10 finite paired IMU measurements")
        mean_accel = np.mean(accel, axis=0)
        norm = float(np.linalg.norm(mean_accel))
        gravity_norm = float(np.linalg.norm(self._filter.gravity))
        norm_std = float(np.std(np.linalg.norm(accel, axis=1)))
        gyro_median = float(np.median(np.linalg.norm(gyro, axis=1)))
        if (
            gravity_norm <= 0
            or abs(norm - gravity_norm) > accel_tolerance_mps2
            or norm_std > accel_std_max_mps2
            or gyro_median > gyro_threshold_rps
        ):
            raise ValueError(
                "IMU initialization window is not sufficiently stationary; "
                "gravity-based attitude is not identifiable from dynamic acceleration"
            )
        source = mean_accel / norm
        target = -self._filter.gravity / gravity_norm
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if dot < -1.0 + 1e-8:
            basis = np.eye(3)[int(np.argmin(np.abs(source)))]
            axis = np.cross(source, basis)
            axis /= np.linalg.norm(axis)
            quat = np.r_[0.0, axis]
        else:
            quat = np.r_[1.0 + dot, np.cross(source, target)]
        self._state.orientation = normalize_quaternion(quat)
        return self.snapshot()

    def process_imu(
        self,
        timestamp_ns: int,
        linear_acceleration_m_s2: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
    ) -> EstimatorState:
        previous_timestamp_ns = self._require_initialized()
        if timestamp_ns <= previous_timestamp_ns:
            raise ValueError("IMU timestamps must be strictly increasing")

        dt = (timestamp_ns - previous_timestamp_ns) * 1e-9
        self._state = self._filter.propagate(
            self._state,
            accel_m=linear_acceleration_m_s2,
            gyro_m=angular_velocity_rad_s,
            dt=dt,
        )
        self._timestamp_ns = int(timestamp_ns)
        self._propagated_imu_samples += 1
        return self.snapshot()

    def process_visual_update(
        self,
        timestamp_ns: int,
        residual: np.ndarray,
        measurement_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> EstimatorState:
        current_timestamp_ns = self._require_initialized()
        if timestamp_ns < current_timestamp_ns:
            raise ValueError("Visual update timestamp cannot precede the estimator state")

        self._state = self._filter.update_linear(
            self._state,
            residual=residual,
            H=measurement_matrix,
            R=measurement_covariance,
        )
        self._timestamp_ns = int(timestamp_ns)
        return self.snapshot()

    def snapshot(self) -> EstimatorState:
        timestamp_ns = self._require_initialized()
        state = self._state
        return EstimatorState(
            timestamp_ns=timestamp_ns,
            position=np.copy(state.position),
            velocity=np.copy(state.velocity),
            orientation_wxyz=np.copy(state.orientation),
            accel_bias=np.copy(state.accel_bias),
            gyro_bias=np.copy(state.gyro_bias),
            covariance=np.copy(state.covariance),
        )

    def health(self) -> EstimatorHealth:
        timestamp_ns = self._require_initialized()
        covariance = self._state.covariance
        symmetric_covariance = 0.5 * (covariance + covariance.T)
        innovation = self._filter.last_innovation
        return EstimatorHealth(
            timestamp_ns=timestamp_ns,
            initialized=True,
            tracking_status="tracking",
            propagated_imu_samples=self._propagated_imu_samples,
            covariance_trace=float(np.trace(covariance)),
            covariance_condition_number=float(np.linalg.cond(covariance)),
            innovation_nis=None if innovation is None else float(innovation.nis),
            covariance_min_eigenvalue=float(np.min(np.linalg.eigvalsh(symmetric_covariance))),
            covariance_symmetry_error=float(np.max(np.abs(covariance - covariance.T))),
            terminal_tracking_loss_observable=False,
            reset_event=False,
            relocalization_event=False,
        )

    def _require_initialized(self) -> int:
        if self._timestamp_ns is None:
            raise RuntimeError("Estimator backend must be initialized before use")
        return self._timestamp_ns
