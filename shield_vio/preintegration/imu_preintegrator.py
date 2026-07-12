"""Bias-aware discrete IMU preintegration research prototype.

Uses piecewise-constant body-frame measurements and first-order covariance/Jacobian
propagation. This implementation is intended for numerical validation, not production use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(theta) / theta * K + (1.0 - np.cos(theta)) / theta**2 * K @ K


@dataclass
class PreintegratedMeasurement:
    delta_t_s: float = 0.0
    delta_position_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((9, 9)))
    jacobian_bias_accel: np.ndarray = field(default_factory=lambda: np.zeros((9, 3)))
    jacobian_bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros((9, 3)))


class IMUPreintegrator:
    def __init__(self, accel_noise_density: float, gyro_noise_density: float) -> None:
        if accel_noise_density <= 0 or gyro_noise_density <= 0:
            raise ValueError("noise densities must be positive")
        self.sigma_a = float(accel_noise_density)
        self.sigma_g = float(gyro_noise_density)
        self.reset()

    def reset(self) -> None:
        self.measurement = PreintegratedMeasurement()

    def integrate(self, acceleration_mps2: np.ndarray, angular_velocity_rps: np.ndarray,
                  dt_s: float, accel_bias_mps2: np.ndarray | None = None,
                  gyro_bias_rps: np.ndarray | None = None) -> None:
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be positive and finite")
        ba = np.zeros(3) if accel_bias_mps2 is None else np.asarray(accel_bias_mps2, dtype=float)
        bg = np.zeros(3) if gyro_bias_rps is None else np.asarray(gyro_bias_rps, dtype=float)
        a = np.asarray(acceleration_mps2, dtype=float) - ba
        w = np.asarray(angular_velocity_rps, dtype=float) - bg
        if a.shape != (3,) or w.shape != (3,):
            raise ValueError("IMU vectors must have shape (3,)")

        m = self.measurement
        R0 = m.delta_rotation.copy()
        v0 = m.delta_velocity_mps.copy()
        m.delta_position_m += v0 * dt_s + 0.5 * (R0 @ a) * dt_s**2
        m.delta_velocity_mps += (R0 @ a) * dt_s
        m.delta_rotation = R0 @ so3_exp(w * dt_s)
        m.delta_t_s += dt_s

        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt_s
        F[0:3, 6:9] = -0.5 * R0 @ skew(a) * dt_s**2
        F[3:6, 6:9] = -R0 @ skew(a) * dt_s
        G = np.zeros((9, 6))
        G[0:3, 0:3] = 0.5 * R0 * dt_s**2
        G[3:6, 0:3] = R0 * dt_s
        G[6:9, 3:6] = np.eye(3) * dt_s
        Q = np.diag([self.sigma_a**2] * 3 + [self.sigma_g**2] * 3)
        m.covariance = F @ m.covariance @ F.T + G @ Q @ G.T
        m.covariance = 0.5 * (m.covariance + m.covariance.T)

        m.jacobian_bias_accel[0:3] += -0.5 * R0 * dt_s**2
        m.jacobian_bias_accel[3:6] += -R0 * dt_s
        m.jacobian_bias_gyro[6:9] += -np.eye(3) * dt_s

    def result(self) -> PreintegratedMeasurement:
        return self.measurement
