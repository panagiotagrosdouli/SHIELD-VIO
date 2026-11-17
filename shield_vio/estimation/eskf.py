"""Error-state EKF scaffold for safety-aware VIO research."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from shield_vio.core.math import quat_multiply, small_angle_quat, quat_to_rot, nearest_psd, mahalanobis_squared, skew
from shield_vio.estimation.state import VIOState, STATE_DIM
from shield_vio.imu.model import IMUNoiseModel

@dataclass
class InnovationRecord:
    residual: np.ndarray
    covariance: np.ndarray
    nis: float

class ErrorStateEKF:
    """Minimal 15-state ESKF scaffold: p, v, theta, accel bias, gyro bias."""
    def __init__(self, noise: IMUNoiseModel | None = None, gravity: np.ndarray | None = None) -> None:
        self.noise = noise or IMUNoiseModel(); self.gravity = np.array([0.0, 0.0, -9.81]) if gravity is None else np.asarray(gravity, dtype=float); self.last_innovation = None
    def propagate(self, state: VIOState, accel_m: np.ndarray, gyro_m: np.ndarray, dt: float) -> VIOState:
        if dt <= 0: raise ValueError("dt must be positive")
        s = state.copy(); accel = np.asarray(accel_m, dtype=float).reshape(3) - s.accel_bias; gyro = np.asarray(gyro_m, dtype=float).reshape(3) - s.gyro_bias
        R = quat_to_rot(s.orientation); a_world = R @ accel + self.gravity
        s.position = s.position + s.velocity * dt + 0.5 * a_world * dt * dt; s.velocity = s.velocity + a_world * dt; s.orientation = quat_multiply(s.orientation, small_angle_quat(gyro * dt))
        F = np.eye(STATE_DIM); F[0:3, 3:6] = np.eye(3) * dt; F[3:6, 6:9] = -(R @ skew(accel)) * dt; F[3:6, 9:12] = -R * dt; F[6:9, 12:15] = -np.eye(3) * dt
        Qd = np.eye(STATE_DIM) * 1e-12; Qd[3:6,3:6] = np.eye(3) * self.noise.accel_noise_density**2 * dt; Qd[6:9,6:9] = np.eye(3) * self.noise.gyro_noise_density**2 * dt; Qd[9:12,9:12] = np.eye(3) * self.noise.accel_bias_random_walk**2 * dt; Qd[12:15,12:15] = np.eye(3) * self.noise.gyro_bias_random_walk**2 * dt
        s.covariance = nearest_psd(F @ s.covariance @ F.T + Qd); s.normalize(); return s
    def update_linear(self, state: VIOState, residual: np.ndarray, H: np.ndarray, R: np.ndarray) -> VIOState:
        r = np.asarray(residual, dtype=float).reshape(-1); H = np.asarray(H, dtype=float); R = np.asarray(R, dtype=float)
        if H.shape[1] != STATE_DIM or H.shape[0] != r.size or R.shape != (r.size, r.size): raise ValueError("Inconsistent visual measurement update dimensions")
        s = state.copy(); S = nearest_psd(H @ s.covariance @ H.T + R); K = s.covariance @ H.T @ np.linalg.inv(S); dx = K @ r
        s.position += dx[0:3]; s.velocity += dx[3:6]; s.orientation = quat_multiply(s.orientation, small_angle_quat(dx[6:9])); s.accel_bias += dx[9:12]; s.gyro_bias += dx[12:15]; s.normalize()
        I_KH = np.eye(STATE_DIM) - K @ H; s.covariance = nearest_psd(I_KH @ s.covariance @ I_KH.T + K @ R @ K.T); self.last_innovation = InnovationRecord(r, S, mahalanobis_squared(r, S)); return s
