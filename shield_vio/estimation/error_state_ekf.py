"""15-dimensional error-state EKF research prototype for p, v, q, b_a, b_g."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shield_vio.preintegration.imu_preintegrator import skew, so3_exp


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    trace = np.trace(rotation)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        j, k = (index + 1) % 3, (index + 2) % 3
        scale = np.sqrt(1 + rotation[index, index] - rotation[j, j] - rotation[k, k]) * 2
        quaternion = np.zeros(4)
        quaternion[index + 1] = 0.25 * scale
        quaternion[0] = (rotation[k, j] - rotation[j, k]) / scale
        quaternion[j + 1] = (rotation[j, index] + rotation[index, j]) / scale
        quaternion[k + 1] = (rotation[k, index] + rotation[index, k]) / scale
    return quaternion / np.linalg.norm(quaternion)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    """Return the rotation vector associated with a valid rotation matrix."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return 0.5 * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
    return (
        angle
        / (2.0 * np.sin(angle))
        * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
    )


@dataclass
class ESKFState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    quaternion_wxyz: np.ndarray
    accel_bias_mps2: np.ndarray
    gyro_bias_rps: np.ndarray
    covariance: np.ndarray
    timestamp_s: float = 0.0


class ErrorStateEKF:
    def __init__(
        self,
        state: ESKFState | None = None,
        gravity_mps2: np.ndarray | None = None,
    ) -> None:
        self.state = state or ESKFState(
            np.zeros(3),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.zeros(3),
            np.zeros(3),
            np.eye(15) * 1e-3,
        )
        self.gravity = (
            np.array([0.0, 0.0, -9.80665])
            if gravity_mps2 is None
            else np.asarray(gravity_mps2, dtype=float)
        )
        self._stabilize()

    def propagate(
        self,
        acceleration_mps2: np.ndarray,
        angular_velocity_rps: np.ndarray,
        dt_s: float,
        accel_noise: float = 0.05,
        gyro_noise: float = 0.005,
        accel_bias_rw: float = 1e-4,
        gyro_bias_rw: float = 1e-5,
    ) -> None:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        state = self.state
        rotation = quat_to_rot(state.quaternion_wxyz)
        acceleration = np.asarray(acceleration_mps2) - state.accel_bias_mps2
        angular_velocity = np.asarray(angular_velocity_rps) - state.gyro_bias_rps
        acceleration_world = rotation @ acceleration + self.gravity
        state.position_m = (
            state.position_m
            + state.velocity_mps * dt_s
            + 0.5 * acceleration_world * dt_s**2
        )
        state.velocity_mps = state.velocity_mps + acceleration_world * dt_s
        state.quaternion_wxyz = rot_to_quat(rotation @ so3_exp(angular_velocity * dt_s))

        transition = np.eye(15)
        transition[0:3, 3:6] = np.eye(3) * dt_s
        transition[3:6, 6:9] = -rotation @ skew(acceleration) * dt_s
        transition[3:6, 9:12] = -rotation * dt_s
        transition[6:9, 12:15] = -np.eye(3) * dt_s
        noise_mapping = np.zeros((15, 12))
        noise_mapping[3:6, 0:3] = rotation * dt_s
        noise_mapping[6:9, 3:6] = np.eye(3) * dt_s
        noise_mapping[9:12, 6:9] = np.eye(3) * dt_s
        noise_mapping[12:15, 9:12] = np.eye(3) * dt_s
        process_noise = np.diag(
            [accel_noise**2] * 3
            + [gyro_noise**2] * 3
            + [accel_bias_rw**2] * 3
            + [gyro_bias_rw**2] * 3
        )
        state.covariance = (
            transition @ state.covariance @ transition.T
            + noise_mapping @ process_noise @ noise_mapping.T
        )
        state.timestamp_s += dt_s
        self._stabilize()

    def update_position(self, position_m: np.ndarray, covariance_m2: np.ndarray) -> float:
        measurement = np.asarray(position_m, dtype=float)
        measurement_covariance = np.asarray(covariance_m2, dtype=float)
        observation = np.zeros((3, 15))
        observation[:, 0:3] = np.eye(3)
        residual = measurement - self.state.position_m
        return self._linear_update(residual, observation, measurement_covariance)

    def update_orientation(
        self,
        rotation_world_from_body: np.ndarray,
        covariance_rad2: np.ndarray,
    ) -> float:
        """Fuse an absolute orientation measurement and return its NIS."""
        measured_rotation = np.asarray(rotation_world_from_body, dtype=float)
        covariance = np.asarray(covariance_rad2, dtype=float)
        if measured_rotation.shape != (3, 3):
            raise ValueError("rotation_world_from_body must be 3x3")
        if covariance.shape != (3, 3):
            raise ValueError("covariance_rad2 must be 3x3")
        if not np.all(np.isfinite(measured_rotation)) or not np.all(np.isfinite(covariance)):
            raise ValueError("orientation measurement must be finite")
        if np.linalg.det(measured_rotation) <= 0:
            raise ValueError("rotation_world_from_body must be a proper rotation")

        predicted_rotation = quat_to_rot(self.state.quaternion_wxyz)
        residual = so3_log(predicted_rotation.T @ measured_rotation)
        observation = np.zeros((3, 15))
        observation[:, 6:9] = np.eye(3)
        return self._linear_update(residual, observation, covariance)

    def _linear_update(
        self,
        residual: np.ndarray,
        observation: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> float:
        residual = np.asarray(residual, dtype=float)
        measurement_covariance = np.asarray(measurement_covariance, dtype=float)
        innovation_covariance = (
            observation @ self.state.covariance @ observation.T + measurement_covariance
        )
        gain = self.state.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        delta = gain @ residual
        self._inject(delta)
        identity = np.eye(15)
        joseph = identity - gain @ observation
        self.state.covariance = (
            joseph @ self.state.covariance @ joseph.T
            + gain @ measurement_covariance @ gain.T
        )
        self._stabilize()
        return float(residual @ np.linalg.solve(innovation_covariance, residual))

    def _inject(self, delta: np.ndarray) -> None:
        state = self.state
        state.position_m += delta[0:3]
        state.velocity_mps += delta[3:6]
        state.quaternion_wxyz = rot_to_quat(
            quat_to_rot(state.quaternion_wxyz) @ so3_exp(delta[6:9])
        )
        state.accel_bias_mps2 += delta[9:12]
        state.gyro_bias_rps += delta[12:15]

    def reset_from_pose(
        self,
        position_m: np.ndarray,
        quaternion_wxyz: np.ndarray,
        covariance_scale: float = 1e-3,
    ) -> None:
        self.state.position_m = np.asarray(position_m, dtype=float)
        self.state.quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=float)
        self.state.velocity_mps = np.zeros(3)
        self.state.covariance = np.eye(15) * covariance_scale
        self._stabilize()

    def _stabilize(self) -> None:
        state = self.state
        state.quaternion_wxyz = np.asarray(state.quaternion_wxyz, dtype=float)
        state.quaternion_wxyz /= np.linalg.norm(state.quaternion_wxyz)
        covariance = np.asarray(state.covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        if minimum_eigenvalue < 1e-12:
            covariance += np.eye(15) * (1e-12 - minimum_eigenvalue)
        state.covariance = covariance
