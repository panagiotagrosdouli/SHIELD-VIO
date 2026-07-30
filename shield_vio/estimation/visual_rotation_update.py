"""Guarded fusion of two-view camera rotation into the ESKF attitude state."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shield_vio.estimation.error_state_ekf import ErrorStateEKF, quat_to_rot, so3_log
from shield_vio.vision.two_view_geometry import RelativePoseMeasurement


@dataclass(frozen=True)
class VisualRotationUpdateResult:
    accepted: bool
    nis: float
    reason: str
    measured_rotation_world_from_body: np.ndarray


def fuse_relative_camera_rotation(
    ekf: ErrorStateEKF,
    previous_rotation_world_from_body: np.ndarray,
    measurement: RelativePoseMeasurement,
    covariance_rad2: np.ndarray,
    *,
    rotation_body_from_camera: np.ndarray | None = None,
    max_nis: float = 11.345,
) -> VisualRotationUpdateResult:
    """Fuse a relative camera rotation after degeneracy and NIS gating.

    ``measurement.rotation`` follows OpenCV ``recoverPose`` and maps points from
    the previous camera frame into the current camera frame. The default NIS
    gate is the 99% chi-square threshold for three degrees of freedom.
    """
    previous_body = _rotation(previous_rotation_world_from_body, "previous_rotation_world_from_body")
    body_from_camera = (
        np.eye(3)
        if rotation_body_from_camera is None
        else _rotation(rotation_body_from_camera, "rotation_body_from_camera")
    )
    covariance = np.asarray(covariance_rad2, dtype=float)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance_rad2 must be a finite 3x3 matrix")
    if np.min(np.linalg.eigvalsh(0.5 * (covariance + covariance.T))) <= 0:
        raise ValueError("covariance_rad2 must be positive definite")
    if max_nis <= 0 or not np.isfinite(max_nis):
        raise ValueError("max_nis must be finite and positive")

    previous_camera = previous_body @ body_from_camera
    current_camera = previous_camera @ measurement.rotation.T
    measured_body = current_camera @ body_from_camera.T

    if measurement.is_degenerate:
        return VisualRotationUpdateResult(False, float("inf"), "degenerate_geometry", measured_body)

    predicted_body = quat_to_rot(ekf.state.quaternion_wxyz)
    residual = so3_log(predicted_body.T @ measured_body)
    innovation_covariance = ekf.state.covariance[6:9, 6:9] + covariance
    nis = float(residual @ np.linalg.solve(innovation_covariance, residual))
    if not np.isfinite(nis) or nis > max_nis:
        return VisualRotationUpdateResult(False, nis, "nis_gate", measured_body)

    applied_nis = ekf.update_orientation(measured_body, covariance)
    return VisualRotationUpdateResult(True, applied_nis, "accepted", measured_body)


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be orthonormal")
    if np.linalg.det(rotation) <= 0:
        raise ValueError(f"{name} must be a proper rotation")
    return rotation
