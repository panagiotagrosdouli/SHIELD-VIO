"""Metric stereo-PnP visual measurements for the EuRoC ESKF runner."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from shield_vio.backends.base import EstimatorState
from shield_vio.datasets.euroc import SynchronizedFrame
from shield_vio.experiments.visual_measurements import LinearVisualMeasurement
from shield_vio.vision.stereo_pnp import MetricPnPResult, StereoLandmarks, StereoPnPFrontend


class StereoPnPVisualProvider:
    """Fuse calibrated metric stereo motion as a 6D body-pose measurement."""

    def __init__(
        self,
        frontend: StereoPnPFrontend,
        transform_body_camera: np.ndarray,
        *,
        right_image_loader: Callable[[SynchronizedFrame], np.ndarray] | None = None,
        min_pnp_inliers: int = 12,
        max_reprojection_rmse_px: float = 2.5,
        position_std_m: float = 0.08,
        rotation_std_rad: float = 0.04,
    ) -> None:
        transform = np.asarray(transform_body_camera, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("transform_body_camera must be a finite 4x4 matrix")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("transform_body_camera must be homogeneous")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("transform_body_camera rotation must be orthonormal")
        if min_pnp_inliers < 4 or max_reprojection_rmse_px <= 0:
            raise ValueError("invalid PnP acceptance thresholds")
        if position_std_m <= 0 or rotation_std_rad <= 0:
            raise ValueError("measurement standard deviations must be positive")
        self.frontend = frontend
        self.transform_body_camera = transform.copy()
        self.right_image_loader = right_image_loader or _load_matching_cam1_image
        self.min_pnp_inliers = int(min_pnp_inliers)
        self.max_reprojection_rmse_px = float(max_reprojection_rmse_px)
        self.position_std_m = float(position_std_m)
        self.rotation_std_rad = float(rotation_std_rad)
        self._previous_landmarks: StereoLandmarks | None = None
        self._previous_world_body: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "opencv_stereo_pnp_metric_pose"

    def measure(
        self,
        packet: SynchronizedFrame,
        state: EstimatorState,
    ) -> LinearVisualMeasurement | None:
        left = cv2.imread(str(Path(packet.frame.image_path)), cv2.IMREAD_GRAYSCALE)
        if left is None:
            raise FileNotFoundError(packet.frame.image_path)
        right = np.asarray(self.right_image_loader(packet))
        if right.ndim != 2 or right.size == 0:
            raise ValueError("right image loader must return a non-empty grayscale image")

        current_landmarks = self.frontend.build_landmarks(left, right)
        current_world_body = _state_world_body(state)
        previous_landmarks = self._previous_landmarks
        previous_world_body = self._previous_world_body
        self._previous_landmarks = current_landmarks
        self._previous_world_body = current_world_body
        if previous_landmarks is None or previous_world_body is None:
            return None

        motion = self.frontend.estimate_motion(previous_landmarks, left)
        if motion is None:
            return None
        if motion.inlier_count < self.min_pnp_inliers:
            return None
        if motion.reprojection_rmse_px > self.max_reprojection_rmse_px:
            return None

        measurement = build_metric_pose_measurement(
            state,
            previous_world_body,
            motion,
            self.transform_body_camera,
            position_std_m=self.position_std_m,
            rotation_std_rad=self.rotation_std_rad,
        )
        return LinearVisualMeasurement(
            residual=measurement.residual,
            measurement_matrix=measurement.measurement_matrix,
            measurement_covariance=measurement.measurement_covariance,
            detected_features=current_landmarks.stereo_match_count,
            tracked_features=motion.correspondence_count,
            correspondence_count=motion.correspondence_count,
            inlier_count=motion.inlier_count,
            status="stereo_pnp_pose_update",
        )


def build_metric_pose_measurement(
    state: EstimatorState,
    previous_world_body: np.ndarray,
    motion: MetricPnPResult,
    transform_body_camera: np.ndarray,
    *,
    position_std_m: float,
    rotation_std_rad: float,
) -> LinearVisualMeasurement:
    """Convert a camera-frame PnP motion into a 6D absolute body-pose update."""
    previous = np.asarray(previous_world_body, dtype=float)
    body_camera = np.asarray(transform_body_camera, dtype=float)
    if previous.shape != (4, 4) or body_camera.shape != (4, 4):
        raise ValueError("pose transforms must be 4x4")

    current_previous_camera = np.eye(4)
    current_previous_camera[:3, :3] = motion.rotation_current_previous
    current_previous_camera[:3, 3] = motion.translation_current_previous_m
    camera_body = np.linalg.inv(body_camera)
    current_previous_body = body_camera @ current_previous_camera @ camera_body
    measured_world_body = previous @ np.linalg.inv(current_previous_body)

    predicted_world_body = _state_world_body(state)
    position_residual = measured_world_body[:3, 3] - predicted_world_body[:3, 3]
    rotation_error = measured_world_body[:3, :3] @ predicted_world_body[:3, :3].T
    rotation_vector, _ = cv2.Rodrigues(rotation_error)
    residual = np.concatenate([position_residual, rotation_vector.reshape(3)])

    jacobian = np.zeros((6, 15), dtype=float)
    jacobian[:3, :3] = np.eye(3)
    jacobian[3:, 6:9] = np.eye(3)
    covariance = np.diag(
        [position_std_m**2] * 3 + [rotation_std_rad**2] * 3
    )
    return LinearVisualMeasurement(residual, jacobian, covariance)


def _state_world_body(state: EstimatorState) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = _quaternion_to_rotation(state.orientation_wxyz)
    transform[:3, 3] = np.asarray(state.position, dtype=float)
    return transform


def _quaternion_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("orientation must be a finite wxyz quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(float).eps:
        raise ValueError("orientation quaternion must be non-zero")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _load_matching_cam1_image(packet: SynchronizedFrame) -> np.ndarray:
    left_path = Path(packet.frame.image_path)
    parts = list(left_path.parts)
    try:
        index = parts.index("cam0")
    except ValueError as exc:
        raise ValueError("cam0 image path is required to derive cam1 path") from exc
    parts[index] = "cam1"
    right_path = Path(*parts)
    image = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(right_path)
    return image
