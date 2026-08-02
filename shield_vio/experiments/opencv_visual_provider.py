"""OpenCV visual provider for calibrated EuRoC rotation updates."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from shield_vio.backends.base import EstimatorState
from shield_vio.datasets.euroc import SynchronizedFrame
from shield_vio.experiments.visual_measurements import LinearVisualMeasurement
from shield_vio.vision.klt_tracker import KLTFeatureTracker
from shield_vio.vision.two_view_geometry import estimate_relative_pose


class OpenCVRotationVisualProvider:
    """Create ESKF orientation updates from real consecutive camera frames.

    Monocular translation is intentionally excluded because essential-matrix
    recovery provides translation direction but not metric scale.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        *,
        tracker: KLTFeatureTracker | None = None,
        min_correspondences: int = 20,
        min_inliers: int = 15,
        min_inlier_ratio: float = 0.35,
        min_median_parallax_px: float = 1.0,
        rotation_std_rad: float = 0.03,
    ) -> None:
        matrix = np.asarray(camera_matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_matrix must be a finite 3x3 matrix")
        if min_correspondences < 8 or min_inliers < 5:
            raise ValueError("invalid correspondence thresholds")
        if not 0.0 <= min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be in [0, 1]")
        if min_median_parallax_px <= 0 or rotation_std_rad <= 0:
            raise ValueError("parallax and rotation noise must be positive")
        self.camera_matrix = matrix.copy()
        self.tracker = tracker or KLTFeatureTracker()
        self.min_correspondences = int(min_correspondences)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.min_median_parallax_px = float(min_median_parallax_px)
        self.rotation_std_rad = float(rotation_std_rad)
        self._previous_orientation: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "opencv_klt_essential_rotation"

    def measure(
        self,
        packet: SynchronizedFrame,
        state: EstimatorState,
    ) -> LinearVisualMeasurement | None:
        image = cv2.imread(str(Path(packet.frame.image_path)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(packet.frame.image_path)
        correspondences = self.tracker.update_correspondences(image, packet.frame.timestamp_s)
        current_orientation = np.asarray(state.orientation_wxyz, dtype=float).copy()
        if self._previous_orientation is None:
            self._previous_orientation = current_orientation
            return None
        if len(correspondences.current_points_px) < self.min_correspondences:
            self._previous_orientation = current_orientation
            return None

        measurement = estimate_relative_pose(
            correspondences.previous_points_px,
            correspondences.current_points_px,
            self.camera_matrix,
            min_correspondences=self.min_correspondences,
            min_inliers=self.min_inliers,
            min_inlier_ratio=self.min_inlier_ratio,
            min_median_parallax_px=self.min_median_parallax_px,
        )
        previous_rotation = _quaternion_to_rotation(self._previous_orientation)
        current_rotation = _quaternion_to_rotation(current_orientation)
        predicted_relative = previous_rotation.T @ current_rotation
        self._previous_orientation = current_orientation
        if measurement.is_degenerate:
            return None

        rotation_error = measurement.rotation @ predicted_relative.T
        residual = _rotation_log(rotation_error)
        jacobian = np.zeros((3, 15), dtype=float)
        jacobian[:, 6:9] = np.eye(3)
        covariance = np.eye(3) * self.rotation_std_rad**2
        tracking = correspondences.measurement
        return LinearVisualMeasurement(
            residual=residual,
            measurement_matrix=jacobian,
            measurement_covariance=covariance,
            detected_features=tracking.detected_features,
            tracked_features=tracking.tracked_features,
            correspondence_count=measurement.correspondence_count,
            inlier_count=measurement.inlier_count,
            status="rotation_update",
        )


def camera_matrix_from_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    """Convert EuRoC ``[fu, fv, cu, cv]`` intrinsics into a 3x3 matrix."""
    values = np.asarray(intrinsics, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("intrinsics must be finite [fu, fv, cu, cv]")
    fu, fv, cu, cv = values
    if fu <= 0 or fv <= 0:
        raise ValueError("focal lengths must be positive")
    return np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]])


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


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    vector, _ = cv2.Rodrigues(matrix)
    return vector.reshape(3)
