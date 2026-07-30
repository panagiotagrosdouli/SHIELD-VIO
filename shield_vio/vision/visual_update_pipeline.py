"""End-to-end sparse visual rotation update for the ESKF."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shield_vio.estimation.error_state_ekf import ErrorStateEKF
from shield_vio.estimation.visual_rotation_update import (
    VisualRotationUpdateResult,
    fuse_relative_camera_rotation,
)
from shield_vio.vision.two_view_geometry import RelativePoseMeasurement, estimate_relative_pose


@dataclass(frozen=True)
class VisualUpdateDiagnostics:
    detected_features: int
    tracked_features: int
    correspondence_count: int
    geometry: RelativePoseMeasurement | None
    update: VisualRotationUpdateResult | None
    reason: str


def run_visual_rotation_update(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    camera_matrix: np.ndarray,
    ekf: ErrorStateEKF,
    previous_rotation_world_from_body: np.ndarray,
    covariance_rad2: np.ndarray,
    *,
    rotation_body_from_camera: np.ndarray | None = None,
    max_features: int = 300,
    quality_level: float = 0.01,
    min_distance_px: float = 10.0,
    forward_backward_threshold_px: float = 1.5,
    max_nis: float = 11.345,
) -> VisualUpdateDiagnostics:
    """Track features, estimate relative pose, and conditionally update attitude."""
    previous = _image(previous_image, "previous_image")
    current = _image(current_image, "current_image")
    if previous.shape != current.shape:
        raise ValueError("images must have matching shape")

    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=int(max_features),
        qualityLevel=float(quality_level),
        minDistance=float(min_distance_px),
        blockSize=7,
    )
    detected = 0 if points is None else len(points)
    if points is None or len(points) < 8:
        return VisualUpdateDiagnostics(detected, 0, 0, None, None, "insufficient_features")

    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if forward is None or status_forward is None:
        return VisualUpdateDiagnostics(detected, 0, 0, None, None, "tracking_failed")
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None)
    if backward is None or status_backward is None:
        return VisualUpdateDiagnostics(detected, 0, 0, None, None, "tracking_failed")

    p0 = points.reshape(-1, 2)
    p1 = forward.reshape(-1, 2)
    fb_error = np.linalg.norm(p0 - backward.reshape(-1, 2), axis=1)
    valid = (
        status_forward.reshape(-1).astype(bool)
        & status_backward.reshape(-1).astype(bool)
        & np.isfinite(fb_error)
        & (fb_error <= float(forward_backward_threshold_px))
    )
    tracked = int(np.count_nonzero(status_forward))
    previous_matches = p0[valid]
    current_matches = p1[valid]
    if len(previous_matches) < 8:
        return VisualUpdateDiagnostics(
            detected, tracked, len(previous_matches), None, None, "insufficient_correspondences"
        )

    geometry = estimate_relative_pose(previous_matches, current_matches, camera_matrix)
    update = fuse_relative_camera_rotation(
        ekf,
        previous_rotation_world_from_body,
        geometry,
        covariance_rad2,
        rotation_body_from_camera=rotation_body_from_camera,
        max_nis=max_nis,
    )
    return VisualUpdateDiagnostics(
        detected_features=detected,
        tracked_features=tracked,
        correspondence_count=len(previous_matches),
        geometry=geometry,
        update=update,
        reason=update.reason,
    )


def _image(value: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"{name} must be a non-empty grayscale image")
    if image.dtype != np.uint8:
        if not np.all(np.isfinite(image)):
            raise ValueError(f"{name} contains non-finite values")
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image
