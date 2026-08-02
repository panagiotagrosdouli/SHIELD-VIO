"""Calibrated stereo triangulation with explicit geometric validity checks."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoTriangulationResult:
    """Metric 3D points expressed in the left-camera frame."""

    points_left_m: np.ndarray
    valid_mask: np.ndarray
    reprojection_error_left_px: np.ndarray
    reprojection_error_right_px: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_left_m, dtype=float)
        valid = np.asarray(self.valid_mask, dtype=bool)
        left_error = np.asarray(self.reprojection_error_left_px, dtype=float)
        right_error = np.asarray(self.reprojection_error_right_px, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("points_left_m must be an Nx3 array")
        expected = (len(points),)
        if valid.shape != expected or left_error.shape != expected or right_error.shape != expected:
            raise ValueError("stereo result arrays must have matching lengths")
        if not np.all(np.isfinite(points)):
            raise ValueError("triangulated points must be finite")
        if not np.all(np.isfinite(left_error)) or not np.all(np.isfinite(right_error)):
            raise ValueError("reprojection errors must be finite")
        object.__setattr__(self, "points_left_m", points.copy())
        object.__setattr__(self, "valid_mask", valid.copy())
        object.__setattr__(self, "reprojection_error_left_px", left_error.copy())
        object.__setattr__(self, "reprojection_error_right_px", right_error.copy())

    @property
    def valid_points_left_m(self) -> np.ndarray:
        return self.points_left_m[self.valid_mask]


def triangulate_stereo_points(
    left_points_px: np.ndarray,
    right_points_px: np.ndarray,
    left_camera_matrix: np.ndarray,
    right_camera_matrix: np.ndarray,
    rotation_right_left: np.ndarray,
    translation_right_left_m: np.ndarray,
    *,
    min_depth_m: float = 0.1,
    max_depth_m: float = 100.0,
    max_reprojection_error_px: float = 2.0,
) -> StereoTriangulationResult:
    """Triangulate synchronized stereo correspondences in metric scale.

    ``rotation_right_left`` and ``translation_right_left_m`` transform a point
    from the left-camera frame into the right-camera frame.
    """

    left = _points(left_points_px, "left_points_px")
    right = _points(right_points_px, "right_points_px")
    if left.shape != right.shape:
        raise ValueError("left and right correspondence arrays must match")
    if len(left) == 0:
        raise ValueError("at least one stereo correspondence is required")

    k_left = _matrix(left_camera_matrix, (3, 3), "left_camera_matrix")
    k_right = _matrix(right_camera_matrix, (3, 3), "right_camera_matrix")
    rotation = _matrix(rotation_right_left, (3, 3), "rotation_right_left")
    translation = np.asarray(translation_right_left_m, dtype=float).reshape(-1)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation_right_left_m must be a finite 3-vector")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("rotation_right_left must be a proper rotation")
    if min_depth_m <= 0 or max_depth_m <= min_depth_m:
        raise ValueError("invalid stereo depth range")
    if max_reprojection_error_px <= 0:
        raise ValueError("max_reprojection_error_px must be positive")

    projection_left = k_left @ np.column_stack([np.eye(3), np.zeros(3)])
    projection_right = k_right @ np.column_stack([rotation, translation])
    homogeneous = cv2.triangulatePoints(
        projection_left,
        projection_right,
        left.T.astype(np.float64),
        right.T.astype(np.float64),
    )
    scale = homogeneous[3]
    finite_scale = np.isfinite(scale) & (np.abs(scale) > np.finfo(float).eps)
    points = np.zeros((len(left), 3), dtype=float)
    points[finite_scale] = (homogeneous[:3, finite_scale] / scale[finite_scale]).T

    points_right = (rotation @ points.T).T + translation
    projected_left = _project(points, k_left)
    projected_right = _project(points_right, k_right)
    left_error = np.linalg.norm(projected_left - left, axis=1)
    right_error = np.linalg.norm(projected_right - right, axis=1)

    depth_left = points[:, 2]
    depth_right = points_right[:, 2]
    valid = (
        finite_scale
        & np.all(np.isfinite(points), axis=1)
        & (depth_left >= min_depth_m)
        & (depth_left <= max_depth_m)
        & (depth_right >= min_depth_m)
        & (depth_right <= max_depth_m)
        & (left_error <= max_reprojection_error_px)
        & (right_error <= max_reprojection_error_px)
    )
    return StereoTriangulationResult(points, valid, left_error, right_error)


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1:] != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite Nx2 array")
    return array


def _matrix(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return array


def _project(points_camera: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    homogeneous = (camera_matrix @ points_camera.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return homogeneous[:, :2] / homogeneous[:, 2:3]
