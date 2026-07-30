"""Robust calibrated two-view geometry for visual odometry measurements.

This module estimates relative camera rotation and translation direction from
pixel correspondences. Translation remains scale-free; callers must obtain
metric scale from inertial propagation, stereo, depth, or another source.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RelativePoseMeasurement:
    rotation: np.ndarray
    translation_direction: np.ndarray
    inlier_mask: np.ndarray
    correspondence_count: int
    inlier_count: int
    inlier_ratio: float
    median_parallax_px: float
    median_epipolar_error_px: float
    is_degenerate: bool

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation_direction, dtype=float)
        mask = np.asarray(self.inlier_mask, dtype=bool)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.0:
            raise ValueError("rotation must be proper")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation_direction must be a finite 3-vector")
        norm = float(np.linalg.norm(translation))
        if norm <= np.finfo(float).eps:
            raise ValueError("translation_direction must be non-zero")
        if mask.shape != (self.correspondence_count,):
            raise ValueError("inlier_mask must match correspondence_count")
        if self.inlier_count != int(np.count_nonzero(mask)):
            raise ValueError("inlier_count must match inlier_mask")
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "translation_direction", translation / norm)
        object.__setattr__(self, "inlier_mask", mask.copy())


def _points(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite Nx2 array")
    return array


def _camera_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(camera_matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or abs(matrix[2, 2]) <= 1e-12:
        raise ValueError("camera_matrix must contain positive focal lengths")
    return matrix


def estimate_relative_pose(
    previous_points_px: np.ndarray,
    current_points_px: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    ransac_threshold_px: float = 1.0,
    confidence: float = 0.999,
    min_correspondences: int = 8,
    min_inliers: int = 15,
    min_inlier_ratio: float = 0.35,
    min_median_parallax_px: float = 1.0,
) -> RelativePoseMeasurement:
    """Estimate calibrated relative pose with RANSAC and cheirality recovery.

    The returned translation is a unit direction only. ``is_degenerate`` is set
    when support or parallax is insufficient for a trustworthy update.
    """
    previous = _points(previous_points_px, "previous_points_px")
    current = _points(current_points_px, "current_points_px")
    intrinsic = _camera_matrix(camera_matrix)
    if previous.shape != current.shape:
        raise ValueError("point arrays must have matching shape")
    if len(previous) < min_correspondences:
        raise ValueError(f"at least {min_correspondences} correspondences are required")
    if ransac_threshold_px <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid RANSAC parameters")
    if min_inliers < 5 or not 0.0 <= min_inlier_ratio <= 1.0:
        raise ValueError("invalid degeneracy thresholds")

    essential, ransac_mask = cv2.findEssentialMat(
        previous,
        current,
        intrinsic,
        method=cv2.RANSAC,
        prob=float(confidence),
        threshold=float(ransac_threshold_px),
    )
    count = len(previous)
    parallax = np.linalg.norm(current - previous, axis=1)
    if essential is None or ransac_mask is None:
        return RelativePoseMeasurement(
            rotation=np.eye(3),
            translation_direction=np.array([1.0, 0.0, 0.0]),
            inlier_mask=np.zeros(count, dtype=bool),
            correspondence_count=count,
            inlier_count=0,
            inlier_ratio=0.0,
            median_parallax_px=float(np.median(parallax)),
            median_epipolar_error_px=float("inf"),
            is_degenerate=True,
        )
    if essential.shape[0] > 3:
        essential = essential[:3, :3]

    recovered, rotation, translation, pose_mask = cv2.recoverPose(
        essential, previous, current, intrinsic, mask=ransac_mask
    )
    if recovered <= 0 or pose_mask is None:
        ransac_inliers = ransac_mask.reshape(-1).astype(bool)
        return RelativePoseMeasurement(
            rotation=np.eye(3),
            translation_direction=np.array([1.0, 0.0, 0.0]),
            inlier_mask=ransac_inliers,
            correspondence_count=count,
            inlier_count=int(np.count_nonzero(ransac_inliers)),
            inlier_ratio=float(np.mean(ransac_inliers)),
            median_parallax_px=float(np.median(parallax[ransac_inliers]))
            if np.any(ransac_inliers)
            else 0.0,
            median_epipolar_error_px=float("inf"),
            is_degenerate=True,
        )

    inlier_mask = pose_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inlier_mask))
    ratio = inlier_count / count
    median_parallax = float(np.median(parallax[inlier_mask])) if inlier_count else 0.0

    p0 = np.column_stack([previous, np.ones(count)])
    p1 = np.column_stack([current, np.ones(count)])
    fundamental = np.linalg.inv(intrinsic).T @ essential @ np.linalg.inv(intrinsic)
    lines1 = (fundamental @ p0.T).T
    lines0 = (fundamental.T @ p1.T).T
    numerator = np.abs(np.sum(p1 * lines1, axis=1))
    denom1 = np.linalg.norm(lines1[:, :2], axis=1)
    denom0 = np.linalg.norm(lines0[:, :2], axis=1)
    valid = inlier_mask & (denom1 > 1e-12) & (denom0 > 1e-12)
    symmetric_error = 0.5 * numerator * (
        1.0 / np.maximum(denom1, 1e-12) + 1.0 / np.maximum(denom0, 1e-12)
    )
    median_epipolar = float(np.median(symmetric_error[valid])) if np.any(valid) else float("inf")

    degenerate = (
        inlier_count < min_inliers
        or ratio < min_inlier_ratio
        or median_parallax < min_median_parallax_px
        or not np.isfinite(median_epipolar)
    )
    return RelativePoseMeasurement(
        rotation=rotation,
        translation_direction=translation.reshape(3),
        inlier_mask=inlier_mask,
        correspondence_count=count,
        inlier_count=inlier_count,
        inlier_ratio=float(ratio),
        median_parallax_px=median_parallax,
        median_epipolar_error_px=median_epipolar,
        is_degenerate=bool(degenerate),
    )
