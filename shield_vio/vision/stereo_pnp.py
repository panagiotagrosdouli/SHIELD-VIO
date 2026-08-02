"""Stereo ORB matching and metric PnP-RANSAC motion estimation."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shield_vio.vision.stereo_geometry import StereoTriangulationResult, triangulate_stereo_points


@dataclass(frozen=True)
class StereoLandmarks:
    """Metric landmarks reconstructed from one synchronized stereo pair."""

    points_left_m: np.ndarray
    left_points_px: np.ndarray
    descriptors: np.ndarray
    stereo_match_count: int

    def __post_init__(self) -> None:
        points = np.asarray(self.points_left_m, dtype=float)
        pixels = np.asarray(self.left_points_px, dtype=float)
        descriptors = np.asarray(self.descriptors, dtype=np.uint8)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("points_left_m must be an Nx3 array")
        if pixels.shape != (len(points), 2):
            raise ValueError("left_points_px must be an Nx2 array")
        if descriptors.ndim != 2 or len(descriptors) != len(points):
            raise ValueError("descriptors must match the landmark count")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(pixels)):
            raise ValueError("landmarks and pixels must be finite")
        if self.stereo_match_count < len(points):
            raise ValueError("stereo_match_count cannot be smaller than valid landmarks")
        object.__setattr__(self, "points_left_m", points.copy())
        object.__setattr__(self, "left_points_px", pixels.copy())
        object.__setattr__(self, "descriptors", descriptors.copy())


@dataclass(frozen=True)
class MetricPnPResult:
    """Metric transform from the landmark frame into the current camera frame."""

    rotation_current_previous: np.ndarray
    translation_current_previous_m: np.ndarray
    inlier_mask: np.ndarray
    correspondence_count: int
    reprojection_rmse_px: float

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_current_previous, dtype=float)
        translation = np.asarray(self.translation_current_previous_m, dtype=float).reshape(-1)
        inliers = np.asarray(self.inlier_mask, dtype=bool)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation_current_previous must be a finite 3x3 matrix")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation_current_previous_m must be a finite 3-vector")
        if inliers.shape != (self.correspondence_count,):
            raise ValueError("inlier_mask must match correspondence_count")
        if not np.isfinite(self.reprojection_rmse_px) or self.reprojection_rmse_px < 0:
            raise ValueError("reprojection_rmse_px must be finite and non-negative")
        object.__setattr__(self, "rotation_current_previous", rotation.copy())
        object.__setattr__(self, "translation_current_previous_m", translation.copy())
        object.__setattr__(self, "inlier_mask", inliers.copy())

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))


class StereoPnPFrontend:
    """Construct metric stereo landmarks and estimate temporal motion with PnP."""

    def __init__(
        self,
        left_camera_matrix: np.ndarray,
        right_camera_matrix: np.ndarray,
        rotation_right_left: np.ndarray,
        translation_right_left_m: np.ndarray,
        *,
        max_features: int = 1000,
        ratio_threshold: float = 0.75,
        max_epipolar_error_px: float = 2.0,
        max_stereo_reprojection_error_px: float = 2.0,
        min_pnp_correspondences: int = 8,
        pnp_reprojection_error_px: float = 3.0,
    ) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        if not 0.0 < ratio_threshold < 1.0:
            raise ValueError("ratio_threshold must be in (0, 1)")
        if max_epipolar_error_px <= 0 or max_stereo_reprojection_error_px <= 0:
            raise ValueError("stereo pixel thresholds must be positive")
        if min_pnp_correspondences < 4 or pnp_reprojection_error_px <= 0:
            raise ValueError("invalid PnP configuration")
        self.left_camera_matrix = _camera_matrix(left_camera_matrix, "left_camera_matrix")
        self.right_camera_matrix = _camera_matrix(right_camera_matrix, "right_camera_matrix")
        self.rotation_right_left = _rotation(rotation_right_left)
        self.translation_right_left_m = _translation(translation_right_left_m)
        self.ratio_threshold = float(ratio_threshold)
        self.max_epipolar_error_px = float(max_epipolar_error_px)
        self.max_stereo_reprojection_error_px = float(max_stereo_reprojection_error_px)
        self.min_pnp_correspondences = int(min_pnp_correspondences)
        self.pnp_reprojection_error_px = float(pnp_reprojection_error_px)
        self._orb = cv2.ORB_create(nfeatures=int(max_features))
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def build_landmarks(self, left_image: np.ndarray, right_image: np.ndarray) -> StereoLandmarks:
        left = _gray(left_image)
        right = _gray(right_image)
        keypoints_left, descriptors_left = self._orb.detectAndCompute(left, None)
        keypoints_right, descriptors_right = self._orb.detectAndCompute(right, None)
        if descriptors_left is None or descriptors_right is None:
            return _empty_landmarks(0)
        matches = _mutual_ratio_matches(
            self._matcher,
            descriptors_left,
            descriptors_right,
            self.ratio_threshold,
        )
        if not matches:
            return _empty_landmarks(0)
        left_points = np.asarray([keypoints_left[m.queryIdx].pt for m in matches], dtype=float)
        right_points = np.asarray([keypoints_right[m.trainIdx].pt for m in matches], dtype=float)
        normalized_left = cv2.undistortPoints(
            left_points.reshape(-1, 1, 2), None, None, P=self.left_camera_matrix
        ).reshape(-1, 2)
        normalized_right = cv2.undistortPoints(
            right_points.reshape(-1, 1, 2), None, None, P=self.right_camera_matrix
        ).reshape(-1, 2)
        fundamental = _fundamental_from_calibration(
            self.left_camera_matrix,
            self.right_camera_matrix,
            self.rotation_right_left,
            self.translation_right_left_m,
        )
        lines_right = cv2.computeCorrespondEpilines(
            normalized_left.reshape(-1, 1, 2), 1, fundamental
        ).reshape(-1, 3)
        numerator = np.abs(
            lines_right[:, 0] * normalized_right[:, 0]
            + lines_right[:, 1] * normalized_right[:, 1]
            + lines_right[:, 2]
        )
        denominator = np.linalg.norm(lines_right[:, :2], axis=1)
        epipolar_error = numerator / np.maximum(denominator, np.finfo(float).eps)
        epipolar_keep = np.isfinite(epipolar_error) & (
            epipolar_error <= self.max_epipolar_error_px
        )
        if not np.any(epipolar_keep):
            return _empty_landmarks(len(matches))
        filtered_matches = [match for match, keep in zip(matches, epipolar_keep) if keep]
        filtered_left = left_points[epipolar_keep]
        filtered_right = right_points[epipolar_keep]
        triangulated: StereoTriangulationResult = triangulate_stereo_points(
            filtered_left,
            filtered_right,
            self.left_camera_matrix,
            self.right_camera_matrix,
            self.rotation_right_left,
            self.translation_right_left_m,
            max_reprojection_error_px=self.max_stereo_reprojection_error_px,
        )
        valid = triangulated.valid_mask
        if not np.any(valid):
            return _empty_landmarks(len(matches))
        valid_descriptors = np.asarray(
            [descriptors_left[match.queryIdx] for match, keep in zip(filtered_matches, valid) if keep],
            dtype=np.uint8,
        )
        return StereoLandmarks(
            triangulated.points_left_m[valid],
            filtered_left[valid],
            valid_descriptors,
            len(matches),
        )

    def estimate_motion(
        self,
        previous_landmarks: StereoLandmarks,
        current_left_image: np.ndarray,
    ) -> MetricPnPResult | None:
        current = _gray(current_left_image)
        keypoints, descriptors = self._orb.detectAndCompute(current, None)
        if descriptors is None or len(previous_landmarks.descriptors) == 0:
            return None
        matches = _mutual_ratio_matches(
            self._matcher,
            previous_landmarks.descriptors,
            descriptors,
            self.ratio_threshold,
        )
        if len(matches) < self.min_pnp_correspondences:
            return None
        object_points = np.asarray(
            [previous_landmarks.points_left_m[m.queryIdx] for m in matches], dtype=np.float64
        )
        image_points = np.asarray([keypoints[m.trainIdx].pt for m in matches], dtype=np.float64)
        success, rotation_vector, translation, inlier_indices = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.left_camera_matrix,
            None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=self.pnp_reprojection_error_px,
            confidence=0.999,
            iterationsCount=200,
        )
        if not success or inlier_indices is None or len(inlier_indices) < self.min_pnp_correspondences:
            return None
        rotation, _ = cv2.Rodrigues(rotation_vector)
        inlier_mask = np.zeros(len(matches), dtype=bool)
        inlier_mask[inlier_indices.reshape(-1)] = True
        projected, _ = cv2.projectPoints(
            object_points[inlier_mask],
            rotation_vector,
            translation,
            self.left_camera_matrix,
            None,
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - image_points[inlier_mask], axis=1)
        return MetricPnPResult(
            rotation,
            translation.reshape(3),
            inlier_mask,
            len(matches),
            float(np.sqrt(np.mean(errors**2))),
        )


def _mutual_ratio_matches(
    matcher: cv2.BFMatcher,
    first: np.ndarray,
    second: np.ndarray,
    ratio: float,
) -> list[cv2.DMatch]:
    forward = matcher.knnMatch(first, second, k=2)
    backward = matcher.knnMatch(second, first, k=2)
    forward_good = {
        pair[0].queryIdx: pair[0]
        for pair in forward
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    backward_good = {
        pair[0].queryIdx: pair[0]
        for pair in backward
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    }
    return [
        match
        for query_index, match in forward_good.items()
        if match.trainIdx in backward_good
        and backward_good[match.trainIdx].trainIdx == query_index
    ]


def _fundamental_from_calibration(
    k_left: np.ndarray,
    k_right: np.ndarray,
    rotation_right_left: np.ndarray,
    translation_right_left_m: np.ndarray,
) -> np.ndarray:
    tx, ty, tz = translation_right_left_m
    skew = np.array([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]])
    essential = skew @ rotation_right_left
    return np.linalg.inv(k_right).T @ essential @ np.linalg.inv(k_left)


def _gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("image must be a non-empty grayscale or BGR array")
    if array.dtype != np.uint8:
        if not np.all(np.isfinite(array)):
            raise ValueError("image contains non-finite values")
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _camera_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"{name} focal lengths must be positive")
    return matrix.copy()


def _rotation(value: np.ndarray) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_right_left must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("rotation_right_left must be a proper rotation")
    return rotation.copy()


def _translation(value: np.ndarray) -> np.ndarray:
    translation = np.asarray(value, dtype=float).reshape(-1)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation_right_left_m must be a finite 3-vector")
    if np.linalg.norm(translation) <= 0:
        raise ValueError("stereo baseline must be non-zero")
    return translation.copy()


def _empty_landmarks(stereo_match_count: int) -> StereoLandmarks:
    return StereoLandmarks(
        np.empty((0, 3), dtype=float),
        np.empty((0, 2), dtype=float),
        np.empty((0, 32), dtype=np.uint8),
        stereo_match_count,
    )
