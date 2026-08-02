"""Sparse KLT tracking with explicit correspondences and health diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackingMeasurement:
    timestamp_s: float
    detected_features: int
    tracked_features: int
    inlier_features: int
    tracking_ratio: float
    median_flow_px: float
    median_forward_backward_error_px: float


@dataclass(frozen=True)
class TrackedCorrespondences:
    """Accepted pixel correspondences between two consecutive frames."""

    timestamp_s: float
    previous_points_px: np.ndarray
    current_points_px: np.ndarray
    forward_backward_error_px: np.ndarray
    measurement: TrackingMeasurement

    def __post_init__(self) -> None:
        previous = np.asarray(self.previous_points_px, dtype=float)
        current = np.asarray(self.current_points_px, dtype=float)
        errors = np.asarray(self.forward_backward_error_px, dtype=float)
        if previous.ndim != 2 or previous.shape[1:] != (2,):
            raise ValueError("previous_points_px must be an Nx2 array")
        if current.shape != previous.shape or errors.shape != (len(previous),):
            raise ValueError("correspondence arrays must have matching lengths")
        if not all(np.all(np.isfinite(value)) for value in (previous, current, errors)):
            raise ValueError("correspondence arrays must be finite")
        object.__setattr__(self, "previous_points_px", previous.copy())
        object.__setattr__(self, "current_points_px", current.copy())
        object.__setattr__(self, "forward_backward_error_px", errors.copy())


class KLTFeatureTracker:
    """Track Shi-Tomasi corners with pyramidal LK and FB rejection."""

    def __init__(
        self,
        *,
        max_features: int = 300,
        quality_level: float = 0.01,
        min_distance_px: float = 10.0,
        forward_backward_threshold_px: float = 1.5,
        replenish_below: int | None = None,
    ) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        if replenish_below is None:
            replenish_below = min(80, max_features)
        elif replenish_below < 0 or replenish_below > max_features:
            raise ValueError("invalid feature-count configuration")
        if not 0.0 < quality_level <= 1.0:
            raise ValueError("quality_level must be in (0, 1]")
        if min_distance_px <= 0 or forward_backward_threshold_px <= 0:
            raise ValueError("pixel thresholds must be positive")
        self.max_features = int(max_features)
        self.quality_level = float(quality_level)
        self.min_distance_px = float(min_distance_px)
        self.forward_backward_threshold_px = float(forward_backward_threshold_px)
        self.replenish_below = int(replenish_below)
        self._previous_image: np.ndarray | None = None
        self._previous_points: np.ndarray | None = None
        self._last_timestamp_s: float | None = None

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 2 or array.size == 0:
            raise ValueError("image must be a non-empty grayscale array")
        if array.dtype != np.uint8:
            if not np.all(np.isfinite(array)):
                raise ValueError("image contains non-finite values")
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _detect(self, image: np.ndarray, existing: np.ndarray | None = None) -> np.ndarray:
        mask = np.full(image.shape, 255, dtype=np.uint8)
        if existing is not None:
            for point in existing.reshape(-1, 2):
                cv2.circle(mask, tuple(np.rint(point).astype(int)), int(self.min_distance_px), 0, -1)
        remaining = self.max_features - (0 if existing is None else len(existing))
        if remaining <= 0:
            return np.empty((0, 1, 2), dtype=np.float32)
        points = cv2.goodFeaturesToTrack(
            image,
            maxCorners=remaining,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance_px,
            mask=mask,
            blockSize=7,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return points.astype(np.float32, copy=False)

    def update_correspondences(
        self, image: np.ndarray, timestamp_s: float
    ) -> TrackedCorrespondences:
        frame = self._validate_image(image)
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            raise ValueError("tracking timestamps must be strictly increasing")

        if self._previous_image is None:
            points = self._detect(frame)
            self._previous_image = frame.copy()
            self._previous_points = points
            self._last_timestamp_s = timestamp
            measurement = TrackingMeasurement(timestamp, len(points), 0, 0, 0.0, 0.0, 0.0)
            empty = np.empty((0, 2), dtype=float)
            return TrackedCorrespondences(timestamp, empty, empty, np.empty(0), measurement)

        previous_points = self._previous_points
        if previous_points is None or len(previous_points) == 0:
            previous_points = self._detect(self._previous_image)
        detected = int(len(previous_points))
        previous_inliers = np.empty((0, 2), dtype=float)
        current_inliers = np.empty((0, 2), dtype=float)
        fb_inliers = np.empty(0, dtype=float)
        tracked = 0

        if detected:
            forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_image, frame, previous_points, None
            )
            if forward is not None and status_forward is not None:
                backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(
                    frame, self._previous_image, forward, None
                )
                valid_forward = status_forward.reshape(-1).astype(bool)
                tracked = int(np.count_nonzero(valid_forward))
                if backward is not None and status_backward is not None:
                    valid_backward = status_backward.reshape(-1).astype(bool)
                    previous_xy = previous_points.reshape(-1, 2)
                    current_xy = forward.reshape(-1, 2)
                    errors = np.linalg.norm(previous_xy - backward.reshape(-1, 2), axis=1)
                    keep = (
                        valid_forward
                        & valid_backward
                        & np.isfinite(errors)
                        & (errors <= self.forward_backward_threshold_px)
                    )
                    previous_inliers = previous_xy[keep]
                    current_inliers = current_xy[keep]
                    fb_inliers = errors[keep]

        inliers = len(current_inliers)
        flow = np.linalg.norm(current_inliers - previous_inliers, axis=1) if inliers else np.empty(0)
        measurement = TrackingMeasurement(
            timestamp,
            detected,
            tracked,
            inliers,
            float(inliers / detected) if detected else 0.0,
            float(np.median(flow)) if inliers else 0.0,
            float(np.median(fb_inliers)) if inliers else 0.0,
        )

        retained = current_inliers.reshape(-1, 1, 2).astype(np.float32)
        if len(retained) < self.replenish_below:
            new_points = self._detect(frame, retained)
            retained = np.concatenate([retained, new_points]) if len(retained) else new_points
        self._previous_image = frame.copy()
        self._previous_points = retained
        self._last_timestamp_s = timestamp
        return TrackedCorrespondences(
            timestamp, previous_inliers, current_inliers, fb_inliers, measurement
        )

    def update(self, image: np.ndarray, timestamp_s: float) -> TrackingMeasurement:
        """Compatibility wrapper returning only health diagnostics."""
        return self.update_correspondences(image, timestamp_s).measurement
