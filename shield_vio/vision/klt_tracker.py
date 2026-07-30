"""Sparse KLT feature tracking and auditable tracking-health measurements."""
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

    def __post_init__(self) -> None:
        counts = (self.detected_features, self.tracked_features, self.inlier_features)
        if any(value < 0 for value in counts):
            raise ValueError("feature counts must be non-negative")
        if self.inlier_features > self.tracked_features or self.tracked_features > self.detected_features:
            raise ValueError("feature counts must satisfy inliers <= tracked <= detected")
        if not 0.0 <= self.tracking_ratio <= 1.0:
            raise ValueError("tracking_ratio must be in [0, 1]")
        for value in (self.timestamp_s, self.median_flow_px, self.median_forward_backward_error_px):
            if not np.isfinite(value) or value < 0:
                raise ValueError("tracking measurements must be finite and non-negative")


class KLTFeatureTracker:
    """Track Shi-Tomasi corners using pyramidal Lucas-Kanade optical flow.

    A forward-backward consistency check rejects unstable correspondences.  The
    class reports health statistics only; it does not claim to produce a visual
    pose constraint or a full VIO update.
    """

    def __init__(
        self,
        *,
        max_features: int = 300,
        quality_level: float = 0.01,
        min_distance_px: float = 10.0,
        forward_backward_threshold_px: float = 1.5,
    ) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        if not 0.0 < quality_level <= 1.0:
            raise ValueError("quality_level must be in (0, 1]")
        if min_distance_px <= 0 or forward_backward_threshold_px <= 0:
            raise ValueError("pixel thresholds must be positive")
        self.max_features = int(max_features)
        self.quality_level = float(quality_level)
        self.min_distance_px = float(min_distance_px)
        self.forward_backward_threshold_px = float(forward_backward_threshold_px)
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

    def _detect(self, image: np.ndarray) -> np.ndarray:
        points = cv2.goodFeaturesToTrack(
            image,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance_px,
            blockSize=7,
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return points.astype(np.float32, copy=False)

    def update(self, image: np.ndarray, timestamp_s: float) -> TrackingMeasurement:
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
            return TrackingMeasurement(timestamp, len(points), 0, 0, 0.0, 0.0, 0.0)

        previous_points = self._previous_points
        if previous_points is None or len(previous_points) == 0:
            previous_points = self._detect(self._previous_image)

        detected = int(len(previous_points))
        tracked = 0
        inliers = 0
        median_flow = 0.0
        median_fb_error = 0.0
        accepted_points = np.empty((0, 1, 2), dtype=np.float32)

        if detected:
            forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_image, frame, previous_points, None
            )
            if forward is not None and forward_status is not None:
                valid_forward = forward_status.reshape(-1).astype(bool)
                tracked = int(np.count_nonzero(valid_forward))
                backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    frame, self._previous_image, forward, None
                )
                if backward is not None and backward_status is not None:
                    valid_backward = backward_status.reshape(-1).astype(bool)
                    fb_error = np.linalg.norm(
                        previous_points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1
                    )
                    inlier_mask = (
                        valid_forward
                        & valid_backward
                        & np.isfinite(fb_error)
                        & (fb_error <= self.forward_backward_threshold_px)
                    )
                    accepted_points = forward.reshape(-1, 1, 2)[inlier_mask].astype(np.float32)
                    inliers = int(np.count_nonzero(inlier_mask))
                    if inliers:
                        flow = np.linalg.norm(
                            forward.reshape(-1, 2)[inlier_mask]
                            - previous_points.reshape(-1, 2)[inlier_mask],
                            axis=1,
                        )
                        median_flow = float(np.median(flow))
                        median_fb_error = float(np.median(fb_error[inlier_mask]))

        if len(accepted_points) < max(20, self.max_features // 5):
            accepted_points = self._detect(frame)

        self._previous_image = frame.copy()
        self._previous_points = accepted_points
        self._last_timestamp_s = timestamp
        ratio = inliers / detected if detected else 0.0
        return TrackingMeasurement(
            timestamp_s=timestamp,
            detected_features=detected,
            tracked_features=tracked,
            inlier_features=inliers,
            tracking_ratio=float(ratio),
            median_flow_px=median_flow,
            median_forward_backward_error_px=median_fb_error,
        )
