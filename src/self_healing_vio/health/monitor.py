"""Compatibility health monitor API.

This module preserves the original repository API used by the existing tests
while the newer SHIELD-VIO prototype uses detector-level health fusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VioHealthSignals:
    """Minimal VIO health signals from a frontend/backend.

    Attributes:
        tracking_ok: Whether the VIO frontend reports valid tracking.
        tracked_features: Number of currently tracked visual features.
        mean_reprojection_error: Mean reprojection error in pixels.
        image_brightness: Normalized image brightness/visibility proxy in [0, 1].
        blur_score: Normalized blur quality proxy in [0, 1], where 1 is sharp.
        imu_consistency: Normalized IMU consistency score in [0, 1].
    """

    tracking_ok: bool
    tracked_features: int
    mean_reprojection_error: float
    image_brightness: float
    blur_score: float
    imu_consistency: float


@dataclass(frozen=True)
class VioHealthEstimate:
    """Aggregate health estimate used by the legacy controller API."""

    score: float
    level: str
    failure_probability: float


class VioHealthMonitor:
    """Computes a compact VIO health estimate from coarse diagnostic signals."""

    def __init__(self, target_features: int = 180, max_reprojection_error: float = 10.0) -> None:
        self.target_features = target_features
        self.max_reprojection_error = max_reprojection_error

    def estimate(self, signals: VioHealthSignals) -> VioHealthEstimate:
        feature_score = np.clip(signals.tracked_features / max(self.target_features, 1), 0.0, 1.0)
        reprojection_score = 1.0 - np.clip(
            signals.mean_reprojection_error / max(self.max_reprojection_error, 1e-6), 0.0, 1.0
        )
        tracking_score = 1.0 if signals.tracking_ok else 0.0
        score = float(
            0.25 * tracking_score
            + 0.20 * feature_score
            + 0.20 * reprojection_score
            + 0.15 * np.clip(signals.image_brightness, 0.0, 1.0)
            + 0.10 * np.clip(signals.blur_score, 0.0, 1.0)
            + 0.10 * np.clip(signals.imu_consistency, 0.0, 1.0)
        )
        failure_probability = float(np.clip(1.0 - score, 0.0, 1.0))
        if score >= 0.7:
            level = "healthy"
        elif score >= 0.35:
            level = "warning"
        else:
            level = "critical"
        return VioHealthEstimate(score=score, level=level, failure_probability=failure_probability)
