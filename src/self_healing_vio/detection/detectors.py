"""Prototype degradation detectors for SHIELD-VIO.

Each detector returns a normalized health score in [0, 1], where 1 is healthy
and 0 is severely degraded. The implementations are intentionally lightweight so
that they can run in the demo and unit tests without ROS2 or a VIO backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class DetectorResult:
    """Normalized detector output.

    Attributes:
        name: Detector name.
        score: Health score in [0, 1], where 1 means healthy.
        raw_value: Unnormalized diagnostic value.
        metadata: Optional additional diagnostic information.
    """

    name: str
    score: float
    raw_value: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ImageEntropyDetector:
    """Detects low-information images using Shannon entropy.

    Low entropy often indicates darkness, saturation, blur, or low-texture
    environments. The detector assumes grayscale or RGB numpy arrays.
    """

    def __init__(self, min_entropy: float = 2.0, max_entropy: float = 7.5) -> None:
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

    def evaluate(self, image: np.ndarray) -> DetectorResult:
        gray = _to_gray(image)
        hist, _ = np.histogram(gray.astype(np.uint8), bins=256, range=(0, 255), density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))
        score = _clip01((entropy - self.min_entropy) / (self.max_entropy - self.min_entropy))
        return DetectorResult("image_entropy", score, entropy)


class MotionBlurDetector:
    """Estimates blur from the variance of an image Laplacian approximation."""

    def __init__(self, low_variance: float = 20.0, high_variance: float = 250.0) -> None:
        self.low_variance = low_variance
        self.high_variance = high_variance

    def evaluate(self, image: np.ndarray) -> DetectorResult:
        gray = _to_gray(image).astype(float)
        lap = -4.0 * gray.copy()
        lap[1:, :] += gray[:-1, :]
        lap[:-1, :] += gray[1:, :]
        lap[:, 1:] += gray[:, :-1]
        lap[:, :-1] += gray[:, 1:]
        variance = float(np.var(lap))
        score = _clip01((variance - self.low_variance) / (self.high_variance - self.low_variance))
        return DetectorResult("motion_blur", score, variance)


class FeatureCollapseDetector:
    """Monitors whether the number of tracked visual features is collapsing."""

    def __init__(self, min_features: int = 40, target_features: int = 180) -> None:
        self.min_features = min_features
        self.target_features = target_features

    def evaluate(self, feature_count: int) -> DetectorResult:
        raw = float(max(feature_count, 0))
        score = _clip01((raw - self.min_features) / (self.target_features - self.min_features))
        return DetectorResult("feature_collapse", score, raw)


class IMUConsistencyDetector:
    """Checks whether IMU residual magnitude is consistent with expected noise."""

    def __init__(self, nominal_residual: float = 0.05, max_residual: float = 1.5) -> None:
        self.nominal_residual = nominal_residual
        self.max_residual = max_residual

    def evaluate(self, residual_norm: float) -> DetectorResult:
        raw = float(abs(residual_norm))
        score = 1.0 - _clip01((raw - self.nominal_residual) / (self.max_residual - self.nominal_residual))
        return DetectorResult("imu_consistency", score, raw)


class ReprojectionErrorMonitor:
    """Monitors visual residual quality through reprojection error in pixels."""

    def __init__(self, nominal_px: float = 1.0, max_px: float = 8.0) -> None:
        self.nominal_px = nominal_px
        self.max_px = max_px

    def evaluate(self, reprojection_error_px: float) -> DetectorResult:
        raw = float(abs(reprojection_error_px))
        score = 1.0 - _clip01((raw - self.nominal_px) / (self.max_px - self.nominal_px))
        return DetectorResult("reprojection_error", score, raw)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return np.mean(image[..., :3], axis=-1)
    raise ValueError("image must be a grayscale or RGB/RGBA numpy array")
