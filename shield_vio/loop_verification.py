"""Geometric verification for appearance-based loop-closure candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shield_vio.loop_closure import PoseGraphEdge
from shield_vio.vision.two_view_geometry import RelativePoseMeasurement, estimate_relative_pose


@dataclass(frozen=True)
class LoopVerificationResult:
    """Decision and diagnostics for one proposed loop closure."""

    accepted: bool
    reason: str
    measurement: RelativePoseMeasurement
    edge: PoseGraphEdge | None


def verify_loop_candidate(
    query_id: int,
    match_id: int,
    query_points_px: np.ndarray,
    match_points_px: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    translation_scale_m: float,
    min_inliers: int = 25,
    min_inlier_ratio: float = 0.45,
    min_median_parallax_px: float = 1.5,
    max_median_epipolar_error_px: float = 1.0,
    rotation_information: float = 100.0,
    translation_information: float = 25.0,
) -> LoopVerificationResult:
    """Verify a loop hypothesis and create a metric pose-graph loop edge.

    The translation scale must come from an independent metric source because
    essential-matrix translation is direction-only.
    """
    if query_id == match_id:
        raise ValueError("loop candidate must connect different frames")
    if translation_scale_m <= 0 or not np.isfinite(translation_scale_m):
        raise ValueError("translation_scale_m must be finite and positive")
    if max_median_epipolar_error_px <= 0:
        raise ValueError("max_median_epipolar_error_px must be positive")
    if rotation_information <= 0 or translation_information <= 0:
        raise ValueError("information weights must be positive")

    measurement = estimate_relative_pose(
        match_points_px,
        query_points_px,
        camera_matrix,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        min_median_parallax_px=min_median_parallax_px,
    )
    if measurement.is_degenerate:
        return LoopVerificationResult(False, "degenerate_geometry", measurement, None)
    if measurement.median_epipolar_error_px > max_median_epipolar_error_px:
        return LoopVerificationResult(False, "epipolar_error", measurement, None)

    transform = np.eye(4)
    transform[:3, :3] = measurement.rotation
    transform[:3, 3] = measurement.translation_direction * translation_scale_m
    information = np.diag([translation_information] * 3 + [rotation_information] * 3)
    edge = PoseGraphEdge(match_id, query_id, transform, information, kind="loop")
    return LoopVerificationResult(True, "accepted", measurement, edge)
