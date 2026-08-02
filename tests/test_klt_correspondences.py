from __future__ import annotations

import cv2
import numpy as np
import pytest

from shield_vio.vision.klt_tracker import KLTFeatureTracker, TrackedCorrespondences


def _dot_image(shift_x: int = 0, shift_y: int = 0) -> np.ndarray:
    image = np.zeros((160, 220), dtype=np.uint8)
    for y in range(25, 145, 30):
        for x in range(25, 205, 30):
            cv2.circle(image, (x + shift_x, y + shift_y), 3, 255, -1)
    return image


def test_returns_real_correspondences_for_translated_frame() -> None:
    tracker = KLTFeatureTracker(
        max_features=80,
        replenish_below=20,
        forward_backward_threshold_px=0.75,
    )
    first = tracker.update_correspondences(_dot_image(), 0.0)
    second = tracker.update_correspondences(_dot_image(3, 2), 0.05)

    assert isinstance(first, TrackedCorrespondences)
    assert len(first.previous_points_px) == 0
    assert second.measurement.inlier_features >= 12
    assert second.previous_points_px.shape == second.current_points_px.shape
    displacement = second.current_points_px - second.previous_points_px
    assert np.median(displacement[:, 0]) == pytest.approx(3.0, abs=0.25)
    assert np.median(displacement[:, 1]) == pytest.approx(2.0, abs=0.25)
    assert np.max(second.forward_backward_error_px) <= 0.75


def test_replenishes_features_without_polluting_current_correspondences() -> None:
    tracker = KLTFeatureTracker(max_features=60, replenish_below=60)
    tracker.update_correspondences(_dot_image(), 0.0)
    result = tracker.update_correspondences(_dot_image(2, 0), 0.1)

    assert result.measurement.inlier_features == len(result.current_points_px)
    assert result.measurement.detected_features <= 60
    follow_up = tracker.update_correspondences(_dot_image(4, 0), 0.2)
    assert follow_up.measurement.detected_features >= result.measurement.inlier_features


def test_update_wrapper_preserves_previous_api() -> None:
    tracker = KLTFeatureTracker(max_features=40)
    measurement = tracker.update(_dot_image(), 1.0)
    assert measurement.timestamp_s == 1.0
    assert measurement.detected_features > 0


def test_rejects_invalid_configuration_and_timestamps() -> None:
    with pytest.raises(ValueError, match="feature-count"):
        KLTFeatureTracker(max_features=10, replenish_below=11)

    tracker = KLTFeatureTracker()
    tracker.update_correspondences(_dot_image(), 1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update_correspondences(_dot_image(), 1.0)
