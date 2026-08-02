from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import shield_vio.experiments.opencv_visual_provider as module
from shield_vio.backends.base import EstimatorState
from shield_vio.datasets.euroc import CameraFrame, SynchronizedFrame
from shield_vio.vision.klt_tracker import TrackedCorrespondences, TrackingMeasurement
from shield_vio.vision.two_view_geometry import RelativePoseMeasurement


class _Tracker:
    def __init__(self) -> None:
        self.calls = 0

    def update_correspondences(self, image: np.ndarray, timestamp_s: float) -> TrackedCorrespondences:
        self.calls += 1
        count = 30 if self.calls > 1 else 0
        points = np.column_stack([np.arange(count), np.arange(count)]).astype(float)
        measurement = TrackingMeasurement(timestamp_s, count, count, count, 1.0, 2.0, 0.1)
        return TrackedCorrespondences(timestamp_s, points, points + 1.0, np.full(count, 0.1), measurement)


def _packet(timestamp_ns: int) -> SynchronizedFrame:
    return SynchronizedFrame(
        frame=CameraFrame(timestamp_ns, timestamp_ns * 1e-9, Path("frame.png")),
        imu_samples=(),
    )


def _state(timestamp_ns: int, orientation: np.ndarray | None = None) -> EstimatorState:
    return EstimatorState(
        timestamp_ns=timestamp_ns,
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation,
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        covariance=np.eye(15),
    )


def _relative_pose(rotation: np.ndarray, *, degenerate: bool = False) -> RelativePoseMeasurement:
    mask = np.ones(30, dtype=bool)
    return RelativePoseMeasurement(
        rotation=rotation,
        translation_direction=np.array([1.0, 0.0, 0.0]),
        inlier_mask=mask,
        correspondence_count=30,
        inlier_count=30,
        inlier_ratio=1.0,
        median_parallax_px=2.0,
        median_epipolar_error_px=0.1,
        is_degenerate=degenerate,
    )


def test_builds_rotation_only_eskf_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module.cv2, "imread", lambda *args, **kwargs: np.zeros((40, 40), np.uint8))
    angle = 0.1
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    monkeypatch.setattr(module, "estimate_relative_pose", lambda *args, **kwargs: _relative_pose(rotation))
    provider = module.OpenCVRotationVisualProvider(np.eye(3), tracker=_Tracker())

    assert provider.measure(_packet(1_000_000_000), _state(1_000_000_000)) is None
    result = provider.measure(_packet(2_000_000_000), _state(2_000_000_000))

    assert result is not None
    assert result.residual == pytest.approx([0.0, 0.0, angle])
    assert result.measurement_matrix.shape == (3, 15)
    assert result.measurement_matrix[:, 6:9] == pytest.approx(np.eye(3))
    assert np.count_nonzero(result.measurement_matrix[:, :6]) == 0
    assert result.correspondence_count == 30
    assert result.inlier_count == 30


def test_rejects_degenerate_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module.cv2, "imread", lambda *args, **kwargs: np.zeros((40, 40), np.uint8))
    monkeypatch.setattr(
        module,
        "estimate_relative_pose",
        lambda *args, **kwargs: _relative_pose(np.eye(3), degenerate=True),
    )
    provider = module.OpenCVRotationVisualProvider(np.eye(3), tracker=_Tracker())
    provider.measure(_packet(1_000_000_000), _state(1_000_000_000))
    assert provider.measure(_packet(2_000_000_000), _state(2_000_000_000)) is None


def test_camera_intrinsic_conversion_and_validation() -> None:
    matrix = module.camera_matrix_from_intrinsics(np.array([458.0, 457.0, 367.0, 248.0]))
    np.testing.assert_allclose(
        matrix,
        np.array([[458.0, 0.0, 367.0], [0.0, 457.0, 248.0], [0.0, 0.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="focal"):
        module.camera_matrix_from_intrinsics(np.array([0.0, 457.0, 367.0, 248.0]))
