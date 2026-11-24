from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from shield_vio.backends.base import EstimatorState
from shield_vio.experiments.stereo_pnp_visual_provider import (
    _load_matching_cam1_image,
    build_metric_pose_measurement,
)
from shield_vio.vision.stereo_pnp import MetricPnPResult


def _state(position: np.ndarray | None = None) -> EstimatorState:
    return EstimatorState(
        timestamp_ns=1,
        position=np.zeros(3) if position is None else np.asarray(position, dtype=float),
        velocity=np.zeros(3),
        orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        covariance=np.eye(15),
    )


def _motion(rotation: np.ndarray, translation: np.ndarray) -> MetricPnPResult:
    return MetricPnPResult(
        rotation_current_previous=rotation,
        translation_current_previous_m=translation,
        inlier_mask=np.ones(20, dtype=bool),
        correspondence_count=20,
        reprojection_rmse_px=0.2,
    )


def test_metric_translation_becomes_body_position_residual() -> None:
    previous_world_body = np.eye(4)
    motion = _motion(np.eye(3), np.array([-1.0, 0.0, 0.0]))

    measurement = build_metric_pose_measurement(
        _state(),
        previous_world_body,
        motion,
        np.eye(4),
        position_std_m=0.1,
        rotation_std_rad=0.05,
    )

    np.testing.assert_allclose(measurement.residual[:3], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(measurement.residual[3:], np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(measurement.measurement_matrix[:3, :3], np.eye(3))
    np.testing.assert_allclose(measurement.measurement_matrix[3:, 6:9], np.eye(3))
    assert np.count_nonzero(measurement.measurement_matrix[:, 3:6]) == 0


def test_metric_rotation_becomes_orientation_residual() -> None:
    angle = 0.2
    current_previous = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    motion = _motion(current_previous, np.zeros(3))

    measurement = build_metric_pose_measurement(
        _state(),
        np.eye(4),
        motion,
        np.eye(4),
        position_std_m=0.1,
        rotation_std_rad=0.05,
    )

    np.testing.assert_allclose(measurement.residual[:3], np.zeros(3), atol=1e-9)
    np.testing.assert_allclose(measurement.residual[3:], [0.0, 0.0, -angle], atol=1e-8)


def test_camera_lever_arm_changes_body_translation() -> None:
    body_camera = np.eye(4)
    body_camera[:3, 3] = [0.2, 0.0, 0.0]
    angle = np.pi / 2
    current_previous = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    measurement = build_metric_pose_measurement(
        _state(),
        np.eye(4),
        _motion(current_previous, np.zeros(3)),
        body_camera,
        position_std_m=0.1,
        rotation_std_rad=0.05,
    )

    assert np.linalg.norm(measurement.residual[:3]) > 0.1
    np.testing.assert_allclose(
        np.diag(measurement.measurement_covariance),
        [0.01, 0.01, 0.01, 0.0025, 0.0025, 0.0025],
    )


def test_cam1_image_is_associated_by_nearest_recorded_timestamp(tmp_path) -> None:
    left_data = tmp_path / "mav0" / "cam0" / "data"
    right_data = tmp_path / "mav0" / "cam1" / "data"
    left_data.mkdir(parents=True)
    right_data.mkdir(parents=True)
    right_path = right_data / "1000000005.png"
    Image.fromarray(np.array([[11, 22], [33, 44]], dtype=np.uint8)).save(right_path)
    (right_data.parent / "data.csv").write_text(
        "#timestamp [ns],filename\n1000000005,1000000005.png\n",
        encoding="utf-8",
    )
    packet = SimpleNamespace(
        frame=SimpleNamespace(
            image_path=left_data / "1000000000.png",
            timestamp_ns=1_000_000_000,
        )
    )

    image = _load_matching_cam1_image(packet)
    np.testing.assert_array_equal(image, [[11, 22], [33, 44]])


def test_cam1_image_association_rejects_large_timestamp_gap(tmp_path) -> None:
    left_data = tmp_path / "mav0" / "cam0" / "data"
    right_data = tmp_path / "mav0" / "cam1" / "data"
    left_data.mkdir(parents=True)
    right_data.mkdir(parents=True)
    (right_data.parent / "data.csv").write_text(
        "#timestamp [ns],filename\n2000000000,2000000000.png\n",
        encoding="utf-8",
    )
    packet = SimpleNamespace(
        frame=SimpleNamespace(
            image_path=left_data / "1000000000.png",
            timestamp_ns=1_000_000_000,
        )
    )

    with pytest.raises(ValueError, match="nearest cam1 image differs"):
        _load_matching_cam1_image(packet)
