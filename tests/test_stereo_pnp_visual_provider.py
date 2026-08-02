from __future__ import annotations

import numpy as np

from shield_vio.backends.base import EstimatorState
from shield_vio.experiments.stereo_pnp_visual_provider import build_metric_pose_measurement
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
