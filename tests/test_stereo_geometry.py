from __future__ import annotations

import numpy as np
import pytest

from shield_vio.vision.stereo_geometry import triangulate_stereo_points


def _camera_matrix() -> np.ndarray:
    return np.array([[460.0, 0.0, 320.0], [0.0, 460.0, 240.0], [0.0, 0.0, 1.0]])


def _project(points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    homogeneous = (camera_matrix @ points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def test_recovers_metric_points_for_rectified_stereo() -> None:
    camera = _camera_matrix()
    baseline_m = 0.11
    points_left = np.array([[0.2, -0.1, 2.0], [-0.4, 0.2, 4.0], [0.1, 0.3, 8.0]])
    rotation = np.eye(3)
    translation = np.array([-baseline_m, 0.0, 0.0])
    points_right = points_left + translation

    result = triangulate_stereo_points(
        _project(points_left, camera),
        _project(points_right, camera),
        camera,
        camera,
        rotation,
        translation,
        max_reprojection_error_px=0.1,
    )

    assert np.all(result.valid_mask)
    np.testing.assert_allclose(result.valid_points_left_m, points_left, atol=1e-6)
    assert np.max(result.reprojection_error_left_px) < 1e-6
    assert np.max(result.reprojection_error_right_px) < 1e-6


def test_rejects_points_behind_camera_and_outside_depth_range() -> None:
    camera = _camera_matrix()
    translation = np.array([-0.11, 0.0, 0.0])
    points_left = np.array([[0.1, 0.0, 2.0], [0.1, 0.0, 20.0], [0.1, 0.0, -2.0]])
    points_right = points_left + translation

    result = triangulate_stereo_points(
        _project(points_left, camera),
        _project(points_right, camera),
        camera,
        camera,
        np.eye(3),
        translation,
        min_depth_m=0.5,
        max_depth_m=10.0,
    )

    assert result.valid_mask.tolist() == [True, False, False]


def test_reprojection_gate_rejects_bad_match() -> None:
    camera = _camera_matrix()
    translation = np.array([-0.11, 0.0, 0.0])
    point = np.array([[0.2, 0.1, 3.0]])
    left = _project(point, camera)
    right = _project(point + translation, camera)
    right[0, 1] += 15.0

    result = triangulate_stereo_points(
        left,
        right,
        camera,
        camera,
        np.eye(3),
        translation,
        max_reprojection_error_px=1.0,
    )

    assert not result.valid_mask[0]


def test_validates_calibration_and_correspondences() -> None:
    camera = _camera_matrix()
    with pytest.raises(ValueError, match="must match"):
        triangulate_stereo_points(
            np.zeros((2, 2)),
            np.zeros((1, 2)),
            camera,
            camera,
            np.eye(3),
            np.array([-0.11, 0.0, 0.0]),
        )
    with pytest.raises(ValueError, match="proper rotation"):
        triangulate_stereo_points(
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            camera,
            camera,
            np.diag([1.0, 1.0, -1.0]),
            np.array([-0.11, 0.0, 0.0]),
        )
