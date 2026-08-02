from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.run_euroc import stereo_transform_right_left


def _calibration(transform_body_sensor: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(transform_body_sensor=np.asarray(transform_body_sensor, dtype=float))


def test_stereo_transform_maps_left_camera_into_right_camera() -> None:
    transform_body_left = np.eye(4)
    transform_body_right = np.eye(4)
    transform_body_right[:3, 3] = [0.11, 0.0, 0.0]

    transform_right_left = stereo_transform_right_left(
        _calibration(transform_body_left),
        _calibration(transform_body_right),
    )

    np.testing.assert_allclose(transform_right_left[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(transform_right_left[:3, 3], [-0.11, 0.0, 0.0], atol=1e-12)


def test_stereo_transform_preserves_extrinsic_rotation() -> None:
    angle = np.deg2rad(10.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform_body_left = np.eye(4)
    transform_body_left[:3, :3] = rotation
    transform_body_right = np.eye(4)

    transform_right_left = stereo_transform_right_left(
        _calibration(transform_body_left),
        _calibration(transform_body_right),
    )

    np.testing.assert_allclose(transform_right_left[:3, :3], rotation, atol=1e-12)
