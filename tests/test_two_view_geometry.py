import numpy as np
import pytest

from shield_vio.vision.two_view_geometry import estimate_relative_pose


def _synthetic_correspondences(seed: int = 7):
    rng = np.random.default_rng(seed)
    intrinsic = np.array([[420.0, 0.0, 320.0], [0.0, 420.0, 240.0], [0.0, 0.0, 1.0]])
    points_world = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, 120),
            rng.uniform(-1.5, 1.5, 120),
            rng.uniform(4.0, 9.0, 120),
        ]
    )
    angle = np.deg2rad(4.0)
    rotation = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    translation = np.array([0.35, 0.02, 0.05])

    def project(points):
        normalized = points[:, :2] / points[:, 2:3]
        return normalized @ intrinsic[:2, :2].T + intrinsic[:2, 2]

    previous = project(points_world)
    current = project((rotation @ points_world.T).T + translation)
    previous += rng.normal(0.0, 0.15, previous.shape)
    current += rng.normal(0.0, 0.15, current.shape)
    current[:12] = rng.uniform([0.0, 0.0], [640.0, 480.0], size=(12, 2))
    return intrinsic, previous, current, rotation, translation / np.linalg.norm(translation)


def test_relative_pose_recovers_rotation_and_translation_direction() -> None:
    intrinsic, previous, current, expected_rotation, expected_translation = _synthetic_correspondences()
    measurement = estimate_relative_pose(previous, current, intrinsic)

    rotation_error = measurement.rotation @ expected_rotation.T
    angle_error = np.arccos(np.clip((np.trace(rotation_error) - 1.0) / 2.0, -1.0, 1.0))
    direction_agreement = abs(float(measurement.translation_direction @ expected_translation))

    assert measurement.inlier_count >= 80
    assert measurement.inlier_ratio > 0.65
    assert angle_error < np.deg2rad(1.5)
    assert direction_agreement > 0.9
    assert measurement.median_epipolar_error_px < 1.0
    assert not measurement.is_degenerate


def test_relative_pose_marks_low_parallax_as_degenerate() -> None:
    intrinsic, previous, _, _, _ = _synthetic_correspondences()
    current = previous + np.array([0.15, 0.0])
    current += np.random.default_rng(1).normal(0.0, 0.01, current.shape)

    measurement = estimate_relative_pose(
        previous,
        current,
        intrinsic,
        min_inliers=8,
        min_median_parallax_px=1.0,
    )
    assert measurement.median_parallax_px < 1.0
    assert measurement.is_degenerate


def test_relative_pose_rejects_invalid_inputs() -> None:
    intrinsic = np.eye(3)
    with pytest.raises(ValueError):
        estimate_relative_pose(np.zeros((7, 2)), np.zeros((7, 2)), intrinsic)
    with pytest.raises(ValueError):
        estimate_relative_pose(np.zeros((8, 2)), np.zeros((9, 2)), intrinsic)
    with pytest.raises(ValueError):
        estimate_relative_pose(np.zeros((8, 2)), np.zeros((8, 2)), np.zeros((3, 3)))
