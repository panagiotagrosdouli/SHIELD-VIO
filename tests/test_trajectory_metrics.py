import numpy as np
import pytest

from shield_vio.evaluation.trajectory_metrics import (
    absolute_trajectory_error,
    align_positions_se3,
    relative_pose_error,
    summarize_trajectory_metrics,
)


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def test_se3_alignment_removes_rigid_transform() -> None:
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 1.0, 0.2], [2.0, 1.5, 0.5]]
    )
    rotation = _rotation_z(0.6)
    translation = np.array([3.0, -2.0, 1.0])
    estimated = reference @ rotation.T + translation

    aligned, recovered_rotation, recovered_translation = align_positions_se3(
        estimated, reference
    )

    np.testing.assert_allclose(aligned, reference, atol=1e-10)
    np.testing.assert_allclose(
        estimated @ recovered_rotation.T + recovered_translation,
        reference,
        atol=1e-10,
    )
    assert np.linalg.det(recovered_rotation) == pytest.approx(1.0)


def test_absolute_trajectory_error_preserves_metric_scale_error() -> None:
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    estimated = 1.2 * reference

    errors = absolute_trajectory_error(estimated, reference)

    assert np.max(errors) > 0.1


def test_relative_pose_error_detects_increment_drift() -> None:
    reference = np.column_stack((np.arange(6.0), np.zeros(6), np.zeros(6)))
    estimated = np.column_stack((1.1 * np.arange(6.0), np.zeros(6), np.zeros(6)))

    errors = relative_pose_error(estimated, reference, delta=2)

    np.testing.assert_allclose(errors, np.full(4, 0.2), atol=1e-12)


def test_metric_summary_reports_expected_statistics() -> None:
    reference = np.column_stack((np.arange(5.0), np.zeros(5), np.zeros(5)))
    estimated = reference.copy()
    estimated[-1, 0] += 0.5

    metrics = summarize_trajectory_metrics(estimated, reference, rpe_delta=1, align=False)

    assert metrics.sample_count == 5
    assert metrics.ate_rmse_m == pytest.approx(np.sqrt(0.25 / 5.0))
    assert metrics.ate_max_m == pytest.approx(0.5)
    assert metrics.rpe_rmse_m == pytest.approx(0.25)
    assert metrics.rpe_max_m == pytest.approx(0.5)
    assert metrics.rpe_delta == 1


def test_relative_pose_error_rejects_invalid_delta() -> None:
    positions = np.zeros((3, 3))
    with pytest.raises(ValueError, match="positive integer"):
        relative_pose_error(positions, positions, delta=0)
    with pytest.raises(ValueError, match="smaller than"):
        relative_pose_error(positions, positions, delta=3)
