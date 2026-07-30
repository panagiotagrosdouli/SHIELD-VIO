import numpy as np

from shield_vio.estimation.error_state_ekf import ErrorStateEKF, quat_to_rot
from shield_vio.estimation.visual_rotation_update import fuse_relative_camera_rotation
from shield_vio.preintegration.imu_preintegrator import so3_exp
from shield_vio.vision.two_view_geometry import RelativePoseMeasurement


def _measurement(rotation: np.ndarray, *, degenerate: bool = False) -> RelativePoseMeasurement:
    return RelativePoseMeasurement(
        rotation=rotation,
        translation_direction=np.array([1.0, 0.0, 0.0]),
        inlier_mask=np.ones(20, dtype=bool),
        correspondence_count=20,
        inlier_count=20,
        inlier_ratio=1.0,
        median_parallax_px=4.0,
        median_epipolar_error_px=0.2,
        is_degenerate=degenerate,
    )


def _rotation_angle(rotation: np.ndarray) -> float:
    return float(
        np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    )


def test_visual_rotation_update_reduces_attitude_error() -> None:
    ekf = ErrorStateEKF()
    previous_body = np.eye(3)
    relative_camera = so3_exp(np.array([0.0, np.deg2rad(3.0), 0.0]))
    expected_current_body = relative_camera.T

    before = _rotation_angle(quat_to_rot(ekf.state.quaternion_wxyz).T @ expected_current_body)
    result = fuse_relative_camera_rotation(
        ekf,
        previous_body,
        _measurement(relative_camera),
        np.eye(3) * np.deg2rad(0.5) ** 2,
        max_nis=1e6,
    )
    after = _rotation_angle(quat_to_rot(ekf.state.quaternion_wxyz).T @ expected_current_body)

    assert result.accepted
    assert result.reason == "accepted"
    assert result.nis >= 0.0
    assert after < before
    assert np.min(np.linalg.eigvalsh(ekf.state.covariance)) >= -1e-12


def test_degenerate_geometry_does_not_modify_state() -> None:
    ekf = ErrorStateEKF()
    quaternion_before = ekf.state.quaternion_wxyz.copy()
    covariance_before = ekf.state.covariance.copy()

    result = fuse_relative_camera_rotation(
        ekf,
        np.eye(3),
        _measurement(np.eye(3), degenerate=True),
        np.eye(3) * 0.01,
    )

    assert not result.accepted
    assert result.reason == "degenerate_geometry"
    assert np.array_equal(ekf.state.quaternion_wxyz, quaternion_before)
    assert np.array_equal(ekf.state.covariance, covariance_before)


def test_nis_gate_rejects_inconsistent_rotation_without_mutation() -> None:
    ekf = ErrorStateEKF()
    quaternion_before = ekf.state.quaternion_wxyz.copy()
    covariance_before = ekf.state.covariance.copy()
    large_rotation = so3_exp(np.array([0.0, np.deg2rad(60.0), 0.0]))

    result = fuse_relative_camera_rotation(
        ekf,
        np.eye(3),
        _measurement(large_rotation),
        np.eye(3) * np.deg2rad(0.1) ** 2,
        max_nis=11.345,
    )

    assert not result.accepted
    assert result.reason == "nis_gate"
    assert result.nis > 11.345
    assert np.array_equal(ekf.state.quaternion_wxyz, quaternion_before)
    assert np.array_equal(ekf.state.covariance, covariance_before)
