import numpy as np
import pytest

from scripts.audit_euroc_imu_frames import (
    audit_gyro_frames,
    audit_imu_frames,
    estimate_acceleration_from_position,
    world_acceleration_from_specific_force,
)


def test_specific_force_and_gravity_cancel_at_rest() -> None:
    accel = np.array([0.0, 0.0, 9.80665])
    result = world_acceleration_from_specific_force(accel, np.array([1.0, 0.0, 0.0, 0.0]))
    assert result == pytest.approx(np.zeros(3), abs=1e-10)


def test_known_orientation_recovers_world_acceleration() -> None:
    angle = np.pi / 2
    q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
    specific_force = np.array([0.0, 0.0, 9.80665]) + np.array([1.0, 0.0, 0.0])
    result = world_acceleration_from_specific_force(specific_force, q)
    assert result == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-8)


def test_audit_rejects_insufficient_timestamp_overlap() -> None:
    imu = np.column_stack([np.arange(10) * 1e9, np.zeros((10, 6))])
    gt = np.column_stack([1e12 + np.arange(10) * 1e9, np.zeros((10, 16))])
    with pytest.raises(ValueError, match="fewer than ten"):
        audit_imu_frames(imu, gt)



def test_local_quadratic_position_fit_recovers_constant_acceleration() -> None:
    t = np.arange(41, dtype=float) * 0.01
    acceleration = np.array([0.5, -1.0, 2.0])
    positions = 0.5 * t[:, None] ** 2 * acceleration
    result = estimate_acceleration_from_position(t, positions, window_samples=11)
    assert result[5:-5] == pytest.approx(np.tile(acceleration, (31, 1)), abs=1e-9)


def test_pose_only_groundtruth_is_supported() -> None:
    t = np.arange(101, dtype=float) * 0.005
    gt = np.column_stack([
        t * 1e9,
        np.zeros((len(t), 3)),
        np.ones(len(t)),
        np.zeros((len(t), 3)),
    ])
    imu = np.column_stack([
        t * 1e9,
        np.zeros((len(t), 3)),
        np.tile([0.0, 0.0, 9.80665], (len(t), 1)),
    ])
    report = audit_imu_frames(imu, gt)
    assert report["associated_samples"] == len(t)
    assert report["reference_acceleration_source"] == "smoothed_groundtruth_position_second_derivative"



def test_gyro_audit_recovers_constant_bias_and_orientation_direction() -> None:
    gt_t = np.arange(101, dtype=float) * 0.005
    imu_t = gt_t[:-1] + 0.0025
    angular_rate = np.array([0.08, -0.04, 0.12])
    bias = np.array([0.01, -0.02, 0.03])
    axis = angular_rate / np.linalg.norm(angular_rate)
    orientations = []
    for t in gt_t:
        angle = np.linalg.norm(angular_rate) * t
        orientations.append(np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)])
    measured = np.tile(angular_rate + bias, (len(imu_t), 1))

    report = audit_gyro_frames(
        imu_t, measured, gt_t, np.asarray(orientations), max_gap_s=0.003
    )

    assert report["estimated_constant_gyro_bias_radps"] == pytest.approx(bias, abs=1e-10)
    assert report["direct_body_rate_rmse_after_constant_bias_removal_radps"] < 1e-10
    assert report["reversed_orientation_rate_rmse_radps"] > report["direct_body_rate_rmse_radps"]
