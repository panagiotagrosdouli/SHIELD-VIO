import numpy as np
import pytest

from scripts.oracle_attitude_initialization_diagnostic import compare_initial_attitudes


def _rotation_x(angle: float) -> np.ndarray:
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)],
    ])


def test_oracle_attitude_ablation_is_explicit_and_reduces_synthetic_drift() -> None:
    count = 101
    t = np.arange(count, dtype=float) * 0.01
    angle = np.deg2rad(35.0)
    rotation = _rotation_x(angle)
    q = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    gravity = np.array([0.0, 0.0, -9.80665])
    world_acceleration = np.array([0.4, 0.0, 0.0])
    specific_force = rotation.T @ (world_acceleration - gravity)
    gyro_bias = np.array([0.0, 0.0, 0.03])
    imu = np.column_stack([
        t * 1e9,
        np.tile(gyro_bias, (count, 1)),
        np.tile(specific_force, (count, 1)),
    ])
    gt_positions = 0.5 * t[:, None] ** 2 * world_acceleration
    gt = np.column_stack([
        t * 1e9,
        gt_positions,
        np.tile(q, (count, 1)),
    ])

    report = compare_initial_attitudes(imu, t, gt_positions, gt[:, 4:8])

    assert report["ground_truth_used_for_initialization"] is True
    assert report["confirmatory"] is False
    assert report["oracle_gyro_bias_radps"] == pytest.approx(gyro_bias, abs=1e-10)
    assert report["oracle_attitude_and_gyro_bias"]["rpe_translation_rmse_m_delta_1"] < 1e-8
    assert (
        report["baseline_identity_zero_bias"]["rpe_translation_rmse_m_delta_1"]
        > report["oracle_attitude_and_gyro_bias"]["rpe_translation_rmse_m_delta_1"]
    )
