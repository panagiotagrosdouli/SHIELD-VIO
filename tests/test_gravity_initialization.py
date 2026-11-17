"""Tests for stationary-gravity attitude initialization without ground truth."""
from __future__ import annotations

import numpy as np
import pytest

from shield_vio.backends.eskf import ESKFBackend
from shield_vio.core.math import quat_to_rot


def test_stationary_tilted_imu_cancels_gravity_after_alignment() -> None:
    backend = ESKFBackend()
    backend.initialize(0)
    accel = np.tile(np.array([9.81, 0.0, 0.0]), (50, 1))
    gyro = np.zeros_like(accel)
    state = backend.initialize_from_stationary_imu(accel, gyro)
    np.testing.assert_allclose(
        quat_to_rot(state.orientation_wxyz) @ accel[0],
        np.array([0.0, 0.0, 9.81]),
        atol=1e-8,
    )
    for i in range(1, 101):
        backend.process_imu(i * 10_000_000, accel[0], gyro[0])
    assert np.linalg.norm(backend.snapshot().position) < 1e-5
    assert np.linalg.norm(backend.snapshot().velocity) < 1e-5


def test_gravity_alignment_rejects_nonstationary_motion() -> None:
    backend = ESKFBackend()
    backend.initialize(0)
    acceleration = np.tile(np.array([0.0, 0.0, 9.81]), (50, 1))
    angular_velocity = np.tile(np.array([1.2, 0.0, 0.0]), (50, 1))
    with pytest.raises(ValueError, match="not sufficiently stationary"):
        backend.initialize_from_stationary_imu(acceleration, angular_velocity)


def test_gravity_alignment_rejects_dynamic_specific_force() -> None:
    backend = ESKFBackend()
    backend.initialize(0)
    acceleration = np.tile(np.array([5.0, 0.0, 9.81]), (50, 1))
    gyro = np.zeros_like(acceleration)
    with pytest.raises(ValueError, match="not sufficiently stationary"):
        backend.initialize_from_stationary_imu(acceleration, gyro)


def test_gravity_alignment_requires_real_stationary_window() -> None:
    backend = ESKFBackend()
    backend.initialize(0)
    with pytest.raises(ValueError, match=">=10"):
        backend.initialize_from_stationary_imu(np.zeros((3, 3)), np.zeros((3, 3)))
