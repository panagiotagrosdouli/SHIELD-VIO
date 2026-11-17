"""Regression tests for physically correct ESKF IMU Jacobian propagation."""
from __future__ import annotations

import numpy as np

from shield_vio.estimation.eskf import ErrorStateEKF
from shield_vio.estimation.state import VIOState
from shield_vio.core.math import quat_multiply, quat_to_rot, small_angle_quat, skew


def test_imu_velocity_attitude_jacobian_matches_finite_difference() -> None:
    """Small orientation errors change velocity via -R[a]x, not -R."""
    acceleration_body = np.array([0.4, -1.3, 9.6])
    orientation = quat_multiply(
        small_angle_quat(np.array([0.2, -0.3, 0.1])),
        small_angle_quat(np.array([0.07, 0.13, -0.2])),
    )
    rotation = quat_to_rot(orientation)
    dt = 0.01
    epsilon = 1e-6
    jacobian = np.empty((3, 3))
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = epsilon
        rotated_plus = quat_to_rot(quat_multiply(orientation, small_angle_quat(delta)))
        rotated_minus = quat_to_rot(quat_multiply(orientation, small_angle_quat(-delta)))
        jacobian[:, axis] = (
            rotated_plus @ acceleration_body - rotated_minus @ acceleration_body
        ) * (0.5 * dt / epsilon)

    expected = -(rotation @ skew(acceleration_body)) * dt
    np.testing.assert_allclose(jacobian, expected, atol=1e-8, rtol=1e-6)

    # Isolate propagation of a single small attitude uncertainty via a cross-covariance.
    # With zero process noise, P_vtheta = F_vtheta * P_thetatheta.
    state = VIOState()
    state.orientation = orientation
    state.covariance = np.zeros((15, 15))
    state.covariance[6:9, 6:9] = np.eye(3) * 1e-4
    result = ErrorStateEKF().propagate(
        state,
        accel_m=acceleration_body,
        gyro_m=np.zeros(3),
        dt=dt,
    )
    np.testing.assert_allclose(
        result.covariance[3:6, 6:9],
        expected * 1e-4,
        atol=1e-8,
        rtol=1e-4,
    )
