from __future__ import annotations

import numpy as np

from shield_vio.navigation.closed_loop import (
    NavigationShield,
    NavigationState,
    ShieldInputs,
    ShieldState,
)
from shield_vio.simulation.degradation import DegradationSpec, degrade_image, degrade_imu


def test_visual_degradation_is_seeded_and_bounded() -> None:
    image = np.full((16, 16), 200, dtype=np.uint8)
    spec = DegradationSpec("noise", start_s=1.0, duration_s=2.0, severity=0.7, seed=4)
    nominal = degrade_image(image, 0.5, spec)
    first = degrade_image(image, 1.5, spec)
    second = degrade_image(image, 1.5, spec)
    assert np.array_equal(nominal, image)
    assert np.array_equal(first, second)
    assert first.dtype == image.dtype
    assert not np.array_equal(first, image)


def test_imu_packet_dropout_has_explicit_validity() -> None:
    spec = DegradationSpec("packet_dropout", 2.0, 1.0, 1.0, seed=2)
    accel = np.array([0.0, 0.0, 9.81])
    gyro = np.zeros(3)
    _, _, valid_before = degrade_imu(accel, gyro, 1.0, spec)
    _, _, valid_during = degrade_imu(accel, gyro, 2.5, spec)
    assert valid_before
    assert not valid_during
    assert spec.manifest()["end_s"] == 3.0


def test_shield_slows_navigation_and_emergency_stops() -> None:
    shield = NavigationShield(dwell_steps=2)
    nav = NavigationState(position=np.zeros(2), goal=np.array([10.0, 0.0]))
    nominal = shield.update(ShieldInputs(0.1, 0.2, 0.9, 0.9))
    p1 = nav.step(1.0, nominal)
    slow = shield.update(
        ShieldInputs(0.5, 0.4, 0.8, 0.9, shift_state="CONFIRMED_SHIFT")
    )
    p2 = nav.step(1.0, slow)
    emergency = shield.update(ShieldInputs(0.2, 0.2, 0.9, 0.9, sensors_stale=True))
    p3 = nav.step(1.0, emergency)
    assert nominal.state is ShieldState.NORMAL
    assert slow.state is ShieldState.SLOW_DOWN
    assert 0.0 < p2[0] - p1[0] < p1[0]
    assert emergency.state is ShieldState.EMERGENCY_STOP
    assert np.array_equal(p3, p2)


def test_shield_requests_recovery_when_visual_tracking_collapses() -> None:
    shield = NavigationShield()
    output = shield.update(ShieldInputs(0.82, 2.0, 0.08, 0.8, recovery_feasible=True))
    assert output.state is ShieldState.RELOCALIZE_REQUESTED
    assert output.recovery_action == "rotate_for_features"
    assert "p_failure=0.820" in output.reason
