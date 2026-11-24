import numpy as np
import pytest

from scripts.audit_euroc_imu_frames import (
    audit_imu_frames,
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
