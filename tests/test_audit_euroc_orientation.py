from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_euroc_orientation import (
    _matrix_to_quaternion_wxyz,
    _quaternion_multiply_wxyz,
    _rotation_from_quaternion_wxyz,
    attitude_alignment_summary,
)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]


def test_position_alignment_removes_global_rotation_from_attitude_error() -> None:
    times = np.arange(20, dtype=float) * 0.05
    groundtruth_p = np.column_stack((times, np.sin(times), 0.1 * times**2))
    groundtruth_q = np.asarray([
        _axis_angle(np.array([1.0, 0.2, 0.1]), 0.2 + 0.01 * t)
        for t in times
    ])
    world_rotation = _rotation_from_quaternion_wxyz(
        _axis_angle(np.array([0.2, 0.3, 1.0]), 0.7)
    )
    global_q = _matrix_to_quaternion_wxyz(world_rotation)
    world_translation = np.array([2.0, -3.0, 0.5])
    estimate_p = (world_rotation.T @ (groundtruth_p - world_translation).T).T
    inverse_global_q = global_q * np.array([1, -1, -1, -1])
    estimate_q = np.asarray([
        _quaternion_multiply_wxyz(inverse_global_q, q) for q in groundtruth_q
    ])

    report = attitude_alignment_summary(
        times,
        estimate_p,
        estimate_q,
        times,
        groundtruth_p,
        groundtruth_q,
        max_gap_s=0.01,
    )

    assert report["associated_samples"] == len(times)
    assert report["attitude_error_deg_mean"] < 1e-5
    assert report["attitude_error_deg_p95"] < 1e-5


def test_attitude_alignment_rejects_fewer_than_three_associations() -> None:
    times = np.array([0.0, 0.05, 0.10])
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
    orientations = np.tile([1.0, 0, 0, 0], (3, 1))
    with pytest.raises(ValueError, match="fewer than three"):
        attitude_alignment_summary(
            times,
            positions,
            orientations,
            np.array([1.0, 1.05, 1.1]),
            positions,
            orientations,
            max_gap_s=0.01,
        )
