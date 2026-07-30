from pathlib import Path

import numpy as np
import pytest

from shield_vio.evaluation.euroc import (
    TimestampedTrajectory,
    associate_ground_truth,
    load_estimator_trajectory,
    load_euroc_ground_truth,
)


def test_timestamped_trajectory_rejects_unsorted_times() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TimestampedTrajectory(np.array([0.0, 0.2, 0.1]), np.zeros((3, 3)))


def test_load_euroc_ground_truth_converts_nanoseconds(tmp_path: Path) -> None:
    path = tmp_path / "mav0/state_groundtruth_estimate0"
    path.mkdir(parents=True)
    (path / "data.csv").write_text(
        "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n"
        "1000000000,0,1,2,1,0,0,0\n"
        "1100000000,1,2,3,1,0,0,0\n",
        encoding="utf-8",
    )
    trajectory = load_euroc_ground_truth(tmp_path)
    np.testing.assert_allclose(trajectory.timestamps, [0.0, 0.1])
    np.testing.assert_allclose(trajectory.positions, [[0, 1, 2], [1, 2, 3]])


def test_load_estimator_trajectory_accepts_tum_style_text(tmp_path: Path) -> None:
    path = tmp_path / "estimate.txt"
    path.write_text(
        "1000000000 0 0 0 0 0 0 1\n"
        "1500000000 1 2 3 0 0 0 1\n",
        encoding="utf-8",
    )
    trajectory = load_estimator_trajectory(path, timestamp_unit="nanoseconds")
    np.testing.assert_allclose(trajectory.timestamps, [0.0, 0.5])
    np.testing.assert_allclose(trajectory.positions[-1], [1.0, 2.0, 3.0])


def test_associate_ground_truth_interpolates_positions() -> None:
    ground_truth = TimestampedTrajectory(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([[0, 0, 0], [1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=float),
    )
    estimate = TimestampedTrajectory(
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([[10, 0, 0], [11, 0, 0], [12, 0, 0], [13, 0, 0]], dtype=float),
    )
    times, estimated, associated_gt = associate_ground_truth(
        estimate, ground_truth, max_gap=0.5
    )
    np.testing.assert_allclose(times, estimate.timestamps)
    np.testing.assert_allclose(estimated, estimate.positions)
    np.testing.assert_allclose(
        associated_gt,
        [[0.25, 0.5, 0.75], [0.75, 1.5, 2.25], [1.25, 2.5, 3.75], [1.75, 3.5, 5.25]],
    )


def test_association_rejects_non_overlapping_trajectories() -> None:
    ground_truth = TimestampedTrajectory(np.array([0.0, 1.0]), np.zeros((2, 3)))
    estimate = TimestampedTrajectory(np.array([2.0, 3.0]), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="do not overlap"):
        associate_ground_truth(estimate, ground_truth)
