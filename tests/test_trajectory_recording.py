import csv

import numpy as np
import pytest

from shield_vio.estimation.error_state_ekf import ErrorStateEKF
from shield_vio.estimation.trajectory import TrajectoryRecorder, TrajectorySample


def test_recorder_copies_state_and_enforces_time_order(tmp_path):
    ekf = ErrorStateEKF()
    recorder = TrajectoryRecorder()

    ekf.propagate(np.array([0.0, 0.0, 9.80665]), np.zeros(3), 0.01)
    first = recorder.append(ekf.state)
    original_position = first.position_m.copy()

    ekf.state.position_m[:] = 123.0
    assert np.allclose(first.position_m, original_position)

    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.append(ekf.state)


def test_sample_normalizes_quaternion():
    sample = TrajectorySample(
        timestamp_s=1.0,
        position_m=np.zeros(3),
        quaternion_wxyz=np.array([2.0, 0.0, 0.0, 0.0]),
        velocity_mps=np.zeros(3),
        accel_bias_mps2=np.zeros(3),
        gyro_bias_rps=np.zeros(3),
        position_covariance_trace_m2=0.1,
        attitude_covariance_trace_rad2=0.2,
    )
    assert np.allclose(sample.quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])


def test_tum_and_csv_exports_have_expected_schema(tmp_path):
    ekf = ErrorStateEKF()
    recorder = TrajectoryRecorder()
    for _ in range(2):
        ekf.propagate(np.array([0.0, 0.0, 9.80665]), np.zeros(3), 0.01)
        recorder.append(ekf.state)

    tum_path = recorder.write_tum(tmp_path / "trajectory.txt")
    rows = tum_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2
    assert all(len(row.split()) == 8 for row in rows)

    csv_path = recorder.write_csv(tmp_path / "trajectory.csv")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        table = list(csv.reader(handle))
    assert table[0][0] == "timestamp_s"
    assert "position_covariance_trace_m2" in table[0]
    assert len(table) == 3


def test_invalid_covariance_is_rejected():
    ekf = ErrorStateEKF()
    ekf.state.covariance = np.eye(3)
    with pytest.raises(ValueError, match="15x15"):
        TrajectorySample.from_eskf_state(ekf.state)
