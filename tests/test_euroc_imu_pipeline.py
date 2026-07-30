from pathlib import Path

import numpy as np
import pytest

from shield_vio.datasets.euroc_imu import IMUSample, load_euroc_imu
from shield_vio.estimation.imu_runner import run_imu_propagation


def _write_imu_csv(path: Path, timestamps_ns: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["#timestamp [ns],wx,wy,wz,ax,ay,az\n"]
    rows.extend(f"{timestamp},0,0,0,0,0,9.80665\n" for timestamp in timestamps_ns)
    path.write_text("".join(rows), encoding="utf-8")


def test_load_euroc_imu_converts_nanoseconds_and_column_order(tmp_path: Path):
    path = tmp_path / "mav0" / "imu0" / "data.csv"
    _write_imu_csv(path, [1_000_000_000, 1_005_000_000])

    samples = load_euroc_imu(path)

    assert len(samples) == 2
    assert samples[0].timestamp_s == pytest.approx(1.0)
    assert samples[1].timestamp_s == pytest.approx(1.005)
    assert np.allclose(samples[0].angular_velocity_rps, 0.0)
    assert np.allclose(samples[0].acceleration_mps2, [0.0, 0.0, 9.80665])


def test_loader_rejects_non_monotonic_timestamps(tmp_path: Path):
    path = tmp_path / "data.csv"
    _write_imu_csv(path, [2_000_000_000, 1_000_000_000])

    with pytest.raises(ValueError, match="strictly increasing"):
        load_euroc_imu(path)


def test_stationary_specific_force_produces_stationary_world_state():
    samples = tuple(
        IMUSample(
            timestamp_s=index * 0.01,
            angular_velocity_rps=np.zeros(3),
            acceleration_mps2=np.array([0.0, 0.0, 9.80665]),
        )
        for index in range(101)
    )

    estimator, recorder, summary = run_imu_propagation(samples, record_stride=10)

    assert summary.sample_count == 101
    assert summary.duration_s == pytest.approx(1.0)
    assert summary.mean_dt_s == pytest.approx(0.01)
    assert np.allclose(estimator.state.position_m, 0.0, atol=1e-10)
    assert np.allclose(estimator.state.velocity_mps, 0.0, atol=1e-10)
    assert recorder.samples[0].timestamp_s == pytest.approx(0.0)
    assert recorder.samples[-1].timestamp_s == pytest.approx(1.0)


def test_runner_rejects_large_dataset_gap():
    samples = (
        IMUSample(0.0, np.zeros(3), np.array([0.0, 0.0, 9.80665])),
        IMUSample(0.5, np.zeros(3), np.array([0.0, 0.0, 9.80665])),
    )

    with pytest.raises(ValueError, match="exceeds"):
        run_imu_propagation(samples, max_dt_s=0.1)
