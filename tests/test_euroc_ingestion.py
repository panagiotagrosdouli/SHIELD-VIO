from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shield_vio.datasets.calibration import load_camera_calibration, load_imu_calibration
from shield_vio.datasets.euroc import (
    estimate_stream_rate_hz,
    read_camera_frames,
    read_imu_samples,
    synchronize_camera_and_imu,
)


def _write_sensor_yaml(path: Path, *, camera: bool) -> None:
    common = """sensor_type: {sensor_type}
comment: fixture
T_BS:
  rows: 4
  cols: 4
  data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
rate_hz: {rate}
""".format(sensor_type="camera" if camera else "imu", rate=20 if camera else 200)
    if camera:
        common += """resolution: [752, 480]
camera_model: pinhole
intrinsics: [458.0, 457.0, 367.0, 248.0]
distortion_model: radial-tangential
distortion_coefficients: [-0.28, 0.07, 0.0, 0.0]
"""
    else:
        common += """gyroscope_noise_density: 0.00017
gyroscope_random_walk: 0.000019
accelerometer_noise_density: 0.002
accelerometer_random_walk: 0.003
"""
    path.write_text(common, encoding="utf-8")


def _make_sequence(root: Path) -> None:
    camera = root / "mav0" / "cam0"
    imu = root / "mav0" / "imu0"
    image_directory = camera / "data"
    image_directory.mkdir(parents=True)
    imu.mkdir(parents=True)

    for name in ("100.png", "200.png", "300.png"):
        (image_directory / name).write_bytes(b"fixture")

    (camera / "data.csv").write_text(
        "#timestamp [ns],filename\n100,100.png\n200,200.png\n300,300.png\n",
        encoding="utf-8",
    )
    (imu / "data.csv").write_text(
        "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n"
        "50,0,0,0,0,0,9.81\n"
        "100,0,0,0,0,0,9.81\n"
        "150,0.1,0,0,0,0,9.81\n"
        "200,0.1,0,0,0,0,9.81\n"
        "250,0.2,0,0,0,0,9.81\n"
        "300,0.2,0,0,0,0,9.81\n",
        encoding="utf-8",
    )
    _write_sensor_yaml(camera / "sensor.yaml", camera=True)
    _write_sensor_yaml(imu / "sensor.yaml", camera=False)


def test_reads_real_euroc_layout_and_units(tmp_path: Path) -> None:
    _make_sequence(tmp_path)

    frames = read_camera_frames(tmp_path)
    samples = read_imu_samples(tmp_path)

    assert [frame.timestamp_ns for frame in frames] == [100, 200, 300]
    assert frames[0].timestamp_s == pytest.approx(1e-7)
    assert samples[-1].angular_velocity_rad_s.tolist() == [0.2, 0.0, 0.0]
    assert samples[-1].linear_acceleration_m_s2.tolist() == [0.0, 0.0, 9.81]


def test_synchronization_uses_open_left_closed_right_intervals(tmp_path: Path) -> None:
    _make_sequence(tmp_path)
    synchronized = synchronize_camera_and_imu(
        read_camera_frames(tmp_path), read_imu_samples(tmp_path)
    )

    assert synchronized[0].imu_samples == ()
    assert [sample.timestamp_ns for sample in synchronized[1].imu_samples] == [150, 200]
    assert [sample.timestamp_ns for sample in synchronized[2].imu_samples] == [250, 300]


def test_optional_first_interval_includes_preceding_imu(tmp_path: Path) -> None:
    _make_sequence(tmp_path)
    synchronized = synchronize_camera_and_imu(
        read_camera_frames(tmp_path),
        read_imu_samples(tmp_path),
        include_pre_first_frame=True,
    )
    assert [sample.timestamp_ns for sample in synchronized[0].imu_samples] == [50, 100]


def test_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    _make_sequence(tmp_path)
    camera_csv = tmp_path / "mav0" / "cam0" / "data.csv"
    camera_csv.write_text(
        "#timestamp [ns],filename\n100,100.png\n100,200.png\n300,300.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        read_camera_frames(tmp_path)


def test_parses_camera_and_imu_calibration(tmp_path: Path) -> None:
    _make_sequence(tmp_path)
    camera = load_camera_calibration(tmp_path / "mav0" / "cam0" / "sensor.yaml")
    imu = load_imu_calibration(tmp_path / "mav0" / "imu0" / "sensor.yaml")

    assert camera.resolution == (752, 480)
    assert camera.intrinsics.shape == (4,)
    assert np.allclose(camera.transform_body_sensor, np.eye(4))
    assert imu.rate_hz == pytest.approx(200.0)
    assert imu.accelerometer_noise_density == pytest.approx(0.002)


def test_estimates_rate_from_nanosecond_timestamps() -> None:
    timestamps = [0, 50_000_000, 100_000_000, 150_000_000]
    assert estimate_stream_rate_hz(timestamps) == pytest.approx(20.0)
