from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.convert_euroc_rosbag import (
    _image_array,
    _stamp_ns,
    _verify_sha256,
    build_asl_calibrations,
    convert_openvins_ground_truth,
)


def test_stamp_ns_accepts_rosbags_style_timestamp() -> None:
    header = SimpleNamespace(stamp=SimpleNamespace(sec=12, nanosec=345))
    assert _stamp_ns(header) == 12_000_000_345


def test_image_array_decodes_padded_mono8() -> None:
    message = SimpleNamespace(
        height=2,
        width=3,
        step=4,
        encoding="mono8",
        data=bytes([1, 2, 3, 99, 4, 5, 6, 99]),
    )
    image = _image_array(message)
    assert image.shape == (2, 3)
    assert image.dtype == np.uint8
    assert image.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_image_array_rejects_unknown_encoding() -> None:
    message = SimpleNamespace(height=1, width=1, step=2, encoding="mono16", data=b"\0\0")
    with pytest.raises(ValueError, match="unsupported EuRoC camera encoding"):
        _image_array(message)


def test_build_asl_calibrations_from_openvins_yaml(tmp_path: Path) -> None:
    imucam = tmp_path / "imucam.yaml"
    imucam.write_text(
        """%YAML:1.0
cam0:
  T_imu_cam:
    - [1.0, 0.0, 0.0, 0.1]
    - [0.0, 1.0, 0.0, 0.2]
    - [0.0, 0.0, 1.0, 0.3]
    - [0.0, 0.0, 0.0, 1.0]
  camera_model: pinhole
  distortion_coeffs: [-0.1, 0.01, 0.0, 0.0]
  distortion_model: radtan
  intrinsics: [458.654, 457.296, 367.215, 248.375]
  resolution: [752, 480]
""",
        encoding="utf-8",
    )
    imu = tmp_path / "imu.yaml"
    imu.write_text(
        """%YAML:1.0
imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: 0.002
  accelerometer_random_walk: 0.003
  gyroscope_noise_density: 0.00016968
  gyroscope_random_walk: 0.000019393
  update_rate: 200.0
""",
        encoding="utf-8",
    )

    camera, inertial = build_asl_calibrations(imucam, imu)
    assert camera["sensor_type"] == "camera"
    assert camera["resolution"] == [752, 480]
    assert camera["rate_hz"] == 20.0
    assert camera["T_BS"]["rows"] == 4
    assert inertial["sensor_type"] == "imu"
    assert inertial["rate_hz"] == 200.0


def test_convert_openvins_ground_truth_to_asl_csv(tmp_path: Path) -> None:
    source = tmp_path / "gt.txt"
    source.write_text(
        "# timestamp(s) tx ty tz qx qy qz qw\n"
        "1.000000001 1 2 3 0.1 0.2 0.3 0.9\n"
        "1.005000001 4 5 6 0.2 0.3 0.4 0.8\n"
        "1.010000001 7 8 9 0.3 0.4 0.5 0.7\n",
        encoding="utf-8",
    )
    destination = tmp_path / "data.csv"
    count = convert_openvins_ground_truth(source, destination)
    rows = destination.read_text(encoding="utf-8").splitlines()
    assert count == 3
    assert rows[0].startswith("#timestamp")
    assert rows[1].startswith("1000000001,1.0,2.0,3.0,0.9,0.1,0.2,0.3")


def test_verify_sha256_rejects_wrong_source(tmp_path: Path) -> None:
    path = tmp_path / "bag.bag"
    path.write_bytes(b"bag bytes")
    correct = hashlib.sha256(b"bag bytes").hexdigest()
    assert _verify_sha256(path, correct) == correct
    with pytest.raises(ValueError, match="source bag SHA-256 mismatch"):
        _verify_sha256(path, "0" * 64)
