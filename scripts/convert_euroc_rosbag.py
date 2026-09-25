#!/usr/bin/env python3
"""Convert a checksum-verified EuRoC ROS1 bag into the minimal ASL tree used by SHIELD-VIO.

This utility is intended for PUBLIC_DATASET_SMOKE portability when the legacy ETH
per-sequence host is unreachable from CI. It does not establish official archive
identity for confirmatory experiments. The source bag, OpenVINS calibration files,
and OpenVINS ground-truth export are all hashed into the conversion manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


CAM0_TOPIC = "/cam0/image_raw"
IMU0_TOPIC = "/imu0"
SUPPORTED_MONO_ENCODINGS = {"mono8", "8uc1", "8UC1"}
SUPPORTED_RGB_ENCODINGS = {"rgb8", "bgr8"}


@dataclass(frozen=True)
class ConversionSummary:
    sequence: str
    camera_frames: int
    imu_samples: int
    ground_truth_samples: int
    first_camera_timestamp_ns: int
    last_camera_timestamp_ns: int
    first_imu_timestamp_ns: int
    last_imu_timestamp_ns: int
    image_pixel_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str) -> str:
    normalized = expected.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError("expected SHA-256 must be a 64-character hexadecimal digest")
    actual = _sha256(path)
    if actual != normalized:
        raise ValueError(f"source bag SHA-256 mismatch: expected {normalized}, got {actual}")
    return actual


def _yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("%YAML:1.0"):
        text = text.split("\n", 1)[1]
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _matrix_payload(matrix: Any) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError("calibration transform must be a finite 4x4 matrix")
    return {"rows": 4, "cols": 4, "data": values.reshape(-1).tolist()}


def build_asl_calibrations(imucam_path: Path, imu_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate pinned OpenVINS EuRoC calibration YAML into SHIELD-VIO's ASL parser schema."""

    imucam = _yaml_mapping(imucam_path)
    imu = _yaml_mapping(imu_path)
    cam0 = imucam.get("cam0")
    imu0 = imu.get("imu0")
    if not isinstance(cam0, dict) or not isinstance(imu0, dict):
        raise ValueError("OpenVINS calibration must contain cam0 and imu0 mappings")

    camera = {
        "sensor_type": "camera",
        "comment": "EuRoC cam0 calibration translated from pinned OpenVINS config",
        "T_BS": _matrix_payload(cam0["T_imu_cam"]),
        "rate_hz": 20.0,
        "resolution": [int(value) for value in cam0["resolution"]],
        "camera_model": str(cam0["camera_model"]),
        "intrinsics": [float(value) for value in cam0["intrinsics"]],
        "distortion_model": str(cam0["distortion_model"]),
        "distortion_coefficients": [float(value) for value in cam0["distortion_coeffs"]],
    }
    inertial = {
        "sensor_type": "imu",
        "comment": "EuRoC imu0 calibration translated from pinned OpenVINS config",
        "T_BS": _matrix_payload(imu0["T_i_b"]),
        "rate_hz": float(imu0["update_rate"]),
        "gyroscope_noise_density": float(imu0["gyroscope_noise_density"]),
        "gyroscope_random_walk": float(imu0["gyroscope_random_walk"]),
        "accelerometer_noise_density": float(imu0["accelerometer_noise_density"]),
        "accelerometer_random_walk": float(imu0["accelerometer_random_walk"]),
    }
    if camera["resolution"] != [752, 480]:
        raise ValueError("unexpected EuRoC cam0 resolution")
    return camera, inertial


def _stamp_ns(header: Any) -> int:
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        raise ValueError("ROS message is missing header.stamp")
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        seconds = int(stamp.sec)
        nanoseconds = int(stamp.nanosec)
    elif hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        seconds = int(stamp.secs)
        nanoseconds = int(stamp.nsecs)
    else:
        raise ValueError("unsupported ROS timestamp representation")
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("invalid ROS header timestamp")
    return seconds * 1_000_000_000 + nanoseconds


def _image_array(message: Any) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding)
    if min(height, width, step) <= 0:
        raise ValueError("image dimensions and step must be positive")
    raw = bytes(message.data)

    if encoding in SUPPORTED_MONO_ENCODINGS:
        if step < width or len(raw) != height * step:
            raise ValueError("malformed mono8 image payload")
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
        return np.ascontiguousarray(rows[:, :width])

    if encoding in SUPPORTED_RGB_ENCODINGS:
        required = width * 3
        if step < required or len(raw) != height * step:
            raise ValueError("malformed rgb8/bgr8 image payload")
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
        image = np.ascontiguousarray(rows[:, :required].reshape(height, width, 3))
        if encoding == "bgr8":
            image = np.ascontiguousarray(image[:, :, ::-1])
        return image

    raise ValueError(f"unsupported EuRoC camera encoding: {encoding!r}")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def convert_openvins_ground_truth(source: Path, destination: Path) -> int:
    """Convert OpenVINS seconds/TUM-like EuRoC trajectory to an ASL-compatible CSV."""

    rows: list[list[object]] = []
    previous: int | None = None
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 8:
                raise ValueError(f"malformed OpenVINS ground-truth row {line_number}")
            timestamp_ns = int(round(float(fields[0]) * 1_000_000_000))
            if previous is not None and timestamp_ns <= previous:
                raise ValueError("ground-truth timestamps must be strictly increasing")
            previous = timestamp_ns
            tx, ty, tz = (float(fields[index]) for index in range(1, 4))
            qx, qy, qz, qw = (float(fields[index]) for index in range(4, 8))
            values = (tx, ty, tz, qw, qx, qy, qz)
            if not all(np.isfinite(value) for value in values):
                raise ValueError("ground-truth trajectory contains non-finite values")
            rows.append([timestamp_ns, *values])
    if len(rows) < 3:
        raise ValueError("ground-truth trajectory contains too few samples")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "#timestamp",
                "p_RS_R_x [m]",
                "p_RS_R_y [m]",
                "p_RS_R_z [m]",
                "q_RS_w []",
                "q_RS_x []",
                "q_RS_y []",
                "q_RS_z []",
            ]
        )
        writer.writerows(rows)
    return len(rows)


def _strictly_increasing(values: Iterable[int], label: str) -> None:
    sequence = list(values)
    if len(sequence) < 2 or any(right <= left for left, right in zip(sequence, sequence[1:])):
        raise ValueError(f"{label} timestamps must be strictly increasing")


def convert_rosbag(
    bag_path: Path,
    output_root: Path,
    *,
    sequence: str,
    expected_bag_sha256: str,
    source_url: str,
    openvins_ground_truth: Path,
    openvins_imucam_calibration: Path,
    openvins_imu_calibration: Path,
    openvins_revision: str,
) -> dict[str, Any]:
    """Convert cam0 and imu0 from one EuRoC ROS bag into a minimal ASL directory."""

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    bag_sha256 = _verify_sha256(bag_path, expected_bag_sha256)
    camera_calibration, imu_calibration = build_asl_calibrations(
        openvins_imucam_calibration, openvins_imu_calibration
    )
    _write_yaml(output_root / "mav0/cam0/sensor.yaml", camera_calibration)
    _write_yaml(output_root / "mav0/imu0/sensor.yaml", imu_calibration)
    gt_count = convert_openvins_ground_truth(
        openvins_ground_truth,
        output_root / "mav0/state_groundtruth_estimate0/data.csv",
    )

    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError(
            "rosbags is required for EuRoC bag conversion; install shield-vio[dataset]"
        ) from exc

    camera_dir = output_root / "mav0/cam0/data"
    camera_dir.mkdir(parents=True, exist_ok=True)
    camera_rows: list[list[object]] = []
    imu_rows: list[list[object]] = []
    camera_timestamps: list[int] = []
    imu_timestamps: list[int] = []
    pixel_digest = hashlib.sha256()

    with AnyReader([bag_path]) as reader:
        by_topic = {connection.topic: connection for connection in reader.connections}
        missing_topics = sorted({CAM0_TOPIC, IMU0_TOPIC} - set(by_topic))
        if missing_topics:
            raise ValueError(f"source bag is missing required topics: {', '.join(missing_topics)}")
        connections = [by_topic[CAM0_TOPIC], by_topic[IMU0_TOPIC]]
        for connection, _bag_timestamp, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            timestamp_ns = _stamp_ns(message.header)

            if connection.topic == CAM0_TOPIC:
                if camera_timestamps and timestamp_ns <= camera_timestamps[-1]:
                    raise ValueError("cam0 timestamps are not strictly increasing")
                image = _image_array(message)
                filename = f"{timestamp_ns}.png"
                Image.fromarray(image).save(camera_dir / filename, format="PNG", compress_level=3)
                camera_timestamps.append(timestamp_ns)
                camera_rows.append([timestamp_ns, filename])
                pixel_digest.update(timestamp_ns.to_bytes(8, "little", signed=False))
                pixel_digest.update(image.tobytes(order="C"))
                continue

            if connection.topic == IMU0_TOPIC:
                if imu_timestamps and timestamp_ns <= imu_timestamps[-1]:
                    raise ValueError("imu0 timestamps are not strictly increasing")
                angular = message.angular_velocity
                linear = message.linear_acceleration
                values = [
                    float(angular.x),
                    float(angular.y),
                    float(angular.z),
                    float(linear.x),
                    float(linear.y),
                    float(linear.z),
                ]
                if not all(np.isfinite(value) for value in values):
                    raise ValueError("IMU message contains non-finite measurements")
                imu_timestamps.append(timestamp_ns)
                imu_rows.append([timestamp_ns, *values])

    _strictly_increasing(camera_timestamps, "cam0")
    _strictly_increasing(imu_timestamps, "imu0")
    if len(camera_timestamps) < 100:
        raise ValueError("unexpectedly few camera frames in EuRoC bag")
    if len(imu_timestamps) < 1000:
        raise ValueError("unexpectedly few IMU samples in EuRoC bag")

    with (output_root / "mav0/cam0/data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["#timestamp [ns]", "filename"])
        writer.writerows(camera_rows)

    with (output_root / "mav0/imu0/data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "#timestamp [ns]",
                "w_RS_S_x [rad s^-1]",
                "w_RS_S_y [rad s^-1]",
                "w_RS_S_z [rad s^-1]",
                "a_RS_S_x [m s^-2]",
                "a_RS_S_y [m s^-2]",
                "a_RS_S_z [m s^-2]",
            ]
        )
        writer.writerows(imu_rows)

    summary = ConversionSummary(
        sequence=sequence,
        camera_frames=len(camera_timestamps),
        imu_samples=len(imu_timestamps),
        ground_truth_samples=gt_count,
        first_camera_timestamp_ns=camera_timestamps[0],
        last_camera_timestamp_ns=camera_timestamps[-1],
        first_imu_timestamp_ns=imu_timestamps[0],
        last_imu_timestamp_ns=imu_timestamps[-1],
        image_pixel_sha256=pixel_digest.hexdigest(),
    )
    tracked_outputs = [
        output_root / "mav0/cam0/data.csv",
        output_root / "mav0/cam0/sensor.yaml",
        output_root / "mav0/imu0/data.csv",
        output_root / "mav0/imu0/sensor.yaml",
        output_root / "mav0/state_groundtruth_estimate0/data.csv",
    ]
    manifest = {
        "schema_version": "SHIELD_VIO_EUROC_PORTABLE_ACQUISITION_V1",
        "dataset": "EuRoC_MAV",
        "sequence": sequence,
        "evidence_scope": "PUBLIC_DATASET_SMOKE_ACQUISITION",
        "confirmatory_identity": False,
        "claim_boundary": (
            "Portable CI acquisition from a checksum-verified ROS bag mirror plus pinned OpenVINS "
            "calibration/ground-truth exports. This does not establish identity with the current "
            "ETH Research Collection archive for confirmatory evaluation."
        ),
        "source": {
            "bag_url": source_url,
            "bag_sha256": bag_sha256,
            "bag_bytes": bag_path.stat().st_size,
            "openvins_revision": openvins_revision,
            "openvins_ground_truth_sha256": _sha256(openvins_ground_truth),
            "openvins_imucam_calibration_sha256": _sha256(openvins_imucam_calibration),
            "openvins_imu_calibration_sha256": _sha256(openvins_imu_calibration),
            "topics": {"camera": CAM0_TOPIC, "imu": IMU0_TOPIC},
        },
        "conversion": {
            **summary.__dict__,
            "generated_file_sha256": {
                str(path.relative_to(output_root)): _sha256(path) for path in tracked_outputs
            },
        },
    }
    (output_root / "portable_acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", default="MH_01_easy")
    parser.add_argument("--expected-bag-sha256", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--openvins-ground-truth", type=Path, required=True)
    parser.add_argument("--openvins-imucam-calibration", type=Path, required=True)
    parser.add_argument("--openvins-imu-calibration", type=Path, required=True)
    parser.add_argument("--openvins-revision", required=True)
    args = parser.parse_args()

    manifest = convert_rosbag(
        args.bag,
        args.output,
        sequence=args.sequence,
        expected_bag_sha256=args.expected_bag_sha256,
        source_url=args.source_url,
        openvins_ground_truth=args.openvins_ground_truth,
        openvins_imucam_calibration=args.openvins_imucam_calibration,
        openvins_imu_calibration=args.openvins_imu_calibration,
        openvins_revision=args.openvins_revision,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
