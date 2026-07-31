"""EuRoC MAV dataset parsing and camera/IMU synchronization.

The reader is deliberately independent from a particular estimator. It converts
EuRoC's nanosecond timestamps and CSV layout into typed records that can feed an
ESKF, an external VIO backend, or an offline diagnostics pipeline.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


_NANOSECONDS_PER_SECOND = 1_000_000_000.0


@dataclass(frozen=True)
class CameraFrame:
    timestamp_ns: int
    timestamp_s: float
    image_path: Path


@dataclass(frozen=True)
class ImuSample:
    timestamp_ns: int
    timestamp_s: float
    angular_velocity_rad_s: np.ndarray
    linear_acceleration_m_s2: np.ndarray


@dataclass(frozen=True)
class SynchronizedFrame:
    frame: CameraFrame
    imu_samples: tuple[ImuSample, ...]


def _iter_data_rows(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0].strip().startswith("#"):
                continue
            yield [field.strip() for field in row]


def _strictly_increasing(timestamps: Sequence[int], stream_name: str) -> None:
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{stream_name} timestamps must be strictly increasing")


def read_camera_frames(sequence_root: str | Path, camera: str = "cam0") -> list[CameraFrame]:
    root = Path(sequence_root)
    csv_path = root / "mav0" / camera / "data.csv"
    image_root = root / "mav0" / camera / "data"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    frames: list[CameraFrame] = []
    for line_number, row in enumerate(_iter_data_rows(csv_path), start=1):
        if len(row) < 2:
            raise ValueError(f"Malformed camera row {line_number} in {csv_path}")
        timestamp_ns = int(row[0])
        image_path = image_root / row[1]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        frames.append(
            CameraFrame(
                timestamp_ns=timestamp_ns,
                timestamp_s=timestamp_ns / _NANOSECONDS_PER_SECOND,
                image_path=image_path,
            )
        )

    if not frames:
        raise ValueError(f"No camera frames found in {csv_path}")
    _strictly_increasing([frame.timestamp_ns for frame in frames], camera)
    return frames


def read_imu_samples(sequence_root: str | Path) -> list[ImuSample]:
    root = Path(sequence_root)
    csv_path = root / "mav0" / "imu0" / "data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    samples: list[ImuSample] = []
    for line_number, row in enumerate(_iter_data_rows(csv_path), start=1):
        if len(row) < 7:
            raise ValueError(f"Malformed IMU row {line_number} in {csv_path}")
        timestamp_ns = int(row[0])
        gyro = np.asarray([float(value) for value in row[1:4]], dtype=float)
        acceleration = np.asarray([float(value) for value in row[4:7]], dtype=float)
        if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(acceleration)):
            raise ValueError(f"Non-finite IMU value at row {line_number} in {csv_path}")
        samples.append(
            ImuSample(
                timestamp_ns=timestamp_ns,
                timestamp_s=timestamp_ns / _NANOSECONDS_PER_SECOND,
                angular_velocity_rad_s=gyro,
                linear_acceleration_m_s2=acceleration,
            )
        )

    if not samples:
        raise ValueError(f"No IMU samples found in {csv_path}")
    _strictly_increasing([sample.timestamp_ns for sample in samples], "imu0")
    return samples


def synchronize_camera_and_imu(
    frames: Sequence[CameraFrame],
    imu_samples: Sequence[ImuSample],
    *,
    include_pre_first_frame: bool = False,
) -> list[SynchronizedFrame]:
    """Assign IMU samples to camera intervals.

    For frame ``k``, samples satisfy ``t[k-1] < t_imu <= t[k]``. The first
    frame receives no samples by default because its left boundary is unknown.
    Set ``include_pre_first_frame`` to include all samples up to the first frame.
    Samples after the last frame remain unconsumed by design.
    """
    if not frames:
        raise ValueError("At least one camera frame is required")
    if not imu_samples:
        raise ValueError("At least one IMU sample is required")

    _strictly_increasing([frame.timestamp_ns for frame in frames], "camera")
    _strictly_increasing([sample.timestamp_ns for sample in imu_samples], "imu")

    synchronized: list[SynchronizedFrame] = []
    imu_index = 0
    previous_frame_timestamp: int | None = None

    for frame in frames:
        interval: list[ImuSample] = []
        while imu_index < len(imu_samples) and imu_samples[imu_index].timestamp_ns <= frame.timestamp_ns:
            sample = imu_samples[imu_index]
            if previous_frame_timestamp is not None or include_pre_first_frame:
                if previous_frame_timestamp is None or sample.timestamp_ns > previous_frame_timestamp:
                    interval.append(sample)
            imu_index += 1
        synchronized.append(SynchronizedFrame(frame=frame, imu_samples=tuple(interval)))
        previous_frame_timestamp = frame.timestamp_ns

    return synchronized


def estimate_stream_rate_hz(timestamps_ns: Sequence[int]) -> float:
    if len(timestamps_ns) < 2:
        raise ValueError("At least two timestamps are required")
    _strictly_increasing(timestamps_ns, "stream")
    intervals_s = np.diff(np.asarray(timestamps_ns, dtype=np.float64)) / _NANOSECONDS_PER_SECOND
    return float(1.0 / np.median(intervals_s))
