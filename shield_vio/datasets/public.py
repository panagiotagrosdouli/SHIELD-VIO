"""Common, strict validation for EuRoC and TUM-VI exported sequences."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from shield_vio.datasets.adapters import DatasetSequence, discover_euroc_sequence, discover_tumvi_sequence
from shield_vio.datasets.provenance import dataset_fingerprint


@dataclass(frozen=True)
class ValidatedPublicSequence:
    dataset_name: str
    sequence_name: str
    sequence: DatasetSequence
    camera_rows: int
    imu_rows: int
    ground_truth_rows: int | None
    fingerprint: dict[str, object]


def discover_public_sequence(dataset: str, root: str | Path) -> DatasetSequence:
    key = dataset.strip().lower()
    if key == "euroc":
        return discover_euroc_sequence(root)
    if key == "tumvi":
        return discover_tumvi_sequence(root)
    raise ValueError(f"unsupported public dataset: {dataset}")


def _rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [row for row in csv.reader(line for line in stream if not line.lstrip().startswith("#")) if row]


def _validate_timestamp_csv(path: Path, *, minimum_columns: int) -> int:
    rows = _rows(path)
    timestamps: list[int] = []
    for row in rows:
        if len(row) < minimum_columns:
            raise ValueError(f"too few columns in {path}: {row}")
        try:
            timestamps.append(int(row[0].strip()))
        except ValueError as exc:
            raise ValueError(f"invalid nanosecond timestamp in {path}: {row[0]!r}") from exc
    if not timestamps:
        raise ValueError(f"no data rows in {path}")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"timestamps are not strictly increasing: {path}")
    return len(timestamps)


def _validate_camera_images(camera_csv: Path) -> None:
    data_dir = camera_csv.parent / "data"
    for row in _rows(camera_csv):
        if len(row) < 2:
            raise ValueError(f"camera row missing image reference: {row}")
        image = data_dir / row[1].strip()
        if not image.is_file():
            raise FileNotFoundError(image)


def validate_public_sequence(dataset: str, root: str | Path) -> ValidatedPublicSequence:
    sequence = discover_public_sequence(dataset, root)
    if len(sequence.calibration_files) < 2:
        raise FileNotFoundError("camera and IMU calibration sensor.yaml files are required")
    camera_rows = _validate_timestamp_csv(sequence.camera_csv, minimum_columns=2)
    _validate_camera_images(sequence.camera_csv)
    imu_rows = _validate_timestamp_csv(sequence.imu_csv, minimum_columns=7)
    ground_truth_rows = None
    if sequence.ground_truth_csv is not None:
        ground_truth_rows = _validate_timestamp_csv(sequence.ground_truth_csv, minimum_columns=4)

    metadata_paths = [sequence.camera_csv, sequence.imu_csv, *sequence.calibration_files]
    if sequence.ground_truth_csv is not None:
        metadata_paths.append(sequence.ground_truth_csv)
    fingerprint = dataset_fingerprint(metadata_paths)
    return ValidatedPublicSequence(
        dataset_name=dataset.strip().lower(), sequence_name=sequence.name, sequence=sequence,
        camera_rows=camera_rows, imu_rows=imu_rows, ground_truth_rows=ground_truth_rows,
        fingerprint=fingerprint,
    )
