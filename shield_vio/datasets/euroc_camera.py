"""Validated EuRoC monocular camera stream loading.

This module provides a dependency-light index over ``mav0/cam0/data.csv`` and
resolves image paths without eagerly decoding the full sequence.  Image decoding
is intentionally delegated to OpenCV at iteration time so callers can stream
large datasets with bounded memory use.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    timestamp_s: float
    image_path: Path

    def load_grayscale(self) -> np.ndarray:
        image = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"failed to decode camera image: {self.image_path}")
        if image.ndim != 2 or image.size == 0:
            raise ValueError(f"camera image must be non-empty grayscale: {self.image_path}")
        return image


def load_euroc_camera_frames(sequence: str | Path, camera: str = "cam0") -> tuple[CameraFrame, ...]:
    """Load and validate a EuRoC camera index.

    Args:
        sequence: EuRoC sequence root containing ``mav0``.
        camera: Camera directory name, normally ``cam0`` or ``cam1``.
    """

    root = Path(sequence)
    camera_root = root / "mav0" / camera
    csv_path = camera_root / "data.csv"
    image_root = camera_root / "data"
    if not csv_path.is_file():
        raise FileNotFoundError(f"EuRoC camera CSV not found: {csv_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"EuRoC camera image directory not found: {image_root}")

    frames: list[CameraFrame] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(line for line in handle if not line.lstrip().startswith("#"))
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 2:
                raise ValueError(f"invalid camera row {row_number}: expected timestamp and filename")
            try:
                timestamp_ns = int(row[0].strip())
            except ValueError as exc:
                raise ValueError(f"invalid camera timestamp on row {row_number}") from exc
            if timestamp_ns < 0:
                raise ValueError(f"negative camera timestamp on row {row_number}")
            filename = row[1].strip()
            if not filename:
                raise ValueError(f"empty camera filename on row {row_number}")
            image_path = image_root / filename
            if not image_path.is_file():
                raise FileNotFoundError(f"camera image listed in CSV is missing: {image_path}")
            timestamp_s = timestamp_ns * 1e-9
            if frames and timestamp_s <= frames[-1].timestamp_s:
                raise ValueError("camera timestamps must be strictly increasing")
            frames.append(CameraFrame(timestamp_s=timestamp_s, image_path=image_path))

    if not frames:
        raise ValueError(f"camera stream is empty: {csv_path}")
    return tuple(frames)


def iter_grayscale_frames(frames: tuple[CameraFrame, ...]) -> Iterator[tuple[float, np.ndarray]]:
    """Decode a validated camera index lazily."""

    for frame in frames:
        yield frame.timestamp_s, frame.load_grayscale()
