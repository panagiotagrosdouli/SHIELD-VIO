from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from scripts.run_euroc_benchmark import _camera_metrics, _imu_metrics


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def test_runner_reads_standard_euroc_layout(tmp_path: Path) -> None:
    sequence = tmp_path / "MH_01_easy"
    image_dir = sequence / "mav0/cam0/data"
    image_dir.mkdir(parents=True)

    image = np.zeros((80, 120), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 60), 220, -1)
    cv2.imwrite(str(image_dir / "100.png"), image)
    cv2.imwrite(str(image_dir / "200.png"), image)
    _write_rows(sequence / "mav0/cam0/data.csv", [[100, "100.png"], [200, "200.png"]])

    imu_rows = []
    for index in range(20):
        imu_rows.append([1_000_000_000 + index * 5_000_000, 0.01, 0.02, 0.03, 0.0, 0.0, 9.81])
    _write_rows(sequence / "mav0/imu0/data.csv", imu_rows)

    camera = _camera_metrics(sequence, "darkness", {"gain": 0.5}, seed=0, max_frames=0)
    imu = _imu_metrics(sequence, "packet_loss", {"probability": 0.2}, seed=0)

    assert camera.frames == 2
    assert 0.0 < camera.brightness_mean < 220.0
    assert camera.feature_count_mean > 0.0
    assert 0 < imu.samples <= 20
    assert imu.accel_norm_mean > 9.0
    assert 0.0 <= imu.dropout_fraction <= 1.0
