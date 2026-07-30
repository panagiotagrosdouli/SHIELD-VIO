from pathlib import Path

import cv2
import numpy as np
import pytest

from shield_vio.datasets.euroc_camera import iter_grayscale_frames, load_euroc_camera_frames
from shield_vio.vision.klt_tracker import KLTFeatureTracker


def _write_frame(path: Path, shift_x: int = 0) -> None:
    image = np.zeros((120, 160), dtype=np.uint8)
    for y in range(20, 101, 20):
        for x in range(20, 141, 20):
            cv2.circle(image, (x + shift_x, y), 3, 255, -1)
    assert cv2.imwrite(str(path), image)


def _make_camera_sequence(tmp_path: Path) -> Path:
    sequence = tmp_path / "MH_00_test"
    data = sequence / "mav0" / "cam0" / "data"
    data.mkdir(parents=True)
    _write_frame(data / "1000000000.png", shift_x=0)
    _write_frame(data / "1050000000.png", shift_x=2)
    csv_path = data.parent / "data.csv"
    csv_path.write_text(
        "#timestamp [ns],filename\n"
        "1000000000,1000000000.png\n"
        "1050000000,1050000000.png\n",
        encoding="utf-8",
    )
    return sequence


def test_camera_loader_and_lazy_decode(tmp_path: Path) -> None:
    sequence = _make_camera_sequence(tmp_path)
    frames = load_euroc_camera_frames(sequence)
    assert len(frames) == 2
    assert frames[0].timestamp_s == pytest.approx(1.0)
    decoded = list(iter_grayscale_frames(frames))
    assert decoded[0][1].shape == (120, 160)
    assert decoded[0][1].dtype == np.uint8


def test_camera_loader_rejects_missing_image(tmp_path: Path) -> None:
    sequence = tmp_path / "broken"
    camera = sequence / "mav0" / "cam0"
    (camera / "data").mkdir(parents=True)
    (camera / "data.csv").write_text("1000000000,missing.png\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_euroc_camera_frames(sequence)


def test_klt_tracker_reports_forward_backward_inliers(tmp_path: Path) -> None:
    sequence = _make_camera_sequence(tmp_path)
    frames = load_euroc_camera_frames(sequence)
    tracker = KLTFeatureTracker(max_features=100, min_distance_px=8.0)

    first = tracker.update(frames[0].load_grayscale(), frames[0].timestamp_s)
    second = tracker.update(frames[1].load_grayscale(), frames[1].timestamp_s)

    assert first.detected_features > 0
    assert first.inlier_features == 0
    assert second.detected_features > 0
    assert second.tracked_features > 0
    assert second.inlier_features > 0
    assert 0.0 < second.tracking_ratio <= 1.0
    assert second.median_flow_px == pytest.approx(2.0, abs=0.5)
    assert second.median_forward_backward_error_px <= 1.5


def test_klt_tracker_rejects_non_monotonic_timestamps() -> None:
    tracker = KLTFeatureTracker()
    image = np.zeros((32, 32), dtype=np.uint8)
    tracker.update(image, 1.0)
    with pytest.raises(ValueError):
        tracker.update(image, 1.0)
