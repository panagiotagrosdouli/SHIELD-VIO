from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shield_vio.calibration_metrics.metrics import summarize_calibration
from shield_vio.datasets.adapters import discover_euroc_sequence, discover_generic_sequence
from shield_vio.failure_detection.baselines import LogisticFailureDetector, RuleBasedDetector


def test_perfect_probabilities_have_zero_brier_and_ece() -> None:
    summary = summarize_calibration(np.array([0.0, 1.0, 0.0, 1.0]), np.array([0, 1, 0, 1]))
    assert summary.brier_score == pytest.approx(0.0)
    assert summary.expected_calibration_error == pytest.approx(0.0)
    assert summary.sample_count == 4


def test_logistic_detector_separates_simple_failure_data() -> None:
    features = np.array(
        [[0.0, 0.1], [0.1, 0.0], [0.2, 0.2], [1.0, 1.1], [1.2, 0.9], [1.4, 1.3]]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    detector = LogisticFailureDetector(iterations=1000).fit(features, labels)
    probabilities = detector.predict_proba(features)
    assert np.mean(probabilities[:3]) < 0.5
    assert np.mean(probabilities[3:]) > 0.5


def test_rule_detector_responds_to_combined_degradation() -> None:
    detector = RuleBasedDetector()
    nominal = detector.score(nis=1.0, covariance_trace=0.1, visual_quality=0.9, imu_health=0.9)
    degraded = detector.score(
        nis=20.0, covariance_trace=2.0, visual_quality=0.1, imu_health=0.2
    )
    assert degraded > nominal
    assert detector.predict(
        nis=20.0, covariance_trace=2.0, visual_quality=0.1, imu_health=0.2
    )


def test_generic_dataset_adapter_uses_real_files(tmp_path: Path) -> None:
    (tmp_path / "camera.csv").write_text("timestamp,path\n", encoding="utf-8")
    (tmp_path / "imu.csv").write_text("timestamp,ax,ay,az,gx,gy,gz\n", encoding="utf-8")
    sequence = discover_generic_sequence(tmp_path)
    assert sequence.camera_csv.name == "camera.csv"
    assert sequence.ground_truth_csv is None


def test_euroc_adapter_fixture(tmp_path: Path) -> None:
    cam = tmp_path / "mav0" / "cam0"
    imu = tmp_path / "mav0" / "imu0"
    gt = tmp_path / "mav0" / "state_groundtruth_estimate0"
    cam.mkdir(parents=True)
    imu.mkdir(parents=True)
    gt.mkdir(parents=True)
    (cam / "data.csv").write_text("#timestamp,filename\n", encoding="utf-8")
    (imu / "data.csv").write_text("#timestamp,w_x,w_y,w_z,a_x,a_y,a_z\n", encoding="utf-8")
    (gt / "data.csv").write_text("#timestamp,p_x,p_y,p_z\n", encoding="utf-8")
    sequence = discover_euroc_sequence(tmp_path)
    assert sequence.ground_truth_csv is not None


def test_dataset_adapter_rejects_missing_streams(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_generic_sequence(tmp_path)
