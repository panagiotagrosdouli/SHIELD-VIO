from __future__ import annotations

import numpy as np
import pytest

from shield_vio.evaluation.prediction_metrics import (
    auprc,
    auroc,
    event_detection_summary,
    select_event_threshold,
)
from shield_vio.evaluation.prediction_targets import (
    build_persistent_failure_events,
    future_failure_targets,
)
from shield_vio.failure_detection.calibration import PlattCalibrator
from shield_vio.features.health_vector import (
    HealthFeatureMatrix,
    assert_no_privileged_features,
    build_causal_health_features,
)


def _health(timestamp: int, covariance: float, nis: str = "") -> dict[str, str]:
    return {
        "frame_timestamp_ns": str(timestamp),
        "state_timestamp_ns": str(timestamp),
        "initialized": "True",
        "tracking_status": "tracking",
        "propagated_imu_samples": "1",
        "covariance_trace": str(covariance),
        "covariance_condition_number": "10.0",
        "innovation_nis": nis,
    }


def _visual(timestamp: int, tracked: int) -> dict[str, str]:
    return {
        "frame_timestamp_ns": str(timestamp),
        "status": "accepted",
        "detected_features": "80",
        "tracked_features": str(tracked),
        "correspondence_count": "60",
        "inlier_count": "50",
        "inlier_ratio": "0.833333",
        "innovation_nis": "1.0",
    }


def test_health_features_are_causal_and_ignore_future_visual_updates() -> None:
    health = [_health(1_000_000_000, 0.1), _health(2_000_000_000, 0.2, "2.0")]
    matrix = build_causal_health_features(health, [_visual(3_000_000_000, 20)])
    tracked_index = matrix.feature_names.index("tracked_features")
    missing_index = matrix.feature_names.index("visual_update_missing")
    assert np.all(matrix.values[:, tracked_index] == 0.0)
    assert np.all(matrix.values[:, missing_index] == 1.0)
    assert np.all(matrix.max_source_timestamps_ns <= matrix.timestamps_ns)


def test_feature_matrix_rejects_future_sources_and_privileged_names() -> None:
    with pytest.raises(ValueError, match="future source"):
        HealthFeatureMatrix(
            np.array([1, 2]),
            np.zeros((2, 1)),
            ("covariance_trace",),
            np.array([1, 3]),
        )
    with pytest.raises(ValueError, match="privileged"):
        assert_no_privileged_features(("ground_truth_position_error",))


def test_persistent_events_and_future_targets_exclude_current_failure() -> None:
    timestamps = np.arange(8, dtype=np.int64) * 1_000_000_000
    exceeded = np.array([False, False, True, True, True, False, False, False])
    events = build_persistent_failure_events(timestamps, exceeded, persistence_seconds=1.0)
    assert events.onsets_ns.tolist() == [3_000_000_000]
    assert events.active_mask.tolist() == [False, False, False, True, True, False, False, False]

    targets = future_failure_targets(timestamps, events, horizon_seconds=2.0)
    assert targets.labels.tolist() == [False, True, True, False, False, False, False, False]
    assert not targets.eligible_mask[3]
    assert not targets.eligible_mask[-1]


def test_rank_metrics_handle_ties_and_perfect_separation() -> None:
    labels = np.array([0, 0, 1, 1])
    assert auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auprc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auroc(labels, np.ones(4)) == pytest.approx(0.5)


def test_event_summary_counts_one_alert_per_event_and_false_alerts() -> None:
    timestamps = np.arange(10, dtype=np.int64) * 1_000_000_000
    exceeded = np.array([False, False, False, False, True, True, False, False, False, False])
    events = build_persistent_failure_events(timestamps, exceeded, persistence_seconds=0.0)
    predictions = np.array([False, True, False, True, True, False, False, False, False, False])
    summary = event_detection_summary(timestamps, predictions, events, warning_horizon_seconds=2.0)
    assert summary.detected_events == 1
    assert summary.false_alerts == 1
    assert summary.lead_times_seconds == (1.0,)

    scores = predictions.astype(float)
    threshold = select_event_threshold(
        timestamps,
        scores,
        events,
        warning_horizon_seconds=2.0,
        max_false_alarms_per_minute=20.0,
    )
    assert threshold in {0.0, 1.0, np.nextafter(1.0, np.inf)}


def test_platt_calibration_fits_held_out_probabilities() -> None:
    raw = np.array([0.05, 0.15, 0.35, 0.65, 0.85, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    calibrator = PlattCalibrator().fit(raw, labels)
    calibrated = calibrator.predict_proba(raw)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert np.all(np.diff(calibrated) > 0.0)
    assert calibrator.to_dict()["method"] == "platt_logit"
