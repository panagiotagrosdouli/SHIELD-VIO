"""Regression tests for trajectory and failure-detection metrics."""
from __future__ import annotations

import numpy as np
import pytest

from shield_vio.evaluation.metrics import (
    align_positions,
    ate,
    failure_detection_metrics,
    rpe,
)


def test_ate_removes_rigid_frame_transform() -> None:
    ground_truth = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 3.0, 1.0]]
    )
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    estimated = (rotation @ ground_truth.T).T + np.asarray([4.0, -3.0, 2.0])

    assert ate(estimated, ground_truth, align=False) > 1.0
    assert ate(estimated, ground_truth) == pytest.approx(0.0, abs=1e-12)


def test_umeyama_alignment_can_recover_scale_when_requested() -> None:
    ground_truth = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 1.0], [3.0, 1.0, 2.0]]
    )
    estimated = 2.5 * ground_truth + np.asarray([8.0, -2.0, 4.0])

    aligned = align_positions(estimated, ground_truth, with_scale=True)
    np.testing.assert_allclose(aligned, ground_truth, atol=1e-12)
    assert ate(estimated, ground_truth, with_scale=True) == pytest.approx(0.0, abs=1e-12)


def test_rpe_reports_relative_translation_drift() -> None:
    ground_truth = np.column_stack((np.arange(5, dtype=float), np.zeros((5, 2))))
    estimated = ground_truth.copy()
    estimated[:, 0] *= 1.1

    assert rpe(estimated, ground_truth) == pytest.approx(0.1)


def test_failure_detection_reports_confusion_and_warning_time() -> None:
    labels = np.asarray([False, False, False, True, True, True])
    predictions = np.asarray([False, False, True, False, True, True])
    timestamps = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    metrics = failure_detection_metrics(
        predictions,
        labels,
        timestamps,
        warning_window=1.0,
    )

    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 2
    assert metrics.false_negatives == 1
    assert metrics.precision == pytest.approx(2.0 / 3.0)
    assert metrics.recall == pytest.approx(2.0 / 3.0)
    assert metrics.f1 == pytest.approx(2.0 / 3.0)
    assert metrics.false_alarm_rate == pytest.approx(1.0 / 3.0)
    assert metrics.specificity == pytest.approx(2.0 / 3.0)
    assert metrics.accuracy == pytest.approx(4.0 / 6.0)
    assert metrics.time_to_detection == pytest.approx(0.5)
    assert metrics.warning_lead_time == pytest.approx(0.5)


def test_metrics_reject_non_monotonic_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        failure_detection_metrics(
            np.asarray([False, True]),
            np.asarray([False, True]),
            np.asarray([1.0, 1.0]),
        )
