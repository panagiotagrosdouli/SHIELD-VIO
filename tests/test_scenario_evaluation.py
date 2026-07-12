from __future__ import annotations

import json

import numpy as np

from scripts.run_scenario_suite import run_suite
from shield_vio.evaluation.failure_labels import (
    FailureKind,
    FailureObservation,
    label_failure,
)
from shield_vio.evaluation.statistics import binary_metrics, summarize


def test_failure_label_uses_observed_thresholds() -> None:
    label = label_failure(
        FailureObservation(
            timestamp_s=2.0,
            position_error_m=1.2,
            relative_pose_error_m=0.1,
            covariance_trace=0.2,
            nis=15.0,
            feature_count=50,
            tracking_valid=True,
            bias_norm=0.01,
            navigation_clearance_m=1.0,
        )
    )
    assert label.failed
    assert label.kinds == (
        FailureKind.POSITION_ERROR,
        FailureKind.INNOVATION_INCONSISTENCY,
    )


def test_statistics_report_expected_counts() -> None:
    summary = summarize([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["median"] == 2.5
    assert summary["iqr"] == 1.5

    metrics = binary_metrics(
        np.array([0, 0, 1, 1], dtype=bool),
        np.array([0, 1, 1, 0], dtype=bool),
    )
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["f1"] == 0.5


def test_scenario_suite_writes_reproducible_summary(tmp_path) -> None:
    report = run_suite(2, tmp_path)
    assert report["requested_runs"] == 2
    assert report["successful_runs"] == 2
    assert report["failed_runs"] == 0
    assert report["seeds"] == [0, 1]
    saved = json.loads((tmp_path / "suite_summary.json").read_text(encoding="utf-8"))
    assert saved["label"] == "Synthetic Validation"
    assert saved["aggregates"]["ate_rmse_m"]["count"] == 2
