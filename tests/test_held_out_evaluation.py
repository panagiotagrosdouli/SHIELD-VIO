from __future__ import annotations

import numpy as np
import pytest

from shield_vio.datasets.splits import load_paper_split
from shield_vio.evaluation.prediction_targets import FailureEventIndex
from shield_vio.experiments.frozen_predictor import FrozenPredictor
from shield_vio.experiments.held_out_evaluation import evaluate_held_out
from test_development_seal import bundle  # noqa: F401, F811


def _evaluate(bundle, sequence="test_seq", split="test", failure=True):  # noqa: F811
    root, config, digest = bundle
    predictor = FrozenPredictor(root, config, digest)
    timestamps = np.arange(8, dtype=np.int64) * 1_000_000_000
    active = np.zeros(8, dtype=bool)
    if failure:
        active[5:7] = True
    events = FailureEventIndex(
        timestamps[5:6] if failure else timestamps[:0],
        timestamps[6:7] if failure else timestamps[:0],
        active,
    )
    return evaluate_held_out(
        predictor,
        load_paper_split(config),
        dataset="euroc",
        sequence=sequence,
        split=split,
        timestamps_ns=timestamps,
        features=np.column_stack((np.arange(8), np.zeros(8))),
        feature_names=predictor.feature_names,
        events=events,
    )


def test_held_out_metrics(bundle):  # noqa: F811
    metrics, predictions = _evaluate(bundle)
    assert metrics["confirmatory"] is False
    assert metrics["eligible_rows"] == 5
    assert metrics["positive_rows"] == 2
    assert metrics["discrimination_defined"]
    assert 0 <= metrics["calibrated"]["auroc"] <= 1
    assert metrics["events"]["event_count"] == 1
    assert len(predictions["timestamp_ns"]) == 8
    assert not predictions["eligible"][-1]


def test_single_class_metrics_are_undefined(bundle):  # noqa: F811
    metrics, _ = _evaluate(bundle, failure=False)
    assert not metrics["discrimination_defined"]
    assert metrics["raw"]["auroc"] is None
    assert metrics["calibrated"]["auprc"] is None
    assert metrics["calibrated"]["calibration"] is not None
    assert metrics["median_lead_time_seconds"] is None


def test_development_partition_rejected(bundle):  # noqa: F811
    with pytest.raises(ValueError, match="requires test"):
        _evaluate(bundle, sequence="train_seq", split="train")


def test_mislabeled_partition_rejected(bundle):  # noqa: F811
    with pytest.raises(ValueError, match="inherits physical-sequence split"):
        _evaluate(bundle, sequence="train_seq")
