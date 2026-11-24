"""Inference-only held-out metrics for a registered sequence.

This evaluator is a development tool, not a confirmatory claim gate.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from shield_vio.calibration_metrics.metrics import summarize_calibration
from shield_vio.datasets.splits import PaperSplit
from shield_vio.evaluation.prediction_metrics import (
    auprc,
    auroc,
    event_detection_summary,
)
from shield_vio.evaluation.prediction_targets import (
    FailureEventIndex,
    future_failure_targets,
)
from shield_vio.experiments.frozen_predictor import FrozenPredictor


def evaluate_held_out(
    predictor: FrozenPredictor,
    registry: PaperSplit,
    *,
    dataset: str,
    sequence: str,
    split: str,
    timestamps_ns: np.ndarray,
    features: np.ndarray,
    feature_names: tuple[str, ...],
    events: FailureEventIndex,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Evaluate one physical sequence without fitting or threshold selection."""
    if split not in ("test", "shifted_test"):
        raise ValueError("held-out evaluation requires test or shifted_test")
    registry.assert_run_membership(dataset, sequence, split)
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    targets = future_failure_targets(
        timestamps, events, horizon_seconds=predictor.horizon_seconds
    )
    raw, calibrated, alerts = predictor.predict(features, feature_names)
    if raw.shape != timestamps.shape:
        raise ValueError("feature rows and timestamps must align")
    mask = targets.eligible_mask
    labels = targets.labels[mask]
    defined = len(np.unique(labels)) == 2

    def probability_metrics(scores: np.ndarray) -> dict:
        eligible_scores = scores[mask]
        return {
            "auroc": auroc(labels, eligible_scores) if defined else None,
            "auprc": auprc(labels, eligible_scores) if defined else None,
            "calibration": (
                asdict(summarize_calibration(eligible_scores, labels))
                if len(labels)
                else None
            ),
        }

    event_summary = event_detection_summary(
        timestamps,
        alerts,
        events,
        warning_horizon_seconds=predictor.horizon_seconds,
    )
    metrics = {
        "dataset": dataset,
        "sequence": sequence,
        "split": split,
        "confirmatory": False,
        "evidence_level": "HELD_OUT_EVALUATION_UNVERIFIED_PROVENANCE",
        "horizon_seconds": predictor.horizon_seconds,
        "threshold": predictor.threshold,
        "eligible_rows": int(mask.sum()),
        "positive_rows": int(labels.sum()),
        "discrimination_defined": defined,
        "raw": probability_metrics(raw),
        "calibrated": probability_metrics(calibrated),
        "events": asdict(event_summary),
        "median_lead_time_seconds": (
            float(np.median(event_summary.lead_times_seconds))
            if event_summary.lead_times_seconds
            else None
        ),
    }
    predictions = {
        "timestamp_ns": timestamps,
        "eligible": mask,
        "label": targets.labels,
        "raw_probability": raw,
        "calibrated_probability": calibrated,
        "alert": alerts,
    }
    return metrics, predictions
