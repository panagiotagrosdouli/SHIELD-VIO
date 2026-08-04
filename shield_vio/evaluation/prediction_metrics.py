"""Discrimination, event-warning, and operating-point metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shield_vio.evaluation.prediction_targets import FailureEventIndex


@dataclass(frozen=True)
class EventDetectionSummary:
    event_count: int
    detected_events: int
    missed_events: int
    false_alerts: int
    event_precision: float
    event_recall: float
    event_f1: float
    false_alarms_per_minute: float
    lead_times_seconds: tuple[float, ...]


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve using average ranks for tied scores."""

    y, s = _binary_scores(labels, scores)
    positives = int(np.sum(y))
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and s[order[end]] == s[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(np.sum(ranks[y]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-integrated precision-recall area (average precision)."""

    y, s = _binary_scores(labels, scores)
    positives = int(np.sum(y))
    if positives == 0 or positives == len(y):
        raise ValueError("AUPRC requires both positive and negative labels")
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    true_positives = np.cumsum(sorted_y)
    false_positives = np.cumsum(~sorted_y)
    distinct = np.r_[s[order][1:] != s[order][:-1], True]
    precision = true_positives[distinct] / (true_positives[distinct] + false_positives[distinct])
    recall = true_positives[distinct] / positives
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def event_detection_summary(
    timestamps_ns: np.ndarray,
    predictions: np.ndarray,
    events: FailureEventIndex,
    *,
    warning_horizon_seconds: float,
) -> EventDetectionSummary:
    """Score alert onsets once per failure event and normalize false alerts by time."""

    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    alerts = np.asarray(predictions, dtype=bool)
    if timestamps.ndim != 1 or alerts.shape != timestamps.shape or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps and predictions must be aligned and ordered")
    if len(events.active_mask) != len(timestamps) or warning_horizon_seconds <= 0:
        raise ValueError("events and warning horizon must be valid")

    rising = alerts & ~np.r_[False, alerts[:-1]]
    alert_indices = np.flatnonzero(rising)
    assigned: set[int] = set()
    lead_times: list[float] = []
    warning_ns = int(round(warning_horizon_seconds * 1e9))
    for onset in events.onsets_ns:
        candidates = [
            int(index)
            for index in alert_indices
            if index not in assigned and onset - warning_ns <= timestamps[index] < onset
        ]
        if candidates:
            selected = candidates[0]
            assigned.add(selected)
            lead_times.append(float(onset - timestamps[selected]) * 1e-9)

    false_alerts = int(
        sum(index not in assigned and not events.active_mask[index] for index in alert_indices)
    )
    detected = len(lead_times)
    event_count = len(events.onsets_ns)
    missed = event_count - detected
    precision = detected / (detected + false_alerts) if detected + false_alerts else 0.0
    recall = detected / event_count if event_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    intervals_s = np.diff(timestamps).astype(float) * 1e-9
    eligible_seconds = float(np.sum(intervals_s[~events.active_mask[:-1]]))
    false_per_minute = false_alerts / (eligible_seconds / 60.0) if eligible_seconds > 0 else 0.0
    return EventDetectionSummary(
        event_count=event_count,
        detected_events=detected,
        missed_events=missed,
        false_alerts=false_alerts,
        event_precision=float(precision),
        event_recall=float(recall),
        event_f1=float(f1),
        false_alarms_per_minute=float(false_per_minute),
        lead_times_seconds=tuple(lead_times),
    )


def select_event_threshold(
    timestamps_ns: np.ndarray,
    scores: np.ndarray,
    events: FailureEventIndex,
    *,
    warning_horizon_seconds: float,
    max_false_alarms_per_minute: float,
) -> float:
    """Select the highest-recall validation threshold under a false-alarm constraint."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite vector")
    if max_false_alarms_per_minute < 0:
        raise ValueError("false-alarm constraint must be non-negative")
    candidates = np.r_[np.nextafter(np.max(values), np.inf), np.unique(values)[::-1]]
    feasible: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        summary = event_detection_summary(
            timestamps_ns,
            values >= threshold,
            events,
            warning_horizon_seconds=warning_horizon_seconds,
        )
        if summary.false_alarms_per_minute <= max_false_alarms_per_minute:
            median_lead = (
                float(np.median(summary.lead_times_seconds)) if summary.lead_times_seconds else 0.0
            )
            feasible.append((summary.event_recall, median_lead, summary.event_precision, threshold))
    if not feasible:
        raise RuntimeError("no threshold satisfies the false-alarm constraint")
    return float(max(feasible)[-1])


def _binary_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=bool).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) == 0 or y.shape != s.shape or not np.all(np.isfinite(s)):
        raise ValueError("labels and finite scores must be non-empty and equally sized")
    return y, s
