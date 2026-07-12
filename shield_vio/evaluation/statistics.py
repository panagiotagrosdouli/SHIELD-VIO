"""Aggregate statistics for repeated SHIELD-VIO experiments."""
from __future__ import annotations

import math

import numpy as np


def summarize(values: list[float] | np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("at least one finite value is required")
    q1, median, q3 = np.percentile(data, [25.0, 50.0, 75.0])
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(data.size) if data.size > 1 else 0.0
    return {
        "count": int(data.size),
        "mean": mean,
        "std": std,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=bool).reshape(-1)
    p = np.asarray(predictions, dtype=bool).reshape(-1)
    if y.size == 0 or y.shape != p.shape:
        raise ValueError("labels and predictions must be non-empty and equally shaped")
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_alarm_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "missed_failure_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }
