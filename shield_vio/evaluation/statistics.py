"""Aggregate statistics for repeated SHIELD-VIO experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Mapping

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


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Run-unit paired comparison with a percentile bootstrap confidence interval."""

    metric: str
    higher_is_better: bool
    paired_unit_count: int
    dropped_unit_count: int
    method_a_mean: float
    method_b_mean: float
    mean_difference: float
    oriented_effect: float
    ci95_low: float
    ci95_high: float
    bootstrap_iterations: int
    seed: int


def paired_grouped_bootstrap(
    method_a: Mapping[Hashable, float],
    method_b: Mapping[Hashable, float],
    *,
    metric: str,
    higher_is_better: bool,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    missing: str = "drop",
) -> PairedBootstrapResult:
    """Compare methods by resampling paired experimental units, never frame rows.

    Mapping keys must identify complete experimental units such as
    (dataset, sequence, estimator, degradation_condition, seed). Only keys
    present for both methods form a paired comparison. The raw difference is
    method_a - method_b; oriented_effect is positive when method A is better.
    """
    if not str(metric).strip():
        raise ValueError("metric must be non-empty")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if missing not in {"drop", "raise"}:
        raise ValueError("missing must be 'drop' or 'raise'")

    keys_a = set(method_a)
    keys_b = set(method_b)
    common = keys_a & keys_b
    unmatched = (keys_a | keys_b) - common
    if unmatched and missing == "raise":
        raise ValueError(f"unpaired experimental units: {len(unmatched)}")
    if not common:
        raise ValueError("at least one paired experimental unit is required")

    keys = sorted(common, key=repr)
    a = np.asarray([method_a[key] for key in keys], dtype=float)
    b = np.asarray([method_b[key] for key in keys], dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    invalid_count = int(np.sum(~finite))
    if invalid_count and missing == "raise":
        raise ValueError(f"non-finite paired metric values: {invalid_count}")
    a = a[finite]
    b = b[finite]
    if a.size == 0:
        raise ValueError("at least one finite paired experimental unit is required")

    differences = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, a.size, size=(n_bootstrap, a.size))
    bootstrap_differences = np.mean(differences[indices], axis=1)
    ci_low, ci_high = np.percentile(bootstrap_differences, [2.5, 97.5])
    mean_difference = float(np.mean(differences))
    direction = 1.0 if higher_is_better else -1.0

    return PairedBootstrapResult(
        metric=str(metric),
        higher_is_better=bool(higher_is_better),
        paired_unit_count=int(a.size),
        dropped_unit_count=len(unmatched) + invalid_count,
        method_a_mean=float(np.mean(a)),
        method_b_mean=float(np.mean(b)),
        mean_difference=mean_difference,
        oriented_effect=direction * mean_difference,
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
        bootstrap_iterations=int(n_bootstrap),
        seed=int(seed),
    )
