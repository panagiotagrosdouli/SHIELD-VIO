"""Timestamp-aware trailing temporal features for canonical health signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalFeatureTable:
    timestamps_ns: np.ndarray
    feature_names: tuple[str, ...]
    values: np.ndarray
    max_source_timestamps_ns: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.timestamps_ns, dtype=np.int64)
        x = np.asarray(self.values, dtype=float)
        s = np.asarray(self.max_source_timestamps_ns, dtype=np.int64)
        if t.ndim != 1 or len(t) < 1 or (len(t) > 1 and np.any(np.diff(t) <= 0)):
            raise ValueError("timestamps must be strictly increasing")
        if x.shape != (len(t), len(self.feature_names)):
            raise ValueError("temporal values do not match timestamps/features")
        if s.shape != t.shape or np.any(s > t):
            raise ValueError("temporal features cannot use future source timestamps")
        object.__setattr__(self, "timestamps_ns", t)
        object.__setattr__(self, "values", x)
        object.__setattr__(self, "max_source_timestamps_ns", s)


def build_trailing_features(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    signal_name: str,
    windows_seconds: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
) -> TemporalFeatureTable:
    """Build causal summary features over [t-W, t] for irregular timestamps."""

    t = np.asarray(timestamps_ns, dtype=np.int64)
    x = np.asarray(values, dtype=float).reshape(-1)
    ok = np.asarray(valid, dtype=bool).reshape(-1)
    if t.shape != x.shape or t.shape != ok.shape or len(t) < 1:
        raise ValueError("timestamps, values and validity must be aligned")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if any(window <= 0 for window in windows_seconds):
        raise ValueError("windows must be positive")

    names: list[str] = []
    columns: list[np.ndarray] = []
    for window_seconds in windows_seconds:
        tag = str(window_seconds).replace(".", "p")
        metrics = {name: np.zeros(len(t), dtype=float) for name in (
            "mean", "std", "min", "max", "slope_per_s", "rate_per_s",
            "persistence_s", "missing_fraction", "time_since_valid_s",
        )}
        window_ns = int(round(window_seconds * 1e9))
        last_valid_index: int | None = None
        persistent_start: int | None = None
        for index, current in enumerate(t):
            if ok[index]:
                last_valid_index = index
                if persistent_start is None:
                    persistent_start = index
            else:
                persistent_start = None
            start = int(np.searchsorted(t, current - window_ns, side="left"))
            indices = np.arange(start, index + 1)
            valid_indices = indices[ok[indices]]
            metrics["missing_fraction"][index] = 1.0 - len(valid_indices) / len(indices)
            if last_valid_index is None:
                metrics["time_since_valid_s"][index] = window_seconds
            else:
                metrics["time_since_valid_s"][index] = max(
                    0.0, (current - t[last_valid_index]) * 1e-9
                )
            if persistent_start is not None:
                metrics["persistence_s"][index] = (current - t[persistent_start]) * 1e-9
            if not len(valid_indices):
                continue
            y = x[valid_indices]
            times_s = (t[valid_indices] - t[valid_indices][0]).astype(float) * 1e-9
            metrics["mean"][index] = float(np.mean(y))
            metrics["std"][index] = float(np.std(y))
            metrics["min"][index] = float(np.min(y))
            metrics["max"][index] = float(np.max(y))
            if len(valid_indices) > 1 and times_s[-1] > 0:
                centered = times_s - np.mean(times_s)
                denom = float(centered @ centered)
                if denom > 0:
                    metrics["slope_per_s"][index] = float(
                        centered @ (y - np.mean(y)) / denom
                    )
                metrics["rate_per_s"][index] = float((y[-1] - y[-2]) / (times_s[-1] - times_s[-2]))
        for metric_name, column in metrics.items():
            names.append(f"{signal_name}__{metric_name}__{tag}s")
            columns.append(column)

    matrix = np.column_stack(columns) if columns else np.empty((len(t), 0))
    return TemporalFeatureTable(t, tuple(names), matrix, t.copy())
