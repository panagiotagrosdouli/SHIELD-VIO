"""Causal, backend-neutral health feature construction for failure prediction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PRIVILEGED_FEATURE_TOKENS = (
    "ground_truth",
    "failure_label",
    "future_failure",
    "degradation",
    "severity",
    "oracle",
    "split",
)


@dataclass(frozen=True)
class HealthFeatureMatrix:
    """Timestamped feature matrix with auditable source-time provenance."""

    timestamps_ns: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...]
    max_source_timestamps_ns: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64)
        values = np.asarray(self.values, dtype=float)
        source_times = np.asarray(self.max_source_timestamps_ns, dtype=np.int64)
        if timestamps.ndim != 1 or len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("feature timestamps must be a strictly increasing vector")
        if values.shape != (len(timestamps), len(self.feature_names)):
            raise ValueError("feature values do not match timestamps and feature names")
        if source_times.shape != timestamps.shape:
            raise ValueError("max source timestamps must match feature timestamps")
        if not np.all(np.isfinite(values)):
            raise ValueError("feature matrix contains non-finite values")
        if np.any(source_times > timestamps):
            raise ValueError("causal feature rows cannot use future source timestamps")
        assert_no_privileged_features(self.feature_names)
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "max_source_timestamps_ns", source_times)


FEATURE_NAMES = (
    "covariance_trace",
    "covariance_log1p",
    "covariance_condition_log1p",
    "innovation_nis",
    "innovation_nis_missing",
    "detected_features",
    "tracked_features",
    "correspondence_count",
    "inlier_count",
    "inlier_ratio",
    "visual_update_missing",
    "seconds_since_visual_update",
    "tracking_lost",
    "covariance_growth_per_s",
    "rolling_covariance_mean",
    "rolling_covariance_std",
    "rolling_covariance_slope_per_s",
    "rolling_nis_mean",
    "rolling_tracked_features_mean",
    "rolling_tracked_features_slope_per_s",
)


def assert_no_privileged_features(feature_names: Sequence[str]) -> None:
    """Reject deployable feature schemas containing privileged experiment fields."""

    for name in feature_names:
        normalized = name.lower()
        if any(token in normalized for token in PRIVILEGED_FEATURE_TOKENS):
            raise ValueError(f"privileged field is forbidden in deployable features: {name}")


def load_causal_health_features(
    run_dir: str | Path,
    *,
    history_seconds: float = 1.0,
) -> HealthFeatureMatrix:
    """Load runner health artifacts and create a causal feature matrix."""

    root = Path(run_dir)
    health_rows = _read_dict_rows(root / "health.csv")
    visual_path = root / "visual_updates.csv"
    visual_rows = _read_dict_rows(visual_path) if visual_path.is_file() else []
    return build_causal_health_features(
        health_rows,
        visual_rows,
        history_seconds=history_seconds,
    )


def build_causal_health_features(
    health_rows: Sequence[Mapping[str, str]],
    visual_rows: Sequence[Mapping[str, str]] = (),
    *,
    history_seconds: float = 1.0,
) -> HealthFeatureMatrix:
    """Merge health and visual records without using any timestamp after the row time."""

    if history_seconds <= 0:
        raise ValueError("history_seconds must be positive")
    if len(health_rows) < 2:
        raise ValueError("at least two health rows are required")

    timestamps = np.asarray(
        [_integer(row, "frame_timestamp_ns") for row in health_rows], dtype=np.int64
    )
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("health timestamps must be strictly increasing")

    visual_sorted = sorted(visual_rows, key=lambda row: _integer(row, "frame_timestamp_ns"))
    visual_timestamps = np.asarray(
        [_integer(row, "frame_timestamp_ns") for row in visual_sorted], dtype=np.int64
    )
    if len(visual_timestamps) and np.any(np.diff(visual_timestamps) <= 0):
        raise ValueError("visual update timestamps must be strictly increasing")

    covariance = np.asarray([_number(row, "covariance_trace") for row in health_rows])
    condition = np.asarray([_number(row, "covariance_condition_number") for row in health_rows])
    nis = np.asarray([_optional_number(row, "innovation_nis") for row in health_rows])
    nis_missing = ~np.isfinite(nis)
    nis_filled = np.where(nis_missing, 0.0, nis)

    detected = np.zeros(len(timestamps))
    tracked = np.zeros(len(timestamps))
    correspondences = np.zeros(len(timestamps))
    inliers = np.zeros(len(timestamps))
    inlier_ratio = np.zeros(len(timestamps))
    visual_missing = np.ones(len(timestamps))
    visual_age = np.zeros(len(timestamps))

    visual_index = -1
    for index, timestamp in enumerate(timestamps):
        while (
            visual_index + 1 < len(visual_sorted)
            and visual_timestamps[visual_index + 1] <= timestamp
        ):
            visual_index += 1
        if visual_index < 0:
            visual_age[index] = history_seconds
            continue
        visual = visual_sorted[visual_index]
        detected[index] = _number(visual, "detected_features")
        tracked[index] = _number(visual, "tracked_features")
        correspondences[index] = _number(visual, "correspondence_count")
        inliers[index] = _number(visual, "inlier_count")
        ratio = _optional_number(visual, "inlier_ratio")
        inlier_ratio[index] = 0.0 if not np.isfinite(ratio) else ratio
        visual_missing[index] = float(visual_timestamps[visual_index] != timestamp)
        visual_age[index] = max(0.0, (timestamp - visual_timestamps[visual_index]) * 1e-9)

    tracking_lost = np.asarray(
        [
            float(str(row.get("tracking_status", "")).lower() not in {"tracking", "ok"})
            for row in health_rows
        ]
    )
    elapsed = (timestamps - timestamps[0]).astype(float) * 1e-9
    covariance_growth = _causal_derivative(elapsed, covariance)

    rolling_covariance_mean = np.zeros(len(timestamps))
    rolling_covariance_std = np.zeros(len(timestamps))
    rolling_covariance_slope = np.zeros(len(timestamps))
    rolling_nis_mean = np.zeros(len(timestamps))
    rolling_tracked_mean = np.zeros(len(timestamps))
    rolling_tracked_slope = np.zeros(len(timestamps))
    for index, current_time in enumerate(elapsed):
        start = int(np.searchsorted(elapsed, current_time - history_seconds, side="left"))
        window = slice(start, index + 1)
        rolling_covariance_mean[index] = float(np.mean(covariance[window]))
        rolling_covariance_std[index] = float(np.std(covariance[window]))
        rolling_covariance_slope[index] = _slope(elapsed[window], covariance[window])
        available_nis = nis[window][np.isfinite(nis[window])]
        rolling_nis_mean[index] = float(np.mean(available_nis)) if len(available_nis) else 0.0
        rolling_tracked_mean[index] = float(np.mean(tracked[window]))
        rolling_tracked_slope[index] = _slope(elapsed[window], tracked[window])

    values = np.column_stack(
        [
            covariance,
            np.log1p(np.maximum(covariance, 0.0)),
            np.log1p(np.maximum(condition, 0.0)),
            nis_filled,
            nis_missing.astype(float),
            detected,
            tracked,
            correspondences,
            inliers,
            inlier_ratio,
            visual_missing,
            visual_age,
            tracking_lost,
            covariance_growth,
            rolling_covariance_mean,
            rolling_covariance_std,
            rolling_covariance_slope,
            rolling_nis_mean,
            rolling_tracked_mean,
            rolling_tracked_slope,
        ]
    )
    return HealthFeatureMatrix(timestamps, values, FEATURE_NAMES, timestamps.copy())


def _read_dict_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"artifact contains no rows: {path}")
    return rows


def _number(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid numeric field: {key}") from exc
    if not np.isfinite(value):
        raise ValueError(f"numeric field must be finite: {key}")
    return value


def _optional_number(row: Mapping[str, str], key: str) -> float:
    raw = row.get(key, "")
    if raw is None or str(raw).strip() == "":
        return float("nan")
    return _number(row, key)


def _integer(row: Mapping[str, str], key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or invalid integer field: {key}") from exc


def _causal_derivative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values))
    delta_time = np.diff(times)
    if len(delta_time):
        output[1:] = np.diff(values) / delta_time
    return output


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2 or times[-1] <= times[0]:
        return 0.0
    centered = times - np.mean(times)
    denominator = float(centered @ centered)
    return 0.0 if denominator <= 0 else float(centered @ (values - np.mean(values)) / denominator)
