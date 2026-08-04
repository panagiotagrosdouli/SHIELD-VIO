"""Persistent failure events and causal future-horizon prediction targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FailureEventIndex:
    onsets_ns: np.ndarray
    offsets_ns: np.ndarray
    active_mask: np.ndarray

    def __post_init__(self) -> None:
        onsets = np.asarray(self.onsets_ns, dtype=np.int64)
        offsets = np.asarray(self.offsets_ns, dtype=np.int64)
        active = np.asarray(self.active_mask, dtype=bool)
        if onsets.ndim != 1 or offsets.shape != onsets.shape:
            raise ValueError("event onsets and offsets must be matching vectors")
        if np.any(offsets < onsets) or np.any(np.diff(onsets) <= 0):
            raise ValueError("failure events must be ordered and non-negative in duration")
        if active.ndim != 1:
            raise ValueError("active failure mask must be one-dimensional")
        object.__setattr__(self, "onsets_ns", onsets)
        object.__setattr__(self, "offsets_ns", offsets)
        object.__setattr__(self, "active_mask", active)


@dataclass(frozen=True)
class FutureFailureTargets:
    labels: np.ndarray
    eligible_mask: np.ndarray
    horizon_seconds: float


def build_persistent_failure_events(
    timestamps_ns: np.ndarray,
    criterion_exceeded: np.ndarray,
    *,
    persistence_seconds: float,
) -> FailureEventIndex:
    """Confirm each criterion after sustained exceedance, then form union events."""

    timestamps = _timestamps(timestamps_ns)
    criteria = np.asarray(criterion_exceeded, dtype=bool)
    if criteria.ndim == 1:
        criteria = criteria[:, None]
    if criteria.ndim != 2 or criteria.shape[0] != len(timestamps):
        raise ValueError("criterion_exceeded must have one row per timestamp")
    if persistence_seconds < 0:
        raise ValueError("persistence_seconds must be non-negative")

    confirmed = np.zeros_like(criteria)
    persistence_ns = int(round(persistence_seconds * 1e9))
    for column in range(criteria.shape[1]):
        start: int | None = None
        for index, exceeded in enumerate(criteria[:, column]):
            if not exceeded:
                start = None
                continue
            if start is None:
                start = index
            if timestamps[index] - timestamps[start] >= persistence_ns:
                confirmed[index, column] = True

        # Once confirmed, retain the active state until the raw criterion clears.
        active = False
        for index, exceeded in enumerate(criteria[:, column]):
            if not exceeded:
                active = False
            elif confirmed[index, column]:
                active = True
            if active:
                confirmed[index, column] = True

    active_union = np.any(confirmed, axis=1)
    onsets: list[int] = []
    offsets: list[int] = []
    start_index: int | None = None
    for index, active in enumerate(active_union):
        if active and start_index is None:
            start_index = index
        if start_index is not None and (not active or index == len(active_union) - 1):
            end_index = index if active else index - 1
            onsets.append(int(timestamps[start_index]))
            offsets.append(int(timestamps[end_index]))
            start_index = None

    return FailureEventIndex(
        np.asarray(onsets, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        active_union,
    )


def future_failure_targets(
    timestamps_ns: np.ndarray,
    events: FailureEventIndex,
    *,
    horizon_seconds: float,
) -> FutureFailureTargets:
    """Label a future onset in ``(t, t + horizon]`` and censor incomplete tails."""

    timestamps = _timestamps(timestamps_ns)
    if len(events.active_mask) != len(timestamps):
        raise ValueError("failure active mask must match timestamps")
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")
    horizon_ns = int(round(horizon_seconds * 1e9))
    eligible = (~events.active_mask) & (timestamps + horizon_ns <= timestamps[-1])
    labels = np.zeros(len(timestamps), dtype=bool)
    for index, timestamp in enumerate(timestamps):
        if not eligible[index] or not len(events.onsets_ns):
            continue
        next_index = int(np.searchsorted(events.onsets_ns, timestamp, side="right"))
        if next_index < len(events.onsets_ns):
            labels[index] = events.onsets_ns[next_index] <= timestamp + horizon_ns
    return FutureFailureTargets(labels, eligible, horizon_seconds)


def _timestamps(values: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.int64)
    if timestamps.ndim != 1 or len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be a strictly increasing vector")
    return timestamps
