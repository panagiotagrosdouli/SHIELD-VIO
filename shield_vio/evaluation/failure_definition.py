"""Versioned observable failure definitions and future-horizon targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from shield_vio.evaluation.prediction_targets import (
    FailureEventIndex,
    FutureFailureTargets,
    build_persistent_failure_events,
    future_failure_targets,
)


@dataclass(frozen=True)
class CriterionDefinition:
    name: str
    persistence_seconds: float
    requires_ground_truth: bool


@dataclass(frozen=True)
class FailureDefinition:
    schema_version: str
    kind: str
    horizons_seconds: tuple[float, ...]
    event_merge_gap_seconds: float
    recovery_confirmation_seconds: float
    criteria: tuple[CriterionDefinition, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"primary", "sensitivity"}:
            raise ValueError("failure definition kind must be primary or sensitivity")
        if not self.schema_version or not self.criteria:
            raise ValueError("failure definition must be versioned and contain criteria")
        if any(value <= 0 for value in self.horizons_seconds):
            raise ValueError("horizons must be positive")
        if self.event_merge_gap_seconds < 0 or self.recovery_confirmation_seconds < 0:
            raise ValueError("event timing values must be non-negative")


@dataclass(frozen=True)
class FailureEventTable:
    definition_version: str
    event_ids: tuple[str, ...]
    onsets_ns: np.ndarray
    offsets_ns: np.ndarray
    active_mask: np.ndarray


@dataclass(frozen=True)
class HorizonTargetSet:
    definition_version: str
    targets: Mapping[float, FutureFailureTargets]


PROHIBITED_CRITERION_TOKENS = (
    "degradation",
    "corruption",
    "severity",
    "seed",
    "injected",
    "oracle",
)


def load_failure_definition(path: str | Path) -> FailureDefinition:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    criteria = []
    for name, settings in payload["criteria"].items():
        if not settings.get("enabled", True):
            continue
        normalized = name.lower()
        if any(token in normalized for token in PROHIBITED_CRITERION_TOKENS):
            raise ValueError(f"experiment oracle cannot define failure: {name}")
        criteria.append(
            CriterionDefinition(
                name=name,
                persistence_seconds=float(settings.get("persistence_seconds", 0.0)),
                requires_ground_truth=bool(settings.get("requires_ground_truth", False)),
            )
        )
    return FailureDefinition(
        schema_version=str(payload["schema_version"]),
        kind=str(payload["kind"]),
        horizons_seconds=tuple(float(value) for value in payload["horizons_seconds"]),
        event_merge_gap_seconds=float(payload["event_merge_gap_seconds"]),
        recovery_confirmation_seconds=float(payload["recovery_confirmation_seconds"]),
        criteria=tuple(criteria),
    )


def build_failure_events_and_targets(
    timestamps_ns: np.ndarray,
    criterion_exceeded: Mapping[str, np.ndarray],
    definition: FailureDefinition,
) -> tuple[FailureEventTable, HorizonTargetSet]:
    """Construct deterministic observable events and `(t, t+tau]` targets."""

    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if timestamps.ndim != 1 or len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be a strictly increasing vector")
    allowed = {criterion.name for criterion in definition.criteria}
    unexpected = set(criterion_exceeded) - allowed
    if unexpected:
        raise ValueError(f"criterion inputs not declared by failure definition: {sorted(unexpected)}")
    missing = allowed - set(criterion_exceeded)
    if missing:
        raise ValueError(f"missing declared failure criteria: {sorted(missing)}")

    active_by_criterion: list[np.ndarray] = []
    for criterion in definition.criteria:
        raw = np.asarray(criterion_exceeded[criterion.name], dtype=bool)
        if raw.shape != timestamps.shape:
            raise ValueError(f"criterion shape mismatch: {criterion.name}")
        index = build_persistent_failure_events(
            timestamps,
            raw,
            persistence_seconds=criterion.persistence_seconds,
        )
        active_by_criterion.append(index.active_mask)
    union = np.any(np.column_stack(active_by_criterion), axis=1)
    stable = _apply_event_timing(
        timestamps,
        union,
        recovery_confirmation_seconds=definition.recovery_confirmation_seconds,
        merge_gap_seconds=definition.event_merge_gap_seconds,
    )
    index = _index_events(timestamps, stable)
    event_ids = tuple(
        f"{definition.schema_version}:event:{number:04d}:{int(onset)}"
        for number, onset in enumerate(index.onsets_ns, start=1)
    )
    table = FailureEventTable(
        definition.schema_version,
        event_ids,
        index.onsets_ns,
        index.offsets_ns,
        index.active_mask,
    )
    targets = {
        horizon: future_failure_targets(timestamps, index, horizon_seconds=horizon)
        for horizon in definition.horizons_seconds
    }
    return table, HorizonTargetSet(definition.schema_version, targets)


def _apply_event_timing(
    timestamps: np.ndarray,
    raw_active: np.ndarray,
    *,
    recovery_confirmation_seconds: float,
    merge_gap_seconds: float,
) -> np.ndarray:
    active = np.asarray(raw_active, dtype=bool).copy()
    recovery_ns = int(round(recovery_confirmation_seconds * 1e9))
    if recovery_ns > 0:
        open_event = False
        clear_start: int | None = None
        for index, value in enumerate(raw_active):
            if value:
                open_event = True
                clear_start = None
                active[index] = True
                continue
            if not open_event:
                continue
            if clear_start is None:
                clear_start = index
            if timestamps[index] - timestamps[clear_start] < recovery_ns:
                active[index] = True
            else:
                open_event = False
                clear_start = None

    merge_ns = int(round(merge_gap_seconds * 1e9))
    if merge_ns > 0:
        runs = _runs(active)
        for left, right in zip(runs[:-1], runs[1:]):
            gap_start = left[1] + 1
            gap_end = right[0] - 1
            if gap_start <= gap_end and timestamps[right[0]] - timestamps[left[1]] < merge_ns:
                active[gap_start : gap_end + 1] = True
    return active


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value else index - 1
            runs.append((start, end))
            start = None
    return runs


def _index_events(timestamps: np.ndarray, active: np.ndarray) -> FailureEventIndex:
    runs = _runs(active)
    return FailureEventIndex(
        np.asarray([timestamps[start] for start, _ in runs], dtype=np.int64),
        np.asarray([timestamps[end] for _, end in runs], dtype=np.int64),
        active,
    )
