"""Fail-closed SHIELD_VIO_FAILURE_V2 event and future-target construction."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from shield_vio.evaluation.failure_definition import (
    FailureDefinition,
    FailureEventTable,
    HorizonTargetSet,
    build_failure_events_and_targets,
)
from shield_vio.evaluation.prediction_targets import (
    FailureEventIndex,
    FutureFailureTargets,
    future_failure_targets,
)
from shield_vio.evaluation.primary_observables import PrimaryObservableTable


@dataclass(frozen=True)
class PrimaryFailureBundle:
    definition_version: str
    joint_available_mask: np.ndarray
    criterion_exceeded: Mapping[str, np.ndarray]
    events: FailureEventTable
    targets: HorizonTargetSet

    def __post_init__(self) -> None:
        mask = np.asarray(self.joint_available_mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("joint availability mask must be one-dimensional")
        if len(self.events.active_mask) != len(mask):
            raise ValueError("event mask and availability mask must align")
        for name, values in self.criterion_exceeded.items():
            if np.asarray(values, dtype=bool).shape != mask.shape:
                raise ValueError(f"criterion exceedance shape mismatch: {name}")


def _criterion_exceedance(
    table: PrimaryObservableTable,
    definition: FailureDefinition,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    declared = {criterion.name for criterion in definition.criteria}
    if declared != set(table.values):
        missing = sorted(declared - set(table.values))
        extra = sorted(set(table.values) - declared)
        raise ValueError(
            f"observable/definition criterion mismatch; missing={missing}, extra={extra}"
        )

    joint_available = np.ones(len(table.timestamps_ns), dtype=bool)
    exceeded: dict[str, np.ndarray] = {}
    for criterion in definition.criteria:
        values = np.asarray(table.values[criterion.name])
        observable = np.asarray(table.observable[criterion.name], dtype=bool)
        applicable = np.asarray(table.applicable[criterion.name], dtype=bool)
        available = (~applicable) | observable
        joint_available &= available

        active = np.zeros(len(values), dtype=bool)
        evaluable = applicable & observable
        if criterion.comparison == "true":
            active[evaluable] = values[evaluable].astype(bool)
        elif criterion.comparison == "greater_than":
            if criterion.threshold is None:
                raise ValueError(f"criterion threshold missing: {criterion.name}")
            numeric = values.astype(float)
            finite = np.isfinite(numeric)
            if np.any(evaluable & ~finite):
                raise ValueError(
                    f"observable criterion contains non-finite value: {criterion.name}"
                )
            active[evaluable] = numeric[evaluable] > criterion.threshold
        else:
            raise ValueError(f"unsupported criterion comparison: {criterion.comparison}")
        exceeded[criterion.name] = active
    return exceeded, joint_available


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open contiguous true runs."""

    values = np.asarray(mask, dtype=bool)
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(values) - 1):
            stop = index + 1 if value else index
            result.append((start, stop))
            start = None
    return result


def _build_events_on_available_runs(
    timestamps_ns: np.ndarray,
    exceeded: Mapping[str, np.ndarray],
    joint_available: np.ndarray,
    definition: FailureDefinition,
) -> FailureEventTable:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    active = np.zeros(len(timestamps), dtype=bool)
    onsets: list[int] = []
    offsets: list[int] = []

    for start, stop in _runs(joint_available):
        # A single isolated sample cannot establish timestamp-aware persistence
        # or a future monitoring interval; leave it censored.
        if stop - start < 2:
            continue
        local_exceeded = {
            name: np.asarray(values, dtype=bool)[start:stop]
            for name, values in exceeded.items()
        }
        local_events, _ = build_failure_events_and_targets(
            timestamps[start:stop],
            local_exceeded,
            definition,
        )
        active[start:stop] = local_events.active_mask
        onsets.extend(int(value) for value in local_events.onsets_ns)
        offsets.extend(int(value) for value in local_events.offsets_ns)

    onset_array = np.asarray(onsets, dtype=np.int64)
    offset_array = np.asarray(offsets, dtype=np.int64)
    event_ids = tuple(
        f"{definition.schema_version}:event:{number:04d}:{onset}"
        for number, onset in enumerate(onsets, start=1)
    )
    return FailureEventTable(
        definition.schema_version,
        event_ids,
        onset_array,
        offset_array,
        active,
    )


def _horizon_availability(
    timestamps_ns: np.ndarray,
    joint_available: np.ndarray,
    horizon_seconds: float,
) -> np.ndarray:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    available = np.asarray(joint_available, dtype=bool)
    horizon_ns = int(round(horizon_seconds * 1e9))
    invalid_prefix = np.r_[0, np.cumsum((~available).astype(np.int64))]
    right = np.searchsorted(timestamps, timestamps + horizon_ns, side="right")
    indices = np.arange(len(timestamps))
    invalid_in_window = invalid_prefix[right] - invalid_prefix[indices]
    complete_tail = timestamps + horizon_ns <= timestamps[-1]
    return available & complete_tail & (invalid_in_window == 0)


def build_primary_failure_bundle(
    table: PrimaryObservableTable,
    definition: FailureDefinition,
) -> PrimaryFailureBundle:
    """Build primary events only across fully available applicable criteria.

    Missing applicable observations break event runs and censor any prediction
    window that intersects them. Explicitly non-applicable criteria are omitted
    from the union by construction.
    """

    if definition.kind != "primary":
        raise ValueError("primary failure bundle requires a primary failure definition")
    exceeded, joint_available = _criterion_exceedance(table, definition)
    if not np.any(joint_available):
        raise ValueError("no samples have complete applicable primary observations")

    events = _build_events_on_available_runs(
        table.timestamps_ns,
        exceeded,
        joint_available,
        definition,
    )
    event_index = FailureEventIndex(events.onsets_ns, events.offsets_ns, events.active_mask)

    targets: dict[float, FutureFailureTargets] = {}
    for horizon in definition.horizons_seconds:
        base = future_failure_targets(
            table.timestamps_ns,
            event_index,
            horizon_seconds=horizon,
        )
        availability = _horizon_availability(
            table.timestamps_ns,
            joint_available,
            horizon,
        )
        eligible = base.eligible_mask & availability
        labels = base.labels & eligible
        targets[horizon] = FutureFailureTargets(labels, eligible, horizon)

    return PrimaryFailureBundle(
        definition_version=definition.schema_version,
        joint_available_mask=joint_available,
        criterion_exceeded=exceeded,
        events=events,
        targets=HorizonTargetSet(definition.schema_version, targets),
    )


def _horizon_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_primary_failure_artifacts(
    bundle: PrimaryFailureBundle,
    table: PrimaryObservableTable,
    output_dir: str | Path,
    *,
    failure_config: str | Path,
    source_run_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    with (destination / "criterion_exceeded.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        names = tuple(bundle.criterion_exceeded)
        writer.writerow(
            ["timestamp_ns", "joint_label_available", *[f"{name}__exceeded" for name in names]]
        )
        for index, timestamp in enumerate(table.timestamps_ns):
            writer.writerow(
                [
                    int(timestamp),
                    int(bundle.joint_available_mask[index]),
                    *[
                        int(np.asarray(bundle.criterion_exceeded[name], dtype=bool)[index])
                        for name in names
                    ],
                ]
            )

    with (destination / "failure_events.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_id", "onset_ns", "offset_ns"])
        writer.writerows(
            zip(
                bundle.events.event_ids,
                bundle.events.onsets_ns,
                bundle.events.offsets_ns,
                strict=True,
            )
        )

    with (destination / "prediction_targets.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        horizons = tuple(bundle.targets.targets)
        header = ["timestamp_ns"]
        for horizon in horizons:
            suffix = _horizon_label(horizon)
            header.extend([f"failure_within_{suffix}s", f"eligible_{suffix}s"])
        writer.writerow(header)
        for index, timestamp in enumerate(table.timestamps_ns):
            row: list[object] = [int(timestamp)]
            for horizon in horizons:
                target = bundle.targets.targets[horizon]
                row.extend([int(target.labels[index]), int(target.eligible_mask[index])])
            writer.writerow(row)

    config_path = Path(failure_config)
    artifacts = [
        destination / "criterion_exceeded.csv",
        destination / "failure_events.csv",
        destination / "prediction_targets.csv",
    ]
    per_horizon = {
        f"{horizon:g}": {
            "eligible_samples": int(np.sum(target.eligible_mask)),
            "positive_windows": int(np.sum(target.labels[target.eligible_mask])),
            "negative_windows": int(np.sum(~target.labels[target.eligible_mask])),
        }
        for horizon, target in bundle.targets.targets.items()
    }
    manifest: dict[str, Any] = {
        "schema_version": "SHIELD_VIO_PRIMARY_FAILURE_ARTIFACTS_V2",
        "failure_definition_schema": bundle.definition_version,
        "failure_config": str(config_path),
        "failure_config_sha256": _sha256(config_path),
        "source_run_dir": str(source_run_dir),
        "sample_count": len(table.timestamps_ns),
        "joint_available_samples": int(np.sum(bundle.joint_available_mask)),
        "joint_unavailable_samples": int(np.sum(~bundle.joint_available_mask)),
        "failure_event_count": len(bundle.events.event_ids),
        "horizons": per_horizon,
        "artifacts": {path.name: _sha256(path) for path in artifacts},
        "confirmatory": False,
        "claim_boundary": (
            "Deterministic primary failure-label construction only. This artifact does not "
            "demonstrate predictive performance, calibration quality, protective utility, "
            "domain-shift handling, or estimator superiority."
        ),
    }
    (destination / "primary_failure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
