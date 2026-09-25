"""Build frozen primary failure events and future targets from observable tables."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shield_vio.evaluation.failure_definition import (
    FailureDefinition,
    evaluate_failure_criteria,
    build_failure_events_and_targets,
)
from shield_vio.evaluation.prediction_targets import FutureFailureTargets
from shield_vio.evaluation.primary_observables import PrimaryObservableTable


@dataclass(frozen=True)
class PrimaryFailureBuildResult:
    definition_version: str
    timestamps_ns: np.ndarray
    complete_observability_mask: np.ndarray
    segment_ids: np.ndarray
    criterion_exceeded: dict[str, np.ndarray]
    event_ids: tuple[str, ...]
    onsets_ns: np.ndarray
    offsets_ns: np.ndarray
    active_mask: np.ndarray
    targets: dict[float, FutureFailureTargets]

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64)
        if timestamps.ndim != 1 or len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("primary failure timestamps must be strictly increasing")
        if np.asarray(self.complete_observability_mask, dtype=bool).shape != timestamps.shape:
            raise ValueError("complete observability mask must match timestamps")
        if np.asarray(self.segment_ids, dtype=int).shape != timestamps.shape:
            raise ValueError("segment ids must match timestamps")
        if np.asarray(self.active_mask, dtype=bool).shape != timestamps.shape:
            raise ValueError("active mask must match timestamps")
        for name, values in self.criterion_exceeded.items():
            if np.asarray(values, dtype=bool).shape != timestamps.shape:
                raise ValueError(f"criterion exceeded shape mismatch: {name}")
        for target in self.targets.values():
            if np.asarray(target.labels, dtype=bool).shape != timestamps.shape:
                raise ValueError("target labels must match timestamps")
            if np.asarray(target.eligible_mask, dtype=bool).shape != timestamps.shape:
                raise ValueError("target eligibility must match timestamps")


def _complete_observability(table: PrimaryObservableTable) -> np.ndarray:
    complete = np.ones(len(table.timestamps_ns), dtype=bool)
    for name in table.values:
        applicable = np.asarray(table.applicable[name], dtype=bool)
        observable = np.asarray(table.observable[name], dtype=bool)
        complete &= (~applicable) | observable
    return complete


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value else index - 1
            runs.append((start, end))
            start = None
    return runs


def build_primary_failure_targets(
    table: PrimaryObservableTable,
    definition: FailureDefinition,
) -> PrimaryFailureBuildResult:
    """Build events/targets only inside contiguous fully observable segments.

    Any sample where an applicable criterion is unavailable breaks the timeline.
    Persistence, recovery timing, event merging, and future horizons are therefore
    never allowed to bridge an observability gap.
    """

    declared = tuple(criterion.name for criterion in definition.criteria)
    if set(declared) != set(table.values):
        missing = sorted(set(declared) - set(table.values))
        extra = sorted(set(table.values) - set(declared))
        raise ValueError(
            f"primary observable/definition mismatch: missing={missing}, extra={extra}"
        )

    timestamps = np.asarray(table.timestamps_ns, dtype=np.int64)
    complete = _complete_observability(table)
    segment_ids = np.full(len(timestamps), -1, dtype=int)
    criterion_exceeded = {
        criterion.name: np.zeros(len(timestamps), dtype=bool)
        for criterion in definition.criteria
    }
    active = np.zeros(len(timestamps), dtype=bool)
    target_labels = {
        horizon: np.zeros(len(timestamps), dtype=bool)
        for horizon in definition.horizons_seconds
    }
    target_eligible = {
        horizon: np.zeros(len(timestamps), dtype=bool)
        for horizon in definition.horizons_seconds
    }

    event_onsets: list[int] = []
    event_offsets: list[int] = []
    segment_number = 0

    for start, end in _runs(complete):
        # failure/target builders require at least two timestamps. A singleton
        # fully-observable island remains censored rather than being bridged.
        if end - start + 1 < 2:
            continue
        segment_number += 1
        segment_ids[start : end + 1] = segment_number
        slc = slice(start, end + 1)
        observations: dict[str, np.ndarray] = {}
        for criterion in definition.criteria:
            name = criterion.name
            values = np.asarray(table.values[name])[slc]
            applicable = np.asarray(table.applicable[name], dtype=bool)[slc]
            if criterion.comparison == "true":
                safe = np.zeros(len(values), dtype=bool)
                safe[applicable] = np.asarray(values[applicable], dtype=bool)
            else:
                safe = np.zeros(len(values), dtype=float)
                safe[applicable] = np.asarray(values[applicable], dtype=float)
            observations[name] = safe

        exceeded = evaluate_failure_criteria(observations, definition)
        events, targets = build_failure_events_and_targets(
            timestamps[slc],
            exceeded,
            definition,
        )
        for name, values in exceeded.items():
            criterion_exceeded[name][slc] = values
        active[slc] = events.active_mask
        event_onsets.extend(int(value) for value in events.onsets_ns)
        event_offsets.extend(int(value) for value in events.offsets_ns)
        for horizon, target in targets.targets.items():
            target_labels[horizon][slc] = target.labels
            target_eligible[horizon][slc] = target.eligible_mask

    ordered = sorted(zip(event_onsets, event_offsets, strict=True))
    onsets = np.asarray([item[0] for item in ordered], dtype=np.int64)
    offsets = np.asarray([item[1] for item in ordered], dtype=np.int64)
    event_ids = tuple(
        f"{definition.schema_version}:event:{number:04d}:{onset}"
        for number, onset in enumerate(onsets, start=1)
    )
    targets = {
        horizon: FutureFailureTargets(
            labels=target_labels[horizon],
            eligible_mask=target_eligible[horizon],
            horizon_seconds=horizon,
        )
        for horizon in definition.horizons_seconds
    }
    return PrimaryFailureBuildResult(
        definition_version=definition.schema_version,
        timestamps_ns=timestamps,
        complete_observability_mask=complete,
        segment_ids=segment_ids,
        criterion_exceeded=criterion_exceeded,
        event_ids=event_ids,
        onsets_ns=onsets,
        offsets_ns=offsets,
        active_mask=active,
        targets=targets,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _horizon_name(horizon: float) -> str:
    return str(float(horizon))


def write_primary_failure_artifacts(
    result: PrimaryFailureBuildResult,
    output_dir: str | Path,
    *,
    failure_config_path: str | Path,
    source_run_dir: str | Path,
    evidence_level: str,
) -> dict[str, Any]:
    """Write deterministic non-confirmatory event/target artifacts."""

    if evidence_level not in {"PUBLIC_DATASET_SMOKE", "PUBLIC_DATASET_DEVELOPMENT"}:
        raise ValueError("primary failure exporter is non-confirmatory by construction")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    failure_config = Path(failure_config_path)
    source_run = Path(source_run_dir)

    events_path = destination / "failure_events.csv"
    with events_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_id", "onset_ns", "offset_ns", "duration_seconds"])
        for event_id, onset, offset in zip(
            result.event_ids,
            result.onsets_ns,
            result.offsets_ns,
            strict=True,
        ):
            writer.writerow(
                [event_id, int(onset), int(offset), float(offset - onset) * 1e-9]
            )

    criteria_path = destination / "criterion_exceeded.csv"
    criterion_names = tuple(result.criterion_exceeded)
    with criteria_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["timestamp_ns", "segment_id", "complete_observability", *criterion_names]
        )
        for index, timestamp in enumerate(result.timestamps_ns):
            writer.writerow(
                [
                    int(timestamp),
                    int(result.segment_ids[index]),
                    int(result.complete_observability_mask[index]),
                    *[
                        int(result.criterion_exceeded[name][index])
                        for name in criterion_names
                    ],
                ]
            )

    targets_path = destination / "prediction_targets.csv"
    with targets_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        header = [
            "timestamp_ns",
            "segment_id",
            "complete_observability",
            "failure_active",
        ]
        for horizon in result.targets:
            label = _horizon_name(horizon)
            header.extend([f"failure_within_{label}s", f"eligible_{label}s"])
        writer.writerow(header)
        for index, timestamp in enumerate(result.timestamps_ns):
            row: list[object] = [
                int(timestamp),
                int(result.segment_ids[index]),
                int(result.complete_observability_mask[index]),
                int(result.active_mask[index]),
            ]
            for horizon, target in result.targets.items():
                row.extend(
                    [
                        int(target.labels[index]),
                        int(target.eligible_mask[index]),
                    ]
                )
            writer.writerow(row)

    source_hashes: dict[str, str] = {}
    for name in ("trajectory.csv", "health.csv", "visual_updates.csv", "experiment_manifest.json"):
        path = source_run / name
        if path.is_file():
            source_hashes[name] = _sha256(path)

    artifact_hashes = {
        path.name: _sha256(path)
        for path in (events_path, criteria_path, targets_path)
    }
    manifest = {
        "schema_version": "SHIELD_VIO_PRIMARY_FAILURE_TARGETS_V2",
        "failure_definition_schema": result.definition_version,
        "failure_config": str(failure_config),
        "failure_config_sha256": _sha256(failure_config),
        "evidence_level": evidence_level,
        "confirmatory": False,
        "sample_count": int(len(result.timestamps_ns)),
        "complete_observability_samples": int(
            np.sum(result.complete_observability_mask)
        ),
        "censored_observability_samples": int(
            np.sum(~result.complete_observability_mask)
        ),
        "labelable_segment_count": int(np.max(result.segment_ids, initial=-1)),
        "failure_event_count": int(len(result.event_ids)),
        "events": [
            {
                "event_id": event_id,
                "onset_ns": int(onset),
                "offset_ns": int(offset),
            }
            for event_id, onset, offset in zip(
                result.event_ids,
                result.onsets_ns,
                result.offsets_ns,
                strict=True,
            )
        ],
        "horizons": {
            _horizon_name(horizon): {
                "eligible_samples": int(np.sum(target.eligible_mask)),
                "positive_windows": int(np.sum(target.labels & target.eligible_mask)),
                "negative_windows": int(
                    np.sum((~target.labels) & target.eligible_mask)
                ),
            }
            for horizon, target in result.targets.items()
        },
        "source_run_artifact_sha256": source_hashes,
        "artifact_sha256": artifact_hashes,
        "claim_boundary": (
            "Non-confirmatory primary-label construction only. Events and future targets "
            "are built exclusively inside contiguous fully observable segments, and no "
            "sealed test partition is opened or evaluated by this exporter."
        ),
    }
    manifest_path = destination / "primary_failure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
