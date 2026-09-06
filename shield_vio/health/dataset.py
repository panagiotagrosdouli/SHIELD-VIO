"""Canonical prediction dataset with structural feature/target/oracle separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from shield_vio.health.schema import HealthSample, flatten_deployable


@dataclass(frozen=True)
class PredictionGroups:
    dataset: str
    sequence: str
    estimator: str
    condition_id: str
    seed: int


@dataclass(frozen=True)
class PredictionDataset:
    samples: tuple[HealthSample, ...]
    horizon_targets: Mapping[float, np.ndarray]
    eligible_masks: Mapping[float, np.ndarray]
    groups: PredictionGroups
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("prediction dataset requires at least one health sample")
        timestamps = np.asarray([sample.timestamp_ns for sample in self.samples], dtype=np.int64)
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("health samples must be strictly time ordered")
        for horizon, labels in self.horizon_targets.items():
            values = np.asarray(labels, dtype=bool)
            eligible = np.asarray(self.eligible_masks[horizon], dtype=bool)
            if horizon <= 0 or values.shape != timestamps.shape or eligible.shape != timestamps.shape:
                raise ValueError("targets and eligibility must align with health samples")

    def features(self) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return only deployable feature values/missingness; never targets or metadata."""

        rows = [flatten_deployable(sample) for sample in self.samples]
        names = tuple(key for key in rows[0] if key != "timestamp_ns")
        values = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
        return values, names

    def targets(self, horizon_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        if horizon_seconds not in self.horizon_targets:
            raise KeyError(f"unknown horizon: {horizon_seconds}")
        return (
            np.asarray(self.horizon_targets[horizon_seconds], dtype=bool).copy(),
            np.asarray(self.eligible_masks[horizon_seconds], dtype=bool).copy(),
        )

    def grouping(self) -> PredictionGroups:
        return self.groups

    def experiment_metadata(self) -> dict[str, object]:
        """Return non-feature metadata, which may contain experimental condition details."""

        return dict(self.metadata)


def build_prediction_dataset(
    samples: Sequence[HealthSample],
    *,
    horizon_targets: Mapping[float, np.ndarray],
    eligible_masks: Mapping[float, np.ndarray],
    groups: PredictionGroups,
    metadata: Mapping[str, object] | None = None,
) -> PredictionDataset:
    return PredictionDataset(
        tuple(samples),
        dict(horizon_targets),
        dict(eligible_masks),
        groups,
        dict(metadata or {}),
    )
