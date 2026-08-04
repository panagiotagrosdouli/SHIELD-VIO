"""Sequence-level split registry and leakage guards for paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

SPLIT_NAMES = ("train", "calibration", "validation", "test", "shifted_test")
STAGE_SPLIT = {
    "preprocessing_fit": "train",
    "detector_fit": "train",
    "calibration_fit": "calibration",
    "conformal_fit": "calibration",
    "threshold_selection": "validation",
    "policy_selection": "validation",
    "confirmatory_evaluation": "test",
    "shifted_evaluation": "shifted_test",
}


@dataclass(frozen=True)
class SequenceSplitRegistry:
    """Immutable mapping that assigns each physical sequence exactly once."""

    assignments: Mapping[str, Sequence[str]]

    def __post_init__(self) -> None:
        unknown = set(self.assignments) - set(SPLIT_NAMES)
        if unknown:
            raise ValueError(f"unknown split names: {sorted(unknown)}")
        seen: dict[str, str] = {}
        for split, sequences in self.assignments.items():
            for sequence in sequences:
                normalized = str(sequence).strip()
                if not normalized:
                    raise ValueError("sequence identifiers must be non-empty")
                if normalized in seen:
                    raise ValueError(
                        f"sequence {normalized!r} appears in both {seen[normalized]!r} and {split!r}"
                    )
                seen[normalized] = split
        if not seen:
            raise ValueError("split registry must contain at least one sequence")
        object.__setattr__(
            self,
            "assignments",
            {
                name: tuple(str(value).strip() for value in values)
                for name, values in self.assignments.items()
            },
        )

    def split_for(self, sequence: str) -> str:
        matches = [name for name, values in self.assignments.items() if sequence in values]
        if not matches:
            raise KeyError(f"sequence is absent from split registry: {sequence}")
        return matches[0]


def assert_stage_uses_declared_split(
    stage: str,
    sequence_ids: Sequence[str],
    registry: SequenceSplitRegistry,
) -> None:
    """Fail when a fitting/evaluation stage receives a forbidden sequence."""

    if stage not in STAGE_SPLIT:
        raise ValueError(f"unknown experiment stage: {stage}")
    required = STAGE_SPLIT[stage]
    offending = sorted(
        sequence for sequence in sequence_ids if registry.split_for(sequence) != required
    )
    if offending:
        raise ValueError(
            f"stage {stage!r} requires split {required!r}; offending sequences: {offending}"
        )


def assert_events_do_not_cross_splits(event_rows: Sequence[tuple[str, str]]) -> None:
    """Reject event windows assigned to more than one split."""

    event_splits: dict[str, str] = {}
    for event_id, split in event_rows:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split name for event {event_id!r}: {split!r}")
        previous = event_splits.setdefault(event_id, split)
        if previous != split:
            raise ValueError(
                f"failure event {event_id!r} is divided between {previous!r} and {split!r}"
            )
