"""Frozen sequence-level split enforcement for publication benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

VALID_SPLITS = ("train", "calibration", "validation", "test", "shifted_test")


@dataclass(frozen=True, order=True)
class SequenceIdentity:
    dataset: str
    sequence: str


@dataclass(frozen=True)
class SplitEntry:
    identity: SequenceIdentity
    split: str


@dataclass(frozen=True)
class PaperSplit:
    entries: tuple[SplitEntry, ...]

    def __post_init__(self) -> None:
        seen: dict[SequenceIdentity, str] = {}
        for entry in self.entries:
            if entry.split not in VALID_SPLITS:
                raise ValueError(f"unknown split: {entry.split}")
            previous = seen.get(entry.identity)
            if previous is not None:
                raise ValueError(
                    f"physical sequence {entry.identity.dataset}/{entry.identity.sequence} "
                    f"appears in both {previous} and {entry.split}"
                )
            seen[entry.identity] = entry.split

    def partition(self, name: str) -> tuple[SequenceIdentity, ...]:
        if name not in VALID_SPLITS:
            raise ValueError(f"unknown split: {name}")
        return tuple(entry.identity for entry in self.entries if entry.split == name)

    def split_for(self, dataset: str, sequence: str) -> str:
        identity = SequenceIdentity(dataset, sequence)
        matches = [entry.split for entry in self.entries if entry.identity == identity]
        if len(matches) != 1:
            raise KeyError(f"unregistered physical sequence: {dataset}/{sequence}")
        return matches[0]

    def assert_run_membership(self, dataset: str, sequence: str, requested_split: str) -> None:
        actual = self.split_for(dataset, sequence)
        if actual != requested_split:
            raise ValueError(
                f"derived run inherits physical-sequence split {actual}, not {requested_split}"
            )


def _entries(partition: str, rows: Iterable[object]) -> list[SplitEntry]:
    result: list[SplitEntry] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{partition} entries must be mappings")
        dataset = str(row.get("dataset", "")).strip()
        sequence = str(row.get("sequence", "")).strip()
        if not dataset or not sequence:
            raise ValueError(f"{partition} entry requires dataset and sequence")
        result.append(SplitEntry(SequenceIdentity(dataset, sequence), partition))
    return result


def load_paper_split(path: str | Path) -> PaperSplit:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("split configuration must be a mapping")
    entries: list[SplitEntry] = []
    for partition in VALID_SPLITS:
        entries.extend(_entries(partition, raw.get(partition, [])))
    return PaperSplit(tuple(entries))
