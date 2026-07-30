"""Batch EuRoC benchmark aggregation utilities."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from shield_vio.evaluation.euroc import (
    associate_ground_truth,
    load_estimator_trajectory,
    load_euroc_ground_truth,
)
from shield_vio.evaluation.trajectory_metrics import summarize_trajectory_metrics


@dataclass(frozen=True)
class SequenceBenchmarkResult:
    """Result for one EuRoC sequence, including recoverable failures."""

    sequence: str
    status: str
    estimate_path: str
    sample_count: int | None = None
    ate_rmse_m: float | None = None
    ate_mean_m: float | None = None
    ate_median_m: float | None = None
    ate_max_m: float | None = None
    rpe_rmse_m: float | None = None
    rpe_mean_m: float | None = None
    rpe_median_m: float | None = None
    rpe_max_m: float | None = None
    rpe_delta: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchBenchmarkReport:
    """Machine-readable collection of per-sequence benchmark results."""

    results: tuple[SequenceBenchmarkResult, ...]

    @property
    def success_count(self) -> int:
        return sum(result.status == "ok" for result in self.results)

    @property
    def failure_count(self) -> int:
        return len(self.results) - self.success_count


def evaluate_euroc_sequence(
    sequence_root: str | Path,
    estimate_path: str | Path,
    *,
    timestamp_unit: str = "seconds",
    max_gap: float = 0.02,
    rpe_delta: int = 1,
    align: bool = True,
) -> SequenceBenchmarkResult:
    """Evaluate one sequence and return a failure record instead of aborting a batch."""

    sequence_path = Path(sequence_root)
    estimate = Path(estimate_path)
    try:
        estimate_trajectory = load_estimator_trajectory(
            estimate, timestamp_unit=timestamp_unit
        )
        ground_truth = load_euroc_ground_truth(sequence_path)
        _, estimated_positions, reference_positions = associate_ground_truth(
            estimate_trajectory, ground_truth, max_gap=max_gap
        )
        metrics = summarize_trajectory_metrics(
            estimated_positions,
            reference_positions,
            rpe_delta=rpe_delta,
            align=align,
        )
    except (OSError, ValueError) as exc:
        return SequenceBenchmarkResult(
            sequence=sequence_path.name,
            status="failed",
            estimate_path=str(estimate),
            error=str(exc),
        )

    return SequenceBenchmarkResult(
        sequence=sequence_path.name,
        status="ok",
        estimate_path=str(estimate),
        **asdict(metrics),
    )


def evaluate_euroc_batch(
    entries: Iterable[tuple[str | Path, str | Path]],
    **kwargs: object,
) -> BatchBenchmarkReport:
    """Evaluate ``(sequence_root, estimate_path)`` pairs in deterministic order."""

    results = tuple(
        evaluate_euroc_sequence(sequence_root, estimate_path, **kwargs)
        for sequence_root, estimate_path in entries
    )
    return BatchBenchmarkReport(results=results)


def write_batch_json(report: BatchBenchmarkReport, path: str | Path) -> None:
    """Write a stable JSON benchmark artifact."""

    payload = {
        "success_count": report.success_count,
        "failure_count": report.failure_count,
        "results": [asdict(result) for result in report.results],
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_batch_csv(report: BatchBenchmarkReport, path: str | Path) -> None:
    """Write a flat CSV benchmark artifact suitable for spreadsheets and CI."""

    rows = [asdict(result) for result in report.results]
    fieldnames = list(SequenceBenchmarkResult.__dataclass_fields__)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
