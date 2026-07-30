"""Compare batch benchmark reports and detect metric regressions."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricRegression:
    """One threshold violation for a sequence metric."""

    sequence: str
    metric: str
    baseline: float
    candidate: float
    absolute_change: float
    relative_change: float
    allowed_absolute: float
    allowed_relative: float


@dataclass(frozen=True)
class RegressionGuardResult:
    """Machine-readable result of a benchmark comparison."""

    passed: bool
    compared_sequences: int
    regressions: tuple[MetricRegression, ...]
    missing_sequences: tuple[str, ...]
    failed_sequences: tuple[str, ...]


def compare_benchmark_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    metrics: tuple[str, ...] = ("ate_rmse_m", "rpe_rmse_m"),
    max_relative_increase: float = 0.05,
    max_absolute_increase: float = 0.01,
) -> RegressionGuardResult:
    """Compare successful sequence metrics using absolute and relative tolerances.

    A metric is considered a regression only when its increase exceeds both the
    absolute and relative tolerance. Candidate failures and missing sequences
    always fail the guard.
    """

    if max_relative_increase < 0 or max_absolute_increase < 0:
        raise ValueError("regression tolerances must be non-negative")
    baseline_results = _index_results(baseline)
    candidate_results = _index_results(candidate)
    regressions: list[MetricRegression] = []
    missing: list[str] = []
    failed: list[str] = []
    compared = 0

    for sequence, baseline_result in sorted(baseline_results.items()):
        if baseline_result.get("status") != "ok":
            continue
        candidate_result = candidate_results.get(sequence)
        if candidate_result is None:
            missing.append(sequence)
            continue
        if candidate_result.get("status") != "ok":
            failed.append(sequence)
            continue
        compared += 1
        for metric in metrics:
            baseline_value = _metric_value(baseline_result, sequence, metric)
            candidate_value = _metric_value(candidate_result, sequence, metric)
            absolute_change = candidate_value - baseline_value
            relative_change = (
                absolute_change / baseline_value if baseline_value > 0 else float("inf")
            )
            if absolute_change > max_absolute_increase and relative_change > max_relative_increase:
                regressions.append(
                    MetricRegression(
                        sequence=sequence,
                        metric=metric,
                        baseline=baseline_value,
                        candidate=candidate_value,
                        absolute_change=absolute_change,
                        relative_change=relative_change,
                        allowed_absolute=max_absolute_increase,
                        allowed_relative=max_relative_increase,
                    )
                )

    return RegressionGuardResult(
        passed=not regressions and not missing and not failed,
        compared_sequences=compared,
        regressions=tuple(regressions),
        missing_sequences=tuple(missing),
        failed_sequences=tuple(failed),
    )


def load_benchmark_report(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a batch benchmark JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("benchmark report must contain a results list")
    return payload


def write_guard_result(result: RegressionGuardResult, path: str | Path) -> None:
    """Write a stable JSON artifact for CI systems."""

    payload = asdict(result)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _index_results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("benchmark report must contain a results list")
    indexed: dict[str, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("sequence"), str):
            raise ValueError("each benchmark result must contain a sequence name")
        sequence = result["sequence"]
        if sequence in indexed:
            raise ValueError(f"duplicate sequence in benchmark report: {sequence}")
        indexed[sequence] = result
    return indexed


def _metric_value(result: dict[str, Any], sequence: str, metric: str) -> float:
    value = result.get(metric)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric metric {metric} for sequence {sequence}")
    return float(value)
