import json
from pathlib import Path

import pytest

from shield_vio.evaluation.regression_guard import (
    compare_benchmark_reports,
    load_benchmark_report,
    write_guard_result,
)


def _report(*results: dict[str, object]) -> dict[str, object]:
    return {"success_count": len(results), "failure_count": 0, "results": list(results)}


def _ok(sequence: str, ate: float, rpe: float) -> dict[str, object]:
    return {
        "sequence": sequence,
        "status": "ok",
        "ate_rmse_m": ate,
        "rpe_rmse_m": rpe,
    }


def test_guard_passes_small_metric_noise() -> None:
    baseline = _report(_ok("MH_01_easy", 0.10, 0.05))
    candidate = _report(_ok("MH_01_easy", 0.104, 0.052))

    result = compare_benchmark_reports(
        baseline,
        candidate,
        max_relative_increase=0.05,
        max_absolute_increase=0.01,
    )

    assert result.passed
    assert result.compared_sequences == 1
    assert result.regressions == ()


def test_guard_detects_ate_regression() -> None:
    baseline = _report(_ok("MH_02_easy", 0.10, 0.05))
    candidate = _report(_ok("MH_02_easy", 0.13, 0.05))

    result = compare_benchmark_reports(baseline, candidate)

    assert not result.passed
    assert len(result.regressions) == 1
    assert result.regressions[0].metric == "ate_rmse_m"
    assert result.regressions[0].absolute_change == pytest.approx(0.03)


def test_guard_fails_missing_and_failed_candidate_sequences() -> None:
    baseline = _report(
        _ok("MH_01_easy", 0.10, 0.05),
        _ok("V1_01_easy", 0.20, 0.08),
    )
    candidate = _report(
        {
            "sequence": "MH_01_easy",
            "status": "failed",
            "error": "trajectory missing",
        }
    )

    result = compare_benchmark_reports(baseline, candidate)

    assert not result.passed
    assert result.failed_sequences == ("MH_01_easy",)
    assert result.missing_sequences == ("V1_01_easy",)


def test_guard_requires_both_tolerances_to_be_exceeded() -> None:
    baseline = _report(_ok("V2_01_easy", 10.0, 1.0))
    candidate = _report(_ok("V2_01_easy", 10.02, 1.0))

    result = compare_benchmark_reports(
        baseline,
        candidate,
        max_relative_increase=0.05,
        max_absolute_increase=0.01,
    )

    assert result.passed


def test_guard_result_round_trip(tmp_path: Path) -> None:
    result = compare_benchmark_reports(
        _report(_ok("MH_01_easy", 0.1, 0.05)),
        _report(_ok("MH_01_easy", 0.1, 0.05)),
    )
    path = tmp_path / "guard.json"

    write_guard_result(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["compared_sequences"] == 1


def test_load_report_rejects_invalid_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="results list"):
        load_benchmark_report(path)
