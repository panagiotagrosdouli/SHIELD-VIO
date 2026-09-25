from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.aggregate_paper_metrics import aggregate, load_metric_table


FIELDS = [
    "dataset", "sequence", "estimator", "degradation_condition", "seed", "method", "auprc"
]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(sequence: str, method: str, value: float, seed: int = 0) -> dict[str, object]:
    return {
        "dataset": "euroc",
        "sequence": sequence,
        "estimator": "eskf",
        "degradation_condition": "nominal",
        "seed": seed,
        "method": method,
        "auprc": value,
    }


def test_aggregate_preserves_exact_pairing_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _write(path, [
        _row("MH_01", "proposed", 0.8), _row("MH_01", "heuristic", 0.6),
        _row("MH_02", "proposed", 0.9), _row("MH_02", "heuristic", 0.7),
    ])
    report = aggregate(
        path, method_a="proposed", method_b="heuristic", metric="auprc",
        higher_is_better=True, n_bootstrap=500, seed=3,
    )
    assert report["schema_version"] == "SHIELD_VIO_PAIRED_AGGREGATE_V1"
    comparison = report["comparison"]
    assert comparison["paired_unit_count"] == 2
    assert comparison["mean_difference"] == pytest.approx(0.2)


def test_aggregate_rejects_unpaired_runs(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _write(path, [
        _row("MH_01", "proposed", 0.8), _row("MH_01", "heuristic", 0.6),
        _row("MH_02", "proposed", 0.9),
    ])
    with pytest.raises(ValueError, match="unpaired experimental units"):
        aggregate(
            path, method_a="proposed", method_b="heuristic", metric="auprc",
            higher_is_better=True, n_bootstrap=100, seed=0,
        )


def test_loader_rejects_duplicate_complete_run_key(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _write(path, [_row("MH_01", "proposed", 0.8), _row("MH_01", "proposed", 0.9)])
    with pytest.raises(ValueError, match="duplicate experimental unit"):
        load_metric_table(path, method_field="method", metric="auprc")


def test_seed_is_part_of_pairing_key(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    _write(path, [
        _row("MH_01", "proposed", 0.8, seed=0), _row("MH_01", "heuristic", 0.6, seed=0),
        _row("MH_01", "proposed", 0.9, seed=1), _row("MH_01", "heuristic", 0.7, seed=1),
    ])
    methods = load_metric_table(path, method_field="method", metric="auprc")
    assert len(methods["proposed"]) == 2
