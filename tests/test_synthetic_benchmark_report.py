from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_synthetic_benchmark_report.py"
SPEC = importlib.util.spec_from_file_location("build_synthetic_benchmark_report", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_trajectory(path: Path, rows: list[tuple[float, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "z"])
        writer.writeheader()
        for x, y, z in rows:
            writer.writerow({"x": x, "y": y, "z": z})


def test_build_report_emits_batch_schema(tmp_path: Path) -> None:
    reference = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    estimated = [(0.1, 0.0, 0.0), (1.1, 0.0, 0.0), (2.1, 0.0, 0.0)]
    _write_trajectory(tmp_path / "ground_truth.csv", reference)
    _write_trajectory(tmp_path / "estimated_trajectory.csv", estimated)

    report = MODULE.build_report(tmp_path)

    assert report["success_count"] == 1
    assert report["failure_count"] == 0
    result = report["results"][0]
    assert result["sequence"] == "synthetic_seed_7"
    assert result["status"] == "ok"
    assert result["sample_count"] == 3
    assert result["ate_rmse_m"] == pytest.approx(0.0, abs=1e-12)
    assert result["rpe_rmse_m"] == pytest.approx(0.0, abs=1e-12)
