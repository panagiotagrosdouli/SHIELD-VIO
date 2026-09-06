from __future__ import annotations

import csv
import json
from pathlib import Path

from shield_vio.experiments.phase_b_dataset import build_synthetic_prediction_dataset
from shield_vio.simulation.synthetic_vio import SyntheticVIOConfig, run_synthetic_vio


def test_phase_b_synthetic_pipeline_writes_trustworthy_dataset(tmp_path: Path) -> None:
    source = tmp_path / "synthetic"
    output = tmp_path / "phase_b"
    run_synthetic_vio(
        SyntheticVIOConfig(
            seed=7,
            duration_s=3.0,
            dt=0.1,
            visual_dt=0.2,
            output_dir=str(source),
            events=[],
        )
    )
    failure_config = Path(__file__).parents[1] / "configs/paper/failure_synthetic_validation_v1.yaml"
    manifest = build_synthetic_prediction_dataset(
        source,
        output,
        failure_config=failure_config,
        seed=7,
    )
    assert manifest["evidence_level"] == "SYNTHETIC_PIPELINE_VALIDATION"
    assert not manifest["confirmatory"]
    required = {
        "health_samples.csv",
        "prediction_targets.csv",
        "prediction_dataset.csv",
        "diagnostic_report.json",
        "manifest.json",
    }
    assert required <= {path.name for path in output.iterdir()}

    with (output / "prediction_dataset.csv").open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    assert not any(
        token in name
        for name in header
        for token in ("ground_truth", "nees", "degradation", "severity", "future_failure")
    )
    report = json.loads((output / "diagnostic_report.json").read_text(encoding="utf-8"))
    assert report["prohibited_fields_excluded_from_predictor_matrix"]
    assert report["samples"] == manifest["samples"]


def test_phase_b_synthetic_export_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "synthetic"
    run_synthetic_vio(
        SyntheticVIOConfig(seed=11, duration_s=2.0, dt=0.1, output_dir=str(source), events=[])
    )
    config = Path(__file__).parents[1] / "configs/paper/failure_synthetic_validation_v1.yaml"
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_synthetic_prediction_dataset(source, first, failure_config=config, seed=11)
    build_synthetic_prediction_dataset(source, second, failure_config=config, seed=11)
    for name in (
        "health_samples.csv",
        "prediction_targets.csv",
        "prediction_dataset.csv",
        "diagnostic_report.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
