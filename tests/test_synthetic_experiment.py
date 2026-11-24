from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_synthetic_experiment import (
    SyntheticExperimentConfig,
    run_experiment,
)


def test_synthetic_experiment_writes_reproducibility_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "motion_blur"
    summary = run_experiment(
        SyntheticExperimentConfig(scenario="motion_blur", seed=42, steps=8, dt=0.05),
        output,
    )

    assert summary.scenario == "motion_blur"
    assert summary.seed == 42
    assert summary.steps == 8
    for name in ("summary.json", "manifest.json", "REPORT.md", "risk_timeline.png", "trajectory.png"):
        assert (output / name).is_file()

    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert saved["seed"] == 42
    assert manifest["config"]["seed"] == 42
    assert manifest["claim_policy"] == "synthetic diagnostic only; not a real benchmark"


@pytest.mark.parametrize(
    "config",
    [
        SyntheticExperimentConfig(steps=0),
        SyntheticExperimentConfig(dt=0.0),
        SyntheticExperimentConfig(dt=float("nan")),
        SyntheticExperimentConfig(min_feature_count=10, visual_feature_target=5),
        SyntheticExperimentConfig(covariance_risk_scale=-0.1),
    ],
)
def test_synthetic_experiment_rejects_invalid_config(
    tmp_path: Path, config: SyntheticExperimentConfig
) -> None:
    with pytest.raises(ValueError):
        run_experiment(config, tmp_path / "invalid")
