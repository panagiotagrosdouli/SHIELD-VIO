import json
from pathlib import Path

from scripts.run_synthetic_experiment import SyntheticExperimentConfig, run_experiment


def test_synthetic_experiment_writes_reproducibility_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "motion_blur"
    summary = run_experiment(
        SyntheticExperimentConfig(scenario="motion_blur", seed=3, steps=8, dt=0.05),
        out_dir,
    )

    assert summary.scenario == "motion_blur"
    assert summary.steps == 8
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "REPORT.md").exists()
    assert (out_dir / "risk_timeline.png").exists()
    assert (out_dir / "trajectory.png").exists()

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["claim_policy"] == "synthetic diagnostic only; not a real benchmark"
