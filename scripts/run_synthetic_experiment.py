"""Run deterministic synthetic SHIELD-VIO experiments.

The synthetic experiment is a diagnostic scaffold for uncertainty monitoring and
safety-shield behaviour. It is not a real EuRoC/TUM-VI/KITTI benchmark and should
not be reported as hardware or dataset validation.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shield_vio.estimation.eskf import ErrorStateEKF
from shield_vio.estimation.state import VIOState
from shield_vio.frontend.features import visual_quality_score
from shield_vio.imu.model import imu_consistency
from shield_vio.safety.shield import SafetyShield
from shield_vio.uncertainty.metrics import covariance_report, normalized_risk
from shield_vio.visualization.plots import plot_risk_timeline, plot_trajectory

SCENARIOS = {
    "nominal",
    "motion_blur",
    "low_light",
    "feature_dropout",
    "imu_bias_drift",
    "aggressive_motion",
}


@dataclass(frozen=True)
class SyntheticExperimentConfig:
    """Configuration for a synthetic VIO degradation experiment."""

    scenario: str = "nominal"
    seed: int = 7
    steps: int = 120
    dt: float = 0.05
    visual_feature_target: int = 170
    min_feature_count: int = 5
    covariance_risk_scale: float = 0.08


@dataclass(frozen=True)
class SyntheticExperimentSummary:
    """Summary metrics for one deterministic synthetic run."""

    scenario: str
    seed: int
    steps: int
    mean_risk: float
    max_risk: float
    mean_visual_quality: float
    min_visual_quality: float
    shield_activations: int
    tracking_failures: int
    final_position_error_m: float
    benchmark_status: str = "synthetic_demo_not_real_benchmark"


def run(scenario: str, seed: int, out_dir: Path) -> dict[str, Any]:
    """Backward-compatible wrapper used by the existing README examples."""

    config = SyntheticExperimentConfig(scenario=scenario, seed=seed)
    return asdict(run_experiment(config, out_dir))


def run_experiment(
    config: SyntheticExperimentConfig,
    out_dir: Path,
) -> SyntheticExperimentSummary:
    """Run a deterministic synthetic SHIELD-VIO scenario and save artifacts.

    Args:
        config: Synthetic experiment parameters.
        out_dir: Output directory for plots, JSON summary, manifest, and report.

    Returns:
        Summary metrics for the synthetic diagnostic run.
    """

    if config.scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{config.scenario}'. Expected one of {sorted(SCENARIOS)}")

    rng = np.random.default_rng(config.seed)
    ekf = ErrorStateEKF()
    shield = SafetyShield()
    state = VIOState()
    time = np.arange(config.steps) * config.dt

    ground_truth: list[list[float]] = []
    estimates: list[NDArray[np.float64]] = []
    risk_values: list[float] = []
    visual_quality_values: list[float] = []
    shield_active: list[bool] = []
    tracking_failures = 0

    for step, timestamp in enumerate(time):
        acc, gyro = _synthetic_imu(timestamp, step, config, rng)
        state = ekf.propagate(state, acc, gyro, config.dt)

        feature_count, outlier_ratio = _synthetic_visual_quality_inputs(step, config)
        tracking_ok = feature_count > 10
        tracking_failures += int(not tracking_ok)

        visual_quality = visual_quality_score(feature_count, 8, 0.75, outlier_ratio)
        imu_quality = imu_consistency(acc, gyro).combined_score
        cov_risk = covariance_report(state.covariance, config.covariance_risk_scale).risk_score
        risk = normalized_risk(cov_risk, visual_quality, imu_quality)
        decision = shield.decide(risk, visual_quality, tracking_ok=tracking_ok)

        ground_truth.append([0.5 * timestamp, 0.1 * np.sin(timestamp), 0.0])
        estimates.append(state.position.copy())
        risk_values.append(float(risk))
        visual_quality_values.append(float(visual_quality))
        shield_active.append(bool(decision.active))

    out_dir.mkdir(parents=True, exist_ok=True)
    estimated_positions = np.asarray(estimates)
    ground_truth_positions = np.asarray(ground_truth)
    plot_risk_timeline(
        time,
        risk_values,
        visual_quality_values,
        shield.warning_threshold,
        out_dir / "risk_timeline.png",
    )
    plot_trajectory(estimated_positions, ground_truth_positions, out_dir / "trajectory.png")

    final_error = float(np.linalg.norm(estimated_positions[-1] - ground_truth_positions[-1]))
    summary = SyntheticExperimentSummary(
        scenario=config.scenario,
        seed=config.seed,
        steps=config.steps,
        mean_risk=float(np.mean(risk_values)),
        max_risk=float(np.max(risk_values)),
        mean_visual_quality=float(np.mean(visual_quality_values)),
        min_visual_quality=float(np.min(visual_quality_values)),
        shield_activations=int(np.sum(shield_active)),
        tracking_failures=tracking_failures,
        final_position_error_m=final_error,
    )

    _write_json(out_dir / "summary.json", asdict(summary))
    _write_json(out_dir / "manifest.json", _manifest(config, out_dir))
    _write_report(out_dir / "REPORT.md", summary)
    return summary


def _synthetic_imu(
    timestamp: float,
    step: int,
    config: SyntheticExperimentConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    acc = np.array([0.2 * np.sin(timestamp), 0.0, 9.81], dtype=np.float64)
    gyro = np.array([0.0, 0.0, 0.05], dtype=np.float64)
    progress = step / max(config.steps - 1, 1)

    if config.scenario == "imu_bias_drift":
        acc += np.array([0.03 * progress, 0.0, 0.0])
        gyro += np.array([0.0, 0.0, 0.01 * progress])
    elif config.scenario == "aggressive_motion":
        acc += rng.normal(0.0, 1.5, 3)
        gyro += rng.normal(0.0, 0.7, 3)
    return acc, gyro


def _synthetic_visual_quality_inputs(
    step: int,
    config: SyntheticExperimentConfig,
) -> tuple[int, float]:
    progress = step / max(config.steps - 1, 1)
    feature_count = config.visual_feature_target
    outlier_ratio = 0.05

    if config.scenario == "feature_dropout":
        feature_count = max(config.min_feature_count, int(config.visual_feature_target * (1.0 - progress)))
    elif config.scenario == "motion_blur":
        outlier_ratio = min(0.8, 0.05 + 0.7 * progress)
    elif config.scenario == "low_light":
        feature_count = max(20, int(config.visual_feature_target - 120 * progress))
        outlier_ratio = 0.2
    return feature_count, outlier_ratio


def _manifest(config: SyntheticExperimentConfig, out_dir: Path) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "out_dir": str(out_dir),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "claim_policy": "synthetic diagnostic only; not a real benchmark",
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - best effort outside git checkouts
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(path: Path, summary: SyntheticExperimentSummary) -> None:
    lines = [
        "# SHIELD-VIO Synthetic Experiment Report",
        "",
        "This report is generated automatically from `scripts/run_synthetic_experiment.py`.",
        "The scenario is synthetic and must not be reported as real VIO benchmark evidence.",
        "",
        "## Summary",
        "",
    ]
    for key, value in asdict(summary).items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="nominal", choices=sorted(SCENARIOS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--out", default="results/synthetic/nominal")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SyntheticExperimentConfig(
        scenario=args.scenario,
        seed=args.seed,
        steps=args.steps,
        dt=args.dt,
    )
    summary = run_experiment(config, Path(args.out))
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
