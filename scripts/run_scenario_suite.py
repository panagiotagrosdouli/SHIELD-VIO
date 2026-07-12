"""Run repeated synthetic SHIELD-VIO scenarios with identical detector seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.evaluation.statistics import summarize
from shield_vio.simulation.synthetic_vio import SyntheticVIOConfig, run_synthetic_vio


def run_suite(num_seeds: int, output_dir: Path) -> dict[str, object]:
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    successful: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for seed in range(num_seeds):
        run_dir = output_dir / f"seed_{seed:04d}"
        try:
            result = run_synthetic_vio(
                SyntheticVIOConfig(seed=seed, output_dir=str(run_dir))
            )
            successful.append(result)
        except Exception as exc:  # continue-after-failure is intentional for experiment batches
            failed.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})

    metric_names = (
        "ate_rmse_m",
        "final_position_error_m",
        "max_risk",
        "mean_visual_quality",
        "shield_activation_rate",
        "failure_detection_precision",
        "failure_detection_recall",
    )
    aggregates: dict[str, object] = {}
    for name in metric_names:
        values = [float(run[name]) for run in successful if name in run]
        if values:
            aggregates[name] = summarize(values)

    report: dict[str, object] = {
        "label": "Synthetic Validation",
        "requested_runs": num_seeds,
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "seeds": list(range(num_seeds)),
        "aggregates": aggregates,
        "failures": failed,
        "disclaimer": "Synthetic validation only; not a public-dataset benchmark.",
    }
    (output_dir / "suite_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/scenario_suite"))
    args = parser.parse_args()
    report = run_suite(args.num_seeds, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["successful_runs"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
