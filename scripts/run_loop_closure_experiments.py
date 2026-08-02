"""Run reproducible multi-seed loop-closure drift experiments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from shield_vio.loop_closure_experiments import (
    LoopClosureExperimentConfig,
    run_loop_closure_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/loop_closure"))
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--odometry-noise", type=float, default=0.04)
    parser.add_argument("--loop-noise", type=float, default=0.02)
    parser.add_argument("--bias-x", type=float, default=0.025)
    parser.add_argument("--bias-y", type=float, default=-0.01)
    args = parser.parse_args()

    if args.num_seeds < 1:
        parser.error("--num-seeds must be positive")

    config = LoopClosureExperimentConfig(
        node_count=args.nodes,
        odometry_noise_std_m=args.odometry_noise,
        loop_noise_std_m=args.loop_noise,
        odometry_bias_m=(args.bias_x, args.bias_y, 0.0),
    )
    seeds = range(args.start_seed, args.start_seed + args.num_seeds)
    summary = run_loop_closure_experiment(seeds, config)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "seed",
                "odometry_rmse_m",
                "loop_corrected_rmse_m",
                "odometry_endpoint_error_m",
                "loop_corrected_endpoint_error_m",
                "improvement_fraction",
                "optimizer_weighted_rmse",
            ],
        )
        writer.writeheader()
        for trial in summary.trials:
            writer.writerow(trial.__dict__)

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
