"""Build the canonical Phase B prediction dataset from synthetic demo artifacts."""

from __future__ import annotations

import argparse
import json

from shield_vio.experiments.phase_b_dataset import build_synthetic_prediction_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("synthetic_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--failure-config",
        default="configs/paper/failure_synthetic_validation_v1.yaml",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = build_synthetic_prediction_dataset(
        args.synthetic_dir,
        args.output,
        failure_config=args.failure_config,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
