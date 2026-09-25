#!/usr/bin/env python3
"""Build a split-checked canonical public V2 development artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.datasets.splits import load_paper_split
from shield_vio.experiments.public_prediction_v2 import (
    build_public_v2_prediction_artifacts,
)


DEVELOPMENT_SPLITS = ("train", "calibration", "validation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="euroc")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--estimator", default="internal_eskf")
    parser.add_argument("--split", choices=DEVELOPMENT_SPLITS, required=True)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/paper/public_dataset_splits.yaml"),
    )
    parser.add_argument(
        "--failure-config",
        type=Path,
        default=Path("configs/paper/failure_primary_v2.yaml"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--degradation-condition", default="clean")
    args = parser.parse_args()

    split = load_paper_split(args.split_config)
    split.assert_run_membership(args.dataset, args.sequence, args.split)

    manifest = build_public_v2_prediction_artifacts(
        args.run_dir,
        args.sequence_root,
        args.output,
        dataset=args.dataset,
        sequence=args.sequence,
        estimator=args.estimator,
        failure_config=args.failure_config,
        evidence_level="PUBLIC_DATASET_DEVELOPMENT",
        seed=args.seed,
        degradation_condition=args.degradation_condition,
    )
    outer = {
        **manifest,
        "split": args.split,
        "split_config": str(args.split_config),
        "degradation_condition": args.degradation_condition,
        "seed": args.seed,
        "test_partition_loaded": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(outer, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(outer, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
