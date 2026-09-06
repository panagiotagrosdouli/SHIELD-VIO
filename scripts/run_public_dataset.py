#!/usr/bin/env python3
"""Validate a local EuRoC/TUM-VI export and record deterministic identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.datasets.public import validate_public_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("euroc", "tumvi"), required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validated = validate_public_sequence(args.dataset, args.root)
    if validated.sequence_name != args.sequence:
        raise SystemExit(
            f"sequence identity mismatch: requested {args.sequence!r}, local root is {validated.sequence_name!r}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "evidence_level": "DATASET_VALIDATED",
        "confirmatory": False,
        "dataset": validated.dataset_name,
        "sequence": validated.sequence_name,
        "camera_rows": validated.camera_rows,
        "imu_rows": validated.imu_rows,
        "ground_truth_rows": validated.ground_truth_rows,
        "dataset_fingerprint": validated.fingerprint,
        "note": "Validation only. PUBLIC_DATASET_SMOKE requires estimator execution plus canonical health/targets.",
    }
    (args.output / "dataset_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
