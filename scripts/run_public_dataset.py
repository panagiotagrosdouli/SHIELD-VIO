#!/usr/bin/env python3
"""Execute a local EuRoC/TUM-VI sequence through the common Phase C smoke path."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from shield_vio.datasets.provenance import RunIdentity, environment_provenance
from shield_vio.datasets.public import validate_public_sequence
from shield_vio.experiments.euroc_runner import run_euroc_sequence
from shield_vio.experiments.public_prediction import build_public_prediction_artifacts
from shield_vio.experiments.tumvi_runner import run_tumvi_sequence


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("euroc", "tumvi"), required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--failure-config", type=Path, default=Path("configs/paper/failure_public_smoke_v1.yaml"))
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    validated = validate_public_sequence(args.dataset, args.root)
    if validated.sequence_name != args.sequence:
        raise SystemExit(f"sequence identity mismatch: requested {args.sequence!r}, local root is {validated.sequence_name!r}")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = RunIdentity(args.dataset, args.sequence, "internal_eskf")
    provenance = {
        "run_id": identity.run_id(), "git_commit": _git_commit(),
        "environment": environment_provenance(), "command": " ".join(sys.argv),
        "working_directory": os.getcwd(), "start_timestamp_utc": started,
        "estimator_configuration": "internal_eskf_default",
    }
    validation = {
        "evidence_level": "DATASET_VALIDATED", "confirmatory": False,
        "dataset": validated.dataset_name, "dataset_source": "local user-provided public dataset copy",
        "sequence": validated.sequence_name, "camera_rows": validated.camera_rows,
        "imu_rows": validated.imu_rows, "ground_truth_rows": validated.ground_truth_rows,
        "dataset_fingerprint": validated.fingerprint, **provenance,
    }
    (args.output / "dataset_validation_manifest.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True)); return
    if validated.sequence.ground_truth_csv is None:
        raise SystemExit("PUBLIC_DATASET_SMOKE requires ground truth/mocap for observable label construction")
    raw_dir = args.output / "estimator"
    if args.dataset == "euroc":
        run_euroc_sequence(args.root, raw_dir, evaluate=True)
    else:
        run_tumvi_sequence(args.root, raw_dir)
    canonical = build_public_prediction_artifacts(
        raw_dir, validated.sequence.ground_truth_csv, args.output,
        dataset=args.dataset, sequence=args.sequence, estimator="internal_eskf", failure_config=args.failure_config,
    )
    manifest = {
        **validation, **canonical, "dataset_fingerprint": validated.fingerprint,
        "degradation_condition": "clean", "degradation_severity": "none", "seed": 0,
        "temporal_windows": "Phase B canonical defaults", "end_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": ["estimator/trajectory.csv", "estimator/health.csv", "failure_events.csv", "prediction_dataset.csv", "canonical_manifest.json"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
