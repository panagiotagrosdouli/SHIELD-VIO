#!/usr/bin/env python3
"""Build fail-closed SHIELD_VIO_FAILURE_V2 events and future targets for one public run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.evaluation.failure_definition import load_failure_definition
from shield_vio.evaluation.primary_failure_pipeline import (
    build_primary_failure_bundle,
    write_primary_failure_artifacts,
)
from shield_vio.evaluation.primary_observables import (
    build_primary_observables,
    write_primary_observable_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--failure-config",
        type=Path,
        default=Path("configs/paper/failure_primary_v2.yaml"),
    )
    parser.add_argument("--max-ground-truth-gap-seconds", type=float, default=0.02)
    parser.add_argument("--rpe-interval-seconds", type=float, default=1.0)
    parser.add_argument("--rpe-pair-tolerance-seconds", type=float, default=0.075)
    args = parser.parse_args()

    definition = load_failure_definition(args.failure_config)
    if definition.schema_version != "SHIELD_VIO_FAILURE_V2":
        raise ValueError("this entrypoint requires SHIELD_VIO_FAILURE_V2")

    table = build_primary_observables(
        args.run_dir,
        args.sequence_root,
        max_ground_truth_gap_seconds=args.max_ground_truth_gap_seconds,
        rpe_interval_seconds=args.rpe_interval_seconds,
        rpe_pair_tolerance_seconds=args.rpe_pair_tolerance_seconds,
        failure_definition=definition,
    )
    audit_dir = args.output / "observable_audit"
    audit = write_primary_observable_artifacts(table, audit_dir)
    if audit["status"] != "READY_FOR_PRIMARY_LABEL_BUILD":
        raise RuntimeError(
            "primary failure labels remain blocked: "
            + ", ".join(audit["blocking_criteria"])
        )

    bundle = build_primary_failure_bundle(table, definition)
    manifest = write_primary_failure_artifacts(
        bundle,
        table,
        args.output,
        failure_config=args.failure_config,
        source_run_dir=args.run_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
