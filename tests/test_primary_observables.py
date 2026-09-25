from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from shield_vio.evaluation.primary_observables import (
    PRIMARY_CRITERIA,
    build_primary_observables,
    write_primary_observable_artifacts,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    sequence = tmp_path / "MH_fixture"
    timestamps = np.arange(8, dtype=np.int64) * 500_000_000 + 1_000_000_000
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.1, 0.0],
            [0.6, 0.2, 0.1],
            [0.9, 0.4, 0.1],
            [1.2, 0.7, 0.2],
            [1.4, 1.0, 0.3],
            [1.5, 1.3, 0.4],
            [1.6, 1.6, 0.5],
        ]
    )
    trajectory = []
    health = []
    for index, (timestamp, position) in enumerate(zip(timestamps, positions, strict=True)):
        state_timestamp = timestamp - (600_000_000 if index == 6 else 0)
        trajectory.append(
            [
                timestamp,
                state_timestamp,
                *position,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        health.append(
            [
                timestamp,
                state_timestamp,
                True,
                "lost" if index == 7 else "tracking",
                index + 1,
                0.1,
                10.0,
                2.0,
                1e-4,
                0.0,
                1,
                0,
                0,
            ]
        )

    _write_csv(
        run / "trajectory.csv",
        [
            "frame_timestamp_ns",
            "state_timestamp_ns",
            "px",
            "py",
            "pz",
            "vx",
            "vy",
            "vz",
            "qw",
            "qx",
            "qy",
            "qz",
            "bax",
            "bay",
            "baz",
            "bgx",
            "bgy",
            "bgz",
        ],
        trajectory,
    )
    _write_csv(
        run / "health.csv",
        [
            "frame_timestamp_ns",
            "state_timestamp_ns",
            "initialized",
            "tracking_status",
            "propagated_imu_samples",
            "covariance_trace",
            "covariance_condition_number",
            "innovation_nis",
            "covariance_min_eigenvalue",
            "covariance_symmetry_error",
            "terminal_tracking_loss_observable",
            "reset_event",
            "relocalization_event",
        ],
        health,
    )
    gt = sequence / "mav0/state_groundtruth_estimate0/data.csv"
    _write_csv(
        gt,
        ["#timestamp", "px", "py", "pz", "qw", "qx", "qy", "qz"],
        [[timestamp, *position, 1.0, 0.0, 0.0, 0.0] for timestamp, position in zip(timestamps, positions, strict=True)],
    )
    return run, sequence


def test_primary_observable_export_computes_available_criteria(tmp_path: Path) -> None:
    run, sequence = _fixture(tmp_path)
    table = build_primary_observables(
        run,
        sequence,
        max_ground_truth_gap_seconds=0.01,
        rpe_interval_seconds=1.0,
        rpe_pair_tolerance_seconds=0.01,
    )

    assert set(table.values) == set(PRIMARY_CRITERIA)
    assert np.nanmax(np.abs(table.values["position_error_m"])) < 1e-9
    assert np.nanmax(np.abs(table.values["orientation_error_deg"])) < 1e-9
    assert np.nanmax(np.abs(table.values["translation_rpe_1s_m"])) < 1e-9
    assert np.nanmax(np.abs(table.values["rotation_rpe_1s_deg"])) < 1e-9
    assert table.values["output_starvation"][6] == 0.6
    assert bool(table.values["terminal_tracking_loss"][7])
    assert np.all(table.observable["terminal_tracking_loss"])
    assert np.all(table.observable["invalid_pose_or_covariance"])
    assert not np.any(table.observable["visual_update_starvation_while_motion"])

    report = write_primary_observable_artifacts(table, tmp_path / "audit")
    assert report["status"] == "BLOCKED"
    assert report["blocking_criteria"] == ["visual_update_starvation_while_motion"]
    serialized = json.loads(
        (tmp_path / "audit/primary_observable_audit.json").read_text(encoding="utf-8")
    )
    assert serialized["criteria"]["translation_rpe_1s_m"]["observable_samples"] == 6


def test_primary_observable_audit_marks_old_covariance_schema_unobservable(
    tmp_path: Path,
) -> None:
    run, sequence = _fixture(tmp_path)
    health_path = run / "health.csv"
    with health_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _write_csv(
        health_path,
        [
            "frame_timestamp_ns",
            "state_timestamp_ns",
            "initialized",
            "tracking_status",
            "propagated_imu_samples",
            "covariance_trace",
            "covariance_condition_number",
            "innovation_nis",
        ],
        [
            [
                row["frame_timestamp_ns"],
                row["state_timestamp_ns"],
                row["initialized"],
                row["tracking_status"],
                row["propagated_imu_samples"],
                row["covariance_trace"],
                row["covariance_condition_number"],
                row["innovation_nis"],
            ]
            for row in rows
        ],
    )

    table = build_primary_observables(run, sequence, max_ground_truth_gap_seconds=0.01)
    assert not np.any(table.observable["invalid_pose_or_covariance"])
    assert not np.any(table.observable["estimator_reset_or_unrecovered_relocalization"])
