from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from shield_vio.experiments.paper_vertical_slice import (
    VerticalSliceConfig,
    run_public_dataset_smoke,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _fixture_run(root: Path) -> tuple[Path, Path]:
    run_dir = root / "run"
    sequence = root / "MH_fixture"
    timestamps = np.arange(40, dtype=np.int64) * 500_000_000 + 1_000_000_000
    ground_truth_positions = np.column_stack(
        [np.arange(40, dtype=float) * 0.1, np.zeros(40), np.zeros(40)]
    )
    error = np.zeros(40)
    error[16:23] = 2.0
    error[30:36] = -2.0
    estimated = ground_truth_positions.copy()
    estimated[:, 1] += error

    trajectory_rows = []
    health_rows = []
    visual_rows = []
    for index, timestamp in enumerate(timestamps):
        covariance = 0.1 + (1.2 if 13 <= index <= 22 or 27 <= index <= 35 else 0.0)
        nis = 16.0 if 13 <= index <= 22 or 27 <= index <= 35 else 2.0
        tracked = 8 if 13 <= index <= 22 or 27 <= index <= 35 else 80
        trajectory_rows.append(
            [
                timestamp,
                timestamp,
                *estimated[index],
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        health_rows.append([timestamp, timestamp, True, "tracking", index + 1, covariance, 10, nis])
        visual_rows.append([timestamp, "accepted", 100, tracked, tracked, tracked - 1, 0.8, nis])

    _write_csv(
        run_dir / "trajectory.csv",
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
        trajectory_rows,
    )
    _write_csv(
        run_dir / "health.csv",
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
        health_rows,
    )
    _write_csv(
        run_dir / "visual_updates.csv",
        [
            "frame_timestamp_ns",
            "status",
            "detected_features",
            "tracked_features",
            "correspondence_count",
            "inlier_count",
            "inlier_ratio",
            "innovation_nis",
        ],
        visual_rows,
    )
    (run_dir / "experiment_manifest.json").write_text("{}\n", encoding="utf-8")

    ground_truth = sequence / "mav0/state_groundtruth_estimate0/data.csv"
    _write_csv(
        ground_truth,
        ["#timestamp", "p_x", "p_y", "p_z"],
        [[timestamp, *position] for timestamp, position in zip(timestamps, ground_truth_positions)],
    )
    _write_csv(sequence / "mav0/cam0/data.csv", ["#timestamp", "filename"], [[1, "1.png"]])
    _write_csv(
        sequence / "mav0/imu0/data.csv",
        ["#timestamp", "wx", "wy", "wz", "ax", "ay", "az"],
        [[1, 0, 0, 0, 0, 0, 9.81]],
    )
    return run_dir, sequence


def test_vertical_slice_writes_complete_nonconfirmatory_artifacts(tmp_path: Path) -> None:
    run_dir, sequence = _fixture_run(tmp_path)
    output = tmp_path / "output"
    manifest = run_public_dataset_smoke(
        run_dir,
        sequence,
        output,
        config=VerticalSliceConfig(
            horizon_seconds=1.0,
            persistence_seconds=0.5,
            max_ground_truth_gap_seconds=0.6,
            synthetic_train_samples=200,
            synthetic_calibration_samples=100,
            synthetic_validation_samples=100,
        ),
        command="fixture vertical slice",
    )

    assert manifest["evidence_level"] == "PUBLIC_DATASET_SMOKE"
    assert not manifest["confirmatory"]
    assert manifest["status"] == "complete"
    assert manifest["sample_counts"]["failure_events"] == 2
    required = {
        "health_features.csv",
        "predictions.csv",
        "metrics.json",
        "model.json",
        "calibration.json",
        "reliability_source.csv",
        "reliability_diagram.pdf",
        "reliability_diagram.svg",
        "prediction_timeline_source.csv",
        "prediction_timeline.pdf",
        "prediction_timeline.svg",
        "experiment_manifest.json",
    }
    assert required <= {path.name for path in output.iterdir()}
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert {
        "covariance_trace",
        "feature_count",
        "innovation_nis",
        "logistic_raw",
        "logistic_platt",
    } <= set(metrics)
    assert 0.0 <= metrics["logistic_platt"]["auprc"] <= 1.0
