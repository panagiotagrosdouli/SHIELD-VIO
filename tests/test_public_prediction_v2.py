from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from shield_vio.experiments.public_prediction_v2 import (
    build_public_v2_prediction_artifacts,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    sequence = tmp_path / "MH_01_easy"
    timestamps = np.arange(16, dtype=np.int64) * 500_000_000 + 1_000_000_000

    trajectory_rows: list[list[object]] = []
    health_rows: list[list[object]] = []
    gt_rows: list[list[object]] = []
    for index, timestamp in enumerate(timestamps):
        gt_position = np.asarray([0.2 * index, 0.05 * index, 0.0])
        estimate = gt_position.copy()
        if index >= 10:
            estimate += np.asarray([2.0 + 0.2 * (index - 10), 0.0, 0.0])
        trajectory_rows.append(
            [
                timestamp,
                timestamp,
                *estimate.tolist(),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.01,
                0.0,
                0.0,
                0.001,
                0.0,
                0.0,
            ]
        )
        health_rows.append(
            [
                timestamp,
                timestamp,
                1,
                "tracking",
                100 + index,
                0.1 + 0.01 * index,
                10.0,
                1.0 + 0.1 * index,
                1e-4,
                0.0,
                0,
                0,
                0,
            ]
        )
        gt_rows.append(
            [timestamp, *gt_position.tolist(), 1.0, 0.0, 0.0, 0.0]
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
        trajectory_rows,
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
        health_rows,
    )
    (run / "experiment_manifest.json").write_text(
        json.dumps({"visual_provider": None}) + "\n",
        encoding="utf-8",
    )

    imu_times = np.arange(
        timestamps[0] - 100_000_000,
        timestamps[-1] + 1,
        20_000_000,
        dtype=np.int64,
    )
    _write_csv(
        sequence / "mav0/imu0/data.csv",
        [
            "#timestamp [ns]",
            "w_RS_S_x [rad s^-1]",
            "w_RS_S_y [rad s^-1]",
            "w_RS_S_z [rad s^-1]",
            "a_RS_S_x [m s^-2]",
            "a_RS_S_y [m s^-2]",
            "a_RS_S_z [m s^-2]",
        ],
        [
            [timestamp, 0.02, 0.01, 0.0, 0.0, 0.0, 9.81]
            for timestamp in imu_times
        ],
    )
    _write_csv(
        sequence / "mav0/state_groundtruth_estimate0/data.csv",
        ["#timestamp", "px", "py", "pz", "qw", "qx", "qy", "qz"],
        gt_rows,
    )
    return run, sequence


def test_public_v2_canonical_uses_deployable_features_and_v2_targets(
    tmp_path: Path,
) -> None:
    run, sequence = _fixture(tmp_path)
    output = tmp_path / "canonical"
    manifest = build_public_v2_prediction_artifacts(
        run,
        sequence,
        output,
        dataset="euroc",
        sequence="MH_01_easy",
        estimator="internal_eskf",
        failure_config=Path(__file__).parents[1]
        / "configs/paper/failure_primary_v2.yaml",
        evidence_level="PUBLIC_DATASET_DEVELOPMENT",
    )

    assert manifest["failure_definition_id"] == "SHIELD_VIO_FAILURE_V2"
    assert manifest["evidence_level"] == "PUBLIC_DATASET_DEVELOPMENT"
    assert manifest["confirmatory"] is False
    assert manifest["ground_truth_is_predictor"] is False
    assert manifest["samples"] == 16
    assert manifest["feature_count"] > 20
    assert manifest["complete_observability_samples"] > 0

    with (output / "prediction_dataset.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.reader(stream)
        header = next(reader)
    assert "estimator.nis" in header
    assert "inertial.accel_norm_m_s2" in header
    assert "inertial.accel_bias_norm_m_s2" in header
    assert "visual.frame_available" in header
    assert not any("ground_truth" in name for name in header)

    with (output / "prediction_targets.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        target_header = next(csv.reader(stream))
    for horizon in (0.5, 1.0, 2.0, 3.0, 5.0):
        assert f"failure_within_{horizon}s" in target_header
        assert f"eligible_{horizon}s" in target_header

    serialized = json.loads(
        (output / "canonical_manifest.json").read_text(encoding="utf-8")
    )
    assert len(serialized["feature_schema_sha256"]) == 64
    assert set(serialized["artifact_sha256"]) == {
        "prediction_dataset.csv",
        "prediction_targets.csv",
        "failure_events.csv",
    }


def test_public_v2_feature_sources_are_causal(tmp_path: Path) -> None:
    run, sequence = _fixture(tmp_path)
    # A future estimator state timestamp must fail through HealthValue validation.
    health_path = run / "health.csv"
    with health_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0])
    rows[3]["state_timestamp_ns"] = str(int(rows[3]["frame_timestamp_ns"]) + 1)
    with health_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    try:
        build_public_v2_prediction_artifacts(
            run,
            sequence,
            tmp_path / "bad",
            dataset="euroc",
            sequence="MH_01_easy",
            estimator="internal_eskf",
            failure_config=Path(__file__).parents[1]
            / "configs/paper/failure_primary_v2.yaml",
            evidence_level="FIXTURE_INTEGRATION_TEST",
        )
    except ValueError as exc:
        assert "future" in str(exc)
    else:
        raise AssertionError("future estimator state timestamp was not rejected")
