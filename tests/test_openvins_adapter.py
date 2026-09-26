from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from shield_vio.experiments.openvins_adapter import import_openvins_total_state


def _write_camera(sequence: Path) -> None:
    root = sequence / "mav0/cam0"
    data = root / "data"
    data.mkdir(parents=True)
    rows = []
    for timestamp in (
        1_000_000_000,
        1_500_000_000,
        2_000_000_000,
        2_500_000_000,
        3_000_000_000,
    ):
        name = f"{timestamp}.png"
        (data / name).write_bytes(b"fixture")
        rows.append([timestamp, name])
    with (root / "data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["#timestamp [ns]", "filename"])
        writer.writerows(rows)


def _write_gt(sequence: Path) -> None:
    root = sequence / "mav0/state_groundtruth_estimate0"
    root.mkdir(parents=True)
    with (root / "data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["#timestamp", "px", "py", "pz", "qw", "qx", "qy", "qz"])
        for index, timestamp in enumerate(
            (
                1_000_000_000,
                1_500_000_000,
                2_000_000_000,
                2_500_000_000,
                3_000_000_000,
            )
        ):
            writer.writerow([timestamp, index * 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def _write_openvins(tmp_path: Path) -> tuple[Path, Path]:
    estimate = tmp_path / "state_estimate.txt"
    deviation = tmp_path / "state_deviation.txt"
    estimate.write_text(
        "# timestamp(s) q p v bg ba cam_imu_dt ...\n"
        "1.200000 0 0 0 1 0 0 0 0 0 0 0.01 0.02 0.03 0.10 0.20 0.30 0.2 1\n"
        "2.200000 0 0 0 1 1 0 0 0 0 0 0.01 0.02 0.03 0.10 0.20 0.30 0.2 1\n"
        "3.200000 0 0 0 1 2 0 0 0 0 0 0.01 0.02 0.03 0.10 0.20 0.30 0.2 1\n",
        encoding="utf-8",
    )
    std = " ".join(["0.1"] * 15)
    deviation.write_text(
        "# timestamp(s) std...\n"
        f"1.200000 {std} 0.0 1\n"
        f"2.200000 {std} 0.0 1\n"
        f"3.200000 {std} 0.0 1\n",
        encoding="utf-8",
    )
    return estimate, deviation


def _config() -> Path:
    return Path(__file__).parents[1] / "configs/paper/openvins_adapter_v1.yaml"


def test_openvins_adapter_maps_total_state_to_camera_frames(tmp_path: Path) -> None:
    sequence = tmp_path / "MH_fixture"
    _write_camera(sequence)
    _write_gt(sequence)
    estimate, deviation = _write_openvins(tmp_path)
    output = tmp_path / "run"

    manifest = import_openvins_total_state(
        estimate,
        deviation,
        sequence,
        output,
        adapter_config=_config(),
        evaluate=False,
    )

    assert manifest["backend"] == "openvins"
    assert manifest["external_state_rows"] == 3
    assert manifest["trajectory_rows"] == 5
    assert manifest["camera_frames"] == 5
    assert manifest["innovation_nis_available"] is False
    assert manifest["covariance_representation"] == (
        "diagonal_approximation_from_published_standard_deviations"
    )
    assert manifest["state_age_seconds"]["max"] == pytest.approx(0.5)

    with (output / "trajectory.csv").open("r", encoding="utf-8", newline="") as stream:
        trajectory = list(csv.DictReader(stream))
    assert [int(row["state_timestamp_ns"]) for row in trajectory] == [
        1_000_000_000,
        1_000_000_000,
        2_000_000_000,
        2_000_000_000,
        3_000_000_000,
    ]
    assert [float(row["px"]) for row in trajectory] == [0.0, 0.0, 1.0, 1.0, 2.0]
    assert all(float(row["qw"]) == pytest.approx(1.0) for row in trajectory)
    assert all(float(row["qx"]) == pytest.approx(0.0) for row in trajectory)

    with (output / "health.csv").open("r", encoding="utf-8", newline="") as stream:
        health = list(csv.DictReader(stream))
    assert all(float(row["covariance_trace"]) == pytest.approx(0.15) for row in health)
    assert all(float(row["covariance_condition_number"]) == pytest.approx(1.0) for row in health)
    assert all(float(row["covariance_min_eigenvalue"]) == pytest.approx(0.01) for row in health)
    assert all(row["innovation_nis"] == "" for row in health)
    assert all(row["terminal_tracking_loss_observable"] == "0" for row in health)
    assert all(row["reset_event"] == "0" for row in health)
    assert all(row["relocalization_event"] == "0" for row in health)


def test_openvins_adapter_outputs_are_evaluable(tmp_path: Path) -> None:
    sequence = tmp_path / "MH_fixture"
    _write_camera(sequence)
    _write_gt(sequence)
    estimate, deviation = _write_openvins(tmp_path)
    output = tmp_path / "run"

    manifest = import_openvins_total_state(
        estimate,
        deviation,
        sequence,
        output,
        adapter_config=_config(),
        evaluate=True,
    )

    assert manifest["metrics_path"] == "metrics.json"
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["associated_samples"] == 5
    assert metrics["alignment"] == "se3"


def test_openvins_adapter_rejects_mismatched_estimate_deviation_timestamps(
    tmp_path: Path,
) -> None:
    sequence = tmp_path / "MH_fixture"
    _write_camera(sequence)
    _write_gt(sequence)
    estimate, deviation = _write_openvins(tmp_path)
    text = deviation.read_text(encoding="utf-8").replace("2.200000", "2.300000")
    deviation.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="timestamps disagree"):
        import_openvins_total_state(
            estimate,
            deviation,
            sequence,
            tmp_path / "run",
            adapter_config=_config(),
            evaluate=False,
        )
