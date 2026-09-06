import csv
from pathlib import Path

from shield_vio.experiments.public_prediction import build_public_prediction_artifacts


def _write(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)


def test_public_run_maps_to_phase_b_canonical_features_without_ground_truth_leakage(tmp_path: Path) -> None:
    run = tmp_path / "run"
    timestamps = [0, 500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000, 2_500_000_000]
    _write(run / "health.csv", ["frame_timestamp_ns", "state_timestamp_ns", "initialized", "tracking_status", "propagated_imu_samples", "covariance_trace", "covariance_condition_number", "innovation_nis"], [[t, t, True, "tracking", i, 1.0 + i, 2.0, 3.0] for i, t in enumerate(timestamps)])
    _write(run / "trajectory.csv", ["frame_timestamp_ns", "state_timestamp_ns", "px", "py", "pz"], [[t, t, 0.0 if i < 2 else 2.0, 0.0, 0.0] for i, t in enumerate(timestamps)])
    gt = tmp_path / "gt.csv"
    gt.write_text("#timestamp,x,y,z\n" + "".join(f"{t},0,0,0\n" for t in timestamps), encoding="utf-8")
    out = tmp_path / "out"
    manifest = build_public_prediction_artifacts(run, gt, out, dataset="euroc", sequence="fixture", estimator="internal_eskf", failure_config="configs/paper/failure_public_smoke_v1.yaml")
    assert manifest["evidence_level"] == "FIXTURE_INTEGRATION_TEST"
    assert manifest["confirmatory"] is False
    header = (out / "prediction_dataset.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "ground_truth" not in header
    assert "position_error" not in header
    assert "degradation" not in header
    assert (out / "failure_events.csv").is_file()
