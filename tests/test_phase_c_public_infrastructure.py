import csv
from pathlib import Path

import pytest

from shield_vio.datasets.provenance import RunIdentity, dataset_fingerprint
from shield_vio.datasets.public import validate_public_sequence
from shield_vio.experiments.public_benchmark import resolve_benchmark_matrix


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def _fixture(root: Path, *, tumvi: bool) -> None:
    _write_csv(root / "mav0/cam0/data.csv", [[100, "100.png"], [200, "200.png"]])
    for name in ("100.png", "200.png"):
        image = root / "mav0/cam0/data" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"fixture-image-reference")
    _write_csv(root / "mav0/imu0/data.csv", [[50, 0, 0, 0, 0, 0, 0], [150, 0, 0, 0, 0, 0, 0], [250, 0, 0, 0, 0, 0, 0]])
    gt = root / ("mav0/mocap0/data.csv" if tumvi else "mav0/state_groundtruth_estimate0/data.csv")
    _write_csv(gt, [[100, 0, 0, 0], [200, 0, 0, 0]])
    (root / "mav0/cam0/sensor.yaml").write_text("sensor_type: camera\n", encoding="utf-8")
    (root / "mav0/imu0/sensor.yaml").write_text("sensor_type: imu\n", encoding="utf-8")


@pytest.mark.parametrize("dataset,tumvi", [("euroc", False), ("tumvi", True)])
def test_public_fixture_validation(dataset: str, tumvi: bool, tmp_path: Path) -> None:
    root = tmp_path / ("room1" if tumvi else "MH_01_easy")
    _fixture(root, tumvi=tumvi)
    validated = validate_public_sequence(dataset, root)
    assert validated.camera_rows == 2
    assert validated.imu_rows == 3
    assert validated.ground_truth_rows == 2
    assert len(validated.fingerprint["fingerprint"]) == 64


def test_nonmonotonic_public_timestamps_fail_early(tmp_path: Path) -> None:
    root = tmp_path / "MH_01_easy"; _fixture(root, tumvi=False)
    _write_csv(root / "mav0/cam0/data.csv", [[200, "200.png"], [100, "100.png"]])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_public_sequence("euroc", root)


def test_missing_image_fails_early(tmp_path: Path) -> None:
    root = tmp_path / "room1"; _fixture(root, tumvi=True)
    (root / "mav0/cam0/data/200.png").unlink()
    with pytest.raises(FileNotFoundError):
        validate_public_sequence("tumvi", root)


def test_run_id_is_deterministic_and_identity_sensitive() -> None:
    base = RunIdentity("euroc", "MH_01_easy", "internal_eskf")
    assert base.run_id() == base.run_id()
    assert base.run_id() != RunIdentity("euroc", "MH_02_easy", "internal_eskf").run_id()


def test_fingerprint_changes_when_metadata_changes(tmp_path: Path) -> None:
    path = tmp_path / "index.csv"; path.write_text("1,a\n", encoding="utf-8")
    before = dataset_fingerprint([path])["fingerprint"]
    path.write_text("1,b\n", encoding="utf-8")
    assert before != dataset_fingerprint([path])["fingerprint"]


def test_benchmark_matrix_is_unique_and_split_checked() -> None:
    runs = resolve_benchmark_matrix("configs/paper/public_benchmark.yaml", "configs/paper/public_dataset_splits.yaml")
    assert len(runs) == 11
    assert len({run.run_id for run in runs}) == len(runs)
    assert all(run.status == "NOT_AVAILABLE" for run in runs)
