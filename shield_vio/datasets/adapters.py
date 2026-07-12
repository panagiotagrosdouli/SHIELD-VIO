"""Filesystem adapters for public visual-inertial datasets.

These adapters discover and validate locally prepared sequences. They never
attempt to download data and do not manufacture benchmark results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSequence:
    name: str
    root: Path
    camera_csv: Path
    imu_csv: Path
    ground_truth_csv: Path | None
    calibration_files: tuple[Path, ...]

    def validate(self) -> None:
        for required in (self.root, self.camera_csv, self.imu_csv):
            if not required.exists():
                raise FileNotFoundError(required)
        if self.ground_truth_csv is not None and not self.ground_truth_csv.exists():
            raise FileNotFoundError(self.ground_truth_csv)


def discover_euroc_sequence(root: str | Path) -> DatasetSequence:
    base = Path(root)
    mav0 = base / "mav0"
    sequence = DatasetSequence(
        name=base.name,
        root=base,
        camera_csv=mav0 / "cam0" / "data.csv",
        imu_csv=mav0 / "imu0" / "data.csv",
        ground_truth_csv=(
            mav0 / "state_groundtruth_estimate0" / "data.csv"
            if (mav0 / "state_groundtruth_estimate0" / "data.csv").exists()
            else None
        ),
        calibration_files=tuple(
            path
            for path in (mav0 / "cam0" / "sensor.yaml", mav0 / "imu0" / "sensor.yaml")
            if path.exists()
        ),
    )
    sequence.validate()
    return sequence


def discover_tumvi_sequence(root: str | Path) -> DatasetSequence:
    base = Path(root)
    mav0 = base / "mav0"
    sequence = DatasetSequence(
        name=base.name,
        root=base,
        camera_csv=mav0 / "cam0" / "data.csv",
        imu_csv=mav0 / "imu0" / "data.csv",
        ground_truth_csv=(
            mav0 / "mocap0" / "data.csv" if (mav0 / "mocap0" / "data.csv").exists() else None
        ),
        calibration_files=tuple(
            path
            for path in (mav0 / "cam0" / "sensor.yaml", mav0 / "imu0" / "sensor.yaml")
            if path.exists()
        ),
    )
    sequence.validate()
    return sequence


def discover_generic_sequence(root: str | Path) -> DatasetSequence:
    base = Path(root)
    camera = base / "camera.csv"
    imu = base / "imu.csv"
    ground_truth = base / "ground_truth.csv"
    sequence = DatasetSequence(
        name=base.name,
        root=base,
        camera_csv=camera,
        imu_csv=imu,
        ground_truth_csv=ground_truth if ground_truth.exists() else None,
        calibration_files=tuple(sorted(base.glob("*calib*.yaml"))),
    )
    sequence.validate()
    return sequence
