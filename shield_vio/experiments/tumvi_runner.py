"""Execute the internal ESKF on a TUM-VI EuRoC/DSO-format export.

TUM-VI exports use the same mav0/cam0/data.csv and mav0/imu0/data.csv stream
shape consumed by the existing synchronized runner. Evaluation is kept
separate because TUM-VI mocap coverage differs from EuRoC ground truth.
"""
from __future__ import annotations

from pathlib import Path

from shield_vio.backends.base import EstimatorBackend
from shield_vio.datasets.euroc import read_camera_frames, read_imu_samples, synchronize_camera_and_imu
from shield_vio.experiments.euroc_runner import EuRoCRunSummary, run_synchronized_frames
from shield_vio.experiments.visual_measurements import VisualMeasurementProvider


def run_tumvi_sequence(
    sequence_root: str | Path,
    output_dir: str | Path,
    *,
    camera: str = "cam0",
    backend: EstimatorBackend | None = None,
    visual_provider: VisualMeasurementProvider | None = None,
) -> EuRoCRunSummary:
    """Run TUM-VI sensor streams without pretending mocap coverage is EuRoC GT."""
    frames = read_camera_frames(sequence_root, camera=camera)
    imu_samples = read_imu_samples(sequence_root)
    packets = synchronize_camera_and_imu(frames, imu_samples, include_pre_first_frame=True)
    return run_synchronized_frames(
        packets,
        output_dir,
        backend=backend,
        visual_provider=visual_provider,
    )
