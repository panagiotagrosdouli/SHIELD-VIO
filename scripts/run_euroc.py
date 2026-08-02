#!/usr/bin/env python3
"""Run the built-in ESKF over a EuRoC sequence and export evaluated artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shield_vio.datasets.calibration import CameraCalibration, load_camera_calibration
from shield_vio.experiments.euroc_runner import run_euroc_sequence
from shield_vio.experiments.opencv_visual_provider import (
    OpenCVRotationVisualProvider,
    camera_matrix_from_intrinsics,
)
from shield_vio.experiments.stereo_pnp_visual_provider import StereoPnPVisualProvider
from shield_vio.vision.stereo_pnp import StereoPnPFrontend


def stereo_transform_right_left(
    left: CameraCalibration,
    right: CameraCalibration,
) -> np.ndarray:
    """Return the rigid transform that maps cam0 coordinates into cam1 coordinates."""
    transform = np.linalg.inv(right.transform_body_sensor) @ left.transform_body_sensor
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("derived stereo transform is not homogeneous")
    return transform


def build_stereo_pnp_provider(sequence_root: Path) -> StereoPnPVisualProvider:
    """Build the metric stereo-PnP provider from EuRoC cam0/cam1 calibration files."""
    calibration_root = sequence_root / "mav0"
    left = load_camera_calibration(calibration_root / "cam0" / "sensor.yaml")
    right = load_camera_calibration(calibration_root / "cam1" / "sensor.yaml")
    transform_right_left = stereo_transform_right_left(left, right)
    frontend = StereoPnPFrontend(
        camera_matrix_from_intrinsics(left.intrinsics),
        camera_matrix_from_intrinsics(right.intrinsics),
        transform_right_left[:3, :3],
        transform_right_left[:3, 3],
    )
    return StereoPnPVisualProvider(frontend, left.transform_body_sensor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_root", type=Path, help="Path containing the EuRoC mav0 directory")
    parser.add_argument("--output", type=Path, required=True, help="Artifact output directory")
    parser.add_argument("--camera", default="cam0", help="EuRoC camera stream used for frame timestamps")
    visual_group = parser.add_mutually_exclusive_group()
    visual_group.add_argument(
        "--opencv-rotation",
        action="store_true",
        help="Fuse KLT/essential-matrix relative-rotation updates",
    )
    visual_group.add_argument(
        "--stereo-pnp",
        action="store_true",
        help="Fuse calibrated metric cam0/cam1 stereo-PnP pose updates",
    )
    parser.add_argument("--skip-evaluation", action="store_true", help="Do not compare against EuRoC ground truth")
    parser.add_argument("--max-gap", type=float, default=0.02, help="Maximum timestamp association gap in seconds")
    parser.add_argument("--rpe-delta", type=int, default=1, help="RPE displacement interval in associated samples")
    parser.add_argument("--with-scale", action="store_true", help="Use Sim(3) instead of SE(3) alignment")
    args = parser.parse_args()

    visual_provider = None
    if args.opencv_rotation:
        calibration = load_camera_calibration(
            args.sequence_root / "mav0" / args.camera / "sensor.yaml"
        )
        visual_provider = OpenCVRotationVisualProvider(
            camera_matrix_from_intrinsics(calibration.intrinsics)
        )
    elif args.stereo_pnp:
        if args.camera != "cam0":
            parser.error("--stereo-pnp requires --camera cam0 because cam1 is the paired right stream")
        visual_provider = build_stereo_pnp_provider(args.sequence_root)

    summary = run_euroc_sequence(
        args.sequence_root,
        args.output,
        camera=args.camera,
        visual_provider=visual_provider,
        evaluate=not args.skip_evaluation,
        max_gap=args.max_gap,
        rpe_delta=args.rpe_delta,
        with_scale=args.with_scale,
    )
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
