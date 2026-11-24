#!/usr/bin/env python3
"""Post-hoc EuRoC IMU/ground-truth frame and unit consistency audit.

Ground truth is used only after estimator runs. This audit is diagnostic and
must not be used to initialize, tune, or score an estimator confirmatorily.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def rotation_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("orientation must be a finite nonzero wxyz quaternion")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def world_acceleration_from_specific_force(
    specific_force_body: np.ndarray,
    orientation_wxyz: np.ndarray,
    gravity_world: np.ndarray | None = None,
    *,
    transpose_orientation: bool = False,
    gravity_sign: float = 1.0,
) -> np.ndarray:
    """Convert accelerometer specific force to world acceleration."""
    gravity = np.array([0.0, 0.0, -9.80665]) if gravity_world is None else np.asarray(gravity_world, dtype=float)
    rotation = rotation_wxyz(orientation_wxyz)
    if transpose_orientation:
        rotation = rotation.T
    return rotation @ np.asarray(specific_force_body, dtype=float) + gravity_sign * gravity


def _read_numeric_csv(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].lstrip().startswith("#"):
                continue
            rows.append([float(value) for value in row])
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"expected finite numeric rows in {path}")
    return values


def audit_imu_frames(
    imu: np.ndarray,
    groundtruth: np.ndarray,
    *,
    max_gap_s: float = 0.01,
) -> dict[str, Any]:
    """Compare IMU-derived acceleration to GT velocity derivatives under conventions."""
    imu = np.asarray(imu, dtype=float)
    gt = np.asarray(groundtruth, dtype=float)
    if imu.ndim != 2 or imu.shape[1] < 7 or gt.ndim != 2 or gt.shape[1] < 11:
        raise ValueError("EuRoC IMU needs 7 columns and state ground truth needs at least 11")
    if not np.all(np.isfinite(imu)) or not np.all(np.isfinite(gt)):
        raise ValueError("inputs must be finite")
    ti = imu[:, 0] * 1e-9
    tg = gt[:, 0] * 1e-9
    if np.any(np.diff(ti) <= 0) or np.any(np.diff(tg) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if max_gap_s <= 0 or not math.isfinite(max_gap_s):
        raise ValueError("max_gap_s must be finite and positive")

    right = np.searchsorted(tg, ti, side="left")
    candidates = np.stack([np.clip(right - 1, 0, len(tg) - 1), np.clip(right, 0, len(tg) - 1)], axis=1)
    distances = np.abs(tg[candidates] - ti[:, None])
    nearest_slot = np.argmin(distances, axis=1)
    indices = candidates[np.arange(len(ti)), nearest_slot]
    valid = (distances[np.arange(len(ti)), nearest_slot] <= max_gap_s)
    if int(valid.sum()) < 10:
        raise ValueError("fewer than ten IMU samples associate with ground truth")

    paired_imu = imu[valid]
    paired_gt = gt[indices[valid]]
    paired_t = ti[valid]
    q = paired_gt[:, 4:8]
    accel = paired_imu[:, 4:7]
    # EuRoC state layout: timestamp, p, q(wxyz), v, gyro bias, accel bias.
    velocity = gt[:, 8:11]
    gt_accel_all = np.gradient(velocity, tg, axis=0, edge_order=2 if len(tg) >= 3 else 1)
    gt_accel = gt_accel_all[indices[valid]]
    accel_bias = paired_gt[:, 14:17] if gt.shape[1] >= 17 else np.zeros_like(accel)

    conventions: dict[str, dict[str, float]] = {}
    for bias_corrected, force in ((False, accel), (True, accel - accel_bias)):
        for transpose in (False, True):
            for gravity_sign in (1.0, -1.0):
                prediction = np.asarray([
                    world_acceleration_from_specific_force(
                        force[i], q[i], transpose_orientation=transpose, gravity_sign=gravity_sign
                    )
                    for i in range(len(force))
                ])
                residual = prediction - gt_accel
                label = f"bias_corrected={bias_corrected},transpose_q={transpose},gravity_sign={int(gravity_sign)}"
                conventions[label] = {
                    "acceleration_rmse_mps2": float(np.sqrt(np.mean(residual**2))),
                    "median_residual_norm_mps2": float(np.median(np.linalg.norm(residual, axis=1))),
                    "p95_residual_norm_mps2": float(np.percentile(np.linalg.norm(residual, axis=1), 95)),
                }

    first_rotation = rotation_wxyz(gt[0, 4:8])
    body_down = first_rotation.T @ np.array([0.0, 0.0, -1.0])
    initial_tilt_deg = math.degrees(math.acos(float(np.clip(body_down[2], -1.0, 1.0))))
    gyro = paired_imu[:, 1:4]
    accel_norm = np.linalg.norm(accel, axis=1)
    best = min(conventions, key=lambda key: conventions[key]["acceleration_rmse_mps2"])
    return {
        "schema_version": "SHIELD_VIO_IMU_FRAME_CONSISTENCY_V1",
        "sequence": "MH_01_easy",
        "associated_samples": int(valid.sum()),
        "association_max_gap_s": max_gap_s,
        "associated_duration_s": float(paired_t[-1] - paired_t[0]),
        "median_raw_accel_norm_mps2": float(np.median(accel_norm)),
        "median_raw_gyro_norm_radps": float(np.median(np.linalg.norm(gyro, axis=1))),
        "initial_gt_body_down_tilt_from_identity_deg": initial_tilt_deg,
        "groundtruth_velocity_derivative_convention_scores": conventions,
        "best_scoring_convention": best,
        "claim_boundary": (
            "Non-confirmatory post-hoc diagnostic. Ground truth is used only to audit "
            "raw IMU units/frame conventions and is never supplied to an estimator."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap", type=float, default=0.01)
    args = parser.parse_args()
    report = audit_imu_frames(
        _read_numeric_csv(args.imu), _read_numeric_csv(args.groundtruth), max_gap_s=args.max_gap
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
