#!/usr/bin/env python3
"""Compare identity and ground-truth attitude starts as a non-confirmatory oracle ablation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from shield_vio.datasets.euroc import read_imu_samples
from shield_vio.estimation.error_state_ekf import ESKFState, ErrorStateEKF
from shield_vio.evaluation.euroc import TimestampedTrajectory, associate_ground_truth, load_euroc_ground_truth
from shield_vio.evaluation.metrics import ate, rpe


def _read_gt_orientations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    orientations: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 8:
                raise ValueError("ground-truth pose rows need timestamp, position, and wxyz quaternion")
            timestamps.append(int(row[0]) * 1e-9)
            orientations.append([float(value) for value in row[4:8]])
    return np.asarray(timestamps), np.asarray(orientations)


def _propagate(
    timestamps_s: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    initial_orientation_wxyz: np.ndarray,
) -> TimestampedTrajectory:
    state = ESKFState(
        position_m=np.zeros(3),
        velocity_mps=np.zeros(3),
        quaternion_wxyz=np.asarray(initial_orientation_wxyz, dtype=float).copy(),
        accel_bias_mps2=np.zeros(3),
        gyro_bias_rps=np.zeros(3),
        covariance=np.eye(15) * 1e-3,
        timestamp_s=float(timestamps_s[0]),
    )
    filter_ = ErrorStateEKF(state=state)
    positions = [state.position_m.copy()]
    for index in range(1, len(timestamps_s)):
        dt = float(timestamps_s[index] - timestamps_s[index - 1])
        if dt <= 0 or dt > 0.1:
            raise ValueError("IMU timestamps must increase with no gap above 0.1 s")
        filter_.propagate(accel[index], gyro[index], dt)
        positions.append(filter_.state.position_m.copy())
    return TimestampedTrajectory(timestamps_s, np.asarray(positions))


def _score(
    estimate: TimestampedTrajectory, ground_truth: TimestampedTrajectory
) -> dict[str, float | int]:
    _, est_p, gt_p = associate_ground_truth(estimate, ground_truth, max_gap=0.01)
    return {
        "associated_samples": int(len(est_p)),
        "ate_rmse_m_se3": ate(est_p, gt_p, align=True),
        "rpe_translation_rmse_m_delta_1": rpe(est_p, gt_p, delta=1, align=True),
    }


def compare_initial_attitudes(
    imu: np.ndarray,
    gt_timestamps_s: np.ndarray,
    gt_positions_m: np.ndarray,
    gt_orientations_wxyz: np.ndarray,
    *,
    max_gap_s: float = 0.01,
) -> dict[str, Any]:
    """Run two identical IMU propagations differing only in starting attitude."""
    imu = np.asarray(imu, dtype=float)
    gt_timestamps_s = np.asarray(gt_timestamps_s, dtype=float)
    gt_positions_m = np.asarray(gt_positions_m, dtype=float)
    gt_orientations_wxyz = np.asarray(gt_orientations_wxyz, dtype=float)
    if imu.ndim != 2 or imu.shape[1] < 7 or len(imu) < 3:
        raise ValueError("IMU input needs >=3 rows and seven EuRoC columns")
    if gt_timestamps_s.ndim != 1 or len(gt_timestamps_s) < 3:
        raise ValueError("ground-truth timestamps need at least three samples")
    if gt_positions_m.shape != (len(gt_timestamps_s), 3):
        raise ValueError("ground-truth positions must have shape (N,3)")
    if gt_orientations_wxyz.shape != (len(gt_timestamps_s), 4):
        raise ValueError("ground-truth orientations must have shape (N,4)")
    ti = imu[:, 0] * 1e-9
    right = np.searchsorted(gt_timestamps_s, ti, side="left")
    candidates = np.stack([
        np.clip(right - 1, 0, len(gt_timestamps_s) - 1),
        np.clip(right, 0, len(gt_timestamps_s) - 1),
    ], axis=1)
    distances = np.abs(gt_timestamps_s[candidates] - ti[:, None])
    nearest = candidates[np.arange(len(ti)), np.argmin(distances, axis=1)]
    valid = np.min(distances, axis=1) <= max_gap_s
    if int(valid.sum()) < 3:
        raise ValueError("fewer than three IMU samples associate with ground truth")
    start = int(np.flatnonzero(valid)[0])
    stop = int(np.flatnonzero(valid)[-1]) + 1
    times = ti[start:stop]
    if np.any(np.diff(times) <= 0):
        raise ValueError("selected IMU timestamps must increase")
    orientation = gt_orientations_wxyz[nearest[start]]
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    baseline = _propagate(times, imu[start:stop, 4:7], imu[start:stop, 1:4], identity)
    oracle = _propagate(times, imu[start:stop, 4:7], imu[start:stop, 1:4], orientation)
    gt = TimestampedTrajectory(gt_timestamps_s, gt_positions_m)
    baseline_score = _score(baseline, gt)
    oracle_score = _score(oracle, gt)
    return {
        "schema_version": "SHIELD_VIO_ORACLE_ATTITUDE_ABLATION_V1",
        "sequence": "MH_01_easy",
        "start_timestamp_s": float(times[0]),
        "end_timestamp_s": float(times[-1]),
        "baseline_identity_initialization": baseline_score,
        "oracle_groundtruth_attitude_initialization": oracle_score,
        "ate_relative_change_oracle_vs_identity": (
            oracle_score["ate_rmse_m_se3"] / baseline_score["ate_rmse_m_se3"] - 1.0
            if baseline_score["ate_rmse_m_se3"] > 0 else None
        ),
        "ground_truth_used_for_initialization": True,
        "confirmatory": False,
        "claim_boundary": (
            "Diagnostic oracle ablation only. Ground-truth attitude is deliberately supplied "
            "to one replay to bound the effect of initial attitude; this is not an estimator result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    imu = read_imu_samples(args.sequence_root)
    imu_array = np.asarray([
        [sample.timestamp_ns, *sample.angular_velocity_rad_s, *sample.linear_acceleration_m_s2]
        for sample in imu
    ], dtype=float)
    ground_truth = load_euroc_ground_truth(args.sequence_root, relative_time=False)
    gt_t, gt_q = _read_gt_orientations(
        args.sequence_root / "mav0/state_groundtruth_estimate0/data.csv"
    )
    report = compare_initial_attitudes(
        imu_array, gt_t, ground_truth.positions, gt_q
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
