#!/usr/bin/env python3
"""Compare EuRoC attitude after the same position-only SE(3) alignment.

This is a non-confirmatory diagnostic. Ground truth is used only for post-hoc
evaluation; it is never passed to the estimator.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _rotation_from_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be a finite wxyz vector")
    norm = float(np.linalg.norm(q))
    if norm <= np.finfo(float).eps:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _quaternion_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=float) / np.linalg.norm(a)
    bw, bx, by, bz = np.asarray(b, dtype=float) / np.linalg.norm(b)
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _position_alignment_rotation(estimate: np.ndarray, groundtruth: np.ndarray) -> np.ndarray:
    if estimate.shape != groundtruth.shape or estimate.ndim != 2 or estimate.shape[1] != 3:
        raise ValueError("paired positions must both have shape (N, 3)")
    if len(estimate) < 3 or not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(groundtruth)):
        raise ValueError("at least three finite position pairs are required")
    e0 = estimate - estimate.mean(axis=0)
    g0 = groundtruth - groundtruth.mean(axis=0)
    u, _, vt = np.linalg.svd(e0.T @ g0)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    return vt.T @ correction @ u.T


def attitude_alignment_summary(
    estimate_t: np.ndarray,
    estimate_p: np.ndarray,
    estimate_q: np.ndarray,
    groundtruth_t: np.ndarray,
    groundtruth_p: np.ndarray,
    groundtruth_q: np.ndarray,
    *,
    max_gap_s: float = 0.02,
) -> dict[str, Any]:
    """Pair samples, position-align globally, then report attitude residuals."""
    et = np.asarray(estimate_t, dtype=float)
    ep = np.asarray(estimate_p, dtype=float)
    eq = np.asarray(estimate_q, dtype=float)
    gt = np.asarray(groundtruth_t, dtype=float)
    gp = np.asarray(groundtruth_p, dtype=float)
    gq = np.asarray(groundtruth_q, dtype=float)
    if max_gap_s <= 0 or not math.isfinite(max_gap_s):
        raise ValueError("max_gap_s must be finite and positive")
    if et.ndim != 1 or gt.ndim != 1 or len(et) != len(ep) or len(et) != len(eq):
        raise ValueError("estimate timestamps, positions, and orientations must have matching lengths")
    if len(gt) != len(gp) or len(gt) != len(gq) or len(gt) < 2:
        raise ValueError("ground-truth timestamps, positions, and orientations must have matching lengths >=2")
    if ep.shape != (len(et), 3) or eq.shape != (len(et), 4) or gp.shape != (len(gt), 3) or gq.shape != (len(gt), 4):
        raise ValueError("positions must have shape (N,3) and wxyz orientations shape (N,4)")
    if (not np.all(np.isfinite(et)) or not np.all(np.isfinite(ep)) or not np.all(np.isfinite(eq))
            or not np.all(np.isfinite(gt)) or not np.all(np.isfinite(gp)) or not np.all(np.isfinite(gq))):
        raise ValueError("trajectory inputs must be finite")
    if np.any(np.diff(et) <= 0) or np.any(np.diff(gt) <= 0):
        raise ValueError("trajectory timestamps must be strictly increasing")

    paired_est: list[int] = []
    paired_gt: list[int] = []
    for i, timestamp in enumerate(et):
        j = int(np.searchsorted(gt, timestamp, side="left"))
        candidates = [k for k in (j - 1, j) if 0 <= k < len(gt)]
        if not candidates:
            continue
        k = min(candidates, key=lambda index: abs(gt[index] - timestamp))
        if abs(gt[k] - timestamp) <= max_gap_s:
            paired_est.append(i)
            paired_gt.append(k)
    if len(paired_est) < 3:
        raise ValueError("fewer than three samples satisfy timestamp association")

    epair = ep[paired_est]
    gpair = gp[paired_gt]
    rotation = _position_alignment_rotation(epair, gpair)
    qe_aligned = np.asarray([
        _quaternion_multiply_wxyz(_matrix_to_quaternion_wxyz(rotation), eq[i])
        for i in paired_est
    ])
    residuals = np.asarray([
        _rotation_error_deg(_rotation_from_quaternion_wxyz(qe), _rotation_from_quaternion_wxyz(gq[j]))
        for qe, j in zip(qe_aligned, paired_gt)
    ])
    paired_times = et[paired_est]
    relative = paired_times - gt[0]
    early = residuals[relative <= 5.0]
    return {
        "associated_samples": len(paired_est),
        "position_alignment_rotation_determinant": float(np.linalg.det(rotation)),
        "attitude_error_deg_mean": float(np.mean(residuals)),
        "attitude_error_deg_median": float(np.median(residuals)),
        "attitude_error_deg_p95": float(np.percentile(residuals, 95)),
        "attitude_error_deg_first_5s_median": None if not len(early) else float(np.median(early)),
        "claim_boundary": (
            "Non-confirmatory diagnostic only. Ground truth is used only for post-hoc evaluation; "
            "one global rotation is derived from positions and applied to estimated attitudes."
        ),
    }


def _matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([0.25*s, (m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s, (m[1, 0]-m[0, 1])/s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = np.array([(m[2, 1]-m[1, 2])/s, 0.25*s, (m[0, 1]+m[1, 0])/s, (m[0, 2]+m[2, 0])/s])
        elif i == 1:
            s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = np.array([(m[0, 2]-m[2, 0])/s, (m[0, 1]+m[1, 0])/s, 0.25*s, (m[1, 2]+m[2, 1])/s])
        else:
            s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = np.array([(m[1, 0]-m[0, 1])/s, (m[0, 2]+m[2, 0])/s, (m[1, 2]+m[2, 1])/s, 0.25*s])
    return q / np.linalg.norm(q)


def _rotation_error_deg(estimate: np.ndarray, groundtruth: np.ndarray) -> float:
    relative = estimate @ groundtruth.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _read_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    positions: list[list[float]] = []
    orientations: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 8:
                raise ValueError("EuRoC ground-truth row requires timestamp, position, and quaternion")
            timestamps.append(int(row[0]) * 1e-9)
            positions.append([float(x) for x in row[1:4]])
            orientations.append([float(x) for x in row[4:8]])
    return np.asarray(timestamps), np.asarray(positions), np.asarray(orientations)


def _read_estimate(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    positions: list[list[float]] = []
    orientations: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            timestamps.append(int(row["frame_timestamp_ns"]) * 1e-9)
            positions.append([float(row[key]) for key in ("px", "py", "pz")])
            orientations.append([float(row[key]) for key in ("qw", "qx", "qy", "qz")])
    return np.asarray(timestamps), np.asarray(positions), np.asarray(orientations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--visual-trajectory", type=Path, required=True)
    parser.add_argument("--max-gap", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gt = _read_groundtruth(args.groundtruth)
    report = {
        "schema_version": "SHIELD_VIO_ATTITUDE_ALIGNMENT_DIAGNOSTIC_V1",
        "sequence": "MH_01_easy",
        "confirmatory": False,
        "ground_truth_used_by_estimator": False,
        "runs": {},
    }
    for name, path in (("imu_only", args.baseline_trajectory), ("opencv_rotation", args.visual_trajectory)):
        estimate = _read_estimate(path)
        report["runs"][name] = attitude_alignment_summary(*estimate, *gt, max_gap_s=args.max_gap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
