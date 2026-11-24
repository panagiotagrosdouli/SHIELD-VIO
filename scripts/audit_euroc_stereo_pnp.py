#!/usr/bin/env python3
"""Summarize stereo-PnP update and ESKF health diagnostics for one EuRoC run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CHI2_DOF6_95 = 12.592
CHI2_DOF6_99 = 16.812


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _numeric(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            values.append(numeric)
    return np.asarray(values, dtype=float)


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {"count": 0, "min": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_run(run_dir: Path) -> dict[str, Any]:
    visual_path = run_dir / "visual_updates.csv"
    health_path = run_dir / "health.csv"
    trajectory_path = run_dir / "trajectory.csv"
    metrics_path = run_dir / "metrics.json"
    required = (visual_path, health_path, trajectory_path, metrics_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing stereo-PnP run artifacts: {missing}")

    visual = _read_rows(visual_path)
    health = _read_rows(health_path)
    trajectory = _read_rows(trajectory_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    inliers = _numeric(visual, "inlier_count")
    correspondences = _numeric(visual, "correspondence_count")
    nis = _numeric(visual, "innovation_nis")
    min_eigenvalue = _numeric(health, "covariance_min_eigenvalue")
    symmetry_error = _numeric(health, "covariance_symmetry_error")
    condition_number = _numeric(health, "covariance_condition_number")

    timestamps = _numeric(trajectory, "frame_timestamp_ns") * 1e-9
    positions = np.asarray(
        [[float(row["px"]), float(row["py"]), float(row["pz"])] for row in trajectory],
        dtype=float,
    )
    if len(positions) < 2 or len(timestamps) != len(positions):
        raise ValueError("trajectory needs at least two finite timestamped positions")
    dt = np.diff(timestamps)
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    valid_dt = np.isfinite(dt) & (dt > 0)
    speed = displacement[valid_dt] / dt[valid_dt]

    status_counts: dict[str, int] = {}
    for row in visual:
        status = row.get("status", "")
        status_counts[status] = status_counts.get(status, 0) + 1

    report = {
        "schema_version": 1,
        "sequence": metrics.get("sequence"),
        "visual_provider": "opencv_stereo_pnp_metric_pose",
        "confirmatory": False,
        "ground_truth_used_for_estimation": False,
        "metrics": metrics,
        "visual_updates": {
            "rows": len(visual),
            "status_counts": status_counts,
            "inlier_count": _distribution(inliers),
            "correspondence_count": _distribution(correspondences),
            "innovation_nis_dof6": _distribution(nis),
            "nis_fraction_above_chi2_95_dof6": (
                float(np.mean(nis > CHI2_DOF6_95)) if len(nis) else None
            ),
            "nis_fraction_above_chi2_99_dof6": (
                float(np.mean(nis > CHI2_DOF6_99)) if len(nis) else None
            ),
        },
        "filter_health": {
            "rows": len(health),
            "covariance_min_eigenvalue": _distribution(min_eigenvalue),
            "covariance_symmetry_error": _distribution(symmetry_error),
            "covariance_condition_number": _distribution(condition_number),
        },
        "trajectory": {
            "rows": len(trajectory),
            "estimated_step_speed_mps": _distribution(speed),
            "max_position_norm_m": float(np.max(np.linalg.norm(positions, axis=1))),
        },
        "artifact_sha256": {
            path.name: _sha256(path) for path in required
        },
        "claim_boundary": (
            "Non-confirmatory development diagnostic. Innovation NIS thresholds assume "
            "six-dimensional chi-square residuals and are descriptive; sequential updates "
            "and frontend selection violate simple independent-sample interpretation."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
