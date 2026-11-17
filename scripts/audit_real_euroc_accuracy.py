"""Audit real EuRoC trajectory accuracy against the observed ground-truth motion.

A passing execution pipeline is NOT evidence of an accurate estimator.
The heuristic below is a warning threshold, not a SLAM benchmark pass criterion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def summarize_accuracy(gt_csv: Path, baseline_metrics: Path, visual_metrics: Path) -> dict[str, object]:
    positions: list[list[float]] = []
    with gt_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 4:
                raise ValueError("EuRoC ground-truth CSV row has fewer than four columns")
            position = [float(x) for x in row[1:4]]
            if not all(math.isfinite(x) for x in position):
                raise ValueError("EuRoC ground truth contains nonfinite coordinates")
            positions.append(position)
    if len(positions) < 2:
        raise ValueError("At least two ground-truth positions are required")
    trajectory = np.asarray(positions)
    length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    if length <= 0:
        raise ValueError("Ground-truth path length must be positive")

    metrics: dict[str, object] = {}
    for label, path in (("imu_only", baseline_metrics), ("opencv_rotation", visual_metrics)):
        report = json.loads(path.read_text(encoding="utf-8"))
        ate = float(report["ate_rmse_m"])
        rpe = float(report["rpe_translation_rmse_m"])
        count = int(report["associated_samples"])
        if not math.isfinite(ate) or not math.isfinite(rpe) or ate < 0 or rpe < 0 or count < 2:
            raise ValueError(f"Invalid estimator metrics in {path}")
        ratio = ate / length
        metrics[label] = {
            "ate_rmse_m": ate,
            "rpe_translation_rmse_m": rpe,
            "associated_samples": count,
            "ate_to_groundtruth_path_length_ratio": ratio,
            "catastrophic_drift_warning": bool(ratio >= 1.0),
            "accuracy_validated": False,
        }
    return {
        "dataset": "EuRoC MAV",
        "groundtruth_path_length_m": length,
        "runs": metrics,
        "pipeline_evidence_only": True,
        "claim_boundary": (
            "ATE >= full ground-truth path length is flagged as catastrophic drift. "
            "This diagnostic is not a publication-quality threshold and does not "
            "establish validated estimation performance when absent."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--visual-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_accuracy(
        args.groundtruth, args.baseline_metrics, args.visual_metrics
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
