"""Flag gross EuRoC trajectory drift without treating CI as accuracy validation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _groundtruth_path_length(path: Path) -> float:
    positions: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 4:
                raise ValueError("EuRoC ground-truth CSV row has fewer than four columns")
            position = [float(value) for value in row[1:4]]
            if not all(math.isfinite(value) for value in position):
                raise ValueError("EuRoC ground truth contains nonfinite coordinates")
            positions.append(position)
    if len(positions) < 2:
        raise ValueError("At least two ground-truth positions are required")
    length = float(np.linalg.norm(np.diff(np.asarray(positions), axis=0), axis=1).sum())
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Ground-truth path length must be finite and positive")
    return length


def summarize_accuracy(
    groundtruth_csv: Path, baseline_metrics: Path, visual_metrics: Path
) -> dict[str, Any]:
    path_length = _groundtruth_path_length(groundtruth_csv)
    runs: dict[str, dict[str, Any]] = {}
    for label, metrics_path in (
        ("imu_only", baseline_metrics),
        ("opencv_rotation", visual_metrics),
    ):
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"Estimator metrics must be a JSON object: {metrics_path}")
        ate = float(report["ate_rmse_m"])
        rpe = float(report["rpe_translation_rmse_m"])
        count = report["associated_samples"]
        if (
            not math.isfinite(ate)
            or not math.isfinite(rpe)
            or ate < 0
            or rpe < 0
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 2
        ):
            raise ValueError(f"Invalid estimator metrics in {metrics_path}")
        ratio = ate / path_length
        runs[label] = {
            "ate_rmse_m": ate,
            "rpe_translation_rmse_m": rpe,
            "associated_samples": count,
            "ate_to_groundtruth_path_length_ratio": ratio,
            "catastrophic_drift_warning": ratio >= 1.0,
            "accuracy_validated": False,
        }
    return {
        "dataset": "EuRoC MAV",
        "sequence": "MH_01_easy",
        "groundtruth_path_length_m": path_length,
        "runs": runs,
        "pipeline_evidence_only": True,
        "claim_boundary": (
            "ATE at least as large as the full ground-truth path length is a "
            "diagnostic warning only; absence of the warning does not validate accuracy."
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
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
