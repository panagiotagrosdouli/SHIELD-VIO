#!/usr/bin/env python3
"""Compare IMU-only and OpenCV-rotation EuRoC runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_metrics(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _visual_statistics(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"visual_updates": 0, "mean_inlier_ratio": None, "median_innovation_nis": None}
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ratios = [float(row["inlier_ratio"]) for row in rows if row.get("inlier_ratio")]
    nis = [float(row["innovation_nis"]) for row in rows if row.get("innovation_nis")]
    return {
        "visual_updates": len(rows),
        "mean_inlier_ratio": None if not ratios else float(np.mean(ratios)),
        "median_innovation_nis": None if not nis else float(np.median(nis)),
    }


def build_report(baseline_dir: Path, visual_dir: Path) -> dict[str, object]:
    baseline = _load_metrics(baseline_dir)
    visual = _load_metrics(visual_dir)
    for key in ("sequence", "associated_samples", "alignment"):
        if baseline[key] != visual[key]:
            raise ValueError(f"paired runs disagree on {key}")

    baseline_ate = float(baseline["ate_rmse_m"])
    visual_ate = float(visual["ate_rmse_m"])
    baseline_rpe = float(baseline["rpe_translation_rmse_m"])
    visual_rpe = float(visual["rpe_translation_rmse_m"])
    return {
        "benchmark": "paired real EuRoC comparison",
        "sequence": baseline["sequence"],
        "alignment": baseline["alignment"],
        "associated_samples": baseline["associated_samples"],
        "baseline": {
            "configuration": "IMU-only ESKF propagation",
            "ate_rmse_m": baseline_ate,
            "rpe_translation_rmse_m": baseline_rpe,
        },
        "opencv_rotation": {
            "configuration": "IMU ESKF plus monocular OpenCV rotation updates",
            "ate_rmse_m": visual_ate,
            "rpe_translation_rmse_m": visual_rpe,
            **_visual_statistics(visual_dir / "visual_updates.csv"),
        },
        "delta": {
            "ate_rmse_m": visual_ate - baseline_ate,
            "rpe_translation_rmse_m": visual_rpe - baseline_rpe,
            "ate_relative_change": None if baseline_ate == 0 else visual_ate / baseline_ate - 1.0,
            "rpe_relative_change": None if baseline_rpe == 0 else visual_rpe / baseline_rpe - 1.0,
        },
        "claim_boundary": (
            "Real EuRoC sensor and ground-truth results. The visual configuration fuses "
            "monocular relative rotation only; it is not a full metric visual-inertial system."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("visual_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.baseline_dir, args.visual_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
