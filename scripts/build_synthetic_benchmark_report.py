"""Build a batch-compatible benchmark report from synthetic demo trajectories."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from shield_vio.evaluation.trajectory_metrics import summarize_trajectory_metrics


def _load_positions(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [[float(row[axis]) for axis in ("x", "y", "z")] for row in reader]
    positions = np.asarray(rows, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"trajectory must contain x, y, z columns: {path}")
    return positions


def build_report(results_dir: str | Path, *, rpe_delta: int = 1) -> dict[str, object]:
    root = Path(results_dir)
    estimated = _load_positions(root / "estimated_trajectory.csv")
    reference = _load_positions(root / "ground_truth.csv")
    metrics = summarize_trajectory_metrics(
        estimated,
        reference,
        rpe_delta=rpe_delta,
        align=True,
    )
    result = {
        "sequence": "synthetic_seed_7",
        "status": "ok",
        "estimate_path": str(root / "estimated_trajectory.csv"),
        **asdict(metrics),
        "error": None,
    }
    return {"success_count": 1, "failure_count": 0, "results": [result]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="Synthetic demo output directory")
    parser.add_argument("--output", required=True, help="Output benchmark JSON path")
    parser.add_argument("--rpe-delta", type=int, default=1)
    args = parser.parse_args()

    payload = build_report(args.results_dir, rpe_delta=args.rpe_delta)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
