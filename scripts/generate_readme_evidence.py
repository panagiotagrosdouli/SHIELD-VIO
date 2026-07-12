"""Generate README evidence panels directly from synthetic run artifacts."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_columns(path: Path, columns: tuple[str, ...]) -> dict[str, list[float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    values = {name: [] for name in columns}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in columns if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row in reader:
            for name in columns:
                values[name].append(float(row[name]))
    if not values[columns[0]]:
        raise ValueError(f"{path} contains no data rows")
    return values


def generate(results_dir: Path, output_dir: Path) -> list[Path]:
    trajectory = _read_columns(results_dir / "estimated_trajectory.csv", ("t", "x", "y"))
    ground_truth = _read_columns(results_dir / "ground_truth.csv", ("t", "x", "y"))
    uncertainty = _read_columns(results_dir / "uncertainty.csv", ("t", "trace", "nis"))
    risk = _read_columns(results_dir / "risk_score.csv", ("t", "risk_score"))

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    trajectory_path = output_dir / "trajectory_evidence.png"
    plt.figure(figsize=(8, 5))
    plt.plot(ground_truth["x"], ground_truth["y"], label="Ground truth")
    plt.plot(trajectory["x"], trajectory["y"], label="Estimate")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Synthetic trajectory evidence")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(trajectory_path, dpi=180)
    plt.close()
    outputs.append(trajectory_path)

    health_path = output_dir / "estimator_health_evidence.png"
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(uncertainty["t"], uncertainty["trace"])
    axes[0].set_ylabel("Cov. trace")
    axes[1].plot(uncertainty["t"], uncertainty["nis"])
    axes[1].set_ylabel("NIS")
    axes[2].plot(risk["t"], risk["risk_score"])
    axes[2].set_ylabel("Risk")
    axes[2].set_xlabel("Time [s]")
    fig.suptitle("Estimator health and shield risk")
    fig.tight_layout()
    fig.savefig(health_path, dpi=180)
    plt.close(fig)
    outputs.append(health_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/synthetic_demo"))
    parser.add_argument("--output", type=Path, default=Path("assets/readme/evidence"))
    args = parser.parse_args()
    for path in generate(args.results, args.output):
        print(path)


if __name__ == "__main__":
    main()
