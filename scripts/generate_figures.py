"""Generate reproducible SHIELD-VIO Synthetic Demo figures from CSV outputs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shield_vio.simulation.synthetic_vio import SyntheticVIOConfig, run_synthetic_vio


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data: dict[str, list] = {}
    for row in rows:
        for key, value in row.items():
            try:
                data.setdefault(key, []).append(float(value))
            except ValueError:
                data.setdefault(key, []).append(value)
    return {k: np.asarray(v, dtype=object if isinstance(v[0], str) else float) for k, v in data.items()}


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def shade_events(ax: plt.Axes, t: np.ndarray, events: np.ndarray) -> None:
    active = events != "nominal"
    start = None
    current = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start, current = float(t[i]), str(events[i])
        end_of_span = start is not None and (i == len(active) - 1 or not active[i + 1] or events[i + 1] != current)
        if end_of_span:
            ax.axvspan(start, float(t[i]), alpha=0.18)
            start = None


def line_plot(x: np.ndarray, y: np.ndarray, title: str, ylabel: str, path: Path, events: np.ndarray | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if events is not None:
        shade_events(ax, x, events)
    return save(fig, path)


def diagram(path: Path, title: str, boxes: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis("off")
    for i, box in enumerate(boxes):
        ax.text(i, 0.5, box, ha="center", va="center", bbox={"boxstyle": "round", "fc": "white"})
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i + 0.72, 0.5), xytext=(i + 0.28, 0.5), arrowprops={"arrowstyle": "->"})
    ax.set_xlim(-0.5, len(boxes) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_title(f"SHIELD-VIO {title} — Synthetic Demo")
    return save(fig, path)


def generate_figures(results_dir: Path, assets_dir: Path = Path("assets/figures"), figures_dir: Path = Path("results/figures")) -> list[Path]:
    if not (results_dir / "ground_truth.csv").exists():
        run_synthetic_vio(SyntheticVIOConfig(output_dir=str(results_dir)))
    gt = read_csv(results_dir / "ground_truth.csv")
    est = read_csv(results_dir / "estimated_trajectory.csv")
    unc = read_csv(results_dir / "uncertainty.csv")
    vq = read_csv(results_dir / "visual_quality.csv")
    risk = read_csv(results_dir / "risk_score.csv")
    shield = read_csv(results_dir / "shield_events.csv")
    written: list[Path] = []

    for root in (assets_dir, figures_dir):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(gt["x"], gt["y"], label="ground truth")
        ax.plot(est["x"], est["y"], label="estimated")
        ax.set_title("SHIELD-VIO Synthetic Demo: trajectory comparison")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)
        written.append(save(fig, root / "trajectory_comparison.png"))

    written.append(line_plot(unc["t"], unc["trace"], "Covariance trace — Synthetic Demo", "trace(P)", figures_dir / "covariance_trace.png", risk["event"]))
    written.append(line_plot(risk["t"], risk["risk_score"], "Risk score — Synthetic Demo", "risk", figures_dir / "risk_score.png", risk["event"]))
    written.append(line_plot(vq["t"], vq["visual_quality"], "Visual quality — Synthetic Demo", "quality", figures_dir / "visual_quality.png", vq["event"]))
    status_map = {"NORMAL": 0, "WARNING": 1, "SLOW_DOWN": 2, "HALT": 3, "RELOCALIZE_REQUESTED": 4}
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.step(shield["t"], [status_map[str(s)] for s in shield["state"]], where="post")
    ax.set_yticks(list(status_map.values()), list(status_map.keys()))
    ax.set_title("Shield activation timeline — Synthetic Demo")
    ax.set_xlabel("time [s]")
    ax.grid(True, alpha=0.3)
    written.append(save(fig, figures_dir / "shield_timeline.png"))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(unc["t"], unc["nees"], label="NEES")
    ax.plot(unc["t"], unc["nis"], label="NIS")
    ax.set_title("NEES/NIS consistency — Synthetic Demo")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlabel("time [s]")
    written.append(save(fig, figures_dir / "nees_nis.png"))
    written.append(line_plot(vq["t"], vq["feature_count"], "Feature dropout timeline — Synthetic Demo", "features", figures_dir / "feature_degradation_timeline.png", vq["event"]))
    written.append(diagram(assets_dir / "architecture_diagram.png", "Architecture", ["Camera/IMU", "Synthetic VIO", "Uncertainty", "Risk", "Shield"]))
    written.append(diagram(assets_dir / "vio_pipeline_diagram.png", "VIO Pipeline", ["IMU", "Propagate", "Visual update", "Covariance", "NEES/NIS"]))
    written.append(diagram(assets_dir / "safety_shield_diagram.png", "Safety Shield", ["NORMAL", "WARNING", "SLOW_DOWN", "HALT", "RELOCALIZE"]))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/synthetic_demo")
    args = parser.parse_args()
    print("\n".join(str(p) for p in generate_figures(Path(args.results))))


if __name__ == "__main__":
    main()
