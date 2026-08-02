"""Reproducible synthetic experiments for translation-only loop-closure correction."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from shield_vio.loop_closure import PoseGraph, PoseGraphEdge, PoseGraphNode
from shield_vio.pose_graph_optimization import optimize_fixed_rotation_translations


@dataclass(frozen=True)
class LoopClosureExperimentConfig:
    """Configuration for a controlled closed-trajectory drift experiment."""

    node_count: int = 40
    radius_m: float = 10.0
    odometry_noise_std_m: float = 0.04
    odometry_bias_m: tuple[float, float, float] = (0.025, -0.01, 0.0)
    loop_noise_std_m: float = 0.02
    odometry_information: float = 625.0
    loop_information: float = 2500.0

    def __post_init__(self) -> None:
        if self.node_count < 4:
            raise ValueError("node_count must be at least four")
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive")
        if self.odometry_noise_std_m < 0 or self.loop_noise_std_m < 0:
            raise ValueError("noise standard deviations must be non-negative")
        if self.odometry_information <= 0 or self.loop_information <= 0:
            raise ValueError("information weights must be positive")


@dataclass(frozen=True)
class LoopClosureTrialResult:
    seed: int
    odometry_rmse_m: float
    loop_corrected_rmse_m: float
    odometry_endpoint_error_m: float
    loop_corrected_endpoint_error_m: float
    improvement_fraction: float
    optimizer_weighted_rmse: float


@dataclass(frozen=True)
class LoopClosureExperimentSummary:
    config: LoopClosureExperimentConfig
    trials: tuple[LoopClosureTrialResult, ...]
    mean_odometry_rmse_m: float
    mean_loop_corrected_rmse_m: float
    mean_improvement_fraction: float
    improved_trial_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "trials": [asdict(trial) for trial in self.trials],
            "mean_odometry_rmse_m": self.mean_odometry_rmse_m,
            "mean_loop_corrected_rmse_m": self.mean_loop_corrected_rmse_m,
            "mean_improvement_fraction": self.mean_improvement_fraction,
            "improved_trial_fraction": self.improved_trial_fraction,
            "claim_boundary": (
                "Synthetic fixed-rotation translation experiment; not public-dataset "
                "or full nonlinear SE(3) evidence."
            ),
        }


def run_loop_closure_trial(
    seed: int, config: LoopClosureExperimentConfig | None = None
) -> LoopClosureTrialResult:
    """Compare odometry-only drift with one terminal loop constraint."""

    cfg = config or LoopClosureExperimentConfig()
    rng = np.random.default_rng(seed)
    ground_truth = _closed_circle(cfg.node_count, cfg.radius_m)
    bias = np.asarray(cfg.odometry_bias_m, dtype=float)

    measured_deltas = []
    estimated = [ground_truth[0].copy()]
    for index in range(cfg.node_count - 1):
        true_delta = ground_truth[index + 1] - ground_truth[index]
        measured = true_delta + bias + rng.normal(0.0, cfg.odometry_noise_std_m, 3)
        measured_deltas.append(measured)
        estimated.append(estimated[-1] + measured)
    estimated_array = np.asarray(estimated)

    odometry_graph = _build_graph(
        estimated_array,
        measured_deltas,
        odometry_information=cfg.odometry_information,
    )
    odometry_nodes = optimize_fixed_rotation_translations(odometry_graph).nodes
    odometry_positions = np.asarray([node.pose[:3, 3] for node in odometry_nodes])

    loop_graph = _build_graph(
        estimated_array,
        measured_deltas,
        odometry_information=cfg.odometry_information,
    )
    loop_delta = ground_truth[0] - ground_truth[-1]
    loop_delta += rng.normal(0.0, cfg.loop_noise_std_m, 3)
    loop_graph.add_edge(
        _edge(
            cfg.node_count - 1,
            0,
            loop_delta,
            cfg.loop_information,
            kind="loop",
        )
    )
    optimized = optimize_fixed_rotation_translations(loop_graph)
    corrected_positions = np.asarray([node.pose[:3, 3] for node in optimized.nodes])

    odometry_rmse = _trajectory_rmse(odometry_positions, ground_truth)
    corrected_rmse = _trajectory_rmse(corrected_positions, ground_truth)
    denominator = max(odometry_rmse, np.finfo(float).eps)
    return LoopClosureTrialResult(
        seed=int(seed),
        odometry_rmse_m=odometry_rmse,
        loop_corrected_rmse_m=corrected_rmse,
        odometry_endpoint_error_m=float(np.linalg.norm(odometry_positions[-1] - ground_truth[-1])),
        loop_corrected_endpoint_error_m=float(
            np.linalg.norm(corrected_positions[-1] - ground_truth[-1])
        ),
        improvement_fraction=float((odometry_rmse - corrected_rmse) / denominator),
        optimizer_weighted_rmse=optimized.final_weighted_rmse,
    )


def run_loop_closure_experiment(
    seeds: range | list[int] | tuple[int, ...],
    config: LoopClosureExperimentConfig | None = None,
) -> LoopClosureExperimentSummary:
    """Run matched multi-seed trials and aggregate correction effectiveness."""

    cfg = config or LoopClosureExperimentConfig()
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("at least one seed is required")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be unique")

    trials = tuple(run_loop_closure_trial(seed, cfg) for seed in seed_values)
    odometry = np.asarray([trial.odometry_rmse_m for trial in trials])
    corrected = np.asarray([trial.loop_corrected_rmse_m for trial in trials])
    improvements = np.asarray([trial.improvement_fraction for trial in trials])
    return LoopClosureExperimentSummary(
        config=cfg,
        trials=trials,
        mean_odometry_rmse_m=float(np.mean(odometry)),
        mean_loop_corrected_rmse_m=float(np.mean(corrected)),
        mean_improvement_fraction=float(np.mean(improvements)),
        improved_trial_fraction=float(np.mean(corrected < odometry)),
    )


def _closed_circle(node_count: int, radius_m: float) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, node_count, endpoint=True)
    return np.column_stack(
        [radius_m * np.cos(angles), radius_m * np.sin(angles), np.zeros(node_count)]
    )


def _build_graph(
    positions: np.ndarray,
    deltas: list[np.ndarray],
    *,
    odometry_information: float,
) -> PoseGraph:
    graph = PoseGraph()
    for index, position in enumerate(positions):
        pose = np.eye(4)
        pose[:3, 3] = position
        graph.add_node(PoseGraphNode(index, pose, fixed=index == 0))
    for index, delta in enumerate(deltas):
        graph.add_edge(_edge(index, index + 1, delta, odometry_information))
    return graph


def _edge(
    source_id: int,
    target_id: int,
    translation: np.ndarray,
    information_weight: float,
    *,
    kind: str = "odometry",
) -> PoseGraphEdge:
    transform = np.eye(4)
    transform[:3, 3] = translation
    information = np.eye(6) * information_weight
    return PoseGraphEdge(source_id, target_id, transform, information, kind=kind)


def _trajectory_rmse(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    errors = np.linalg.norm(estimated - ground_truth, axis=1)
    return float(np.sqrt(np.mean(errors**2)))
