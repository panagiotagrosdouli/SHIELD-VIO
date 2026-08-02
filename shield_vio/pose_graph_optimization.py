"""Dependency-light translation optimization for validated pose graphs.

This module solves node translations while keeping every node rotation fixed.
It is an auditable intermediate step between pose-graph construction and a full
nonlinear SE(3) optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shield_vio.loop_closure import PoseGraph, PoseGraphNode


@dataclass(frozen=True)
class PoseGraphOptimizationResult:
    """Optimized graph state and residual diagnostics."""

    nodes: tuple[PoseGraphNode, ...]
    initial_weighted_rmse: float
    final_weighted_rmse: float
    rank: int
    variable_count: int


def optimize_fixed_rotation_translations(graph: PoseGraph) -> PoseGraphOptimizationResult:
    """Optimize graph translations with weighted linear least squares.

    For an edge from source ``i`` to target ``j``, the stored relative
    translation is interpreted in the source-node frame. With rotations held
    fixed, each constraint is

    ``p_j - p_i = R_i t_ij``.

    At least one node must be marked ``fixed`` to remove gauge freedom. The
    returned nodes preserve input rotations and fixed-node poses exactly.
    """

    nodes = graph.nodes
    edges = graph.edges
    if not nodes:
        raise ValueError("pose graph must contain at least one node")
    if not edges:
        raise ValueError("pose graph must contain at least one edge")

    by_id = {node.node_id: node for node in nodes}
    fixed_ids = {node.node_id for node in nodes if node.fixed}
    if not fixed_ids:
        raise ValueError("pose graph requires at least one fixed node")

    variable_ids = [node.node_id for node in nodes if not node.fixed]
    index = {node_id: offset for offset, node_id in enumerate(variable_ids)}
    variable_count = 3 * len(variable_ids)

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for edge in edges:
        source = by_id[edge.source_id]
        target = by_id[edge.target_id]
        source_rotation = source.pose[:3, :3]
        measured_world_delta = source_rotation @ edge.transform[:3, 3]

        equation = np.zeros((3, variable_count), dtype=float)
        right_hand_side = measured_world_delta.copy()

        if source.node_id in index:
            start = 3 * index[source.node_id]
            equation[:, start : start + 3] -= np.eye(3)
        else:
            right_hand_side += source.pose[:3, 3]

        if target.node_id in index:
            start = 3 * index[target.node_id]
            equation[:, start : start + 3] += np.eye(3)
        else:
            right_hand_side -= target.pose[:3, 3]

        translational_information = edge.information[:3, :3]
        whitening = np.linalg.cholesky(translational_information)
        rows.append(whitening @ equation)
        targets.append(whitening @ right_hand_side)

    design = np.vstack(rows)
    observation = np.concatenate(targets)

    if variable_count:
        solution, _, rank, _ = np.linalg.lstsq(design, observation, rcond=None)
        if rank < variable_count:
            raise ValueError("pose graph translation system is underconstrained")
        initial_vector = np.concatenate(
            [by_id[node_id].pose[:3, 3] for node_id in variable_ids]
        )
    else:
        solution = np.empty(0, dtype=float)
        initial_vector = solution.copy()
        rank = 0

    optimized_nodes: list[PoseGraphNode] = []
    for node in nodes:
        if node.node_id in index:
            start = 3 * index[node.node_id]
            pose = node.pose.copy()
            pose[:3, 3] = solution[start : start + 3]
            optimized_nodes.append(PoseGraphNode(node.node_id, pose, fixed=False))
        else:
            optimized_nodes.append(PoseGraphNode(node.node_id, node.pose, fixed=True))

    initial_rmse = _weighted_rmse(design, observation, initial_vector)
    final_rmse = _weighted_rmse(design, observation, solution)
    return PoseGraphOptimizationResult(
        nodes=tuple(optimized_nodes),
        initial_weighted_rmse=initial_rmse,
        final_weighted_rmse=final_rmse,
        rank=int(rank),
        variable_count=variable_count,
    )


def _weighted_rmse(design: np.ndarray, observation: np.ndarray, state: np.ndarray) -> float:
    residual = design @ state - observation
    return float(np.sqrt(np.mean(residual**2)))
