from __future__ import annotations

import numpy as np
import pytest

from shield_vio.loop_closure import PoseGraph, PoseGraphEdge, PoseGraphNode
from shield_vio.pose_graph_optimization import optimize_fixed_rotation_translations


def _pose(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = [x, y, z]
    return pose


def _edge(source: int, target: int, dx: float, *, weight: float = 1.0) -> PoseGraphEdge:
    transform = np.eye(4)
    transform[0, 3] = dx
    return PoseGraphEdge(source, target, transform, np.eye(6) * weight)


def test_loop_constraint_reduces_translation_drift() -> None:
    graph = PoseGraph()
    graph.add_node(PoseGraphNode(0, _pose(0.0), fixed=True))
    graph.add_node(PoseGraphNode(1, _pose(1.2)))
    graph.add_node(PoseGraphNode(2, _pose(2.5)))
    graph.add_edge(_edge(0, 1, 1.0))
    graph.add_edge(_edge(1, 2, 1.0))
    graph.add_edge(_edge(0, 2, 2.0, weight=4.0))

    result = optimize_fixed_rotation_translations(graph)
    optimized = {node.node_id: node for node in result.nodes}

    assert optimized[0].pose == pytest.approx(_pose(0.0))
    assert optimized[1].pose[:3, 3] == pytest.approx([1.0, 0.0, 0.0])
    assert optimized[2].pose[:3, 3] == pytest.approx([2.0, 0.0, 0.0])
    assert result.final_weighted_rmse < result.initial_weighted_rmse
    assert result.rank == result.variable_count == 6


def test_relative_translation_is_rotated_from_source_frame() -> None:
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    source_pose = np.eye(4)
    source_pose[:3, :3] = rotation

    graph = PoseGraph()
    graph.add_node(PoseGraphNode(10, source_pose, fixed=True))
    graph.add_node(PoseGraphNode(11, _pose(0.5, 0.5)))
    graph.add_edge(_edge(10, 11, 2.0))

    result = optimize_fixed_rotation_translations(graph)
    target = next(node for node in result.nodes if node.node_id == 11)
    assert target.pose[:3, 3] == pytest.approx([0.0, 2.0, 0.0])
    assert target.pose[:3, :3] == pytest.approx(np.eye(3))


def test_requires_anchor_edges_and_full_translation_rank() -> None:
    graph = PoseGraph()
    graph.add_node(PoseGraphNode(0, _pose(0.0)))
    graph.add_node(PoseGraphNode(1, _pose(1.0)))
    graph.add_edge(_edge(0, 1, 1.0))
    with pytest.raises(ValueError, match="fixed node"):
        optimize_fixed_rotation_translations(graph)

    empty_edges = PoseGraph()
    empty_edges.add_node(PoseGraphNode(0, _pose(0.0), fixed=True))
    with pytest.raises(ValueError, match="at least one edge"):
        optimize_fixed_rotation_translations(empty_edges)

    disconnected = PoseGraph()
    disconnected.add_node(PoseGraphNode(0, _pose(0.0), fixed=True))
    disconnected.add_node(PoseGraphNode(1, _pose(1.0)))
    disconnected.add_node(PoseGraphNode(2, _pose(2.0)))
    disconnected.add_edge(_edge(0, 1, 1.0))
    with pytest.raises(ValueError, match="underconstrained"):
        optimize_fixed_rotation_translations(disconnected)
