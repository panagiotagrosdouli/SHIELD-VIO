from __future__ import annotations

import numpy as np
import pytest

from shield_vio.loop_closure import (
    DescriptorDatabase,
    PoseGraph,
    PoseGraphEdge,
    PoseGraphNode,
)


def _pose(x: float = 0.0) -> np.ndarray:
    transform = np.eye(4)
    transform[0, 3] = x
    return transform


def test_descriptor_database_ranks_and_excludes_nearby_frames() -> None:
    database = DescriptorDatabase()
    database.add(0, np.array([1.0, 0.0]))
    database.add(10, np.array([0.8, 0.2]))
    database.add(100, np.array([0.99, 0.01]))

    candidates = database.query(
        110,
        np.array([1.0, 0.0]),
        top_k=2,
        min_score=0.7,
        exclusion_window=20,
    )

    assert [candidate.match_id for candidate in candidates] == [0, 10]
    assert candidates[0].score > candidates[1].score


def test_descriptor_database_rejects_invalid_entries() -> None:
    database = DescriptorDatabase()
    database.add(1, np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="duplicate"):
        database.add(1, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="dimension"):
        database.add(2, np.ones(3))
    with pytest.raises(ValueError, match="norm"):
        database.query(5, np.zeros(2))


def test_pose_graph_accepts_odometry_and_loop_constraints() -> None:
    graph = PoseGraph()
    graph.add_node(PoseGraphNode(0, _pose(), fixed=True))
    graph.add_node(PoseGraphNode(1, _pose(1.0)))
    graph.add_node(PoseGraphNode(2, _pose(2.0)))
    information = np.eye(6)

    graph.add_edge(PoseGraphEdge(0, 1, _pose(1.0), information))
    graph.add_edge(PoseGraphEdge(2, 0, _pose(-2.0), information, kind="loop"))

    assert [node.node_id for node in graph.nodes] == [0, 1, 2]
    assert [edge.kind for edge in graph.edges] == ["odometry", "loop"]


def test_pose_graph_rejects_missing_nodes_and_bad_information() -> None:
    graph = PoseGraph()
    graph.add_node(PoseGraphNode(0, _pose(), fixed=True))

    with pytest.raises(ValueError, match="missing nodes"):
        graph.add_edge(PoseGraphEdge(0, 1, _pose(), np.eye(6)))
    with pytest.raises(ValueError, match="positive definite"):
        PoseGraphEdge(0, 1, _pose(), np.zeros((6, 6)))


def test_pose_validation_rejects_non_rotation() -> None:
    invalid = np.eye(4)
    invalid[0, 0] = 2.0

    with pytest.raises(ValueError, match="orthonormal"):
        PoseGraphNode(0, invalid)
