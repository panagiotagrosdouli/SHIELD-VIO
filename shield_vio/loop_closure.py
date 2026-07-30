"""Loop-closure candidate search and pose-graph data structures.

This module provides a dependency-light foundation for later geometric
verification and nonlinear pose-graph optimization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoopCandidate:
    """One appearance-based loop-closure candidate."""

    query_id: int
    match_id: int
    score: float


class DescriptorDatabase:
    """Deterministic cosine-similarity database for global descriptors."""

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._descriptors: list[np.ndarray] = []
        self._dimension: int | None = None

    def add(self, frame_id: int, descriptor: np.ndarray) -> None:
        """Insert one finite, non-zero descriptor under a unique frame id."""

        if frame_id in self._ids:
            raise ValueError(f"duplicate frame id: {frame_id}")
        normalized = _normalized_descriptor(descriptor)
        if self._dimension is None:
            self._dimension = len(normalized)
        elif len(normalized) != self._dimension:
            raise ValueError("descriptor dimension does not match database")
        self._ids.append(frame_id)
        self._descriptors.append(normalized)

    def query(
        self,
        frame_id: int,
        descriptor: np.ndarray,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        exclusion_window: int = 30,
    ) -> tuple[LoopCandidate, ...]:
        """Return ranked candidates outside the temporal exclusion window."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        if exclusion_window < 0:
            raise ValueError("exclusion_window must be non-negative")
        query_descriptor = _normalized_descriptor(descriptor)
        if self._dimension is not None and len(query_descriptor) != self._dimension:
            raise ValueError("descriptor dimension does not match database")

        candidates: list[LoopCandidate] = []
        for match_id, stored in zip(self._ids, self._descriptors, strict=True):
            if abs(frame_id - match_id) <= exclusion_window:
                continue
            score = float(query_descriptor @ stored)
            if score >= min_score:
                candidates.append(LoopCandidate(frame_id, match_id, score))
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.match_id))
        return tuple(candidates[:top_k])


@dataclass(frozen=True)
class PoseGraphNode:
    """Pose-graph node represented by a world-frame SE(3) transform."""

    node_id: int
    pose: np.ndarray
    fixed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", _validated_transform(self.pose))


@dataclass(frozen=True)
class PoseGraphEdge:
    """Relative-pose constraint between two graph nodes."""

    source_id: int
    target_id: int
    transform: np.ndarray
    information: np.ndarray
    kind: str = "odometry"

    def __post_init__(self) -> None:
        if self.source_id == self.target_id:
            raise ValueError("pose-graph edge must connect different nodes")
        if self.kind not in {"odometry", "loop"}:
            raise ValueError("edge kind must be 'odometry' or 'loop'")
        object.__setattr__(self, "transform", _validated_transform(self.transform))
        information = np.asarray(self.information, dtype=float)
        if information.shape != (6, 6) or not np.all(np.isfinite(information)):
            raise ValueError("information matrix must be finite with shape (6, 6)")
        if not np.allclose(information, information.T, atol=1e-10):
            raise ValueError("information matrix must be symmetric")
        if np.min(np.linalg.eigvalsh(information)) <= 0:
            raise ValueError("information matrix must be positive definite")
        object.__setattr__(self, "information", information.copy())


class PoseGraph:
    """Validated in-memory pose graph, independent of optimizer backend."""

    def __init__(self) -> None:
        self._nodes: dict[int, PoseGraphNode] = {}
        self._edges: list[PoseGraphEdge] = []

    @property
    def nodes(self) -> tuple[PoseGraphNode, ...]:
        return tuple(self._nodes[node_id] for node_id in sorted(self._nodes))

    @property
    def edges(self) -> tuple[PoseGraphEdge, ...]:
        return tuple(self._edges)

    def add_node(self, node: PoseGraphNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate pose-graph node: {node.node_id}")
        self._nodes[node.node_id] = node

    def add_edge(self, edge: PoseGraphEdge) -> None:
        missing = {
            node_id
            for node_id in (edge.source_id, edge.target_id)
            if node_id not in self._nodes
        }
        if missing:
            raise ValueError(f"edge references missing nodes: {sorted(missing)}")
        self._edges.append(edge)


def _normalized_descriptor(descriptor: np.ndarray) -> np.ndarray:
    vector = np.asarray(descriptor, dtype=float)
    if vector.ndim != 1 or len(vector) == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("descriptor must be a non-empty finite vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("descriptor norm must be positive")
    return vector / norm


def _validated_transform(transform: np.ndarray) -> np.ndarray:
    pose = np.asarray(transform, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("SE(3) transform must be finite with shape (4, 4)")
    if not np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-10):
        raise ValueError("invalid homogeneous transform bottom row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError("transform rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError("transform rotation determinant must be +1")
    return pose.copy()
