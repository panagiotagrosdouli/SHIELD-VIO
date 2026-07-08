"""Causal failure graph for interpretable VIO degradation diagnosis.

This is a lightweight causal model, not a learned structural causal model. It
encodes the research hypothesis that several observable detector failures are
intermediate symptoms between root causes and navigation failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True)
class CausalEdge:
    """Directed weighted relation from one degradation variable to another."""

    source: str
    target: str
    weight: float


@dataclass
class FailureCausalGraph:
    """Interpretable causal graph over degradation symptoms and failure risks."""

    edges: list[CausalEdge] = field(default_factory=lambda: [
        CausalEdge("motion_blur", "feature_collapse", 0.55),
        CausalEdge("low_texture", "feature_collapse", 0.75),
        CausalEdge("feature_collapse", "reprojection_error", 0.60),
        CausalEdge("imu_inconsistency", "estimator_inconsistency", 0.70),
        CausalEdge("reprojection_error", "pose_drift", 0.65),
        CausalEdge("estimator_inconsistency", "pose_drift", 0.50),
        CausalEdge("pose_drift", "localization_failure", 0.80),
    ])

    def infer_risk(self, symptoms: Mapping[str, float]) -> dict[str, float]:
        """Propagate symptom severities through the graph.

        Args:
            symptoms: Mapping from symptom name to severity in [0, 1]. Here 1 is
                severe degradation, unlike detector health scores where 1 is good.
        """
        risk = {name: _clip01(value) for name, value in symptoms.items()}
        for edge in self.edges:
            propagated = risk.get(edge.source, 0.0) * _clip01(edge.weight)
            risk[edge.target] = max(risk.get(edge.target, 0.0), _clip01(propagated))
        return risk

    def from_health_scores(self, health_scores: Mapping[str, float]) -> dict[str, float]:
        """Convert detector health scores to causal symptom risks."""
        symptoms = {
            "low_texture": 1.0 - _clip01(health_scores.get("image_entropy", 1.0)),
            "motion_blur": 1.0 - _clip01(health_scores.get("motion_blur", 1.0)),
            "feature_collapse": 1.0 - _clip01(health_scores.get("feature_collapse", 1.0)),
            "imu_inconsistency": 1.0 - _clip01(health_scores.get("imu_consistency", 1.0)),
            "reprojection_error": 1.0 - _clip01(health_scores.get("reprojection_error", 1.0)),
        }
        return self.infer_risk(symptoms)
