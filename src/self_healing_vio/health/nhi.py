"""Navigation Health Index computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class NavigationHealthIndex:
    """Weighted health fusion into a scalar index in [0, 100]."""

    weights: Mapping[str, float] = field(default_factory=dict)
    risk_penalty: float = 0.0

    def compute(self, health_scores: Mapping[str, float], risk: float = 0.0) -> float:
        if not health_scores:
            raise ValueError("health_scores must not be empty")
        weights = self.weights or {name: 1.0 for name in health_scores}
        total_weight = float(sum(max(weights.get(name, 0.0), 0.0) for name in health_scores))
        if total_weight <= 0.0:
            raise ValueError("at least one positive weight is required")
        fused = 0.0
        for name, score in health_scores.items():
            fused += max(weights.get(name, 0.0), 0.0) * float(np.clip(score, 0.0, 1.0))
        fused /= total_weight
        fused -= self.risk_penalty * float(np.clip(risk, 0.0, 1.0))
        return float(np.clip(100.0 * fused, 0.0, 100.0))
