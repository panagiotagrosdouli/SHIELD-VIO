"""Recovery policy definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class RecoveryAction(str, Enum):
    """Supported prototype recovery actions."""

    NO_ACTION = "no_action"
    REDUCE_SPEED = "reduce_speed"
    INCREASE_FEATURE_BUDGET = "increase_feature_budget"
    RELAX_RANSAC = "relax_ransac"
    RESET_VISUAL_FRONTEND = "reset_visual_frontend"
    REINITIALIZE_BIAS = "reinitialize_bias"
    INFLATE_COVARIANCE = "inflate_covariance"
    REQUEST_HOVER = "request_hover"
    RELOCALIZE = "relocalize"
    FALLBACK_INERTIAL = "fallback_inertial"
    EMERGENCY_LAND = "emergency_land"


@dataclass(frozen=True)
class RecoveryDecision:
    """Legacy controller recovery decision."""

    action: str
    priority: int
    reason: str = ""


class RecoveryPolicy:
    """Compatibility policy used by the original controller tests."""

    def decide(self, health: Any) -> RecoveryDecision:
        if health.level == "healthy":
            return RecoveryDecision("continue_nominal_tracking", 1, "VIO health is nominal")
        if health.level == "warning":
            return RecoveryDecision("increase_inertial_weight", 2, "VIO health is degraded")
        return RecoveryDecision("trigger_relocalization", 3, "VIO health is critical")


@dataclass
class RuleBasedRecoveryPolicy:
    """Transparent diagnosis-conditioned recovery baseline."""

    critical_nhi: float = 25.0
    degraded_nhi: float = 55.0
    posterior_threshold: float = 0.35

    def select_action(self, health_scores: Mapping[str, float], posterior: Mapping[str, float], nhi: float) -> RecoveryAction:
        if nhi <= self.critical_nhi:
            return RecoveryAction.REQUEST_HOVER

        dominant_cause = max(posterior, key=posterior.get)
        confidence = posterior[dominant_cause]

        if confidence < self.posterior_threshold and nhi < self.degraded_nhi:
            return RecoveryAction.INFLATE_COVARIANCE
        if dominant_cause == "camera":
            return RecoveryAction.RESET_VISUAL_FRONTEND if health_scores.get("motion_blur", 1.0) < 0.3 else RecoveryAction.INCREASE_FEATURE_BUDGET
        if dominant_cause == "environment":
            return RecoveryAction.INCREASE_FEATURE_BUDGET
        if dominant_cause == "motion":
            return RecoveryAction.REDUCE_SPEED
        if dominant_cause == "sensor":
            return RecoveryAction.REINITIALIZE_BIAS
        if dominant_cause == "algorithm":
            return RecoveryAction.RELOCALIZE if nhi < self.degraded_nhi else RecoveryAction.RELAX_RANSAC
        return RecoveryAction.NO_ACTION

    @property
    def valid_actions(self) -> set[str]:
        return {action.value for action in RecoveryAction}
