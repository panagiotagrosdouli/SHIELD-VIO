"""Explicit, reproducible failure labels for estimator-health experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class FailureKind(str, Enum):
    POSITION_ERROR = "position_error"
    RELATIVE_POSE_ERROR = "relative_pose_error"
    COVARIANCE_INSTABILITY = "covariance_instability"
    INNOVATION_INCONSISTENCY = "innovation_inconsistency"
    TRACKING_LOSS = "tracking_loss"
    BIAS_INSTABILITY = "bias_instability"
    UNSAFE_NAVIGATION = "unsafe_navigation"


@dataclass(frozen=True)
class FailureThresholds:
    position_error_m: float = 1.0
    relative_pose_error_m: float = 0.5
    covariance_trace: float = 2.0
    nis: float = 11.345
    min_feature_count: int = 12
    max_bias_norm: float = 0.5
    unsafe_clearance_m: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.position_error_m,
            self.relative_pose_error_m,
            self.covariance_trace,
            self.nis,
            self.max_bias_norm,
            self.unsafe_clearance_m,
        )
        if any(value <= 0.0 for value in values) or self.min_feature_count < 0:
            raise ValueError("failure thresholds must be positive")


@dataclass(frozen=True)
class FailureObservation:
    timestamp_s: float
    position_error_m: float
    relative_pose_error_m: float
    covariance_trace: float
    nis: float
    feature_count: int
    tracking_valid: bool
    bias_norm: float
    navigation_clearance_m: float


@dataclass(frozen=True)
class FailureLabel:
    timestamp_s: float
    failed: bool
    kinds: tuple[FailureKind, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["kinds"] = [kind.value for kind in self.kinds]
        return data


def label_failure(
    observation: FailureObservation,
    thresholds: FailureThresholds | None = None,
) -> FailureLabel:
    """Label a sample from observable criteria, never degradation metadata."""
    cfg = thresholds or FailureThresholds()
    kinds: list[FailureKind] = []
    if observation.position_error_m > cfg.position_error_m:
        kinds.append(FailureKind.POSITION_ERROR)
    if observation.relative_pose_error_m > cfg.relative_pose_error_m:
        kinds.append(FailureKind.RELATIVE_POSE_ERROR)
    if observation.covariance_trace > cfg.covariance_trace:
        kinds.append(FailureKind.COVARIANCE_INSTABILITY)
    if observation.nis > cfg.nis:
        kinds.append(FailureKind.INNOVATION_INCONSISTENCY)
    if not observation.tracking_valid or observation.feature_count < cfg.min_feature_count:
        kinds.append(FailureKind.TRACKING_LOSS)
    if observation.bias_norm > cfg.max_bias_norm:
        kinds.append(FailureKind.BIAS_INSTABILITY)
    if observation.navigation_clearance_m < cfg.unsafe_clearance_m:
        kinds.append(FailureKind.UNSAFE_NAVIGATION)
    return FailureLabel(observation.timestamp_s, bool(kinds), tuple(kinds))
