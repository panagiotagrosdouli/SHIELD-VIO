"""Typed, role-aware health schema with explicit missingness and provenance."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Iterable


class FeatureRole(str, Enum):
    DEPLOYABLE_FEATURE = "DEPLOYABLE_FEATURE"
    TARGET = "TARGET"
    GROUPING_METADATA = "GROUPING_METADATA"
    EXPERIMENT_ORACLE = "EXPERIMENT_ORACLE"
    EVALUATION_ONLY = "EVALUATION_ONLY"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    category: str
    unit: str
    role: FeatureRole
    source: str
    requires_ground_truth: bool = False
    backend_dependency: str = "optional"
    missing_semantics: str = "unavailable_or_not_observed"


@dataclass(frozen=True)
class HealthValue:
    value: float | bool | None
    valid: bool
    source_timestamp_ns: int | None

    @classmethod
    def missing(cls) -> "HealthValue":
        return cls(value=None, valid=False, source_timestamp_ns=None)

    def validate_at(self, timestamp_ns: int) -> None:
        if self.valid and self.value is None:
            raise ValueError("valid health values must contain a value")
        if self.source_timestamp_ns is not None and self.source_timestamp_ns > timestamp_ns:
            raise ValueError("health value cannot depend on a future source timestamp")


@dataclass(frozen=True)
class VisualHealth:
    tracked_feature_count: HealthValue = HealthValue.missing()
    surviving_feature_count: HealthValue = HealthValue.missing()
    feature_survival_ratio: HealthValue = HealthValue.missing()
    mean_track_age_s: HealthValue = HealthValue.missing()
    median_track_age_s: HealthValue = HealthValue.missing()
    optical_flow_mean_px: HealthValue = HealthValue.missing()
    optical_flow_std_px: HealthValue = HealthValue.missing()
    forward_backward_error_px: HealthValue = HealthValue.missing()
    inlier_ratio: HealthValue = HealthValue.missing()
    outlier_ratio: HealthValue = HealthValue.missing()
    reprojection_residual_px: HealthValue = HealthValue.missing()
    blur_score: HealthValue = HealthValue.missing()
    brightness: HealthValue = HealthValue.missing()
    contrast: HealthValue = HealthValue.missing()
    entropy: HealthValue = HealthValue.missing()
    frame_available: HealthValue = HealthValue.missing()
    visual_update_available: HealthValue = HealthValue.missing()
    rejected_visual_update: HealthValue = HealthValue.missing()


@dataclass(frozen=True)
class InertialHealth:
    accel_norm_m_s2: HealthValue = HealthValue.missing()
    gyro_norm_rad_s: HealthValue = HealthValue.missing()
    accel_bias_norm_m_s2: HealthValue = HealthValue.missing()
    gyro_bias_norm_rad_s: HealthValue = HealthValue.missing()
    bias_change: HealthValue = HealthValue.missing()
    saturated: HealthValue = HealthValue.missing()
    clipped: HealthValue = HealthValue.missing()
    packet_available: HealthValue = HealthValue.missing()
    seconds_since_previous_packet: HealthValue = HealthValue.missing()
    packet_loss_detected: HealthValue = HealthValue.missing()
    scale_anomaly: HealthValue = HealthValue.missing()


@dataclass(frozen=True)
class EstimatorHealth:
    innovation_norm: HealthValue = HealthValue.missing()
    nis: HealthValue = HealthValue.missing()
    covariance_trace: HealthValue = HealthValue.missing()
    covariance_condition_number: HealthValue = HealthValue.missing()
    covariance_logdet: HealthValue = HealthValue.missing()
    covariance_growth_per_s: HealthValue = HealthValue.missing()
    visual_update_accepted: HealthValue = HealthValue.missing()
    visual_update_rejected: HealthValue = HealthValue.missing()
    seconds_since_last_successful_update: HealthValue = HealthValue.missing()
    reset: HealthValue = HealthValue.missing()
    relocalization: HealthValue = HealthValue.missing()


@dataclass(frozen=True)
class EvaluationDiagnostics:
    ground_truth_position_error_m: HealthValue = HealthValue.missing()
    ground_truth_rotation_error_deg: HealthValue = HealthValue.missing()
    nees: HealthValue = HealthValue.missing()


@dataclass(frozen=True)
class HealthSample:
    timestamp_ns: int
    sequence_id: str | None = None
    estimator_id: str | None = None
    schema_version: str = "SHIELD_VIO_HEALTH_V1"
    source_available: bool = True
    visual: VisualHealth = VisualHealth()
    inertial: InertialHealth = InertialHealth()
    estimator: EstimatorHealth = EstimatorHealth()
    evaluation: EvaluationDiagnostics = EvaluationDiagnostics()

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        for group in (self.visual, self.inertial, self.estimator, self.evaluation):
            for item in fields(group):
                getattr(group, item.name).validate_at(self.timestamp_ns)


FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    FeatureSpec("visual.tracked_feature_count", "visual", "count", FeatureRole.DEPLOYABLE_FEATURE, "tracker"),
    FeatureSpec("visual.surviving_feature_count", "visual", "count", FeatureRole.DEPLOYABLE_FEATURE, "tracker"),
    FeatureSpec("visual.feature_survival_ratio", "visual", "ratio", FeatureRole.DEPLOYABLE_FEATURE, "tracker"),
    FeatureSpec("visual.mean_track_age_s", "visual", "s", FeatureRole.DEPLOYABLE_FEATURE, "tracker"),
    FeatureSpec("visual.median_track_age_s", "visual", "s", FeatureRole.DEPLOYABLE_FEATURE, "tracker"),
    FeatureSpec("visual.optical_flow_mean_px", "visual", "px", FeatureRole.DEPLOYABLE_FEATURE, "tracker"),
    FeatureSpec("visual.optical_flow_std_px", "visual", "px", FeatureRole.DEPLOYABLE_FEATURE, "tracker"),
    FeatureSpec("visual.forward_backward_error_px", "visual", "px", FeatureRole.DEPLOYABLE_FEATURE, "tracker"),
    FeatureSpec("visual.inlier_ratio", "visual", "ratio", FeatureRole.DEPLOYABLE_FEATURE, "visual_update"),
    FeatureSpec("visual.outlier_ratio", "visual", "ratio", FeatureRole.DEPLOYABLE_FEATURE, "visual_update"),
    FeatureSpec("visual.reprojection_residual_px", "visual", "px", FeatureRole.DEPLOYABLE_FEATURE, "visual_update"),
    FeatureSpec("visual.blur_score", "visual", "score", FeatureRole.DEPLOYABLE_FEATURE, "image"),
    FeatureSpec("visual.brightness", "visual", "intensity", FeatureRole.DEPLOYABLE_FEATURE, "image"),
    FeatureSpec("visual.contrast", "visual", "intensity", FeatureRole.DEPLOYABLE_FEATURE, "image"),
    FeatureSpec("visual.entropy", "visual", "bits", FeatureRole.DEPLOYABLE_FEATURE, "image"),
    FeatureSpec("visual.frame_available", "visual", "bool", FeatureRole.DEPLOYABLE_FEATURE, "camera"),
    FeatureSpec("visual.visual_update_available", "visual", "bool", FeatureRole.DEPLOYABLE_FEATURE, "estimator"),
    FeatureSpec("visual.rejected_visual_update", "visual", "bool", FeatureRole.DEPLOYABLE_FEATURE, "estimator"),
    FeatureSpec("inertial.accel_norm_m_s2", "inertial", "m/s^2", FeatureRole.DEPLOYABLE_FEATURE, "imu"),
    FeatureSpec("inertial.gyro_norm_rad_s", "inertial", "rad/s", FeatureRole.DEPLOYABLE_FEATURE, "imu"),
    FeatureSpec("inertial.accel_bias_norm_m_s2", "inertial", "m/s^2", FeatureRole.DEPLOYABLE_FEATURE, "estimator"),
    FeatureSpec("inertial.gyro_bias_norm_rad_s", "inertial", "rad/s", FeatureRole.DEPLOYABLE_FEATURE, "estimator"),
    FeatureSpec("inertial.bias_change", "inertial", "norm", FeatureRole.DEPLOYABLE_FEATURE, "estimator"),
    FeatureSpec("inertial.saturated", "inertial", "bool", FeatureRole.DEPLOYABLE_FEATURE, "imu"),
    FeatureSpec("inertial.clipped", "inertial", "bool", FeatureRole.DEPLOYABLE_FEATURE, "imu"),
    FeatureSpec("inertial.packet_available", "inertial", "bool", FeatureRole.DEPLOYABLE_FEATURE, "imu"),
    FeatureSpec("inertial.seconds_since_previous_packet", "inertial", "s", FeatureRole.DEPLOYABLE_FEATURE, "imu"),
    FeatureSpec("inertial.packet_loss_detected", "inertial", "bool", FeatureRole.DEPLOYABLE_FEATURE, "imu"),
    FeatureSpec("inertial.scale_anomaly", "inertial", "bool", FeatureRole.DEPLOYABLE_FEATURE, "imu"),
    FeatureSpec("estimator.innovation_norm", "estimator", "norm", FeatureRole.DEPLOYABLE_FEATURE, "innovation"),
    FeatureSpec("estimator.nis", "estimator", "dimensionless", FeatureRole.DEPLOYABLE_FEATURE, "innovation"),
    FeatureSpec("estimator.covariance_trace", "estimator", "state_units^2", FeatureRole.DEPLOYABLE_FEATURE, "covariance"),
    FeatureSpec("estimator.covariance_condition_number", "estimator", "ratio", FeatureRole.DEPLOYABLE_FEATURE, "covariance"),
    FeatureSpec("estimator.covariance_logdet", "estimator", "log", FeatureRole.DEPLOYABLE_FEATURE, "covariance"),
    FeatureSpec("estimator.covariance_growth_per_s", "estimator", "state_units^2/s", FeatureRole.DEPLOYABLE_FEATURE, "covariance"),
    FeatureSpec("estimator.visual_update_accepted", "estimator", "bool", FeatureRole.DEPLOYABLE_FEATURE, "estimator"),
    FeatureSpec("estimator.visual_update_rejected", "estimator", "bool", FeatureRole.DEPLOYABLE_FEATURE, "estimator"),
    FeatureSpec("estimator.seconds_since_last_successful_update", "estimator", "s", FeatureRole.DEPLOYABLE_FEATURE, "estimator"),
    FeatureSpec("estimator.reset", "estimator", "bool", FeatureRole.DEPLOYABLE_FEATURE, "estimator"),
    FeatureSpec("estimator.relocalization", "estimator", "bool", FeatureRole.DEPLOYABLE_FEATURE, "estimator"),
    FeatureSpec("evaluation.ground_truth_position_error_m", "evaluation", "m", FeatureRole.EVALUATION_ONLY, "ground_truth+estimate", True),
    FeatureSpec("evaluation.ground_truth_rotation_error_deg", "evaluation", "deg", FeatureRole.EVALUATION_ONLY, "ground_truth+estimate", True),
    FeatureSpec("evaluation.nees", "evaluation", "dimensionless", FeatureRole.EVALUATION_ONLY, "ground_truth+covariance", True),
)


def deployable_feature_specs() -> tuple[FeatureSpec, ...]:
    return tuple(spec for spec in FEATURE_REGISTRY if spec.role is FeatureRole.DEPLOYABLE_FEATURE)


def flatten_deployable(sample: HealthSample) -> dict[str, float | int]:
    """Return deterministic deployable values plus explicit missingness indicators."""

    row: dict[str, float | int] = {"timestamp_ns": sample.timestamp_ns}
    for spec in deployable_feature_specs():
        group_name, field_name = spec.name.split(".", 1)
        value: HealthValue = getattr(getattr(sample, group_name), field_name)
        row[spec.name] = float(value.value) if value.valid and value.value is not None else 0.0
        row[f"{spec.name}__missing"] = int(not value.valid)
    return row


def validate_feature_roles(specs: Iterable[FeatureSpec]) -> None:
    """Reject any attempt to treat oracle/evaluation fields as deployable features."""

    for spec in specs:
        if spec.role is FeatureRole.DEPLOYABLE_FEATURE and spec.requires_ground_truth:
            raise ValueError(f"ground-truth feature cannot be deployable: {spec.name}")
