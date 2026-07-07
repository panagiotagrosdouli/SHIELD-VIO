"""Shared enumerations for SHIELD-VIO."""

from __future__ import annotations

from enum import Enum


class FailureCause(str, Enum):
    """High-level degradation causes considered by SHIELD-VIO."""

    NOMINAL = "nominal"
    CAMERA = "camera"
    ENVIRONMENT = "environment"
    MOTION = "motion"
    SENSOR = "sensor"
    ALGORITHM = "algorithm"
    UNKNOWN = "unknown"


class HealthLevel(str, Enum):
    """Discrete operational health levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    """Canonical recovery action vocabulary."""

    NO_ACTION = "no_action"
    CONTINUE_NOMINAL_TRACKING = "continue_nominal_tracking"
    REDUCE_SPEED = "reduce_speed"
    INCREASE_FEATURE_BUDGET = "increase_feature_budget"
    INCREASE_INERTIAL_WEIGHT = "increase_inertial_weight"
    RELAX_RANSAC = "relax_ransac"
    RESET_VISUAL_FRONTEND = "reset_visual_frontend"
    REINITIALIZE_BIAS = "reinitialize_bias"
    INFLATE_COVARIANCE = "inflate_covariance"
    REQUEST_HOVER = "request_hover"
    RELOCALIZE = "relocalize"
    TRIGGER_RELOCALIZATION = "trigger_relocalization"
    FALLBACK_INERTIAL = "fallback_inertial"
    EMERGENCY_LAND = "emergency_land"


class SensorType(str, Enum):
    """Sensor categories used by future backend and ROS2 adapters."""

    CAMERA = "camera"
    IMU = "imu"
    VIO_BACKEND = "vio_backend"
    CLOCK = "clock"
    UNKNOWN = "unknown"


class DetectorStatus(str, Enum):
    """Execution status for a detector evaluation."""

    OK = "ok"
    DEGRADED = "degraded"
    INVALID_INPUT = "invalid_input"
    UNAVAILABLE = "unavailable"
