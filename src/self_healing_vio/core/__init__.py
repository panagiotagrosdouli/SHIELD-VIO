"""Core research API for SHIELD-VIO.

The core package defines stable enums, dataclasses, interfaces, metrics, and
exceptions shared across detection, diagnosis, health fusion, recovery, and
future backend integrations. It is intentionally dependency-light so that other
modules can depend on it without creating circular imports.
"""

from .exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    DetectorError,
    DiagnosisError,
    RecoveryPolicyError,
    ShieldVioError,
)
from .state import (
    DiagnosisResult,
    HealthVector,
    NavigationHealthState,
    RecoveryDecision,
    VIOState,
)
from .types import DetectorStatus, FailureCause, HealthLevel, RecoveryAction, SensorType

__all__ = [
    "BackendUnavailableError",
    "ConfigurationError",
    "DetectorError",
    "DiagnosisError",
    "RecoveryPolicyError",
    "ShieldVioError",
    "DetectorStatus",
    "FailureCause",
    "HealthLevel",
    "RecoveryAction",
    "SensorType",
    "DiagnosisResult",
    "HealthVector",
    "NavigationHealthState",
    "RecoveryDecision",
    "VIOState",
]
