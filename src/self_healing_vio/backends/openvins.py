"""OpenVINS integration contract for SHIELD-VIO.

This module deliberately avoids pretending that OpenVINS is already integrated.
It defines the adapter boundary that a ROS2/C++ bridge must satisfy: expose
backend residuals, tracked feature counts, covariance surrogates, and accept
recovery commands from SHIELD-VIO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class OpenVINSAdapterConfig:
    """Configuration for a planned OpenVINS bridge."""

    feature_count_topic: str = "/openvins/features/count"
    reprojection_error_topic: str = "/openvins/residuals/reprojection_error"
    covariance_topic: str = "/openvins/state/covariance_trace"
    reset_service: str = "/openvins/reset"
    relocalize_service: str = "/openvins/relocalize"


@dataclass
class OpenVINSHealthAdapter:
    """Simulation-safe OpenVINS adapter stub.

    The class is useful for tests and architecture validation. A real adapter
    should implement the same public methods but bind them to ROS2 topics,
    services, and backend instrumentation.
    """

    config: OpenVINSAdapterConfig = OpenVINSAdapterConfig()
    feature_count: int = 0
    reprojection_error_px: float = 0.0
    covariance_trace: float | None = None
    last_action: str | None = None

    def get_reprojection_error(self) -> float:
        """Return current mean reprojection error in pixels."""
        return float(self.reprojection_error_px)

    def get_feature_count(self) -> int:
        """Return current number of tracked visual features."""
        return int(self.feature_count)

    def get_covariance_trace(self) -> float | None:
        """Return a covariance trace surrogate if the backend exposes one."""
        return self.covariance_trace

    def snapshot(self) -> Mapping[str, float | int | None]:
        """Return backend health signals as a serializable snapshot."""
        return {
            "feature_count": self.feature_count,
            "reprojection_error_px": self.reprojection_error_px,
            "covariance_trace": self.covariance_trace,
        }

    def apply_recovery_action(self, action: str) -> None:
        """Record a recovery action; real adapters should call backend services."""
        self.last_action = action
