"""Abstract interfaces for SHIELD-VIO modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

from .state import DiagnosisResult, HealthVector, NavigationHealthState, RecoveryDecision, VIOState


class Detector(ABC):
    """Base interface for degradation detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique detector name used in the health vector."""

    @abstractmethod
    def evaluate(self, *args: object, **kwargs: object) -> float:
        """Return a normalized health score in [0, 1]."""


class DiagnosisModel(ABC):
    """Base interface for failure diagnosis models."""

    @abstractmethod
    def infer(self, health: HealthVector | Mapping[str, float]) -> DiagnosisResult | Mapping[str, float]:
        """Infer degradation cause probabilities from health signals."""


class RecoveryPolicy(ABC):
    """Base interface for recovery policies."""

    @abstractmethod
    def decide(self, health: NavigationHealthState | object) -> RecoveryDecision | object:
        """Return a recovery decision for the current health state."""


class HealthFusion(ABC):
    """Base interface for health fusion and NHI computation."""

    @abstractmethod
    def compute(self, health: HealthVector | Mapping[str, float]) -> float:
        """Compute a scalar Navigation Health Index in [0, 100]."""


class BackendAdapter(ABC):
    """Minimal interface for VIO backend adapters."""

    @abstractmethod
    def read_state(self) -> VIOState:
        """Read the current VIO backend state summary."""

    @abstractmethod
    def apply_recovery(self, decision: RecoveryDecision) -> None:
        """Apply a backend-specific recovery decision."""
