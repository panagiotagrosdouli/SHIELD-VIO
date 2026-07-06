from dataclasses import dataclass

from self_healing_vio.health import VioHealthMonitor, VioHealthSignals
from self_healing_vio.recovery import RecoveryDecision, RecoveryPolicy


@dataclass
class ControllerOutput:
    health_level: str
    health_score: float
    failure_probability: float
    recovery_decision: RecoveryDecision


class SelfHealingVioController:
    def __init__(
        self,
        health_monitor: VioHealthMonitor | None = None,
        recovery_policy: RecoveryPolicy | None = None,
    ) -> None:
        self.health_monitor = health_monitor or VioHealthMonitor()
        self.recovery_policy = recovery_policy or RecoveryPolicy()

    def step(self, signals: VioHealthSignals) -> ControllerOutput:
        health = self.health_monitor.estimate(signals)
        decision = self.recovery_policy.decide(health)
        return ControllerOutput(
            health_level=health.level,
            health_score=health.score,
            failure_probability=health.failure_probability,
            recovery_decision=decision,
        )
