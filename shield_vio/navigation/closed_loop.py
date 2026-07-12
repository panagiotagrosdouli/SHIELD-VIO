"""Lightweight closed-loop navigation protection for controlled experiments."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ShieldState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    SLOW_DOWN = "SLOW_DOWN"
    HOLD_POSITION = "HOLD_POSITION"
    RELOCALIZE_REQUESTED = "RELOCALIZE_REQUESTED"
    RETURN_TO_SAFE_STATE = "RETURN_TO_SAFE_STATE"
    HALT = "HALT"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True)
class ShieldInputs:
    failure_probability: float
    covariance_trace: float
    visual_quality: float
    imu_health: float
    shift_state: str = "IN_DISTRIBUTION"
    sensors_stale: bool = False
    recovery_feasible: bool = True


@dataclass(frozen=True)
class ShieldOutput:
    state: ShieldState
    speed_scale: float
    recovery_action: str
    reason: str


class NavigationShield:
    """Stateful shield with hysteresis and minimum dwell time."""

    _severity = {state: index for index, state in enumerate(ShieldState)}

    def __init__(self, dwell_steps: int = 2, release_margin: float = 0.08) -> None:
        if dwell_steps < 1:
            raise ValueError("dwell_steps must be positive")
        self.dwell_steps = dwell_steps
        self.release_margin = release_margin
        self.state = ShieldState.NORMAL
        self._age = 0

    def _candidate(self, x: ShieldInputs) -> ShieldState:
        p = float(np.clip(x.failure_probability, 0.0, 1.0))
        if x.sensors_stale or p >= 0.98:
            return ShieldState.EMERGENCY_STOP
        if p >= 0.90 or x.imu_health < 0.15:
            return ShieldState.HALT
        if p >= 0.78 or x.visual_quality < 0.12:
            return ShieldState.RELOCALIZE_REQUESTED if x.recovery_feasible else ShieldState.HALT
        if p >= 0.62 or x.covariance_trace > 4.0:
            return ShieldState.HOLD_POSITION
        if p >= 0.45 or x.shift_state in {"CONFIRMED_SHIFT", "SEVERE_SHIFT"}:
            return ShieldState.SLOW_DOWN
        if p >= 0.28 or x.visual_quality < 0.35:
            return ShieldState.WARNING
        return ShieldState.NORMAL

    def update(self, x: ShieldInputs) -> ShieldOutput:
        candidate = self._candidate(x)
        current_rank = self._severity[self.state]
        candidate_rank = self._severity[candidate]
        escalating = candidate_rank > current_rank
        can_release = self._age >= self.dwell_steps and x.failure_probability < 0.28 - self.release_margin
        if escalating or (candidate_rank == current_rank):
            next_state = candidate
        elif can_release:
            next_state = candidate
        else:
            next_state = self.state
        if next_state == self.state:
            self._age += 1
        else:
            self.state = next_state
            self._age = 0

        speed = {
            ShieldState.NORMAL: 1.0,
            ShieldState.WARNING: 0.8,
            ShieldState.SLOW_DOWN: 0.45,
            ShieldState.HOLD_POSITION: 0.0,
            ShieldState.RELOCALIZE_REQUESTED: 0.0,
            ShieldState.RETURN_TO_SAFE_STATE: 0.25,
            ShieldState.HALT: 0.0,
            ShieldState.EMERGENCY_STOP: 0.0,
        }[self.state]
        recovery = {
            ShieldState.RELOCALIZE_REQUESTED: "rotate_for_features",
            ShieldState.RETURN_TO_SAFE_STATE: "return_to_last_safe_pose",
            ShieldState.HALT: "abort_mission",
            ShieldState.EMERGENCY_STOP: "emergency_stop",
        }.get(self.state, "none")
        reason = (
            f"state={self.state.value}; p_failure={x.failure_probability:.3f}; "
            f"cov_trace={x.covariance_trace:.3f}; visual_quality={x.visual_quality:.3f}; "
            f"imu_health={x.imu_health:.3f}; shift={x.shift_state}"
        )
        return ShieldOutput(self.state, speed, recovery, reason)


@dataclass
class NavigationState:
    position: np.ndarray
    goal: np.ndarray
    nominal_speed_mps: float = 1.0

    def step(self, dt: float, shield: ShieldOutput) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be positive")
        delta = np.asarray(self.goal, dtype=float) - np.asarray(self.position, dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance == 0.0 or shield.speed_scale == 0.0:
            return self.position.copy()
        travel = min(distance, self.nominal_speed_mps * shield.speed_scale * dt)
        self.position = self.position + delta / distance * travel
        return self.position.copy()
