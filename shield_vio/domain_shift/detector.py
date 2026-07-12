"""Transparent rolling z-score domain-shift detector."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ShiftState(str, Enum):
    IN_DISTRIBUTION="IN_DISTRIBUTION"; POSSIBLE_SHIFT="POSSIBLE_SHIFT"; CONFIRMED_SHIFT="CONFIRMED_SHIFT"; SEVERE_SHIFT="SEVERE_SHIFT"


@dataclass(frozen=True)
class ShiftAssessment:
    state: ShiftState
    max_abs_z: float
    confidence_multiplier: float
    shifted_features: tuple[str, ...]


class RollingShiftDetector:
    def __init__(self, reference: dict[str, tuple[float,float]], window: int=20) -> None:
        self.reference=reference; self.buffers={k:deque(maxlen=window) for k in reference}; self.window=window

    def update(self, diagnostics: dict[str,float]) -> ShiftAssessment:
        zs={}
        for name,(mean,std) in self.reference.items():
            if name in diagnostics:
                self.buffers[name].append(float(diagnostics[name]))
                if len(self.buffers[name]) >= max(3,self.window//2): zs[name]=(float(np.mean(self.buffers[name]))-mean)/max(std,1e-9)
        m=max((abs(v) for v in zs.values()),default=0.0)
        state=ShiftState.SEVERE_SHIFT if m>=5 else ShiftState.CONFIRMED_SHIFT if m>=3 else ShiftState.POSSIBLE_SHIFT if m>=2 else ShiftState.IN_DISTRIBUTION
        mult={ShiftState.IN_DISTRIBUTION:1.0,ShiftState.POSSIBLE_SHIFT:0.85,ShiftState.CONFIRMED_SHIFT:0.65,ShiftState.SEVERE_SHIFT:0.4}[state]
        return ShiftAssessment(state,m,mult,tuple(sorted(k for k,v in zs.items() if abs(v)>=2)))
