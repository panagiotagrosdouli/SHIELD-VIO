"""Finite-sample split-conformal bounds for scalar failure risk.

Exchangeability is required for nominal coverage; coverage under domain shift must be
reported empirically and is not a formal safety guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConformalResult:
    lower: np.ndarray
    upper: np.ndarray
    target_coverage: float
    quantile: float


class SplitConformalRisk:
    def __init__(self, coverage: float = 0.9) -> None:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be in (0,1)")
        self.coverage=coverage; self.quantile_: float | None=None

    def fit(self, predicted_probability: np.ndarray, labels: np.ndarray) -> "SplitConformalRisk":
        p=np.asarray(predicted_probability,dtype=float); y=np.asarray(labels,dtype=float)
        if p.shape != y.shape or p.ndim != 1 or len(p) < 2: raise ValueError("calibration arrays must be equal 1-D arrays")
        if np.any((p<0)|(p>1)) or np.any((y<0)|(y>1)): raise ValueError("probabilities and labels must be in [0,1]")
        scores=np.abs(y-p); n=len(scores); rank=int(np.ceil((n+1)*self.coverage)); rank=min(rank,n)
        self.quantile_=float(np.partition(scores,rank-1)[rank-1]); return self

    def predict(self, predicted_probability: np.ndarray) -> ConformalResult:
        if self.quantile_ is None: raise RuntimeError("fit must be called before predict")
        p=np.asarray(predicted_probability,dtype=float); q=self.quantile_
        return ConformalResult(np.clip(p-q,0,1),np.clip(p+q,0,1),self.coverage,q)

    @staticmethod
    def empirical_coverage(result: ConformalResult, labels: np.ndarray) -> float:
        y=np.asarray(labels,dtype=float)
        return float(np.mean((y>=result.lower)&(y<=result.upper)))
