"""Temporal filtering for Navigation Health Index estimates.

The prototype detectors operate frame-by-frame. This module adds a lightweight
Bayesian-style recursive filter so health estimates do not jump erratically from
single-frame noise. The filter is intentionally simple, deterministic, and easy
to replace with a full hidden Markov model or particle filter later.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


@dataclass
class TemporalHealthFilter:
    """Exponential Bayesian surrogate for smoothing scalar health estimates.

    Args:
        process_noise: Allows the latent health state to change over time.
        measurement_gain: Weight assigned to the current measurement.
        initial_health: Initial latent health value in [0, 100].
    """

    process_noise: float = 2.0
    measurement_gain: float = 0.35
    initial_health: float = 100.0

    def __post_init__(self) -> None:
        self.mean = _clip(self.initial_health, 0.0, 100.0)
        self.variance = 25.0

    def update(self, measured_nhi: float) -> tuple[float, float]:
        """Update the latent health estimate.

        Returns:
            Tuple of filtered health mean and standard deviation.
        """
        z = _clip(measured_nhi, 0.0, 100.0)
        self.variance += max(self.process_noise, 0.0)
        gain = _clip(self.measurement_gain, 0.0, 1.0)
        self.mean = _clip((1.0 - gain) * self.mean + gain * z, 0.0, 100.0)
        self.variance = max((1.0 - gain) * self.variance, 1e-9)
        return self.mean, self.variance ** 0.5
