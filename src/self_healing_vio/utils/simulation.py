"""Synthetic degradation helpers for demos and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticDegradationScenario:
    """Generates simple signals that mimic VIO degradation over time."""

    image_shape: tuple[int, int] = (120, 160)
    rng_seed: int = 7

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.rng_seed)

    def sample(self, step: int, total_steps: int) -> dict[str, object]:
        alpha = step / max(total_steps - 1, 1)
        texture = max(0.05, 1.0 - 0.75 * alpha)
        image = self.rng.normal(127.0, 60.0 * texture, self.image_shape)
        image = np.clip(image, 0, 255).astype(np.uint8)
        feature_count = int(220 * (1.0 - 0.8 * alpha))
        imu_residual = 0.04 + 0.9 * max(0.0, alpha - 0.45)
        reprojection_error = 1.0 + 7.0 * max(0.0, alpha - 0.35)
        return {
            "image": image,
            "feature_count": feature_count,
            "imu_residual": imu_residual,
            "reprojection_error": reprojection_error,
        }
