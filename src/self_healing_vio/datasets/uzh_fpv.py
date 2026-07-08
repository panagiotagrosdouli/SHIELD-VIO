"""UZH FPV dataset adapter skeleton for aggressive UAV motion."""

from __future__ import annotations

from pathlib import Path

from .base import DatasetSequence


class UzhFpvAdapter:
    """Discover UZH FPV sequences for aggressive-flight degradation studies."""

    dataset_name = "UZH FPV"

    def discover_sequences(self, root: Path) -> list[DatasetSequence]:
        return [
            DatasetSequence(
                name=path.name,
                root=path,
                camera_topic="/camera/image_raw",
                imu_topic="/imu/data",
                notes="UZH FPV adapter skeleton; intended for aggressive-motion robustness studies.",
            )
            for path in sorted(root.glob("*"))
            if path.is_dir()
        ]
