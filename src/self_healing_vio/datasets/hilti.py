"""Hilti SLAM Challenge dataset adapter skeleton."""

from __future__ import annotations

from pathlib import Path

from .base import DatasetSequence


class HiltiAdapter:
    """Discover Hilti-style sequences for future robustness benchmarks."""

    dataset_name = "Hilti SLAM Challenge"

    def discover_sequences(self, root: Path) -> list[DatasetSequence]:
        return [
            DatasetSequence(
                name=path.name,
                root=path,
                camera_topic="/camera/image_raw",
                imu_topic="/imu/data",
                notes="Hilti adapter skeleton; exact topic mapping depends on release format.",
            )
            for path in sorted(root.glob("*"))
            if path.is_dir()
        ]
