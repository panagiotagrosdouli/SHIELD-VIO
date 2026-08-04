from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shield_vio.evaluation.splits import (
    SequenceSplitRegistry,
    assert_events_do_not_cross_splits,
    assert_stage_uses_declared_split,
)


def _registry() -> SequenceSplitRegistry:
    return SequenceSplitRegistry(
        {
            "train": ("MH_01_easy",),
            "calibration": ("MH_03_medium",),
            "validation": ("V2_01_easy",),
            "test": ("MH_04_difficult",),
            "shifted_test": ("TUM_room_test",),
        }
    )


def test_registry_rejects_sequence_overlap() -> None:
    with pytest.raises(ValueError, match="appears in both"):
        SequenceSplitRegistry(
            {
                "train": ("MH_01_easy",),
                "test": ("MH_01_easy",),
            }
        )


def test_calibration_and_threshold_stages_reject_test_sequences() -> None:
    registry = _registry()
    with pytest.raises(ValueError, match="calibration_fit"):
        assert_stage_uses_declared_split("calibration_fit", ["MH_04_difficult"], registry)
    with pytest.raises(ValueError, match="threshold_selection"):
        assert_stage_uses_declared_split("threshold_selection", ["MH_04_difficult"], registry)


def test_detector_fit_accepts_only_training_sequences() -> None:
    assert_stage_uses_declared_split("detector_fit", ["MH_01_easy"], _registry())


def test_failure_event_cannot_be_randomly_split_by_window() -> None:
    with pytest.raises(ValueError, match="divided"):
        assert_events_do_not_cross_splits([("event-7", "train"), ("event-7", "validation")])


def test_checked_in_euroc_registry_is_sequence_disjoint() -> None:
    path = Path(__file__).parents[1] / "configs/datasets/euroc_paper_v1.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    registry = SequenceSplitRegistry(payload["splits"])
    assert registry.split_for("MH_01_easy") == "train"
    assert registry.split_for("MH_04_difficult") == "test"
