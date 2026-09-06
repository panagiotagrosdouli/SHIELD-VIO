from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shield_vio.evaluation.failure_definition import (
    CriterionDefinition,
    FailureDefinition,
    build_failure_events_and_targets,
    evaluate_failure_criteria,
    load_failure_definition,
)


def _definition() -> FailureDefinition:
    return FailureDefinition(
        schema_version="TEST_FAILURE_V1",
        kind="primary",
        horizons_seconds=(0.5, 1.0, 2.0),
        event_merge_gap_seconds=0.0,
        recovery_confirmation_seconds=0.0,
        criteria=(
            CriterionDefinition(
                name="position_error_m",
                persistence_seconds=0.5,
                requires_ground_truth=True,
                comparison="greater_than",
                threshold=1.0,
            ),
            CriterionDefinition(
                name="terminal_tracking_loss",
                persistence_seconds=0.0,
                requires_ground_truth=False,
                comparison="true",
            ),
        ),
    )


def test_checked_in_primary_definition_is_frozen_and_complete() -> None:
    path = Path(__file__).parents[1] / "configs/paper/failure_primary_v1.yaml"
    definition = load_failure_definition(path)
    assert definition.schema_version == "SHIELD_VIO_FAILURE_V1"
    assert definition.kind == "primary"
    assert definition.horizons_seconds == (0.5, 1.0, 2.0, 3.0, 5.0)
    names = {criterion.name for criterion in definition.criteria}
    assert "position_error_m" in names
    assert "orientation_error_deg" in names
    assert "terminal_tracking_loss" in names


def test_thresholds_persistence_and_event_ids_are_deterministic() -> None:
    timestamps = np.arange(8, dtype=np.int64) * 500_000_000
    observations = {
        "position_error_m": np.array([0.0, 0.0, 1.2, 1.3, 1.4, 0.0, 0.0, 0.0]),
        "terminal_tracking_loss": np.zeros(8, dtype=bool),
    }
    exceeded = evaluate_failure_criteria(observations, _definition())
    first, targets_first = build_failure_events_and_targets(timestamps, exceeded, _definition())
    second, targets_second = build_failure_events_and_targets(timestamps, exceeded, _definition())
    assert first.event_ids == second.event_ids
    assert first.onsets_ns.tolist() == [1_500_000_000]
    np.testing.assert_array_equal(first.active_mask, second.active_mask)
    for horizon in _definition().horizons_seconds:
        np.testing.assert_array_equal(
            targets_first.targets[horizon].labels,
            targets_second.targets[horizon].labels,
        )


def test_current_failure_is_not_future_positive_and_horizon_boundary_is_inclusive() -> None:
    timestamps = np.arange(7, dtype=np.int64) * 500_000_000
    observations = {
        "position_error_m": np.zeros(7),
        "terminal_tracking_loss": np.array([False, False, False, False, True, False, False]),
    }
    events, targets = build_failure_events_and_targets(
        timestamps,
        evaluate_failure_criteria(observations, _definition()),
        _definition(),
    )
    assert events.onsets_ns.tolist() == [2_000_000_000]
    one_second = targets.targets[1.0]
    assert one_second.labels[2]
    assert one_second.labels[3]
    assert not one_second.eligible_mask[4]
    assert not one_second.labels[4]
    assert not one_second.eligible_mask[-1]


def test_transient_violation_does_not_create_persistent_event() -> None:
    timestamps = np.arange(6, dtype=np.int64) * 500_000_000
    observations = {
        "position_error_m": np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        "terminal_tracking_loss": np.zeros(6, dtype=bool),
    }
    events, _ = build_failure_events_and_targets(
        timestamps,
        evaluate_failure_criteria(observations, _definition()),
        _definition(),
    )
    assert len(events.event_ids) == 0


def test_oracle_and_undeclared_fields_are_rejected() -> None:
    observations = {
        "position_error_m": np.zeros(4),
        "terminal_tracking_loss": np.zeros(4, dtype=bool),
        "degradation_severity": np.ones(4),
    }
    with pytest.raises(ValueError, match="oracle"):
        evaluate_failure_criteria(observations, _definition())


def test_undeclared_failure_label_cannot_be_used_as_criterion() -> None:
    observations = {
        "position_error_m": np.zeros(4),
        "terminal_tracking_loss": np.zeros(4, dtype=bool),
        "failure_label": np.ones(4, dtype=bool),
    }
    with pytest.raises(ValueError, match="not declared"):
        evaluate_failure_criteria(observations, _definition())
