from __future__ import annotations

from pathlib import Path

import numpy as np

from shield_vio.evaluation.failure_definition import load_failure_definition
from shield_vio.evaluation.primary_failure_pipeline import build_primary_failure_bundle
from shield_vio.evaluation.primary_observables import PRIMARY_CRITERIA, PrimaryObservableTable


def _definition():
    return load_failure_definition(
        Path(__file__).parents[1] / "configs/paper/failure_primary_v2.yaml"
    )


def _table(*, missing_index: int | None = None) -> PrimaryObservableTable:
    timestamps = np.arange(10, dtype=np.int64) * 500_000_000
    values: dict[str, np.ndarray] = {}
    observable: dict[str, np.ndarray] = {}
    applicable: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    boolean = {
        "invalid_pose_or_covariance",
        "terminal_tracking_loss",
        "estimator_reset_or_unrecovered_relocalization",
    }
    for name in PRIMARY_CRITERIA:
        if name in boolean:
            values[name] = np.zeros(len(timestamps), dtype=bool)
        else:
            values[name] = np.zeros(len(timestamps), dtype=float)
        observable[name] = np.ones(len(timestamps), dtype=bool)
        applicable[name] = np.ones(len(timestamps), dtype=bool)
        sources[name] = "fixture"

    # These backend/stream-specific criteria are explicitly not applicable in this fixture.
    applicable["terminal_tracking_loss"][:] = False
    observable["terminal_tracking_loss"][:] = False
    applicable["visual_update_starvation_while_motion"][:] = False
    observable["visual_update_starvation_while_motion"][:] = False

    # Position-error exceedance is long enough to satisfy the 0.5 s persistence
    # only if the middle sample is actually available.
    values["position_error_m"][2:5] = 2.0
    if missing_index is not None:
        observable["position_error_m"][missing_index] = False

    return PrimaryObservableTable(
        timestamps,
        values,
        observable,
        applicable,
        sources,
    )


def test_not_applicable_criteria_are_ignored_in_primary_union() -> None:
    bundle = build_primary_failure_bundle(_table(), _definition())
    assert len(bundle.events.event_ids) == 1
    assert bundle.events.onsets_ns.tolist() == [1_500_000_000]
    assert np.all(~bundle.criterion_exceeded["terminal_tracking_loss"])
    assert np.all(~bundle.criterion_exceeded["visual_update_starvation_while_motion"])


def test_missing_applicable_sample_breaks_persistence_and_censors_horizons() -> None:
    bundle = build_primary_failure_bundle(_table(missing_index=3), _definition())
    assert len(bundle.events.event_ids) == 0
    assert not bundle.joint_available_mask[3]

    one_second = bundle.targets.targets[1.0]
    # t=0.5s and t=1.0s both look through the missing t=1.5s sample.
    assert not one_second.eligible_mask[1]
    assert not one_second.eligible_mask[2]
    assert not np.any(one_second.labels)


def test_missing_nonapplicable_criterion_does_not_censor_labels() -> None:
    table = _table()
    assert not np.any(table.observable["terminal_tracking_loss"])
    bundle = build_primary_failure_bundle(table, _definition())
    assert np.all(bundle.joint_available_mask)
