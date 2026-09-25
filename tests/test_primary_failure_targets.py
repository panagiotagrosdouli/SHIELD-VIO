from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shield_vio.evaluation.failure_definition import load_failure_definition
from shield_vio.evaluation.primary_failure_targets import (
    build_primary_failure_targets,
    write_primary_failure_artifacts,
)
from shield_vio.evaluation.primary_observables import (
    PRIMARY_CRITERIA,
    PrimaryObservableTable,
)


def _definition():
    return load_failure_definition(
        Path(__file__).parents[1] / "configs/paper/failure_primary_v2.yaml"
    )


def _table(*, observability_gap: int | None = None) -> PrimaryObservableTable:
    timestamps = np.arange(14, dtype=np.int64) * 500_000_000
    values: dict[str, np.ndarray] = {}
    observable: dict[str, np.ndarray] = {}
    applicable: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}

    for name in PRIMARY_CRITERIA:
        if name in {
            "invalid_pose_or_covariance",
            "terminal_tracking_loss",
            "estimator_reset_or_unrecovered_relocalization",
        }:
            values[name] = np.zeros(len(timestamps), dtype=bool)
        else:
            values[name] = np.zeros(len(timestamps), dtype=float)
        applicable[name] = np.ones(len(timestamps), dtype=bool)
        observable[name] = np.ones(len(timestamps), dtype=bool)
        sources[name] = "test fixture"

    # The internal ESKF does not declare a terminal tracking-loss semantic.
    applicable["terminal_tracking_loss"][:] = False
    observable["terminal_tracking_loss"][:] = False
    values["terminal_tracking_loss"][:] = False

    # This unit fixture does not model a visual-update stream. The V2 policy
    # therefore treats visual starvation as explicitly not applicable.
    applicable["visual_update_starvation_while_motion"][:] = False
    observable["visual_update_starvation_while_motion"][:] = False

    values["position_error_m"][3:6] = 1.5
    if observability_gap is not None:
        observable["position_error_m"][observability_gap] = False

    return PrimaryObservableTable(
        timestamps_ns=timestamps,
        values=values,
        observable=observable,
        applicable=applicable,
        sources=sources,
    )


def test_primary_v2_event_targets_are_deterministic() -> None:
    definition = _definition()
    first = build_primary_failure_targets(_table(), definition)
    second = build_primary_failure_targets(_table(), definition)

    assert first.definition_version == "SHIELD_VIO_FAILURE_V2"
    assert first.event_ids == second.event_ids
    np.testing.assert_array_equal(first.onsets_ns, second.onsets_ns)
    np.testing.assert_array_equal(first.offsets_ns, second.offsets_ns)
    np.testing.assert_array_equal(first.active_mask, second.active_mask)
    assert first.event_ids == ("SHIELD_VIO_FAILURE_V2:event:0001:2000000000",)
    assert first.onsets_ns.tolist() == [2_000_000_000]

    for horizon in definition.horizons_seconds:
        np.testing.assert_array_equal(
            first.targets[horizon].labels,
            second.targets[horizon].labels,
        )
        np.testing.assert_array_equal(
            first.targets[horizon].eligible_mask,
            second.targets[horizon].eligible_mask,
        )


def test_observability_gap_breaks_persistence_and_future_targets() -> None:
    definition = _definition()
    table = _table(observability_gap=4)
    # Leave only one above-threshold sample on either side of the unknown sample.
    table.values["position_error_m"][5] = 0.0

    result = build_primary_failure_targets(table, definition)

    assert result.event_ids == ()
    assert not result.active_mask.any()
    assert result.segment_ids[4] == -1
    assert result.segment_ids[3] != result.segment_ids[5]
    for target in result.targets.values():
        assert not target.eligible_mask[4]
        assert not target.labels[4]


def test_primary_failure_writer_records_censoring_and_hashes(tmp_path: Path) -> None:
    definition = _definition()
    result = build_primary_failure_targets(_table(observability_gap=0), definition)
    run = tmp_path / "run"
    run.mkdir()
    for name in ("trajectory.csv", "health.csv", "experiment_manifest.json"):
        (run / name).write_text(f"fixture:{name}\n", encoding="utf-8")

    output = tmp_path / "labels"
    manifest = write_primary_failure_artifacts(
        result,
        output,
        failure_config_path=Path(__file__).parents[1]
        / "configs/paper/failure_primary_v2.yaml",
        source_run_dir=run,
        evidence_level="PUBLIC_DATASET_SMOKE",
    )

    assert manifest["schema_version"] == "SHIELD_VIO_PRIMARY_FAILURE_TARGETS_V2"
    assert manifest["failure_definition_schema"] == "SHIELD_VIO_FAILURE_V2"
    assert manifest["confirmatory"] is False
    assert manifest["censored_observability_samples"] == 1
    assert manifest["complete_observability_samples"] == 13
    assert len(manifest["failure_config_sha256"]) == 64
    assert set(manifest["artifact_sha256"]) == {
        "failure_events.csv",
        "criterion_exceeded.csv",
        "prediction_targets.csv",
    }
    serialized = json.loads(
        (output / "primary_failure_manifest.json").read_text(encoding="utf-8")
    )
    assert serialized["claim_boundary"].startswith(
        "Non-confirmatory primary-label construction only."
    )
