from __future__ import annotations

import numpy as np
import pytest

from shield_vio.health.schema import (
    EvaluationDiagnostics,
    FeatureRole,
    FeatureSpec,
    HealthSample,
    HealthValue,
    VisualHealth,
    deployable_feature_specs,
    flatten_deployable,
    validate_feature_roles,
)
from shield_vio.health.temporal import build_trailing_features


def test_health_sample_preserves_missingness_and_excludes_evaluation_fields() -> None:
    sample = HealthSample(
        timestamp_ns=2_000_000_000,
        sequence_id="fixture",
        estimator_id="eskf",
        visual=VisualHealth(
            tracked_feature_count=HealthValue(42.0, True, 1_900_000_000),
            inlier_ratio=HealthValue.missing(),
        ),
        evaluation=EvaluationDiagnostics(
            ground_truth_position_error_m=HealthValue(3.0, True, 2_000_000_000),
            nees=HealthValue(8.0, True, 2_000_000_000),
        ),
    )
    row = flatten_deployable(sample)
    assert row["visual.tracked_feature_count"] == 42.0
    assert row["visual.tracked_feature_count__missing"] == 0
    assert row["visual.inlier_ratio__missing"] == 1
    assert not any("ground_truth" in key or "nees" in key for key in row)
    assert all(spec.role is FeatureRole.DEPLOYABLE_FEATURE for spec in deployable_feature_specs())


def test_health_sample_rejects_future_source_timestamp() -> None:
    with pytest.raises(ValueError, match="future source"):
        HealthSample(
            timestamp_ns=10,
            visual=VisualHealth(tracked_feature_count=HealthValue(3.0, True, 11)),
        )


def test_role_validation_rejects_ground_truth_deployable_feature() -> None:
    with pytest.raises(ValueError, match="ground-truth"):
        validate_feature_roles(
            (
                FeatureSpec(
                    "bad",
                    "evaluation",
                    "m",
                    FeatureRole.DEPLOYABLE_FEATURE,
                    "ground_truth",
                    requires_ground_truth=True,
                ),
            )
        )


def test_future_modification_cannot_change_past_temporal_features() -> None:
    timestamps = np.array([0, 200, 550, 900, 1_400, 2_000], dtype=np.int64) * 1_000_000
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    valid = np.array([True, True, False, True, True, True])
    baseline = build_trailing_features(
        timestamps,
        values,
        valid,
        signal_name="nis",
        windows_seconds=(0.5, 1.0),
    )
    modified = values.copy()
    modified[4:] = [5000.0, -9000.0]
    changed = build_trailing_features(
        timestamps,
        modified,
        valid,
        signal_name="nis",
        windows_seconds=(0.5, 1.0),
    )
    np.testing.assert_allclose(baseline.values[:4], changed.values[:4], rtol=0.0, atol=0.0)
    assert np.all(baseline.max_source_timestamps_ns <= baseline.timestamps_ns)


def test_irregular_timestamp_windows_and_missing_fraction_are_causal() -> None:
    timestamps = np.array([0, 100, 450, 1_000], dtype=np.int64) * 1_000_000
    values = np.array([1.0, 2.0, 100.0, 4.0])
    valid = np.array([True, True, False, True])
    table = build_trailing_features(
        timestamps,
        values,
        valid,
        signal_name="tracked",
        windows_seconds=(0.5,),
    )
    missing = table.feature_names.index("tracked__missing_fraction__0p5s")
    mean = table.feature_names.index("tracked__mean__0p5s")
    assert table.values[2, missing] == pytest.approx(1.0 / 3.0)
    assert table.values[2, mean] == pytest.approx(1.5)
    assert table.values[3, mean] == pytest.approx(4.0)
