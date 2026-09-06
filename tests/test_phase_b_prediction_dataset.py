from __future__ import annotations

import numpy as np

from shield_vio.health.dataset import PredictionGroups, build_prediction_dataset
from shield_vio.health.schema import EvaluationDiagnostics, HealthSample, HealthValue, VisualHealth


def test_prediction_dataset_structurally_separates_features_targets_and_oracles() -> None:
    samples = [
        HealthSample(
            timestamp_ns=index * 1_000_000_000,
            visual=VisualHealth(
                tracked_feature_count=HealthValue(float(40 - index), True, index * 1_000_000_000)
            ),
            evaluation=EvaluationDiagnostics(
                ground_truth_position_error_m=HealthValue(
                    float(index), True, index * 1_000_000_000
                ),
                nees=HealthValue(float(index + 1), True, index * 1_000_000_000),
            ),
        )
        for index in range(3)
    ]
    dataset = build_prediction_dataset(
        samples,
        horizon_targets={1.0: np.array([False, True, False])},
        eligible_masks={1.0: np.array([True, True, False])},
        groups=PredictionGroups("synthetic", "seq", "eskf", "darkness", 7),
        metadata={
            "degradation_type": "darkness",
            "degradation_severity": 0.8,
            "degradation_onset": 2.0,
        },
    )
    values, names = dataset.features()
    labels, eligible = dataset.targets(1.0)
    assert values.shape[0] == 3
    assert labels.tolist() == [False, True, False]
    assert eligible.tolist() == [True, True, False]
    assert not any(
        token in name
        for name in names
        for token in ("ground_truth", "nees", "failure", "degradation", "severity", "onset")
    )
    assert dataset.experiment_metadata()["degradation_type"] == "darkness"
