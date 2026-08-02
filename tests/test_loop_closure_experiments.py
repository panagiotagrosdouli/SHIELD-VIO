from __future__ import annotations

import pytest

from shield_vio.loop_closure_experiments import (
    LoopClosureExperimentConfig,
    run_loop_closure_experiment,
    run_loop_closure_trial,
)


def test_loop_constraint_reduces_drift_for_deterministic_bias() -> None:
    config = LoopClosureExperimentConfig(
        node_count=30,
        odometry_noise_std_m=0.0,
        odometry_bias_m=(0.04, -0.015, 0.0),
        loop_noise_std_m=0.0,
    )
    result = run_loop_closure_trial(3, config)

    assert result.loop_corrected_rmse_m < result.odometry_rmse_m
    assert result.loop_corrected_endpoint_error_m < result.odometry_endpoint_error_m
    assert result.improvement_fraction > 0.0


def test_multi_seed_experiment_is_reproducible_and_aggregated() -> None:
    config = LoopClosureExperimentConfig(node_count=20)
    first = run_loop_closure_experiment(range(5), config)
    second = run_loop_closure_experiment(range(5), config)

    assert first == second
    assert len(first.trials) == 5
    assert 0.0 <= first.improved_trial_fraction <= 1.0
    assert first.mean_loop_corrected_rmse_m < first.mean_odometry_rmse_m
    assert first.to_dict()["claim_boundary"].startswith("Synthetic")


def test_experiment_rejects_invalid_configuration_and_seed_sets() -> None:
    with pytest.raises(ValueError, match="node_count"):
        LoopClosureExperimentConfig(node_count=3)
    with pytest.raises(ValueError, match="information"):
        LoopClosureExperimentConfig(loop_information=0.0)
    with pytest.raises(ValueError, match="at least one seed"):
        run_loop_closure_experiment([])
    with pytest.raises(ValueError, match="unique"):
        run_loop_closure_experiment([1, 1])
