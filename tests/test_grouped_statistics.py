from __future__ import annotations

import numpy as np
import pytest

from shield_vio.evaluation.statistics import paired_grouped_bootstrap


def _key(sequence: str, seed: int = 0) -> tuple[str, str, str, str, int]:
    return ("euroc", sequence, "eskf", "nominal", seed)


def test_grouped_bootstrap_is_reproducible_with_fixed_seed() -> None:
    a = {_key("MH_01"): 0.8, _key("MH_02"): 0.7, _key("MH_03"): 0.9}
    b = {_key("MH_01"): 0.6, _key("MH_02"): 0.6, _key("MH_03"): 0.7}
    first = paired_grouped_bootstrap(a, b, metric="auprc", higher_is_better=True, seed=7)
    second = paired_grouped_bootstrap(a, b, metric="auprc", higher_is_better=True, seed=7)
    assert first == second


def test_pairing_uses_complete_experimental_unit_keys() -> None:
    a = {_key("MH_01", 0): 1.0, _key("MH_01", 1): 3.0}
    b = {_key("MH_01", 0): 0.0, _key("MH_01", 1): 2.0}
    result = paired_grouped_bootstrap(a, b, metric="lead_time", higher_is_better=True)
    assert result.paired_unit_count == 2
    assert result.mean_difference == pytest.approx(1.0)


def test_bootstrap_resamples_runs_not_frame_rows() -> None:
    # One scalar per run is accepted; frame arrays are deliberately invalid metric values.
    a = {_key("MH_01"): np.array([0.9, 0.8])}
    b = {_key("MH_01"): 0.7}
    with pytest.raises((TypeError, ValueError)):
        paired_grouped_bootstrap(a, b, metric="auprc", higher_is_better=True)


def test_known_systematic_effect_has_expected_interval() -> None:
    a = {_key(f"S{i}"): float(i) + 0.5 for i in range(8)}
    b = {_key(f"S{i}"): float(i) for i in range(8)}
    result = paired_grouped_bootstrap(
        a, b, metric="auprc", higher_is_better=True, n_bootstrap=2000, seed=11
    )
    assert result.mean_difference == pytest.approx(0.5)
    assert result.ci95_low == pytest.approx(0.5)
    assert result.ci95_high == pytest.approx(0.5)
    assert result.oriented_effect > 0.0


def test_lower_is_better_metric_orients_effect_without_changing_raw_difference() -> None:
    a = {_key("A"): 0.10, _key("B"): 0.20}
    b = {_key("A"): 0.20, _key("B"): 0.30}
    result = paired_grouped_bootstrap(a, b, metric="brier", higher_is_better=False)
    assert result.mean_difference == pytest.approx(-0.10)
    assert result.oriented_effect == pytest.approx(0.10)


def test_missing_and_nonfinite_pairs_can_be_dropped_or_rejected() -> None:
    a = {_key("A"): 0.8, _key("B"): np.nan, _key("C"): 0.6}
    b = {_key("A"): 0.7, _key("B"): 0.5, _key("D"): 0.4}
    result = paired_grouped_bootstrap(a, b, metric="auroc", higher_is_better=True)
    assert result.paired_unit_count == 1
    assert result.dropped_unit_count == 3
    with pytest.raises(ValueError, match="unpaired experimental units"):
        paired_grouped_bootstrap(
            a, b, metric="auroc", higher_is_better=True, missing="raise"
        )


def test_invalid_inputs_and_small_samples() -> None:
    key = _key("A")
    with pytest.raises(ValueError, match="paired experimental unit"):
        paired_grouped_bootstrap({}, {}, metric="auprc", higher_is_better=True)
    with pytest.raises(ValueError, match="positive"):
        paired_grouped_bootstrap(
            {key: 1.0}, {key: 0.0}, metric="auprc", higher_is_better=True, n_bootstrap=0
        )
    one = paired_grouped_bootstrap(
        {key: 1.0}, {key: 0.75}, metric="auprc", higher_is_better=True, seed=2
    )
    assert one.paired_unit_count == 1
    assert one.ci95_low == pytest.approx(0.25)
    assert one.ci95_high == pytest.approx(0.25)
