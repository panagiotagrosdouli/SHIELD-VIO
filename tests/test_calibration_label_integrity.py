"""Regression tests for calibration target integrity."""
import numpy as np
import pytest

from shield_vio.calibration_metrics.metrics import (
    brier_score, calibration_error, negative_log_likelihood, summarize_calibration,
)


@pytest.mark.parametrize("metric", [
    brier_score, calibration_error, negative_log_likelihood, summarize_calibration,
])
@pytest.mark.parametrize("invalid", [0.5, -0.1, 1.9, np.nan, np.inf, -1])
def test_reject_invalid_calibration_labels(metric, invalid):
    with pytest.raises(ValueError, match="labels must be binary"):
        metric(np.array([0.2, 0.8]), np.array([0, invalid]))


def test_valid_binary_labels_are_preserved():
    assert brier_score(np.array([0.25, 0.75]), np.array([False, True])) == 0.0625
