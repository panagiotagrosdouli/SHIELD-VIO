from __future__ import annotations

import numpy as np
import pytest

from shield_vio.experiments.frozen_predictor import FrozenPredictor
from test_development_seal import bundle  # noqa: F401, F811


def test_frozen_predictions(bundle):  # noqa: F811
    root, split, digest = bundle
    predictor = FrozenPredictor(root, split, digest)
    features = np.array([[0.1, 0.0], [2.5, 0.0]])
    raw, calibrated, alerts = predictor.predict(features, predictor.feature_names)
    assert raw.shape == calibrated.shape == alerts.shape == (2,)
    assert np.all((raw >= 0) & (raw <= 1))
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    np.testing.assert_array_equal(alerts, calibrated >= predictor.threshold)
    np.testing.assert_array_equal(
        predictor.predict(features, predictor.feature_names)[1], calibrated
    )


def test_frozen_rejects_reordered_schema(bundle):  # noqa: F811
    root, split, digest = bundle
    predictor = FrozenPredictor(root, split, digest)
    with pytest.raises(ValueError, match="feature schema mismatch"):
        predictor.predict(np.zeros((2, 2)), predictor.feature_names[::-1])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_frozen_rejects_nonfinite_features(bundle, value):  # noqa: F811
    root, split, digest = bundle
    predictor = FrozenPredictor(root, split, digest)
    with pytest.raises(ValueError, match="invalid inference features"):
        predictor.predict(np.array([[value, 0.0]]), predictor.feature_names)


def test_frozen_rejects_wrong_shape(bundle):  # noqa: F811
    root, split, digest = bundle
    predictor = FrozenPredictor(root, split, digest)
    with pytest.raises(ValueError, match="invalid inference features"):
        predictor.predict(np.zeros((2, 1)), predictor.feature_names)
