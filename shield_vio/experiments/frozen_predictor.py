"""Inference-only loading of a verified development model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shield_vio.experiments.development_seal import verify_development_seal
from shield_vio.failure_detection.baselines import LogisticFailureDetector
from shield_vio.failure_detection.calibration import PlattCalibrator


class FrozenPredictor:
    """Apply frozen preprocessing, model, calibration and threshold without fitting."""

    def __init__(self, root: Path, split_config: Path, expected_sha256: str):
        manifest = verify_development_seal(root, split_config, expected_sha256)
        model = json.loads((root / "model.json").read_text(encoding="utf-8"))
        calibration = json.loads(
            (root / "calibration.json").read_text(encoding="utf-8")
        )
        threshold = json.loads((root / "threshold.json").read_text(encoding="utf-8"))
        self.feature_names = tuple(manifest["feature_schema"])
        self.horizon_seconds = float(manifest["horizon_seconds"])
        self.threshold = float(threshold["threshold"])
        n = len(self.feature_names)
        if (
            not n
            or len(set(self.feature_names)) != n
            or not np.isfinite(self.horizon_seconds)
            or self.horizon_seconds <= 0
            or not np.isfinite(self.threshold)
            or not 0 <= self.threshold <= np.nextafter(1.0, np.inf)
            or model.get("method") != "P-LOG"
            or calibration.get("method") != "platt_logit"
        ):
            raise ValueError("invalid frozen predictor metadata")
        detector = LogisticFailureDetector()
        detector.weights = np.asarray(model["weights"], dtype=float)
        detector.mean = np.asarray(model["mean"], dtype=float)
        detector.scale = np.asarray(model["scale"], dtype=float)
        for array, shape in (
            (detector.weights, (n + 1,)),
            (detector.mean, (n,)),
            (detector.scale, (n,)),
        ):
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError("invalid frozen model parameters")
            array.setflags(write=False)
        if np.any(detector.scale <= 0):
            raise ValueError("invalid frozen feature scale")
        calibrator = PlattCalibrator()
        calibrator.intercept_ = float(calibration["intercept"])
        calibrator.slope_ = float(calibration["slope"])
        if not np.all(np.isfinite([calibrator.intercept_, calibrator.slope_])):
            raise ValueError("invalid frozen calibration parameters")
        self._detector = detector
        self._calibrator = calibrator

    def predict(
        self, features: np.ndarray, feature_names: tuple[str, ...]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw probabilities, calibrated probabilities, and binary alerts."""
        if tuple(feature_names) != self.feature_names:
            raise ValueError("feature schema mismatch")
        values = np.asarray(features, dtype=float)
        if (
            values.ndim != 2
            or not len(values)
            or values.shape[1] != len(self.feature_names)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("invalid inference features")
        raw = self._detector.predict_proba(values)
        calibrated = self._calibrator.predict_proba(raw)
        return raw, calibrated, calibrated >= self.threshold
