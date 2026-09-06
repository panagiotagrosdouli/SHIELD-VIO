"""Canonical causal health representation for publication experiments."""

from .dataset import PredictionDataset, PredictionGroups, build_prediction_dataset
from .schema import (
    EstimatorHealth,
    EvaluationDiagnostics,
    FeatureRole,
    FeatureSpec,
    HealthSample,
    HealthValue,
    InertialHealth,
    VisualHealth,
    deployable_feature_specs,
    flatten_deployable,
)
from .temporal import TemporalFeatureTable, build_trailing_features

__all__ = [
    "EstimatorHealth",
    "EvaluationDiagnostics",
    "FeatureRole",
    "FeatureSpec",
    "HealthSample",
    "HealthValue",
    "InertialHealth",
    "PredictionDataset",
    "PredictionGroups",
    "TemporalFeatureTable",
    "VisualHealth",
    "build_prediction_dataset",
    "build_trailing_features",
    "deployable_feature_specs",
    "flatten_deployable",
]
