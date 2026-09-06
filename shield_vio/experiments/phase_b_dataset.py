"""Build canonical Phase B prediction datasets from deterministic synthetic artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from shield_vio.evaluation.failure_definition import (
    build_failure_events_and_targets,
    evaluate_failure_criteria,
    load_failure_definition,
)
from shield_vio.health.dataset import PredictionGroups, build_prediction_dataset
from shield_vio.health.schema import (
    EstimatorHealth,
    EvaluationDiagnostics,
    HealthSample,
    HealthValue,
    VisualHealth,
)


def build_synthetic_prediction_dataset(
    synthetic_dir: str | Path,
    output_dir: str | Path,
    *,
    failure_config: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Create deterministic canonical outputs without using degradation metadata as features."""

    source = Path(synthetic_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    gt = _rows(source / "ground_truth.csv")
    est = _rows(source / "estimated_trajectory.csv")
    unc = _rows(source / "uncertainty.csv")
    visual = _rows(source / "visual_quality.csv")
    if not (len(gt) == len(est) == len(unc) == len(visual)):
        raise ValueError("synthetic artifact rows must align")

    timestamps_ns = np.asarray([round(float(row["t"]) * 1e9) for row in gt], dtype=np.int64)
    if len(timestamps_ns) < 2 or np.any(np.diff(timestamps_ns) <= 0):
        raise ValueError("synthetic timestamps must be strictly increasing")

    position_error = np.asarray(
        [
            np.linalg.norm(
                np.asarray([float(e["x"]), float(e["y"]), float(e["z"])])
                - np.asarray([float(g["x"]), float(g["y"]), float(g["z"])])
            )
            for g, e in zip(gt, est)
        ],
        dtype=float,
    )

    samples: list[HealthSample] = []
    for index, timestamp in enumerate(timestamps_ns):
        feature_count = float(visual[index]["feature_count"])
        outlier_ratio = float(visual[index]["outlier_rate"])
        samples.append(
            HealthSample(
                timestamp_ns=int(timestamp),
                sequence_id="synthetic_demo",
                estimator_id="synthetic_ekf",
                visual=VisualHealth(
                    tracked_feature_count=HealthValue(feature_count, True, int(timestamp)),
                    inlier_ratio=HealthValue(1.0 - outlier_ratio, True, int(timestamp)),
                    outlier_ratio=HealthValue(outlier_ratio, True, int(timestamp)),
                    blur_score=HealthValue(float(visual[index]["blur_proxy"]), True, int(timestamp)),
                    brightness=HealthValue(float(visual[index]["light_proxy"]), True, int(timestamp)),
                    frame_available=HealthValue(True, True, int(timestamp)),
                ),
                estimator=EstimatorHealth(
                    nis=HealthValue(float(unc[index]["nis"]), True, int(timestamp)),
                    covariance_trace=HealthValue(float(unc[index]["trace"]), True, int(timestamp)),
                    covariance_logdet=HealthValue(float(unc[index]["logdet"]), True, int(timestamp)),
                ),
                evaluation=EvaluationDiagnostics(
                    ground_truth_position_error_m=HealthValue(
                        float(position_error[index]), True, int(timestamp)
                    ),
                    nees=HealthValue(float(unc[index]["nees"]), True, int(timestamp)),
                ),
            )
        )

    definition = load_failure_definition(failure_config)
    exceeded = evaluate_failure_criteria({"position_error_m": position_error}, definition)
    events, target_set = build_failure_events_and_targets(timestamps_ns, exceeded, definition)
    dataset = build_prediction_dataset(
        samples,
        horizon_targets={h: target.labels for h, target in target_set.targets.items()},
        eligible_masks={h: target.eligible_mask for h, target in target_set.targets.items()},
        groups=PredictionGroups("synthetic", "synthetic_demo", "synthetic_ekf", "mixed", seed),
        metadata={
            "evidence_level": "SYNTHETIC_PIPELINE_VALIDATION",
            "failure_definition": definition.schema_version,
            "source_run": str(source),
        },
    )

    _write_health_samples(samples, destination / "health_samples.csv")
    _write_targets(timestamps_ns, target_set.targets, destination / "prediction_targets.csv")
    _write_prediction_dataset(dataset, timestamps_ns, destination / "prediction_dataset.csv")
    report = _diagnostic_report(dataset, events.event_ids, events.onsets_ns)
    (destination / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "evidence_level": "SYNTHETIC_PIPELINE_VALIDATION",
        "confirmatory": False,
        "health_schema_version": samples[0].schema_version,
        "failure_definition_id": definition.schema_version,
        "horizons_seconds": list(definition.horizons_seconds),
        "source_run": str(source),
        "sequence": "synthetic_demo",
        "estimator": "synthetic_ekf",
        "seed": seed,
        "samples": len(samples),
        "failure_events": len(events.event_ids),
        "prohibited_predictor_fields": [
            "ground_truth_position_error_m",
            "ground_truth_rotation_error_deg",
            "nees",
            "failure_label",
            "future_failure",
            "degradation_type",
            "degradation_severity",
            "degradation_onset",
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"artifact contains no rows: {path}")
    return rows


def _write_health_samples(samples: list[HealthSample], path: Path) -> None:
    fieldnames = [
        "timestamp_ns",
        "schema_version",
        "tracked_feature_count",
        "inlier_ratio",
        "outlier_ratio",
        "blur_score",
        "brightness",
        "nis",
        "covariance_trace",
        "covariance_logdet",
        "ground_truth_position_error_m__evaluation_only",
        "nees__evaluation_only",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "timestamp_ns": sample.timestamp_ns,
                    "schema_version": sample.schema_version,
                    "tracked_feature_count": sample.visual.tracked_feature_count.value,
                    "inlier_ratio": sample.visual.inlier_ratio.value,
                    "outlier_ratio": sample.visual.outlier_ratio.value,
                    "blur_score": sample.visual.blur_score.value,
                    "brightness": sample.visual.brightness.value,
                    "nis": sample.estimator.nis.value,
                    "covariance_trace": sample.estimator.covariance_trace.value,
                    "covariance_logdet": sample.estimator.covariance_logdet.value,
                    "ground_truth_position_error_m__evaluation_only": sample.evaluation.ground_truth_position_error_m.value,
                    "nees__evaluation_only": sample.evaluation.nees.value,
                }
            )


def _write_targets(timestamps: np.ndarray, targets: Any, path: Path) -> None:
    horizons = sorted(targets)
    with path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["timestamp_ns"]
        for horizon in horizons:
            tag = str(horizon).replace(".", "p")
            fieldnames.extend([f"failure_within_{tag}s", f"eligible_{tag}s"])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, timestamp in enumerate(timestamps):
            row: dict[str, object] = {"timestamp_ns": int(timestamp)}
            for horizon in horizons:
                tag = str(horizon).replace(".", "p")
                row[f"failure_within_{tag}s"] = int(targets[horizon].labels[index])
                row[f"eligible_{tag}s"] = int(targets[horizon].eligible_mask[index])
            writer.writerow(row)


def _write_prediction_dataset(dataset: Any, timestamps: np.ndarray, path: Path) -> None:
    values, names = dataset.features()
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_ns", *names])
        for timestamp, row in zip(timestamps, values):
            writer.writerow([int(timestamp), *row.tolist()])


def _diagnostic_report(dataset: Any, event_ids: tuple[str, ...], onsets_ns: np.ndarray) -> dict[str, Any]:
    features, names = dataset.features()
    missing = {
        name: float(np.mean(features[:, index]))
        for index, name in enumerate(names)
        if name.endswith("__missing")
    }
    positive_rate = {}
    excluded = {}
    for horizon in sorted(dataset.horizon_targets):
        labels, eligible = dataset.targets(horizon)
        positive_rate[str(horizon)] = float(np.mean(labels[eligible])) if np.any(eligible) else 0.0
        excluded[str(horizon)] = int(np.sum(~eligible))
    duration = (dataset.samples[-1].timestamp_ns - dataset.samples[0].timestamp_ns) * 1e-9
    return {
        "samples": len(dataset.samples),
        "duration_seconds": float(duration),
        "available_feature_columns": list(names),
        "missingness_fraction_by_indicator": missing,
        "failure_events": len(event_ids),
        "failure_event_ids": list(event_ids),
        "failure_onsets_seconds": [float(value) * 1e-9 for value in onsets_ns],
        "positive_rate_by_horizon": positive_rate,
        "excluded_or_censored_by_horizon": excluded,
        "prohibited_fields_excluded_from_predictor_matrix": True,
    }
