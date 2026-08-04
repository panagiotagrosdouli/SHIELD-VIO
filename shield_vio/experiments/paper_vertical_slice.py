"""First public-data failure-prediction slice from completed EuRoC runner artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from shield_vio.calibration_metrics.metrics import summarize_calibration
from shield_vio.evaluation.prediction_metrics import (
    auprc,
    auroc,
    event_detection_summary,
)
from shield_vio.evaluation.prediction_targets import (
    FailureEventIndex,
    FutureFailureTargets,
    build_persistent_failure_events,
    future_failure_targets,
)
from shield_vio.evaluation.statistics import binary_metrics
from shield_vio.evaluation.trajectory_metrics import align_positions_se3
from shield_vio.failure_detection.baselines import LogisticFailureDetector
from shield_vio.failure_detection.calibration import PlattCalibrator
from shield_vio.features.health_vector import HealthFeatureMatrix, load_causal_health_features


@dataclass(frozen=True)
class VerticalSliceConfig:
    history_seconds: float = 1.0
    horizon_seconds: float = 2.0
    position_error_threshold_m: float = 1.0
    persistence_seconds: float = 0.5
    max_ground_truth_gap_seconds: float = 0.02
    covariance_threshold: float = 1.0
    feature_count_threshold: float = 12.0
    nis_threshold: float = 11.345
    synthetic_train_samples: int = 2000
    synthetic_calibration_samples: int = 1000
    synthetic_validation_samples: int = 1000
    seed: int = 7

    def __post_init__(self) -> None:
        positive = (
            self.history_seconds,
            self.horizon_seconds,
            self.position_error_threshold_m,
            self.persistence_seconds,
            self.max_ground_truth_gap_seconds,
            self.covariance_threshold,
            self.feature_count_threshold,
            self.nis_threshold,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("vertical-slice durations and thresholds must be positive")
        if (
            min(
                self.synthetic_train_samples,
                self.synthetic_calibration_samples,
                self.synthetic_validation_samples,
            )
            < 20
        ):
            raise ValueError("synthetic reference splits require at least 20 samples")


@dataclass(frozen=True)
class PublicSequenceSlice:
    sequence: str
    features: HealthFeatureMatrix
    position_error_m: np.ndarray
    events: FailureEventIndex
    targets: FutureFailureTargets


def build_public_sequence_slice(
    run_dir: str | Path,
    sequence_root: str | Path,
    config: VerticalSliceConfig,
) -> PublicSequenceSlice:
    """Build deployable features and privileged offline labels as separate objects."""

    features = load_causal_health_features(run_dir, history_seconds=config.history_seconds)
    error_timestamps, position_error = _aligned_position_errors(
        Path(run_dir),
        Path(sequence_root),
        max_gap_seconds=config.max_ground_truth_gap_seconds,
    )
    feature_lookup = {int(value): index for index, value in enumerate(features.timestamps_ns)}
    shared_pairs = [
        (feature_lookup[int(timestamp)], error_index)
        for error_index, timestamp in enumerate(error_timestamps)
        if int(timestamp) in feature_lookup
    ]
    if len(shared_pairs) < 10:
        raise ValueError("fewer than ten health rows have associated ground truth")
    feature_indices = np.asarray([item[0] for item in shared_pairs], dtype=int)
    error_indices = np.asarray([item[1] for item in shared_pairs], dtype=int)
    selected_features = HealthFeatureMatrix(
        features.timestamps_ns[feature_indices],
        features.values[feature_indices],
        features.feature_names,
        features.max_source_timestamps_ns[feature_indices],
    )
    selected_error = position_error[error_indices]
    exceeded = selected_error > config.position_error_threshold_m
    events = build_persistent_failure_events(
        selected_features.timestamps_ns,
        exceeded,
        persistence_seconds=config.persistence_seconds,
    )
    targets = future_failure_targets(
        selected_features.timestamps_ns,
        events,
        horizon_seconds=config.horizon_seconds,
    )
    return PublicSequenceSlice(
        sequence=Path(sequence_root).name,
        features=selected_features,
        position_error_m=selected_error,
        events=events,
        targets=targets,
    )


def run_public_dataset_smoke(
    run_dir: str | Path,
    sequence_root: str | Path,
    output_dir: str | Path,
    *,
    config: VerticalSliceConfig | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Execute a synthetic-train/public-test smoke study without confirmatory claims."""

    cfg = config or VerticalSliceConfig()
    dataset = build_public_sequence_slice(run_dir, sequence_root, cfg)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    x_train, y_train = _synthetic_health_reference(
        dataset.features.feature_names, cfg.synthetic_train_samples, cfg.seed
    )
    x_calibration, y_calibration = _synthetic_health_reference(
        dataset.features.feature_names, cfg.synthetic_calibration_samples, cfg.seed + 1
    )
    x_validation, y_validation = _synthetic_health_reference(
        dataset.features.feature_names, cfg.synthetic_validation_samples, cfg.seed + 2
    )
    detector = LogisticFailureDetector(iterations=1000).fit(x_train, y_train)
    calibration_raw = detector.predict_proba(x_calibration)
    calibrator = PlattCalibrator().fit(calibration_raw, y_calibration)
    validation_probability = calibrator.predict_proba(detector.predict_proba(x_validation))
    learned_threshold = _select_sample_threshold(
        validation_probability, y_validation, max_false_positive_rate=0.05
    )

    features = dataset.features.values
    index = {name: position for position, name in enumerate(dataset.features.feature_names)}
    covariance_score = features[:, index["covariance_trace"]]
    feature_score = -features[:, index["tracked_features"]]
    nis_score = features[:, index["innovation_nis"]]
    raw_probability = detector.predict_proba(features)
    calibrated_probability = calibrator.predict_proba(raw_probability)
    predictions = {
        "covariance_trace": covariance_score >= cfg.covariance_threshold,
        "feature_count": -feature_score < cfg.feature_count_threshold,
        "innovation_nis": nis_score >= cfg.nis_threshold,
        "logistic_raw": raw_probability >= 0.5,
        "logistic_platt": calibrated_probability >= learned_threshold,
    }
    scores = {
        "covariance_trace": covariance_score,
        "feature_count": feature_score,
        "innovation_nis": nis_score,
        "logistic_raw": raw_probability,
        "logistic_platt": calibrated_probability,
    }

    eligible = dataset.targets.eligible_mask
    labels = dataset.targets.labels
    if len(np.unique(labels[eligible])) != 2:
        raise ValueError(
            "public smoke test requires both future-failure classes; choose a sequence/threshold "
            "with at least one observable event and eligible nominal monitoring interval"
        )

    method_metrics: dict[str, dict[str, Any]] = {}
    for method, method_scores in scores.items():
        method_metrics[method] = _method_metrics(
            dataset,
            method_scores,
            predictions[method],
        )
    method_metrics["logistic_platt"]["calibration"] = asdict(
        summarize_calibration(calibrated_probability[eligible], labels[eligible])
    )
    method_metrics["logistic_raw"]["calibration"] = asdict(
        summarize_calibration(raw_probability[eligible], labels[eligible])
    )

    _write_feature_table(dataset, destination / "health_features.csv")
    _write_prediction_table(
        dataset,
        scores,
        predictions,
        learned_threshold,
        destination / "predictions.csv",
    )
    reliability_source = _write_reliability_artifacts(
        calibrated_probability[eligible],
        labels[eligible],
        destination,
    )
    _write_timeline_artifacts(dataset, calibrated_probability, destination)
    (destination / "metrics.json").write_text(
        json.dumps(method_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "model.json").write_text(
        json.dumps(
            {
                "method": "logistic_regression",
                "feature_names": dataset.features.feature_names,
                "weights": detector.weights.tolist() if detector.weights is not None else None,
                "mean": detector.mean.tolist() if detector.mean is not None else None,
                "scale": detector.scale.tolist() if detector.scale is not None else None,
                "training_domain": "ANALYTIC_SYNTHETIC_HEALTH_V1",
                "seed": cfg.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "calibration.json").write_text(
        json.dumps(
            {
                **calibrator.to_dict(),
                "fit_domain": "ANALYTIC_SYNTHETIC_HEALTH_V1",
                "fit_samples": cfg.synthetic_calibration_samples,
                "validation_threshold": learned_threshold,
                "validation_false_positive_constraint": 0.05,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = _manifest(
        dataset,
        cfg,
        Path(run_dir),
        Path(sequence_root),
        destination,
        command or " ".join(sys.argv),
        reliability_source,
    )
    (destination / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _method_metrics(
    dataset: PublicSequenceSlice,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    eligible = dataset.targets.eligible_mask
    labels = dataset.targets.labels
    event_summary = event_detection_summary(
        dataset.features.timestamps_ns,
        predictions,
        dataset.events,
        warning_horizon_seconds=dataset.targets.horizon_seconds,
    )
    return {
        "auroc": auroc(labels[eligible], scores[eligible]),
        "auprc": auprc(labels[eligible], scores[eligible]),
        "frame": binary_metrics(labels[eligible], predictions[eligible]),
        "event": asdict(event_summary),
        "eligible_samples": int(np.sum(eligible)),
        "positive_windows": int(np.sum(labels[eligible])),
        "negative_windows": int(np.sum(~labels[eligible])),
    }


def _aligned_position_errors(
    run_dir: Path,
    sequence_root: Path,
    *,
    max_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = _dict_rows(run_dir / "trajectory.csv")
    estimate_timestamps = np.asarray(
        [int(row["frame_timestamp_ns"]) for row in trajectory], dtype=np.int64
    )
    estimated_positions = np.asarray(
        [[float(row["px"]), float(row["py"]), float(row["pz"])] for row in trajectory]
    )
    ground_truth_path = sequence_root / "mav0/state_groundtruth_estimate0/data.csv"
    ground_truth_rows = _raw_data_rows(ground_truth_path)
    ground_truth_timestamps = np.asarray([int(row[0]) for row in ground_truth_rows], dtype=np.int64)
    ground_truth_positions = np.asarray(
        [[float(row[1]), float(row[2]), float(row[3])] for row in ground_truth_rows]
    )
    if np.any(np.diff(estimate_timestamps) <= 0) or np.any(np.diff(ground_truth_timestamps) <= 0):
        raise ValueError("trajectory and ground-truth timestamps must be strictly increasing")

    inside = (estimate_timestamps >= ground_truth_timestamps[0]) & (
        estimate_timestamps <= ground_truth_timestamps[-1]
    )
    candidates = np.flatnonzero(inside)
    right = np.searchsorted(ground_truth_timestamps, estimate_timestamps[candidates], side="left")
    right = np.clip(right, 1, len(ground_truth_timestamps) - 1)
    left = right - 1
    gaps = np.minimum(
        estimate_timestamps[candidates] - ground_truth_timestamps[left],
        ground_truth_timestamps[right] - estimate_timestamps[candidates],
    )
    selected = candidates[gaps <= int(round(max_gap_seconds * 1e9))]
    if len(selected) < 3:
        raise ValueError("fewer than three trajectory samples have nearby ground truth")
    selected_timestamps = estimate_timestamps[selected]
    reference = np.column_stack(
        [
            np.interp(
                selected_timestamps.astype(float),
                ground_truth_timestamps.astype(float),
                ground_truth_positions[:, axis],
            )
            for axis in range(3)
        ]
    )
    aligned, _, _ = align_positions_se3(estimated_positions[selected], reference)
    return selected_timestamps, np.linalg.norm(aligned - reference, axis=1)


def _synthetic_health_reference(
    feature_names: tuple[str, ...],
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an explicit analytic reference domain without reading public test data."""

    rng = np.random.default_rng(seed)
    labels = np.zeros(sample_count, dtype=int)
    labels[sample_count // 2 :] = 1
    rng.shuffle(labels)
    failed = labels.astype(bool)
    values = rng.normal(0.0, 0.05, (sample_count, len(feature_names)))
    columns = {name: index for index, name in enumerate(feature_names)}

    covariance = np.where(
        failed, rng.lognormal(0.3, 0.35, sample_count), rng.lognormal(-2.3, 0.25, sample_count)
    )
    nis = np.where(failed, rng.normal(16.0, 3.0, sample_count), rng.normal(2.0, 0.8, sample_count))
    tracked = np.where(
        failed, rng.normal(14.0, 6.0, sample_count), rng.normal(80.0, 10.0, sample_count)
    )
    detected = tracked + np.where(
        failed, rng.normal(8.0, 3.0, sample_count), rng.normal(20.0, 5.0, sample_count)
    )
    correspondences = np.maximum(0.0, tracked - rng.normal(4.0, 2.0, sample_count))
    inliers = np.maximum(0.0, correspondences * np.where(failed, 0.45, 0.85))
    ratio = np.divide(inliers, np.maximum(correspondences, 1.0))

    assignments = {
        "covariance_trace": covariance,
        "covariance_log1p": np.log1p(covariance),
        "covariance_condition_log1p": np.where(failed, 5.0, 2.0) + rng.normal(0, 0.3, sample_count),
        "innovation_nis": np.maximum(0.0, nis),
        "innovation_nis_missing": rng.binomial(1, np.where(failed, 0.2, 0.05)),
        "detected_features": np.maximum(0.0, detected),
        "tracked_features": np.maximum(0.0, tracked),
        "correspondence_count": correspondences,
        "inlier_count": inliers,
        "inlier_ratio": np.clip(ratio, 0.0, 1.0),
        "visual_update_missing": rng.binomial(1, np.where(failed, 0.35, 0.03)),
        "seconds_since_visual_update": np.maximum(
            0.0, np.where(failed, 0.4, 0.03) + rng.normal(0, 0.03, sample_count)
        ),
        "tracking_lost": rng.binomial(1, np.where(failed, 0.25, 0.01)),
        "covariance_growth_per_s": np.where(failed, 0.5, 0.01) + rng.normal(0, 0.08, sample_count),
        "rolling_covariance_mean": covariance * np.where(failed, 0.8, 1.0),
        "rolling_covariance_std": np.where(failed, 0.3, 0.02)
        + np.abs(rng.normal(0, 0.03, sample_count)),
        "rolling_covariance_slope_per_s": np.where(failed, 0.35, 0.0)
        + rng.normal(0, 0.06, sample_count),
        "rolling_nis_mean": np.maximum(0.0, nis * np.where(failed, 0.8, 1.0)),
        "rolling_tracked_features_mean": np.maximum(0.0, tracked + rng.normal(0, 3, sample_count)),
        "rolling_tracked_features_slope_per_s": np.where(failed, -12.0, 0.0)
        + rng.normal(0, 2.0, sample_count),
    }
    for name, column in assignments.items():
        values[:, columns[name]] = column
    return values, labels


def _select_sample_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    max_false_positive_rate: float,
) -> float:
    scores = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=bool)
    candidates = np.r_[np.nextafter(np.max(scores), np.inf), np.unique(scores)[::-1]]
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        metrics = binary_metrics(y, scores >= threshold)
        if float(metrics["false_alarm_rate"]) <= max_false_positive_rate:
            feasible.append((float(metrics["recall"]), float(metrics["precision"]), threshold))
    if not feasible:
        raise RuntimeError("no validation threshold satisfies the false-positive constraint")
    return float(max(feasible)[-1])


def _write_feature_table(dataset: PublicSequenceSlice, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["timestamp_ns", "max_source_timestamp_ns", *dataset.features.feature_names]
        )
        for timestamp, source, values in zip(
            dataset.features.timestamps_ns,
            dataset.features.max_source_timestamps_ns,
            dataset.features.values,
            strict=True,
        ):
            writer.writerow([int(timestamp), int(source), *values.tolist()])


def _write_prediction_table(
    dataset: PublicSequenceSlice,
    scores: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    learned_threshold: float,
    path: Path,
) -> None:
    methods = tuple(scores)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "timestamp_ns",
                "timestamp_s",
                "position_error_m",
                "failure_active",
                "future_failure",
                "eligible",
                "learned_validation_threshold",
                *[f"{method}_score" for method in methods],
                *[f"{method}_prediction" for method in methods],
            ]
        )
        origin = dataset.features.timestamps_ns[0]
        for index, timestamp in enumerate(dataset.features.timestamps_ns):
            writer.writerow(
                [
                    int(timestamp),
                    float(timestamp - origin) * 1e-9,
                    dataset.position_error_m[index],
                    int(dataset.events.active_mask[index]),
                    int(dataset.targets.labels[index]),
                    int(dataset.targets.eligible_mask[index]),
                    learned_threshold,
                    *[scores[method][index] for method in methods],
                    *[int(predictions[method][index]) for method in methods],
                ]
            )


def _write_reliability_artifacts(
    probabilities: np.ndarray,
    labels: np.ndarray,
    destination: Path,
    *,
    bins: int = 10,
) -> str:
    edges = np.linspace(0.0, 1.0, bins + 1)
    source: list[tuple[float, float, int]] = []
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            (probabilities < edges[index + 1]) if index < bins - 1 else (probabilities <= 1.0)
        )
        if np.any(mask):
            source.append(
                (
                    float(np.mean(probabilities[mask])),
                    float(np.mean(labels[mask])),
                    int(np.sum(mask)),
                )
            )
    source_path = destination / "reliability_source.csv"
    with source_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mean_probability", "empirical_frequency", "sample_count"])
        writer.writerows(source)

    figure, axis = plt.subplots(figsize=(5.0, 4.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="ideal")
    axis.plot(
        [item[0] for item in source],
        [item[1] for item in source],
        marker="o",
        label="logistic + Platt",
    )
    axis.set(xlabel="Predicted failure probability", ylabel="Empirical failure frequency")
    axis.set_title(f"Public-data smoke reliability (n={len(labels)})")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "reliability_diagram.pdf")
    figure.savefig(destination / "reliability_diagram.svg")
    plt.close(figure)
    return source_path.name


def _write_timeline_artifacts(
    dataset: PublicSequenceSlice,
    probability: np.ndarray,
    destination: Path,
) -> None:
    time_s = (dataset.features.timestamps_ns - dataset.features.timestamps_ns[0]) * 1e-9
    source_path = destination / "prediction_timeline_source.csv"
    with source_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "failure_probability", "position_error_m", "failure_active"])
        writer.writerows(
            zip(
                time_s,
                probability,
                dataset.position_error_m,
                dataset.events.active_mask.astype(int),
                strict=True,
            )
        )

    figure, probability_axis = plt.subplots(figsize=(9.0, 4.2))
    probability_axis.plot(time_s, probability, color="tab:red", label="calibrated risk")
    probability_axis.set(xlabel="Time (s)", ylabel="Failure probability")
    probability_axis.set_ylim(0, 1)
    error_axis = probability_axis.twinx()
    error_axis.plot(
        time_s, dataset.position_error_m, color="tab:blue", alpha=0.7, label="position error"
    )
    error_axis.set_ylabel("Aligned position error (m)")
    if np.any(dataset.events.active_mask):
        probability_axis.fill_between(
            time_s,
            0,
            1,
            where=dataset.events.active_mask,
            color="0.7",
            alpha=0.25,
            label="failure active",
        )
    handles_a, labels_a = probability_axis.get_legend_handles_labels()
    handles_b, labels_b = error_axis.get_legend_handles_labels()
    probability_axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper left")
    probability_axis.grid(alpha=0.2)
    probability_axis.set_title(
        f"{dataset.sequence}: prediction timeline (public-data smoke, n={len(time_s)})"
    )
    figure.tight_layout()
    figure.savefig(destination / "prediction_timeline.pdf")
    figure.savefig(destination / "prediction_timeline.svg")
    plt.close(figure)


def _manifest(
    dataset: PublicSequenceSlice,
    config: VerticalSliceConfig,
    run_dir: Path,
    sequence_root: Path,
    destination: Path,
    command: str,
    reliability_source: str,
) -> dict[str, Any]:
    artifacts = [
        "health_features.csv",
        "predictions.csv",
        "metrics.json",
        "model.json",
        "calibration.json",
        reliability_source,
        "reliability_diagram.pdf",
        "reliability_diagram.svg",
        "prediction_timeline_source.csv",
        "prediction_timeline.pdf",
        "prediction_timeline.svg",
    ]
    config_payload = asdict(config)
    return {
        "schema_version": 1,
        "experiment_id": f"euroc-smoke-{dataset.sequence}-h{config.horizon_seconds:g}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_level": "PUBLIC_DATASET_SMOKE",
        "confirmatory": False,
        "claim_boundary": (
            "One real EuRoC sequence tested after analytic synthetic health training, calibration, "
            "and validation. This verifies pipeline execution only and does not confirm H1-H5."
        ),
        "git": _git_state(),
        "python": sys.version,
        "packages": _package_versions(),
        "operating_system": platform.platform(),
        "hardware": _hardware(),
        "dataset": "EuRoC_MAV",
        "sequence": dataset.sequence,
        "dataset_index_sha256": _sha256_files(
            [
                sequence_root / "mav0/cam0/data.csv",
                sequence_root / "mav0/imu0/data.csv",
                sequence_root / "mav0/state_groundtruth_estimate0/data.csv",
            ]
        ),
        "estimator_manifest": str(run_dir / "experiment_manifest.json"),
        "detector": "logistic_regression",
        "detector_training_domain": "ANALYTIC_SYNTHETIC_HEALTH_V1",
        "calibration_method": "platt_logit",
        "shift_method": "none",
        "policy": "not_evaluated",
        "failure_definition": {
            "criterion": "se3_aligned_position_error",
            "threshold_m": config.position_error_threshold_m,
            "persistence_seconds": config.persistence_seconds,
        },
        "prediction_horizon_seconds": config.horizon_seconds,
        "split_definition": {
            "train": "analytic_synthetic_seed_7",
            "calibration": "analytic_synthetic_seed_8",
            "validation": "analytic_synthetic_seed_9",
            "test": dataset.sequence,
        },
        "seed": config.seed,
        "command": command,
        "configuration": config_payload,
        "configuration_sha256": hashlib.sha256(
            json.dumps(config_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "sample_counts": {
            "associated": len(dataset.features.timestamps_ns),
            "eligible": int(np.sum(dataset.targets.eligible_mask)),
            "positive_windows": int(np.sum(dataset.targets.labels[dataset.targets.eligible_mask])),
            "negative_windows": int(np.sum(~dataset.targets.labels[dataset.targets.eligible_mask])),
            "failure_events": len(dataset.events.onsets_ns),
        },
        "artifact_sha256": {name: _sha256_files([destination / name]) for name in artifacts},
        "status": "complete",
    }


def _dict_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"artifact contains no rows: {path}")
    return rows


def _raw_data_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append([item.strip() for item in next(csv.reader([stripped]))])
    if not rows:
        raise ValueError(f"artifact contains no data rows: {path}")
    return rows


def _git_state() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit_sha": sha, "dirty_tree": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit_sha": "unknown", "dirty_tree": None}


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "matplotlib", "PyYAML", "opencv-python-headless", "shield-vio")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _hardware() -> dict[str, Any]:
    memory_bytes: int | None = None
    try:
        memory_bytes = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": memory_bytes,
    }


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
