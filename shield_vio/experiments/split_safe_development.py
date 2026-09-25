"""Leakage-resistant public-data development fitting for the paper pipeline.

This module intentionally consumes only train/calibration/validation sequence artifacts.
It does not expose a confirmatory-test evaluation path.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from shield_vio.datasets.splits import PaperSplit, SequenceIdentity, load_paper_split
from shield_vio.evaluation.prediction_metrics import event_detection_summary
from shield_vio.evaluation.prediction_targets import FailureEventIndex
from shield_vio.failure_detection.baselines import LogisticFailureDetector
from shield_vio.failure_detection.calibration import PlattCalibrator


DEVELOPMENT_SPLITS = ("train", "calibration", "validation")


@dataclass(frozen=True)
class CanonicalPredictionRun:
    root: Path
    dataset: str
    sequence: str
    estimator: str
    split: str
    timestamps_ns: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    labels: np.ndarray
    eligible_mask: np.ndarray
    events: FailureEventIndex
    evidence_level: str
    degradation_condition: str
    seed: int

    def eligible_features_labels(self) -> tuple[np.ndarray, np.ndarray]:
        return self.features[self.eligible_mask], self.labels[self.eligible_mask]


@dataclass(frozen=True)
class GroupedThresholdSummary:
    threshold: float
    event_count: int
    detected_events: int
    false_alerts: int
    event_recall: float
    event_precision: float
    false_alarms_per_minute: float
    median_lead_time_seconds: float


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _dict_rows(path: Path, *, allow_empty: bool = False) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        names = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not names:
        raise ValueError(f"CSV has no header: {path}")
    if not rows and not allow_empty:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows, names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_columns(fieldnames: Iterable[str], horizon_seconds: float) -> tuple[str, str]:
    targets: dict[float, str] = {}
    eligible: dict[float, str] = {}
    for name in fieldnames:
        if name.startswith("failure_within_") and name.endswith("s"):
            text = name[len("failure_within_") : -1].replace("p", ".")
            try:
                targets[float(text)] = name
            except ValueError:
                continue
        if name.startswith("eligible_") and name.endswith("s"):
            text = name[len("eligible_") : -1].replace("p", ".")
            try:
                eligible[float(text)] = name
            except ValueError:
                continue
    matches = [value for value in targets if abs(value - horizon_seconds) < 1e-12]
    if len(matches) != 1 or matches[0] not in eligible:
        raise ValueError(f"prediction targets do not contain horizon {horizon_seconds:g}s")
    horizon = matches[0]
    return targets[horizon], eligible[horizon]


def _active_mask(timestamps_ns: np.ndarray, onsets: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    active = np.zeros(len(timestamps_ns), dtype=bool)
    for onset, offset in zip(onsets, offsets, strict=True):
        active |= (timestamps_ns >= int(onset)) & (timestamps_ns <= int(offset))
    return active


def load_canonical_prediction_run(
    root: str | Path,
    *,
    paper_split: PaperSplit,
    requested_split: str,
    horizon_seconds: float,
    expected_dataset: str | None = None,
    expected_sequence: str | None = None,
    expected_estimator: str | None = None,
) -> CanonicalPredictionRun:
    """Load and validate one canonical public-sequence prediction artifact."""

    if requested_split not in DEVELOPMENT_SPLITS:
        raise ValueError(
            "development loader accepts only train/calibration/validation; "
            "test and shifted_test are intentionally sealed"
        )

    path = Path(root)
    manifest = _json(path / "canonical_manifest.json")
    dataset = str(manifest.get("dataset", "")).strip()
    sequence = str(manifest.get("sequence", "")).strip()
    estimator = str(manifest.get("estimator", "")).strip()
    if not dataset or not sequence or not estimator:
        raise ValueError("canonical manifest must identify dataset, sequence, and estimator")
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError(f"dataset mismatch: {dataset!r} != {expected_dataset!r}")
    if expected_sequence is not None and sequence != expected_sequence:
        raise ValueError(f"sequence mismatch: {sequence!r} != {expected_sequence!r}")
    if expected_estimator is not None and estimator != expected_estimator:
        raise ValueError(f"estimator mismatch: {estimator!r} != {expected_estimator!r}")
    paper_split.assert_run_membership(dataset, sequence, requested_split)

    feature_rows, feature_fields = _dict_rows(path / "prediction_dataset.csv")
    if feature_fields[0] != "timestamp_ns" or len(feature_fields) < 2:
        raise ValueError("prediction_dataset.csv must begin with timestamp_ns and contain features")
    timestamps = np.asarray([int(row["timestamp_ns"]) for row in feature_rows], dtype=np.int64)
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("prediction dataset timestamps must be strictly increasing")
    feature_names = feature_fields[1:]
    features = np.asarray(
        [[float(row[name]) for name in feature_names] for row in feature_rows],
        dtype=float,
    )
    if np.any(~np.isfinite(features)):
        raise ValueError("prediction features must be finite; missingness must be explicit columns")
    prohibited = (
        "ground_truth",
        "failure_label",
        "future_failure",
        "degradation_type",
        "degradation_severity",
        "degradation_onset",
    )
    for name in feature_names:
        lowered = name.lower()
        if any(token in lowered for token in prohibited):
            raise ValueError(f"prohibited predictor column in deployable matrix: {name}")

    target_rows, target_fields = _dict_rows(path / "prediction_targets.csv")
    if len(target_rows) != len(timestamps):
        raise ValueError("prediction target rows must align with prediction feature rows")
    target_timestamps = np.asarray([int(row["timestamp_ns"]) for row in target_rows], dtype=np.int64)
    if not np.array_equal(target_timestamps, timestamps):
        raise ValueError("prediction target timestamps must exactly match feature timestamps")
    target_column, eligible_column = _target_columns(target_fields, horizon_seconds)
    labels = np.asarray([bool(int(row[target_column])) for row in target_rows], dtype=bool)
    eligible_mask = np.asarray(
        [bool(int(row[eligible_column])) for row in target_rows], dtype=bool
    )
    if not np.any(eligible_mask):
        raise ValueError("canonical run has no eligible prediction rows for requested horizon")

    event_rows, event_fields = _dict_rows(path / "failure_events.csv", allow_empty=True)
    required_event_fields = {"event_id", "onset_ns", "offset_ns"}
    if not required_event_fields <= set(event_fields):
        raise ValueError("failure_events.csv has an invalid schema")
    onsets = np.asarray([int(row["onset_ns"]) for row in event_rows], dtype=np.int64)
    offsets = np.asarray([int(row["offset_ns"]) for row in event_rows], dtype=np.int64)
    events = FailureEventIndex(onsets, offsets, _active_mask(timestamps, onsets, offsets))

    outer_manifest = _json(path / "manifest.json") if (path / "manifest.json").is_file() else {}
    evidence_level = str(
        outer_manifest.get("evidence_level", manifest.get("evidence_level", "UNKNOWN"))
    )
    degradation = str(outer_manifest.get("degradation_condition", "clean"))
    seed = int(outer_manifest.get("seed", 0))

    return CanonicalPredictionRun(
        root=path,
        dataset=dataset,
        sequence=sequence,
        estimator=estimator,
        split=requested_split,
        timestamps_ns=timestamps,
        features=features,
        feature_names=feature_names,
        labels=labels,
        eligible_mask=eligible_mask,
        events=events,
        evidence_level=evidence_level,
        degradation_condition=degradation,
        seed=seed,
    )


def _partition_identities(
    paper_split: PaperSplit, partition: str, dataset: str
) -> tuple[SequenceIdentity, ...]:
    identities = tuple(
        identity for identity in paper_split.partition(partition) if identity.dataset == dataset
    )
    if not identities:
        raise ValueError(f"no {dataset!r} sequences registered in {partition!r}")
    return identities


def load_development_runs(
    artifacts_root: str | Path,
    split_path: str | Path,
    *,
    dataset: str,
    estimator: str,
    horizon_seconds: float,
) -> dict[str, tuple[CanonicalPredictionRun, ...]]:
    """Load all required development sequences, never test or shifted-test."""

    root = Path(artifacts_root)
    paper_split = load_paper_split(split_path)
    result: dict[str, tuple[CanonicalPredictionRun, ...]] = {}
    schema: tuple[str, ...] | None = None
    for partition in DEVELOPMENT_SPLITS:
        runs: list[CanonicalPredictionRun] = []
        for identity in _partition_identities(paper_split, partition, dataset):
            run = load_canonical_prediction_run(
                root / identity.sequence,
                paper_split=paper_split,
                requested_split=partition,
                horizon_seconds=horizon_seconds,
                expected_dataset=dataset,
                expected_sequence=identity.sequence,
                expected_estimator=estimator,
            )
            if schema is None:
                schema = run.feature_names
            elif run.feature_names != schema:
                raise ValueError(
                    f"feature schema mismatch for {run.dataset}/{run.sequence}; "
                    "all development runs must use an identical ordered schema"
                )
            runs.append(run)
        result[partition] = tuple(runs)
    return result


def _pooled_eligible(
    runs: Iterable[CanonicalPredictionRun],
) -> tuple[np.ndarray, np.ndarray]:
    x: list[np.ndarray] = []
    y: list[np.ndarray] = []
    for run in runs:
        features, labels = run.eligible_features_labels()
        x.append(features)
        y.append(labels)
    if not x:
        raise ValueError("no runs supplied")
    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)


def _eligible_seconds(run: CanonicalPredictionRun) -> float:
    intervals_s = np.diff(run.timestamps_ns).astype(float) * 1e-9
    return float(np.sum(intervals_s[~run.events.active_mask[:-1]]))


def grouped_event_threshold(
    runs: tuple[CanonicalPredictionRun, ...],
    scores: tuple[np.ndarray, ...],
    *,
    warning_horizon_seconds: float,
    max_false_alarms_per_minute: float,
) -> GroupedThresholdSummary:
    """Choose one threshold across validation sequences without concatenating timelines."""

    if len(runs) != len(scores) or not runs:
        raise ValueError("runs and scores must be aligned and non-empty")
    vectors: list[np.ndarray] = []
    for run, values in zip(runs, scores, strict=True):
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.shape != run.timestamps_ns.shape or np.any(~np.isfinite(array)):
            raise ValueError("each score vector must align with its validation run")
        vectors.append(array)
    pooled = np.concatenate(vectors)
    candidates = np.r_[np.nextafter(np.max(pooled), np.inf), np.unique(pooled)[::-1]]
    total_eligible_seconds = sum(_eligible_seconds(run) for run in runs)
    total_events = sum(len(run.events.onsets_ns) for run in runs)
    if total_events <= 0:
        raise ValueError("validation partition requires at least one persistent failure event")
    if total_eligible_seconds <= 0:
        raise ValueError("validation partition has no eligible monitoring duration")

    feasible: list[tuple[float, float, float, float, GroupedThresholdSummary]] = []
    for threshold in candidates:
        detected = 0
        false_alerts = 0
        lead_times: list[float] = []
        for run, values in zip(runs, vectors, strict=True):
            summary = event_detection_summary(
                run.timestamps_ns,
                values >= threshold,
                run.events,
                warning_horizon_seconds=warning_horizon_seconds,
            )
            detected += summary.detected_events
            false_alerts += summary.false_alerts
            lead_times.extend(summary.lead_times_seconds)
        recall = detected / total_events
        precision = detected / (detected + false_alerts) if detected + false_alerts else 0.0
        false_per_minute = false_alerts / (total_eligible_seconds / 60.0)
        median_lead = float(np.median(lead_times)) if lead_times else 0.0
        result = GroupedThresholdSummary(
            threshold=float(threshold),
            event_count=total_events,
            detected_events=detected,
            false_alerts=false_alerts,
            event_recall=float(recall),
            event_precision=float(precision),
            false_alarms_per_minute=float(false_per_minute),
            median_lead_time_seconds=median_lead,
        )
        if false_per_minute <= max_false_alarms_per_minute:
            feasible.append((recall, median_lead, precision, float(threshold), result))
    if not feasible:
        raise RuntimeError("no grouped validation threshold satisfies false-alarm constraint")
    return max(feasible, key=lambda item: item[:4])[-1]


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_manifest_hash(run: CanonicalPredictionRun) -> str:
    candidates = [run.root / "manifest.json", run.root / "canonical_manifest.json"]
    digest = hashlib.sha256()
    for path in candidates:
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def fit_development_bundle(
    artifacts_root: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    *,
    dataset: str = "euroc",
    estimator: str = "internal_eskf",
    horizon_seconds: float = 2.0,
    max_false_alarms_per_minute: float = 0.2,
    logistic_iterations: int = 1000,
) -> dict[str, Any]:
    """Fit detector, calibrator, and operating point using development partitions only."""

    runs = load_development_runs(
        artifacts_root,
        split_path,
        dataset=dataset,
        estimator=estimator,
        horizon_seconds=horizon_seconds,
    )
    training_x, training_y = _pooled_eligible(runs["train"])
    calibration_x, calibration_y = _pooled_eligible(runs["calibration"])
    if len(np.unique(training_y)) != 2:
        raise ValueError("pooled training partition requires both future-failure classes")
    if len(np.unique(calibration_y)) != 2:
        raise ValueError("pooled calibration partition requires both future-failure classes")

    detector = LogisticFailureDetector(iterations=logistic_iterations).fit(training_x, training_y)
    calibration_raw = detector.predict_proba(calibration_x)
    calibrator = PlattCalibrator().fit(calibration_raw, calibration_y)

    validation_scores = tuple(
        calibrator.predict_proba(detector.predict_proba(run.features))
        for run in runs["validation"]
    )
    selected = grouped_event_threshold(
        runs["validation"],
        validation_scores,
        warning_horizon_seconds=horizon_seconds,
        max_false_alarms_per_minute=max_false_alarms_per_minute,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    feature_names = runs["train"][0].feature_names
    model_payload = {
        "schema_version": 1,
        "method": "P-LOG",
        "feature_names": list(feature_names),
        "weights": detector.weights.tolist() if detector.weights is not None else None,
        "mean": detector.mean.tolist() if detector.mean is not None else None,
        "scale": detector.scale.tolist() if detector.scale is not None else None,
        "learning_rate": detector.learning_rate,
        "iterations": detector.iterations,
        "l2": detector.l2,
        "fit_split": "train",
    }
    calibration_payload = {
        "schema_version": 1,
        **calibrator.to_dict(),
        "fit_split": "calibration",
        "fit_rows": int(len(calibration_y)),
        "positive_rows": int(np.sum(calibration_y)),
        "negative_rows": int(np.sum(~calibration_y)),
    }
    threshold_payload = {
        "schema_version": 1,
        "method": "validation_grouped_event_threshold",
        "selected_on": "validation",
        "horizon_seconds": horizon_seconds,
        "max_false_alarms_per_minute": max_false_alarms_per_minute,
        "threshold": selected.threshold,
        "event_count": selected.event_count,
        "detected_events": selected.detected_events,
        "false_alerts": selected.false_alerts,
        "event_recall": selected.event_recall,
        "event_precision": selected.event_precision,
        "false_alarms_per_minute": selected.false_alarms_per_minute,
        "median_lead_time_seconds": selected.median_lead_time_seconds,
    }
    _save_json(destination / "model.json", model_payload)
    _save_json(destination / "calibration.json", calibration_payload)
    _save_json(destination / "threshold.json", threshold_payload)

    split_file = Path(split_path)
    per_partition: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    for partition in DEVELOPMENT_SPLITS:
        partition_runs = runs[partition]
        pooled_x, pooled_y = _pooled_eligible(partition_runs)
        per_partition[partition] = {
            "sequences": [run.sequence for run in partition_runs],
            "sequence_count": len(partition_runs),
            "eligible_rows": int(len(pooled_y)),
            "positive_rows": int(np.sum(pooled_y)),
            "negative_rows": int(np.sum(~pooled_y)),
            "failure_events": int(sum(len(run.events.onsets_ns) for run in partition_runs)),
            "input_evidence_levels": sorted({run.evidence_level for run in partition_runs}),
        }
        for run in partition_runs:
            input_hashes[f"{partition}:{run.dataset}/{run.sequence}"] = _run_manifest_hash(run)

    manifest = {
        "schema_version": "SHIELD_VIO_PUBLIC_DEVELOPMENT_BUNDLE_V1",
        "evidence_level": "PUBLIC_DATASET_DEVELOPMENT",
        "confirmatory": False,
        "test_partition_loaded": False,
        "claim_boundary": (
            "Real public train/calibration/validation orchestration only. Test and shifted-test "
            "artifacts are intentionally not loaded; this bundle cannot support H1-H5 test claims."
        ),
        "dataset": dataset,
        "estimator": estimator,
        "horizon_seconds": horizon_seconds,
        "feature_schema": list(feature_names),
        "split_config": str(split_file),
        "split_config_sha256": _sha256(split_file),
        "partitions": per_partition,
        "input_manifest_sha256": input_hashes,
        "artifacts": ["model.json", "calibration.json", "threshold.json"],
    }
    _save_json(destination / "development_manifest.json", manifest)

    seal_digest = hashlib.sha256()
    for name in ("model.json", "calibration.json", "threshold.json", "development_manifest.json"):
        seal_digest.update(name.encode("utf-8"))
        seal_digest.update((destination / name).read_bytes())
    seal = {
        "schema_version": "SHIELD_VIO_DEVELOPMENT_SEAL_V1",
        "sha256": seal_digest.hexdigest(),
        "sealed_files": [
            "model.json",
            "calibration.json",
            "threshold.json",
            "development_manifest.json",
        ],
        "test_partition_loaded": False,
    }
    _save_json(destination / "development_seal.json", seal)
    return {**manifest, "development_seal_sha256": seal["sha256"]}
