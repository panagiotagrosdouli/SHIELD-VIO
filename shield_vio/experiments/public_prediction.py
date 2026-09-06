"""Map public estimator artifacts into the Phase B canonical prediction pipeline."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from shield_vio.evaluation.failure_definition import build_failure_events_and_targets, evaluate_failure_criteria, load_failure_definition
from shield_vio.health.dataset import PredictionGroups, build_prediction_dataset
from shield_vio.health.schema import EstimatorHealth, EvaluationDiagnostics, HealthSample, HealthValue

ALLOWED_EVIDENCE_LEVELS = {"FIXTURE_INTEGRATION_TEST", "PUBLIC_DATASET_SMOKE"}


def _dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _ground_truth_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    positions: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(line for line in stream if not line.lstrip().startswith("#"))
        for row in reader:
            if not row:
                continue
            try:
                timestamps.append(int(row[0].strip()))
                positions.append([float(row[1]), float(row[2]), float(row[3])])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"invalid ground-truth row in {path}: {row}") from exc
    ts = np.asarray(timestamps, dtype=np.int64)
    xyz = np.asarray(positions, dtype=float)
    if len(ts) < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError("ground-truth timestamps must be strictly increasing")
    return ts, xyz


def _nearest_position_error(timestamps: np.ndarray, estimate: np.ndarray, gt_ts: np.ndarray, gt_xyz: np.ndarray, max_gap_ns: int) -> np.ndarray:
    result = np.full(len(timestamps), np.nan, dtype=float)
    for index, timestamp in enumerate(timestamps):
        right = int(np.searchsorted(gt_ts, timestamp))
        candidates = [candidate for candidate in (right - 1, right) if 0 <= candidate < len(gt_ts)]
        if not candidates:
            continue
        best = min(candidates, key=lambda candidate: abs(int(gt_ts[candidate]) - int(timestamp)))
        if abs(int(gt_ts[best]) - int(timestamp)) <= max_gap_ns:
            result[index] = float(np.linalg.norm(estimate[index] - gt_xyz[best]))
    return result


def build_public_prediction_artifacts(run_dir: str | Path, ground_truth_csv: str | Path, output_dir: str | Path, *, dataset: str, sequence: str, estimator: str, failure_config: str | Path, seed: int = 0, max_ground_truth_gap_seconds: float = 0.02, evidence_level: str = "FIXTURE_INTEGRATION_TEST") -> dict[str, object]:
    if evidence_level not in ALLOWED_EVIDENCE_LEVELS:
        raise ValueError(f"invalid evidence level: {evidence_level}")
    source = Path(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    health_rows = _dict_rows(source / "health.csv")
    trajectory_rows = _dict_rows(source / "trajectory.csv")
    if len(health_rows) != len(trajectory_rows) or len(health_rows) < 2:
        raise ValueError("health and trajectory artifacts must align and contain at least two rows")
    timestamps = np.asarray([int(row["frame_timestamp_ns"]) for row in health_rows], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("public run timestamps must be strictly increasing")
    estimate = np.asarray([[float(row["px"]), float(row["py"]), float(row["pz"])] for row in trajectory_rows])
    gt_ts, gt_xyz = _ground_truth_positions(Path(ground_truth_csv))
    position_error = _nearest_position_error(timestamps, estimate, gt_ts, gt_xyz, int(round(max_ground_truth_gap_seconds * 1e9)))
    samples: list[HealthSample] = []
    for timestamp, health, error in zip(timestamps, health_rows, position_error):
        nis_text = health.get("innovation_nis", "")
        samples.append(HealthSample(timestamp_ns=int(timestamp), sequence_id=sequence, estimator_id=estimator,
            estimator=EstimatorHealth(nis=HealthValue.missing() if nis_text == "" else HealthValue(float(nis_text), True, int(timestamp)), covariance_trace=HealthValue(float(health["covariance_trace"]), True, int(timestamp)), covariance_condition_number=HealthValue(float(health["covariance_condition_number"]), True, int(timestamp))),
            evaluation=EvaluationDiagnostics(ground_truth_position_error_m=HealthValue.missing() if not np.isfinite(error) else HealthValue(float(error), True, int(timestamp)))))
    definition = load_failure_definition(failure_config)
    if tuple(criterion.name for criterion in definition.criteria) != ("position_error_m",):
        raise ValueError("public smoke builder only accepts the declared position-error sensitivity definition")
    exceeded = evaluate_failure_criteria({"position_error_m": position_error}, definition)
    events, targets = build_failure_events_and_targets(timestamps, exceeded, definition)
    dataset_obj = build_prediction_dataset(samples, horizon_targets={h: target.labels for h, target in targets.targets.items()}, eligible_masks={h: target.eligible_mask for h, target in targets.targets.items()}, groups=PredictionGroups(dataset, sequence, estimator, "clean", seed), metadata={"evidence_level": evidence_level, "confirmatory": False})
    values, names = dataset_obj.features()
    with (destination / "health_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_ns", "schema_version", "nis", "covariance_trace", "covariance_condition_number", "ground_truth_position_error_m__evaluation_only"])
        for sample in samples:
            writer.writerow([sample.timestamp_ns, sample.schema_version, sample.estimator.nis.value, sample.estimator.covariance_trace.value, sample.estimator.covariance_condition_number.value, sample.evaluation.ground_truth_position_error_m.value])
    with (destination / "prediction_dataset.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_ns", *names])
        writer.writerows([[int(timestamp), *row.tolist()] for timestamp, row in zip(timestamps, values)])
    with (destination / "prediction_targets.csv").open("w", encoding="utf-8", newline="") as stream:
        horizons = sorted(targets.targets)
        writer = csv.writer(stream)
        writer.writerow(["timestamp_ns", *[f"failure_within_{h}s" for h in horizons], *[f"eligible_{h}s" for h in horizons]])
        for index, timestamp in enumerate(timestamps):
            writer.writerow([int(timestamp), *[int(targets.targets[h].labels[index]) for h in horizons], *[int(targets.targets[h].eligible_mask[index]) for h in horizons]])
    with (destination / "failure_events.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_id", "onset_ns", "offset_ns"])
        writer.writerows(zip(events.event_ids, events.onsets_ns.tolist(), events.offsets_ns.tolist()))
    manifest = {"evidence_level": evidence_level, "confirmatory": False, "dataset": dataset, "sequence": sequence, "estimator": estimator, "health_schema_version": samples[0].schema_version, "failure_definition_id": definition.schema_version, "prediction_horizons_seconds": list(definition.horizons_seconds), "samples": len(samples), "failure_events": len(events.event_ids), "ground_truth_associated_samples": int(np.sum(np.isfinite(position_error))), "ground_truth_is_predictor": False}
    (destination / "canonical_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
