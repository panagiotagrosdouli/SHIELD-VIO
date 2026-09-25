"""Canonical deployable features paired with frozen primary V2 failure targets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from shield_vio.datasets.euroc import read_imu_samples
from shield_vio.evaluation.failure_definition import load_failure_definition
from shield_vio.evaluation.primary_failure_targets import build_primary_failure_targets
from shield_vio.evaluation.primary_observables import build_primary_observables
from shield_vio.health.dataset import PredictionGroups, build_prediction_dataset
from shield_vio.health.schema import (
    EstimatorHealth,
    HealthSample,
    HealthValue,
    InertialHealth,
    VisualHealth,
)


ALLOWED_EVIDENCE_LEVELS = {
    "FIXTURE_INTEGRATION_TEST",
    "PUBLIC_DATASET_SMOKE",
    "PUBLIC_DATASET_DEVELOPMENT",
}


def _dict_rows(path: Path, *, allow_missing: bool = False) -> list[dict[str, str]]:
    if not path.is_file():
        if allow_missing:
            return []
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows and not allow_missing:
        raise ValueError(f"artifact contains no rows: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _health_value(
    text: str | None,
    *,
    source_timestamp_ns: int,
    boolean: bool = False,
) -> HealthValue:
    if text is None or str(text).strip() == "":
        return HealthValue.missing()
    value: float | bool
    if boolean:
        value = bool(int(str(text)))
    else:
        value = float(str(text))
        if not np.isfinite(value):
            return HealthValue.missing()
    return HealthValue(value, True, source_timestamp_ns)


def _visual_rows_by_timestamp(path: Path) -> dict[int, dict[str, str]]:
    rows = _dict_rows(path, allow_missing=True)
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        timestamp = int(row["frame_timestamp_ns"])
        if timestamp in result:
            raise ValueError("duplicate visual update timestamp")
        result[timestamp] = row
    return result


def _latest_imu_indices(frame_timestamps: np.ndarray, imu_timestamps: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(imu_timestamps, frame_timestamps, side="right") - 1
    return indices.astype(int)


def _build_deployable_health_samples(
    run_dir: Path,
    sequence_root: Path,
    *,
    sequence: str,
    estimator: str,
) -> list[HealthSample]:
    health_rows = _dict_rows(run_dir / "health.csv")
    trajectory_rows = _dict_rows(run_dir / "trajectory.csv")
    if len(health_rows) != len(trajectory_rows) or len(health_rows) < 2:
        raise ValueError("health and trajectory artifacts must align and contain at least two rows")

    timestamps = np.asarray(
        [int(row["frame_timestamp_ns"]) for row in health_rows],
        dtype=np.int64,
    )
    trajectory_timestamps = np.asarray(
        [int(row["frame_timestamp_ns"]) for row in trajectory_rows],
        dtype=np.int64,
    )
    if not np.array_equal(timestamps, trajectory_timestamps):
        raise ValueError("health and trajectory timestamps must match exactly")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("public feature timestamps must be strictly increasing")

    visual_by_timestamp = _visual_rows_by_timestamp(run_dir / "visual_updates.csv")
    imu = read_imu_samples(sequence_root)
    imu_timestamps = np.asarray([sample.timestamp_ns for sample in imu], dtype=np.int64)
    if len(imu_timestamps) < 2 or np.any(np.diff(imu_timestamps) <= 0):
        raise ValueError("IMU timestamps must be strictly increasing")
    latest_imu = _latest_imu_indices(timestamps, imu_timestamps)

    last_visual_update_ns: int | None = None
    previous_bias: np.ndarray | None = None
    samples: list[HealthSample] = []

    for index, (timestamp, health, trajectory) in enumerate(
        zip(timestamps, health_rows, trajectory_rows, strict=True)
    ):
        state_timestamp = int(health["state_timestamp_ns"])
        trajectory_state_timestamp = int(trajectory["state_timestamp_ns"])
        if state_timestamp != trajectory_state_timestamp:
            raise ValueError("health and trajectory state timestamps must match")
        if state_timestamp > int(timestamp):
            raise ValueError("state timestamp cannot be in the future")

        visual_row = visual_by_timestamp.get(int(timestamp))
        frame_available = HealthValue(True, True, int(timestamp))
        if visual_row is None:
            visual_update_available = HealthValue(False, True, int(timestamp))
            tracked = HealthValue.missing()
            inlier_ratio = HealthValue.missing()
            accepted = HealthValue(False, True, int(timestamp))
            seconds_since_update = (
                HealthValue.missing()
                if last_visual_update_ns is None
                else HealthValue(
                    float(int(timestamp) - last_visual_update_ns) * 1e-9,
                    True,
                    int(timestamp),
                )
            )
        else:
            status = str(visual_row.get("status", "")).strip().lower()
            is_accepted = status == "accepted"
            visual_update_available = HealthValue(True, True, int(timestamp))
            tracked = _health_value(
                visual_row.get("tracked_features"),
                source_timestamp_ns=int(timestamp),
            )
            inlier_ratio = _health_value(
                visual_row.get("inlier_ratio"),
                source_timestamp_ns=int(timestamp),
            )
            accepted = HealthValue(is_accepted, True, int(timestamp))
            if is_accepted:
                last_visual_update_ns = int(timestamp)
                seconds_since_update = HealthValue(0.0, True, int(timestamp))
            else:
                seconds_since_update = (
                    HealthValue.missing()
                    if last_visual_update_ns is None
                    else HealthValue(
                        float(int(timestamp) - last_visual_update_ns) * 1e-9,
                        True,
                        int(timestamp),
                    )
                )

        imu_index = int(latest_imu[index])
        if imu_index < 0:
            accel_norm = HealthValue.missing()
            gyro_norm = HealthValue.missing()
            packet_available = HealthValue(False, True, int(timestamp))
            packet_interval = HealthValue.missing()
        else:
            imu_sample = imu[imu_index]
            source_timestamp = int(imu_sample.timestamp_ns)
            accel_norm = HealthValue(
                float(np.linalg.norm(imu_sample.linear_acceleration_m_s2)),
                True,
                source_timestamp,
            )
            gyro_norm = HealthValue(
                float(np.linalg.norm(imu_sample.angular_velocity_rad_s)),
                True,
                source_timestamp,
            )
            packet_available = HealthValue(True, True, source_timestamp)
            if imu_index == 0:
                packet_interval = HealthValue.missing()
            else:
                interval = float(
                    imu_timestamps[imu_index] - imu_timestamps[imu_index - 1]
                ) * 1e-9
                packet_interval = HealthValue(interval, True, source_timestamp)

        accel_bias = np.asarray(
            [float(trajectory["bax"]), float(trajectory["bay"]), float(trajectory["baz"])],
            dtype=float,
        )
        gyro_bias = np.asarray(
            [float(trajectory["bgx"]), float(trajectory["bgy"]), float(trajectory["bgz"])],
            dtype=float,
        )
        if np.any(~np.isfinite(accel_bias)) or np.any(~np.isfinite(gyro_bias)):
            raise ValueError("estimator bias state must be finite")
        bias_vector = np.r_[accel_bias, gyro_bias]
        bias_change = (
            HealthValue.missing()
            if previous_bias is None
            else HealthValue(
                float(np.linalg.norm(bias_vector - previous_bias)),
                True,
                state_timestamp,
            )
        )
        previous_bias = bias_vector

        reset = _health_value(
            health.get("reset_event"),
            source_timestamp_ns=state_timestamp,
            boolean=True,
        )
        relocalization = _health_value(
            health.get("relocalization_event"),
            source_timestamp_ns=state_timestamp,
            boolean=True,
        )

        samples.append(
            HealthSample(
                timestamp_ns=int(timestamp),
                sequence_id=sequence,
                estimator_id=estimator,
                visual=VisualHealth(
                    tracked_feature_count=tracked,
                    inlier_ratio=inlier_ratio,
                    frame_available=frame_available,
                    visual_update_available=visual_update_available,
                ),
                inertial=InertialHealth(
                    accel_norm_m_s2=accel_norm,
                    gyro_norm_rad_s=gyro_norm,
                    accel_bias_norm_m_s2=HealthValue(
                        float(np.linalg.norm(accel_bias)),
                        True,
                        state_timestamp,
                    ),
                    gyro_bias_norm_rad_s=HealthValue(
                        float(np.linalg.norm(gyro_bias)),
                        True,
                        state_timestamp,
                    ),
                    bias_change=bias_change,
                    packet_available=packet_available,
                    seconds_since_previous_packet=packet_interval,
                ),
                estimator=EstimatorHealth(
                    nis=_health_value(
                        health.get("innovation_nis"),
                        source_timestamp_ns=state_timestamp,
                    ),
                    covariance_trace=_health_value(
                        health.get("covariance_trace"),
                        source_timestamp_ns=state_timestamp,
                    ),
                    covariance_condition_number=_health_value(
                        health.get("covariance_condition_number"),
                        source_timestamp_ns=state_timestamp,
                    ),
                    visual_update_accepted=accepted,
                    seconds_since_last_successful_update=seconds_since_update,
                    reset=reset,
                    relocalization=relocalization,
                ),
            )
        )

    return samples


def build_public_v2_prediction_artifacts(
    run_dir: str | Path,
    sequence_root: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    sequence: str,
    estimator: str,
    failure_config: str | Path,
    evidence_level: str,
    seed: int = 0,
    degradation_condition: str = "clean",
) -> dict[str, Any]:
    """Build canonical deployable features with primary V2 labels."""

    if evidence_level not in ALLOWED_EVIDENCE_LEVELS:
        raise ValueError(f"invalid evidence level: {evidence_level}")

    source = Path(run_dir)
    sequence_path = Path(sequence_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    definition = load_failure_definition(failure_config)
    if definition.schema_version != "SHIELD_VIO_FAILURE_V2":
        raise ValueError("public V2 canonical builder requires SHIELD_VIO_FAILURE_V2")

    observable_table = build_primary_observables(
        source,
        sequence_path,
        failure_definition=definition,
    )
    primary = build_primary_failure_targets(observable_table, definition)
    samples = _build_deployable_health_samples(
        source,
        sequence_path,
        sequence=sequence,
        estimator=estimator,
    )
    sample_timestamps = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
    if not np.array_equal(sample_timestamps, primary.timestamps_ns):
        raise ValueError("deployable features and primary V2 targets must align exactly")

    prediction = build_prediction_dataset(
        samples,
        horizon_targets={
            horizon: target.labels for horizon, target in primary.targets.items()
        },
        eligible_masks={
            horizon: target.eligible_mask for horizon, target in primary.targets.items()
        },
        groups=PredictionGroups(
            dataset,
            sequence,
            estimator,
            degradation_condition,
            seed,
        ),
        metadata={
            "evidence_level": evidence_level,
            "confirmatory": False,
            "failure_definition": definition.schema_version,
        },
    )
    values, feature_names = prediction.features()

    dataset_path = destination / "prediction_dataset.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_ns", *feature_names])
        for timestamp, row in zip(sample_timestamps, values, strict=True):
            writer.writerow([int(timestamp), *row.tolist()])

    target_path = destination / "prediction_targets.csv"
    horizons = sorted(primary.targets)
    with target_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        header = ["timestamp_ns"]
        header.extend(f"failure_within_{h}s" for h in horizons)
        header.extend(f"eligible_{h}s" for h in horizons)
        writer.writerow(header)
        for index, timestamp in enumerate(sample_timestamps):
            writer.writerow(
                [
                    int(timestamp),
                    *[int(primary.targets[h].labels[index]) for h in horizons],
                    *[int(primary.targets[h].eligible_mask[index]) for h in horizons],
                ]
            )

    events_path = destination / "failure_events.csv"
    with events_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_id", "onset_ns", "offset_ns"])
        writer.writerows(
            zip(
                primary.event_ids,
                primary.onsets_ns.tolist(),
                primary.offsets_ns.tolist(),
                strict=True,
            )
        )

    source_hashes: dict[str, str] = {}
    for name in (
        "trajectory.csv",
        "health.csv",
        "visual_updates.csv",
        "experiment_manifest.json",
    ):
        path = source / name
        if path.is_file():
            source_hashes[name] = _sha256(path)

    feature_schema_sha256 = hashlib.sha256(
        "\n".join(feature_names).encode("utf-8")
    ).hexdigest()
    failure_config_path = Path(failure_config)
    manifest = {
        "schema_version": "SHIELD_VIO_PUBLIC_CANONICAL_V2",
        "evidence_level": evidence_level,
        "confirmatory": False,
        "dataset": dataset,
        "sequence": sequence,
        "estimator": estimator,
        "degradation_condition": degradation_condition,
        "seed": seed,
        "health_schema_version": samples[0].schema_version,
        "failure_definition_id": definition.schema_version,
        "failure_config_sha256": _sha256(failure_config_path),
        "prediction_horizons_seconds": horizons,
        "samples": len(samples),
        "complete_observability_samples": int(
            np.sum(primary.complete_observability_mask)
        ),
        "censored_observability_samples": int(
            np.sum(~primary.complete_observability_mask)
        ),
        "failure_events": len(primary.event_ids),
        "feature_count": len(feature_names),
        "feature_schema_sha256": feature_schema_sha256,
        "ground_truth_is_predictor": False,
        "source_run_artifact_sha256": source_hashes,
        "artifact_sha256": {
            dataset_path.name: _sha256(dataset_path),
            target_path.name: _sha256(target_path),
            events_path.name: _sha256(events_path),
        },
        "claim_boundary": (
            "Non-confirmatory canonical public-data development artifact. Ground truth is "
            "used only by offline V2 label construction and never enters deployable features."
        ),
    }
    (destination / "canonical_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
