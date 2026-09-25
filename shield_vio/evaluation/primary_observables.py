"""Offline construction and availability audit for primary paper failure observables."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shield_vio.core.math import quat_to_rot
from shield_vio.datasets.euroc import read_imu_samples
from shield_vio.evaluation.failure_definition import FailureDefinition
from shield_vio.evaluation.trajectory_metrics import align_positions_se3


PRIMARY_CRITERIA = (
    "position_error_m",
    "orientation_error_deg",
    "translation_rpe_1s_m",
    "rotation_rpe_1s_deg",
    "invalid_pose_or_covariance",
    "output_starvation",
    "terminal_tracking_loss",
    "visual_update_starvation_while_motion",
    "estimator_reset_or_unrecovered_relocalization",
)


@dataclass(frozen=True)
class PrimaryObservableTable:
    timestamps_ns: np.ndarray
    values: dict[str, np.ndarray]
    observable: dict[str, np.ndarray]
    applicable: dict[str, np.ndarray]
    sources: dict[str, str]

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64)
        if timestamps.ndim != 1 or len(timestamps) < 3 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("primary observable timestamps must be strictly increasing")
        if set(self.values) != set(PRIMARY_CRITERIA):
            raise ValueError("primary observable value keys do not match frozen criteria")
        if set(self.observable) != set(PRIMARY_CRITERIA):
            raise ValueError("primary observable masks do not match frozen criteria")
        if set(self.applicable) != set(PRIMARY_CRITERIA):
            raise ValueError("primary observable applicability masks do not match frozen criteria")
        if set(self.sources) != set(PRIMARY_CRITERIA):
            raise ValueError("primary observable sources do not match frozen criteria")
        for name in PRIMARY_CRITERIA:
            if np.asarray(self.values[name]).shape != timestamps.shape:
                raise ValueError(f"observable value shape mismatch: {name}")
            if np.asarray(self.observable[name], dtype=bool).shape != timestamps.shape:
                raise ValueError(f"observable mask shape mismatch: {name}")
            if np.asarray(self.applicable[name], dtype=bool).shape != timestamps.shape:
                raise ValueError(f"applicability mask shape mismatch: {name}")
            if np.any(
                np.asarray(self.observable[name], dtype=bool)
                & ~np.asarray(self.applicable[name], dtype=bool)
            ):
                raise ValueError(f"criterion cannot be observable where it is not applicable: {name}")


def _dict_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"artifact contains no rows: {path}")
    return rows


def _ground_truth_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    timestamps: list[int] = []
    positions: list[list[float]] = []
    quaternions: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw in stream:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = next(csv.reader([stripped]))
            if len(row) < 8:
                raise ValueError("EuRoC ground truth requires timestamp, position, and quaternion")
            timestamps.append(int(row[0]))
            positions.append([float(row[1]), float(row[2]), float(row[3])])
            # EuRoC ASL ground truth stores quaternion as qw,qx,qy,qz after position.
            quaternions.append([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
    ts = np.asarray(timestamps, dtype=np.int64)
    pos = np.asarray(positions, dtype=float)
    quat = np.asarray(quaternions, dtype=float)
    if len(ts) < 3 or np.any(np.diff(ts) <= 0):
        raise ValueError("ground-truth timestamps must be strictly increasing")
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
        raise ValueError("ground truth contains non-finite values")
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("ground truth contains zero quaternion")
    quat = quat / norms[:, None]
    return ts, pos, quat


def _nearest_indices(
    query_ns: np.ndarray,
    reference_ns: np.ndarray,
    *,
    max_gap_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive")
    right = np.searchsorted(reference_ns, query_ns, side="left")
    right = np.clip(right, 0, len(reference_ns) - 1)
    left = np.clip(right - 1, 0, len(reference_ns) - 1)
    right_gap = np.abs(reference_ns[right] - query_ns)
    left_gap = np.abs(query_ns - reference_ns[left])
    choose_right = right_gap < left_gap
    selected = np.where(choose_right, right, left)
    gap = np.minimum(right_gap, left_gap)
    inside = (query_ns >= reference_ns[0]) & (query_ns <= reference_ns[-1])
    observable = inside & (gap <= int(round(max_gap_seconds * 1e9)))
    return selected.astype(int), observable


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    cosine = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _estimate_rotations(quaternions_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.zeros((len(quaternions_wxyz), 3, 3), dtype=float)
    valid = np.ones(len(quaternions_wxyz), dtype=bool)
    for index, quaternion in enumerate(quaternions_wxyz):
        if not np.all(np.isfinite(quaternion)) or np.linalg.norm(quaternion) <= np.finfo(float).eps:
            valid[index] = False
            rotations[index] = np.eye(3)
        else:
            rotations[index] = quat_to_rot(quaternion)
    return rotations, valid


def _one_second_predecessors(
    timestamps_ns: np.ndarray,
    *,
    interval_seconds: float,
    tolerance_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if interval_seconds <= 0 or tolerance_seconds <= 0:
        raise ValueError("RPE interval and tolerance must be positive")
    target = timestamps_ns - int(round(interval_seconds * 1e9))
    indices = np.searchsorted(timestamps_ns, target, side="left")
    indices = np.clip(indices, 0, len(timestamps_ns) - 1)
    left = np.clip(indices - 1, 0, len(timestamps_ns) - 1)
    right_gap = np.abs(timestamps_ns[indices] - target)
    left_gap = np.abs(timestamps_ns[left] - target)
    selected = np.where(right_gap < left_gap, indices, left)
    valid = (
        (timestamps_ns >= timestamps_ns[0] + int(round(interval_seconds * 1e9)))
        & (np.abs(timestamps_ns[selected] - target) <= int(round(tolerance_seconds * 1e9)))
        & (selected < np.arange(len(timestamps_ns)))
    )
    return selected.astype(int), valid


def _criterion(definition: FailureDefinition | None, name: str):
    if definition is None:
        return None
    matches = [criterion for criterion in definition.criteria if criterion.name == name]
    if len(matches) != 1:
        raise ValueError(f"failure definition must declare criterion exactly once: {name}")
    return matches[0]


def _optional_csv_rows(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if not path.is_file():
        return [], ()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        return list(reader), fieldnames


def _visual_starvation_observable(
    run: Path,
    sequence: Path,
    timestamps: np.ndarray,
    definition: FailureDefinition | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    values = np.full(len(timestamps), np.nan, dtype=float)
    observable = np.zeros(len(timestamps), dtype=bool)
    applicable = np.ones(len(timestamps), dtype=bool)
    criterion = _criterion(definition, "visual_update_starvation_while_motion")

    manifest_path = run / "experiment_manifest.json"
    if not manifest_path.is_file():
        return values, observable, applicable, "NOT_OBSERVABLE: estimator manifest missing"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provider = manifest.get("visual_provider")
    if provider in {None, "", "none"}:
        if (
            criterion is not None
            and criterion.applicability == "visual_update_stream"
            and criterion.explicitly_unsupported_policy == "not_applicable"
        ):
            applicable[:] = False
            return (
                values,
                observable,
                applicable,
                "NOT_APPLICABLE: run declares no visual-update provider",
            )
        return values, observable, applicable, "NOT_OBSERVABLE: visual-update provider unknown"

    if definition is None or definition.imu_motion_gate is None:
        return values, observable, applicable, "NOT_OBSERVABLE: versioned IMU motion gate missing"
    gate = definition.imu_motion_gate

    update_rows, update_fields = _optional_csv_rows(run / "visual_updates.csv")
    if not update_fields or "frame_timestamp_ns" not in update_fields:
        return values, observable, applicable, "NOT_OBSERVABLE: visual update log missing"
    accepted_ns = np.asarray(
        [int(row["frame_timestamp_ns"]) for row in update_rows], dtype=np.int64
    )
    if len(accepted_ns) and np.any(np.diff(accepted_ns) <= 0):
        raise ValueError("visual update timestamps must be strictly increasing")

    imu = read_imu_samples(sequence)
    imu_ns = np.asarray([sample.timestamp_ns for sample in imu], dtype=np.int64)
    gyro_norm = np.asarray(
        [np.linalg.norm(sample.angular_velocity_rad_s) for sample in imu], dtype=float
    )
    accel_deviation = np.asarray(
        [
            abs(np.linalg.norm(sample.linear_acceleration_m_s2) - gate.gravity_m_s2)
            for sample in imu
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(gyro_norm)) or np.any(~np.isfinite(accel_deviation)):
        raise ValueError("IMU motion-gate inputs must be finite")

    window_ns = int(round(gate.window_seconds * 1e9))
    left = np.searchsorted(imu_ns, timestamps - window_ns, side="right")
    right = np.searchsorted(imu_ns, timestamps, side="right")
    motion = np.zeros(len(timestamps), dtype=bool)
    motion_observable = np.zeros(len(timestamps), dtype=bool)
    for index, (start, stop) in enumerate(zip(left, right, strict=True)):
        if stop <= start:
            continue
        gyro_rms = float(np.sqrt(np.mean(np.square(gyro_norm[start:stop]))))
        accel_rms = float(np.sqrt(np.mean(np.square(accel_deviation[start:stop]))))
        motion[index] = (
            gyro_rms > gate.gyroscope_rms_threshold_rad_s
            or accel_rms > gate.accelerometer_norm_deviation_rms_threshold_m_s2
        )
        motion_observable[index] = True

    last_update_ns = int(timestamps[0])
    update_index = 0
    for index, timestamp in enumerate(timestamps):
        while update_index < len(accepted_ns) and accepted_ns[update_index] <= timestamp:
            last_update_ns = int(accepted_ns[update_index])
            update_index += 1
        if not motion_observable[index]:
            continue
        observable[index] = True
        if motion[index]:
            values[index] = max(0.0, float(timestamp - last_update_ns) * 1e-9)
        else:
            values[index] = 0.0

    source = (
        f"visual_updates.csv + trailing {gate.window_seconds:g}s IMU RMS motion gate "
        f"({gate.schema_version}; gyro>{gate.gyroscope_rms_threshold_rad_s:g} rad/s OR "
        f"|a|-g RMS>{gate.accelerometer_norm_deviation_rms_threshold_m_s2:g} m/s^2)"
    )
    return values, observable, applicable, source


def build_primary_observables(
    run_dir: str | Path,
    sequence_root: str | Path,
    *,
    max_ground_truth_gap_seconds: float = 0.02,
    rpe_interval_seconds: float = 1.0,
    rpe_pair_tolerance_seconds: float = 0.075,
    covariance_psd_tolerance: float = 1e-9,
    covariance_symmetry_tolerance: float = 1e-9,
    failure_definition: FailureDefinition | None = None,
) -> PrimaryObservableTable:
    """Build all currently defensible primary failure observations.

    Unsupported criteria are represented with an all-false observable mask rather
    than being silently treated as healthy.
    """

    run = Path(run_dir)
    sequence = Path(sequence_root)
    trajectory = _dict_rows(run / "trajectory.csv")
    health = _dict_rows(run / "health.csv")
    if len(trajectory) != len(health):
        raise ValueError("trajectory and health rows must have identical length")

    timestamps = np.asarray([int(row["frame_timestamp_ns"]) for row in trajectory], dtype=np.int64)
    health_timestamps = np.asarray([int(row["frame_timestamp_ns"]) for row in health], dtype=np.int64)
    if not np.array_equal(timestamps, health_timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("trajectory and health timestamps must align exactly and increase")

    estimated_positions = np.asarray(
        [[float(row["px"]), float(row["py"]), float(row["pz"])] for row in trajectory],
        dtype=float,
    )
    estimated_quaternions = np.asarray(
        [[float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])] for row in trajectory],
        dtype=float,
    )
    state_timestamps = np.asarray([int(row["state_timestamp_ns"]) for row in trajectory], dtype=np.int64)
    estimated_rotations, orientation_valid = _estimate_rotations(estimated_quaternions)
    pose_finite = np.all(np.isfinite(estimated_positions), axis=1) & orientation_valid

    gt_ts, gt_positions, gt_quaternions = _ground_truth_rows(
        sequence / "mav0/state_groundtruth_estimate0/data.csv"
    )
    gt_indices, gt_observable = _nearest_indices(
        timestamps, gt_ts, max_gap_seconds=max_ground_truth_gap_seconds
    )
    gt_reference_positions = gt_positions[gt_indices]
    gt_reference_rotations, gt_rotation_valid = _estimate_rotations(gt_quaternions[gt_indices])
    gt_observable &= gt_rotation_valid

    alignment_mask = gt_observable & pose_finite
    if int(np.sum(alignment_mask)) < 3:
        raise ValueError("fewer than three finite estimator poses have nearby ground truth")
    _aligned_subset, alignment_rotation, alignment_translation = align_positions_se3(
        estimated_positions[alignment_mask],
        gt_reference_positions[alignment_mask],
    )
    aligned_positions = estimated_positions @ alignment_rotation.T + alignment_translation
    aligned_rotations = np.einsum("ij,njk->nik", alignment_rotation, estimated_rotations)

    position_error = np.full(len(timestamps), np.nan, dtype=float)
    position_error[alignment_mask] = np.linalg.norm(
        aligned_positions[alignment_mask] - gt_reference_positions[alignment_mask], axis=1
    )
    orientation_error = np.full(len(timestamps), np.nan, dtype=float)
    for index in np.flatnonzero(alignment_mask):
        orientation_error[index] = _rotation_angle_deg(
            gt_reference_rotations[index].T @ aligned_rotations[index]
        )

    predecessor, pair_valid = _one_second_predecessors(
        timestamps,
        interval_seconds=rpe_interval_seconds,
        tolerance_seconds=rpe_pair_tolerance_seconds,
    )
    rpe_observable = pair_valid & alignment_mask & alignment_mask[predecessor]
    translation_rpe = np.full(len(timestamps), np.nan, dtype=float)
    rotation_rpe = np.full(len(timestamps), np.nan, dtype=float)
    for index in np.flatnonzero(rpe_observable):
        previous = int(predecessor[index])
        estimated_delta = aligned_positions[index] - aligned_positions[previous]
        reference_delta = gt_reference_positions[index] - gt_reference_positions[previous]
        translation_rpe[index] = float(np.linalg.norm(estimated_delta - reference_delta))
        estimated_relative = aligned_rotations[previous].T @ aligned_rotations[index]
        reference_relative = (
            gt_reference_rotations[previous].T @ gt_reference_rotations[index]
        )
        rotation_rpe[index] = _rotation_angle_deg(reference_relative.T @ estimated_relative)

    covariance_columns = (
        "covariance_trace",
        "covariance_condition_number",
        "covariance_min_eigenvalue",
        "covariance_symmetry_error",
    )
    covariance_observable = all(column in health[0] for column in covariance_columns)
    invalid_pose_or_covariance = ~pose_finite
    invalid_observable = np.full(len(timestamps), covariance_observable, dtype=bool)
    if covariance_observable:
        for index, row in enumerate(health):
            texts = [row.get(column, "") for column in covariance_columns]
            if any(text == "" for text in texts):
                invalid_observable[index] = False
                continue
            trace, condition, minimum_eigenvalue, symmetry_error = (
                float(text) for text in texts
            )
            covariance_invalid = (
                not all(
                    np.isfinite(value)
                    for value in (trace, condition, minimum_eigenvalue, symmetry_error)
                )
                or trace < 0.0
                or condition < 1.0
                or minimum_eigenvalue < -covariance_psd_tolerance
                or symmetry_error > covariance_symmetry_tolerance
            )
            invalid_pose_or_covariance[index] |= covariance_invalid

    state_age_ns = timestamps - state_timestamps
    output_observable = state_age_ns >= 0
    output_starvation = np.full(len(timestamps), np.nan, dtype=float)
    output_starvation[output_observable] = state_age_ns[output_observable].astype(float) * 1e-9

    tracking_values = np.zeros(len(timestamps), dtype=bool)
    tracking_observable = np.zeros(len(timestamps), dtype=bool)
    tracking_applicable = np.ones(len(timestamps), dtype=bool)
    lost_states = {"lost", "terminal", "failed", "tracking_lost"}
    tracking_criterion = _criterion(failure_definition, "terminal_tracking_loss")
    support_text = [
        str(row.get("terminal_tracking_loss_observable", "")).strip() for row in health
    ]
    if all(text in {"0", "1"} for text in support_text):
        support = np.asarray([bool(int(text)) for text in support_text], dtype=bool)
        if np.any(support) and not np.all(support):
            raise ValueError("terminal tracking-loss capability must be constant within a run")
        if (
            not np.any(support)
            and tracking_criterion is not None
            and tracking_criterion.applicability == "backend_declared"
            and tracking_criterion.explicitly_unsupported_policy == "not_applicable"
        ):
            tracking_applicable[:] = False
        else:
            tracking_observable = support.copy()
    for index, row in enumerate(health):
        if not tracking_applicable[index] or not tracking_observable[index]:
            continue
        status = str(row.get("tracking_status", "")).strip().lower()
        if not status:
            tracking_observable[index] = False
            continue
        tracking_values[index] = status in lost_states

    reset_relocalization = np.zeros(len(timestamps), dtype=bool)
    reset_observable = np.ones(len(timestamps), dtype=bool)
    for index, row in enumerate(health):
        reset = row.get("reset_event", "")
        relocalization = row.get("relocalization_event", "")
        if reset == "" or relocalization == "":
            reset_observable[index] = False
        else:
            reset_relocalization[index] = bool(int(reset)) or bool(int(relocalization))

    (
        visual_starvation,
        visual_starvation_observable,
        visual_starvation_applicable,
        visual_starvation_source,
    ) = _visual_starvation_observable(
        run,
        sequence,
        timestamps,
        failure_definition,
    )

    values: dict[str, np.ndarray] = {
        "position_error_m": position_error,
        "orientation_error_deg": orientation_error,
        "translation_rpe_1s_m": translation_rpe,
        "rotation_rpe_1s_deg": rotation_rpe,
        "invalid_pose_or_covariance": invalid_pose_or_covariance,
        "output_starvation": output_starvation,
        "terminal_tracking_loss": tracking_values,
        "visual_update_starvation_while_motion": visual_starvation,
        "estimator_reset_or_unrecovered_relocalization": reset_relocalization,
    }
    observable: dict[str, np.ndarray] = {
        "position_error_m": alignment_mask.copy(),
        "orientation_error_deg": alignment_mask.copy(),
        "translation_rpe_1s_m": rpe_observable.copy(),
        "rotation_rpe_1s_deg": rpe_observable.copy(),
        "invalid_pose_or_covariance": invalid_observable,
        "output_starvation": output_observable,
        "terminal_tracking_loss": tracking_observable,
        "visual_update_starvation_while_motion": visual_starvation_observable,
        "estimator_reset_or_unrecovered_relocalization": reset_observable,
    }
    always_applicable = np.ones(len(timestamps), dtype=bool)
    applicable: dict[str, np.ndarray] = {
        "position_error_m": always_applicable.copy(),
        "orientation_error_deg": always_applicable.copy(),
        "translation_rpe_1s_m": always_applicable.copy(),
        "rotation_rpe_1s_deg": always_applicable.copy(),
        "invalid_pose_or_covariance": always_applicable.copy(),
        "output_starvation": always_applicable.copy(),
        "terminal_tracking_loss": tracking_applicable,
        "visual_update_starvation_while_motion": visual_starvation_applicable,
        "estimator_reset_or_unrecovered_relocalization": always_applicable.copy(),
    }
    sources = {
        "position_error_m": "trajectory.csv + EuRoC ground truth + global SE(3) alignment",
        "orientation_error_deg": "trajectory.csv + EuRoC ground truth + alignment rotation",
        "translation_rpe_1s_m": "timestamp-paired aligned trajectory + EuRoC ground truth",
        "rotation_rpe_1s_deg": "timestamp-paired estimator and ground-truth attitudes",
        "invalid_pose_or_covariance": "trajectory.csv + health covariance validity diagnostics",
        "output_starvation": "frame_timestamp_ns - state_timestamp_ns",
        "terminal_tracking_loss": (
            "health.csv tracking_status gated by terminal_tracking_loss_observable"
        ),
        "visual_update_starvation_while_motion": visual_starvation_source,
        "estimator_reset_or_unrecovered_relocalization": (
            "health.csv reset_event OR relocalization_event"
        ),
    }
    return PrimaryObservableTable(timestamps, values, observable, applicable, sources)


def write_primary_observable_artifacts(
    table: PrimaryObservableTable,
    output_dir: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "primary_observables.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        columns = ["timestamp_ns"]
        for name in PRIMARY_CRITERIA:
            columns.extend([name, f"{name}__observable", f"{name}__applicable"])
        writer.writerow(columns)
        for index, timestamp in enumerate(table.timestamps_ns):
            row: list[object] = [int(timestamp)]
            for name in PRIMARY_CRITERIA:
                value = table.values[name][index]
                if isinstance(value, (np.bool_, bool)):
                    serialized: object = int(bool(value))
                else:
                    numeric = float(value)
                    serialized = "" if not np.isfinite(numeric) else numeric
                row.extend(
                    [
                        serialized,
                        int(table.observable[name][index]),
                        int(table.applicable[name][index]),
                    ]
                )
            writer.writerow(row)

    criteria: dict[str, Any] = {}
    blocking: list[str] = []
    for name in PRIMARY_CRITERIA:
        observable = np.asarray(table.observable[name], dtype=bool)
        applicable = np.asarray(table.applicable[name], dtype=bool)
        applicable_count = int(np.sum(applicable))
        observable_count = int(np.sum(observable & applicable))
        if applicable_count == 0:
            status = "NOT_APPLICABLE"
        elif observable_count == 0:
            status = "NOT_OBSERVABLE"
            blocking.append(name)
        else:
            status = "OBSERVABLE"
        criteria[name] = {
            "observable_samples": observable_count,
            "applicable_samples": applicable_count,
            "total_samples": len(observable),
            "observable_fraction_of_applicable": (
                observable_count / applicable_count if applicable_count else None
            ),
            "source": table.sources[name],
            "status": status,
        }
    report = {
        "schema_version": "SHIELD_VIO_PRIMARY_OBSERVABLE_AUDIT_V2",
        "status": "BLOCKED" if blocking else "READY_FOR_PRIMARY_LABEL_BUILD",
        "sample_count": len(table.timestamps_ns),
        "criteria": criteria,
        "blocking_criteria": blocking,
        "claim_boundary": (
            "Availability audit only. Primary SHIELD_VIO_FAILURE_V2 events may be built only "
            "when every enabled criterion is observable where applicable or is explicitly "
            "NOT_APPLICABLE under the frozen backend-capability policy."
        ),
    }
    (destination / "primary_observable_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
