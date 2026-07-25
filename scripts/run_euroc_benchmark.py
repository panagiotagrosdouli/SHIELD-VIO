"""Run reproducible EuRoC sensor-health degradation experiments.

This runner consumes the standard EuRoC ``mav0`` layout, evaluates camera and
IMU health before and after configured degradations, and writes per-run metrics
plus a provenance manifest. It is intentionally estimator-independent: it
provides executable public-dataset evidence while the full VIO benchmark backend
is integrated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class CameraMetrics:
    frames: int
    brightness_mean: float
    contrast_mean: float
    blur_laplacian_mean: float
    feature_count_mean: float
    dropout_fraction: float


@dataclass(frozen=True)
class ImuMetrics:
    samples: int
    accel_norm_mean: float
    accel_norm_std: float
    gyro_norm_mean: float
    gyro_norm_std: float
    packet_gap_p95_ms: float
    dropout_fraction: float


def _read_csv(path: Path) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing EuRoC file: {path}")
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(line for line in handle if not line.startswith("#")):
            if row:
                rows.append(row)
    return rows


def _sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _degrade_image(
    image: np.ndarray, degradation: str, params: dict[str, Any], rng: np.random.Generator
) -> np.ndarray | None:
    if degradation == "none":
        return image
    if degradation == "darkness":
        return np.clip(image.astype(np.float32) * float(params["gain"]), 0, 255).astype(np.uint8)
    if degradation == "overexposure":
        return np.clip(image.astype(np.float32) * float(params["gain"]), 0, 255).astype(np.uint8)
    if degradation == "contrast_reduction":
        mean = float(np.mean(image))
        factor = float(params["factor"])
        return np.clip(mean + factor * (image.astype(np.float32) - mean), 0, 255).astype(np.uint8)
    if degradation == "additive_noise":
        noise = rng.normal(0.0, float(params["sigma"]), image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if degradation == "feature_dropout":
        if rng.random() < float(params["probability"]):
            return np.full_like(image, int(np.mean(image)))
        return image
    if degradation == "occlusion":
        output = image.copy()
        fraction = float(params["area_fraction"])
        height, width = output.shape[:2]
        box_w = max(1, int(width * np.sqrt(fraction)))
        box_h = max(1, int(height * np.sqrt(fraction)))
        x0 = int(rng.integers(0, max(1, width - box_w + 1)))
        y0 = int(rng.integers(0, max(1, height - box_h + 1)))
        output[y0 : y0 + box_h, x0 : x0 + box_w] = 0
        return output
    if degradation == "frame_dropout":
        return None if rng.random() < float(params["probability"]) else image
    raise ValueError(f"Unsupported visual degradation: {degradation}")


def _camera_metrics(
    sequence: Path,
    degradation: str,
    params: dict[str, Any],
    seed: int,
    max_frames: int,
) -> CameraMetrics:
    rows = _read_csv(sequence / "mav0/cam0/data.csv")
    rng = np.random.default_rng(seed)
    brightness: list[float] = []
    contrast: list[float] = []
    blur: list[float] = []
    features: list[float] = []
    dropped = 0

    selected = rows if max_frames <= 0 else rows[:max_frames]
    for row in selected:
        image_path = sequence / "mav0/cam0/data" / row[1].strip()
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = _degrade_image(image, degradation, params, rng)
        if image is None:
            dropped += 1
            continue
        brightness.append(float(np.mean(image)))
        contrast.append(float(np.std(image)))
        blur.append(float(cv2.Laplacian(image, cv2.CV_64F).var()))
        corners = cv2.goodFeaturesToTrack(image, 500, 0.01, 7)
        features.append(float(0 if corners is None else len(corners)))

    total = len(selected)
    return CameraMetrics(
        frames=total - dropped,
        brightness_mean=float(np.mean(brightness)) if brightness else 0.0,
        contrast_mean=float(np.mean(contrast)) if contrast else 0.0,
        blur_laplacian_mean=float(np.mean(blur)) if blur else 0.0,
        feature_count_mean=float(np.mean(features)) if features else 0.0,
        dropout_fraction=float(dropped / total) if total else 0.0,
    )


def _degrade_imu(
    values: np.ndarray,
    degradation: str,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(values), dtype=bool)
    output = values.copy()
    if degradation == "none":
        return output, keep
    if degradation in {"accelerometer_noise", "gyroscope_noise"}:
        columns = slice(3, 6) if degradation == "accelerometer_noise" else slice(0, 3)
        baseline = np.std(output[:, columns], axis=0) + 1e-9
        multiplier = max(0.0, float(params["multiplier"]) - 1.0)
        output[:, columns] += rng.normal(0.0, baseline * multiplier, (len(output), 3))
    elif degradation == "bias_drift":
        drift = np.linspace(0.0, float(params["multiplier"]) * 0.01, len(output))[:, None]
        output += drift
    elif degradation == "scale_factor_error":
        output *= 1.0 + float(params["fraction"])
    elif degradation == "saturation":
        fraction = float(params["clip_fraction"])
        limit = np.quantile(np.abs(output), fraction, axis=0)
        output = np.clip(output, -limit, limit)
    elif degradation == "axis_failure":
        count = int(round(len(output) * float(params["duty_cycle"])))
        output[:count, 0] = 0.0
    elif degradation == "packet_loss":
        keep = rng.random(len(output)) >= float(params["probability"])
    else:
        raise ValueError(f"Unsupported inertial degradation: {degradation}")
    return output[keep], keep


def _imu_metrics(
    sequence: Path, degradation: str, params: dict[str, Any], seed: int
) -> ImuMetrics:
    rows = _read_csv(sequence / "mav0/imu0/data.csv")
    timestamps = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    values = np.asarray([[float(value) for value in row[1:7]] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    degraded, keep = _degrade_imu(values, degradation, params, rng)
    kept_timestamps = timestamps[keep]
    gaps_ms = np.diff(kept_timestamps).astype(float) / 1e6
    gyro = np.linalg.norm(degraded[:, :3], axis=1) if len(degraded) else np.asarray([])
    accel = np.linalg.norm(degraded[:, 3:6], axis=1) if len(degraded) else np.asarray([])
    return ImuMetrics(
        samples=len(degraded),
        accel_norm_mean=float(np.mean(accel)) if len(accel) else 0.0,
        accel_norm_std=float(np.std(accel)) if len(accel) else 0.0,
        gyro_norm_mean=float(np.mean(gyro)) if len(gyro) else 0.0,
        gyro_norm_std=float(np.std(gyro)) if len(gyro) else 0.0,
        packet_gap_p95_ms=float(np.quantile(gaps_ms, 0.95)) if len(gaps_ms) else 0.0,
        dropout_fraction=float(1.0 - np.mean(keep)) if len(keep) else 0.0,
    )


def _jobs(config: dict[str, Any], split: str) -> Iterable[tuple[str, str, str, dict[str, Any], int]]:
    dataset = config["dataset"]
    sequences = dataset[f"{split}_sequences"]
    degradation = config["degradation"]
    seeds = degradation["seeds"]
    yield from ((seq, "none", "none", {}, seed) for seq in sequences for seed in seeds[:1])
    for family in ("visual", "inertial"):
        for name, levels in degradation[family].items():
            for severity in degradation["severity_levels"]:
                for seed in seeds:
                    yield name, family, severity, levels[severity], seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/euroc/benchmark_v1.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/euroc_benchmark"))
    parser.add_argument("--split", choices=("development", "calibration", "test"), default="development")
    parser.add_argument("--sequence", action="append", help="Restrict to one or more EuRoC sequence names")
    parser.add_argument("--max-frames", type=int, default=300, help="0 processes all camera frames")
    parser.add_argument("--quick", action="store_true", help="Baseline plus one medium degradation per family")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    allowed = set(config["dataset"][f"{args.split}_sequences"])
    requested = set(args.sequence or allowed)
    unknown = requested - allowed
    if unknown:
        raise ValueError(f"Sequences not declared in {args.split} split: {sorted(unknown)}")

    results: list[dict[str, Any]] = []
    for item in _jobs(config, args.split):
        sequence_name, family, severity, params, seed = item
        if family == "none":
            sequence_name = next(iter(requested)) if len(requested) == 1 else sequence_name
        if sequence_name not in requested and family == "none":
            continue
        if family != "none":
            # Apply every configured degradation to every requested sequence.
            degradation_name = sequence_name
            for actual_sequence in sorted(requested):
                if args.quick and not (severity == "medium" and seed == config["degradation"]["seeds"][0]):
                    continue
                sequence = args.dataset_root / actual_sequence
                camera = _camera_metrics(
                    sequence,
                    degradation_name if family == "visual" else "none",
                    params if family == "visual" else {},
                    seed,
                    args.max_frames,
                )
                imu = _imu_metrics(
                    sequence,
                    degradation_name if family == "inertial" else "none",
                    params if family == "inertial" else {},
                    seed,
                )
                results.append({
                    "sequence": actual_sequence,
                    "family": family,
                    "degradation": degradation_name,
                    "severity": severity,
                    "seed": seed,
                    "camera": asdict(camera),
                    "imu": asdict(imu),
                })
            continue

        sequence = args.dataset_root / sequence_name
        camera = _camera_metrics(sequence, "none", {}, seed, args.max_frames)
        imu = _imu_metrics(sequence, "none", {}, seed)
        results.append({
            "sequence": sequence_name,
            "family": "baseline",
            "degradation": "none",
            "severity": "none",
            "seed": seed,
            "camera": asdict(camera),
            "imu": asdict(imu),
        })

    data_files = [
        args.dataset_root / sequence / "mav0/cam0/data.csv" for sequence in requested
    ] + [args.dataset_root / sequence / "mav0/imu0/data.csv" for sequence in requested]
    manifest = {
        "protocol": config["protocol"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "sequences": sorted(requested),
        "dataset_index_sha256": _sha256(data_files),
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": str(args.config),
        "max_frames": args.max_frames,
        "quick": args.quick,
        "completed_runs": len(results),
        "status": "complete",
    }
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
