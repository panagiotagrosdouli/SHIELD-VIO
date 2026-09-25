from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from shield_vio.datasets.splits import load_paper_split
from shield_vio.experiments.split_safe_development import (
    fit_development_bundle,
    load_canonical_prediction_run,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _write_split(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "protocol": "TEST_SPLIT",
        "train": [{"dataset": "euroc", "sequence": "train_seq"}],
        "calibration": [{"dataset": "euroc", "sequence": "cal_seq"}],
        "validation": [{"dataset": "euroc", "sequence": "val_seq"}],
        "test": [{"dataset": "euroc", "sequence": "test_seq"}],
        "shifted_test": [],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_run(
    root: Path,
    sequence: str,
    *,
    feature_name: str = "estimator.nis",
) -> Path:
    path = root / sequence
    path.mkdir(parents=True)
    timestamps = [index * 1_000_000_000 for index in range(8)]
    signal = [0.1, 0.15, 0.2, 2.0, 2.5, 3.0, 2.0, 0.2]
    _write_csv(
        path / "prediction_dataset.csv",
        ["timestamp_ns", feature_name, f"{feature_name}__missing"],
        [[timestamp, value, 0] for timestamp, value in zip(timestamps, signal, strict=True)],
    )
    labels = [0, 0, 0, 1, 1, 0, 0, 0]
    eligible = [1, 1, 1, 1, 1, 0, 0, 0]
    _write_csv(
        path / "prediction_targets.csv",
        ["timestamp_ns", "failure_within_2.0s", "eligible_2.0s"],
        [
            [timestamp, label, is_eligible]
            for timestamp, label, is_eligible in zip(
                timestamps, labels, eligible, strict=True
            )
        ],
    )
    _write_csv(
        path / "failure_events.csv",
        ["event_id", "onset_ns", "offset_ns"],
        [["event-1", 5_000_000_000, 6_000_000_000]],
    )
    canonical = {
        "evidence_level": "PUBLIC_DATASET_SMOKE",
        "confirmatory": False,
        "dataset": "euroc",
        "sequence": sequence,
        "estimator": "internal_eskf",
    }
    (path / "canonical_manifest.json").write_text(
        json.dumps(canonical, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        **canonical,
        "degradation_condition": "clean",
        "seed": 0,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return path


def test_fit_development_bundle_never_loads_test_partition(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.yaml"
    _write_split(split_path)
    artifacts = tmp_path / "artifacts"
    _write_run(artifacts, "train_seq")
    _write_run(artifacts, "cal_seq")
    _write_run(artifacts, "val_seq")

    # Deliberately broken test artifact: development fitting must never touch it.
    test_dir = artifacts / "test_seq"
    test_dir.mkdir(parents=True)
    (test_dir / "THIS_MUST_NOT_BE_READ.txt").write_text("sealed test\n", encoding="utf-8")

    output = tmp_path / "bundle"
    report = fit_development_bundle(
        artifacts,
        split_path,
        output,
        horizon_seconds=2.0,
        max_false_alarms_per_minute=0.2,
        logistic_iterations=200,
    )

    assert report["evidence_level"] == "PUBLIC_DATASET_DEVELOPMENT"
    assert not report["confirmatory"]
    assert report["test_partition_loaded"] is False
    assert report["partitions"]["train"]["sequences"] == ["train_seq"]
    assert report["partitions"]["calibration"]["sequences"] == ["cal_seq"]
    assert report["partitions"]["validation"]["sequences"] == ["val_seq"]
    serialized = json.loads((output / "development_manifest.json").read_text(encoding="utf-8"))
    assert "test_seq" not in json.dumps(serialized)
    seal = json.loads((output / "development_seal.json").read_text(encoding="utf-8"))
    assert len(seal["sha256"]) == 64
    assert seal["test_partition_loaded"] is False


def test_development_loader_rejects_test_partition(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.yaml"
    _write_split(split_path)
    artifacts = tmp_path / "artifacts"
    test_run = _write_run(artifacts, "test_seq")
    split = load_paper_split(split_path)

    with pytest.raises(ValueError, match="intentionally sealed"):
        load_canonical_prediction_run(
            test_run,
            paper_split=split,
            requested_split="test",
            horizon_seconds=2.0,
        )


def test_fit_development_bundle_rejects_feature_schema_drift(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.yaml"
    _write_split(split_path)
    artifacts = tmp_path / "artifacts"
    _write_run(artifacts, "train_seq")
    _write_run(artifacts, "cal_seq", feature_name="estimator.covariance_trace")
    _write_run(artifacts, "val_seq")

    with pytest.raises(ValueError, match="feature schema mismatch"):
        fit_development_bundle(
            artifacts,
            split_path,
            tmp_path / "bundle",
            horizon_seconds=2.0,
            logistic_iterations=100,
        )


def test_fit_development_bundle_rejects_sequence_in_wrong_split(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.yaml"
    _write_split(split_path)
    artifacts = tmp_path / "artifacts"
    train = _write_run(artifacts, "train_seq")
    manifest_path = train / "canonical_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sequence"] = "test_seq"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    _write_run(artifacts, "cal_seq")
    _write_run(artifacts, "val_seq")

    with pytest.raises(ValueError, match="sequence mismatch"):
        fit_development_bundle(
            artifacts,
            split_path,
            tmp_path / "bundle",
            horizon_seconds=2.0,
            logistic_iterations=100,
        )
