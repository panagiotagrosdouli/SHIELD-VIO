from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.validate_public_smoke import REQUIRED_ARTIFACTS, _sha256_file, validate_smoke


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _bundle(root: Path) -> Path:
    output = root / "smoke"
    output.mkdir(parents=True)
    estimator = root / "estimator"
    estimator.mkdir()
    (estimator / "experiment_manifest.json").write_text("{}\n", encoding="utf-8")

    _write_csv(
        output / "health_features.csv",
        ["timestamp_ns", "max_source_timestamp_ns", "covariance_trace"],
        [[100, 100, 0.1], [200, 190, 1.2]],
    )
    _write_csv(
        output / "predictions.csv",
        ["timestamp_ns", "future_failure", "eligible"],
        [[100, 0, 1], [200, 1, 1]],
    )
    _write_csv(
        output / "reliability_source.csv",
        ["mean_probability", "empirical_frequency", "sample_count"],
        [[0.2, 0.0, 1], [0.8, 1.0, 1]],
    )
    _write_csv(
        output / "prediction_timeline_source.csv",
        ["time_s", "failure_probability", "position_error_m", "failure_active"],
        [[0.0, 0.2, 0.1, 0], [0.1, 0.8, 2.0, 1]],
    )

    metrics = {}
    for method in (
        "covariance_trace",
        "feature_count",
        "innovation_nis",
        "logistic_raw",
        "logistic_platt",
    ):
        metrics[method] = {
            "auroc": 0.75,
            "auprc": 0.8,
            "discrimination_defined": True,
            "eligible_samples": 2,
            "positive_windows": 1,
            "negative_windows": 1,
        }
    metrics["logistic_raw"]["calibration"] = {"brier": 0.2}
    metrics["logistic_platt"]["calibration"] = {"brier": 0.1}
    (output / "metrics.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    (output / "model.json").write_text("{}\n", encoding="utf-8")
    (output / "calibration.json").write_text("{}\n", encoding="utf-8")
    for name in (
        "reliability_diagram.pdf",
        "reliability_diagram.svg",
        "prediction_timeline.pdf",
        "prediction_timeline.svg",
    ):
        (output / name).write_text(f"fixture {name}\n", encoding="utf-8")

    hashes = {name: _sha256_file(output / name) for name in REQUIRED_ARTIFACTS}
    manifest = {
        "schema_version": 1,
        "evidence_level": "PUBLIC_DATASET_SMOKE",
        "confirmatory": False,
        "status": "complete",
        "claim_boundary": "Pipeline evidence only; this does not confirm H1-H5.",
        "dataset": "EuRoC_MAV",
        "sequence": "MH_01_easy",
        "dataset_index_sha256": "0" * 64,
        "configuration_sha256": "1" * 64,
        "detector_training_domain": "ANALYTIC_SYNTHETIC_HEALTH_V1",
        "policy": "not_evaluated",
        "split_definition": {
            "train": "analytic_train",
            "calibration": "analytic_calibration",
            "validation": "analytic_validation",
            "test": "MH_01_easy",
        },
        "sample_counts": {
            "associated": 2,
            "eligible": 2,
            "positive_windows": 1,
            "negative_windows": 1,
            "failure_events": 1,
        },
        "discrimination_defined": True,
        "target_status": "TWO_CLASS",
        "artifact_sha256": hashes,
        "estimator_manifest": str(estimator / "experiment_manifest.json"),
        "git": {"commit_sha": "0123456789abcdef", "dirty_tree": False},
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output


def test_validate_public_smoke_accepts_complete_bundle(tmp_path: Path) -> None:
    output = _bundle(tmp_path)
    report = validate_smoke(output, expected_sequence="MH_01_easy")
    assert report["status"] == "pass"
    assert report["evidence_level"] == "PUBLIC_DATASET_SMOKE"
    assert not report["confirmatory"]
    assert report["causal_provenance_checked"]
    assert report["artifact_hashes_checked"]


def test_validate_public_smoke_rejects_future_feature_source(tmp_path: Path) -> None:
    output = _bundle(tmp_path)
    _write_csv(
        output / "health_features.csv",
        ["timestamp_ns", "max_source_timestamp_ns", "covariance_trace"],
        [[100, 101, 0.1], [200, 190, 1.2]],
    )
    manifest_path = output / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["health_features.csv"] = _sha256_file(
        output / "health_features.csv"
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="causal provenance violation"):
        validate_smoke(output, expected_sequence="MH_01_easy")


def test_validate_public_smoke_rejects_artifact_tampering(tmp_path: Path) -> None:
    output = _bundle(tmp_path)
    with (output / "metrics.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_smoke(output, expected_sequence="MH_01_easy")


def test_validate_public_smoke_accepts_single_class_negative_pipeline_evidence(
    tmp_path: Path,
) -> None:
    output = _bundle(tmp_path)
    _write_csv(
        output / "predictions.csv",
        ["timestamp_ns", "future_failure", "eligible"],
        [[100, 0, 1], [200, 0, 1]],
    )
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for payload in metrics.values():
        payload["auroc"] = None
        payload["auprc"] = None
        payload["discrimination_defined"] = False
        payload["positive_windows"] = 0
        payload["negative_windows"] = 2
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")

    manifest_path = output / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sample_counts"]["positive_windows"] = 0
    manifest["sample_counts"]["negative_windows"] = 2
    manifest["sample_counts"]["failure_events"] = 0
    manifest["discrimination_defined"] = False
    manifest["target_status"] = "SINGLE_CLASS_NEGATIVE"
    manifest["artifact_sha256"]["predictions.csv"] = _sha256_file(
        output / "predictions.csv"
    )
    manifest["artifact_sha256"]["metrics.json"] = _sha256_file(metrics_path)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    report = validate_smoke(output, expected_sequence="MH_01_easy")
    assert report["status"] == "pass"
    assert report["target_status"] == "SINGLE_CLASS_NEGATIVE"
    assert not report["discrimination_defined"]
    assert report["failure_events"] == 0
