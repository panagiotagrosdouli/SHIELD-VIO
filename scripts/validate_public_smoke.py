#!/usr/bin/env python3
"""Validate a completed PUBLIC_DATASET_SMOKE artifact bundle fail-closed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACTS = {
    "health_features.csv",
    "predictions.csv",
    "metrics.json",
    "model.json",
    "calibration.json",
    "reliability_source.csv",
    "reliability_diagram.pdf",
    "reliability_diagram.svg",
    "prediction_timeline_source.csv",
    "prediction_timeline.pdf",
    "prediction_timeline.svg",
}

REQUIRED_METHODS = {
    "covariance_trace",
    "feature_count",
    "innovation_nis",
    "logistic_raw",
    "logistic_platt",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def _require_sha256(value: object, field: str) -> None:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise ValueError(f"{field} is not a SHA-256 hex digest")


def _require_probability_metric(value: object, field: str) -> None:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")


def validate_smoke(
    output_dir: Path,
    *,
    expected_dataset: str = "EuRoC_MAV",
    expected_sequence: str | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "experiment_manifest.json"
    manifest = _load_json(manifest_path)

    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported smoke manifest schema_version")
    if manifest.get("evidence_level") != "PUBLIC_DATASET_SMOKE":
        raise ValueError("evidence_level must be PUBLIC_DATASET_SMOKE")
    if manifest.get("confirmatory") is not False:
        raise ValueError("public smoke artifact must be explicitly non-confirmatory")
    if manifest.get("status") != "complete":
        raise ValueError("public smoke artifact status must be complete")
    if manifest.get("dataset") != expected_dataset:
        raise ValueError(
            f"unexpected dataset: {manifest.get('dataset')!r}; expected {expected_dataset!r}"
        )

    sequence = str(manifest.get("sequence", "")).strip()
    if not sequence:
        raise ValueError("manifest sequence must be non-empty")
    if expected_sequence is not None and sequence != expected_sequence:
        raise ValueError(f"unexpected sequence: {sequence!r}; expected {expected_sequence!r}")

    claim_boundary = str(manifest.get("claim_boundary", "")).strip()
    if not claim_boundary:
        raise ValueError("claim_boundary must be explicit")
    if "does not confirm" not in claim_boundary.lower():
        raise ValueError("claim_boundary must state that the smoke does not confirm paper hypotheses")

    if manifest.get("detector_training_domain") != "ANALYTIC_SYNTHETIC_HEALTH_V1":
        raise ValueError("smoke detector training domain must remain explicit and synthetic")
    if manifest.get("policy") != "not_evaluated":
        raise ValueError("public smoke must not imply closed-loop policy evidence")

    _require_sha256(manifest.get("dataset_index_sha256"), "dataset_index_sha256")
    _require_sha256(manifest.get("configuration_sha256"), "configuration_sha256")

    split = manifest.get("split_definition")
    if not isinstance(split, dict):
        raise ValueError("split_definition must be an object")
    if set(("train", "calibration", "validation", "test")) - set(split):
        raise ValueError("split_definition is incomplete")
    if split.get("test") != sequence:
        raise ValueError("test split must name the real public sequence")
    if any(split.get(role) == sequence for role in ("train", "calibration", "validation")):
        raise ValueError("public test sequence leaked into a development role")

    counts = manifest.get("sample_counts")
    if not isinstance(counts, dict):
        raise ValueError("sample_counts must be an object")
    associated = int(counts.get("associated", 0))
    eligible = int(counts.get("eligible", 0))
    positive = int(counts.get("positive_windows", 0))
    negative = int(counts.get("negative_windows", 0))
    events = int(counts.get("failure_events", 0))
    if min(associated, eligible, positive, negative, events) <= 0:
        raise ValueError("smoke evidence requires associated/eligible rows, both classes, and events")
    if positive + negative != eligible:
        raise ValueError("positive_windows + negative_windows must equal eligible")
    if eligible > associated:
        raise ValueError("eligible sample count cannot exceed associated sample count")

    missing_files = sorted(name for name in REQUIRED_ARTIFACTS if not (output_dir / name).is_file())
    if missing_files:
        raise ValueError(f"missing required smoke artifacts: {', '.join(missing_files)}")

    recorded_hashes = manifest.get("artifact_sha256")
    if not isinstance(recorded_hashes, dict):
        raise ValueError("artifact_sha256 must be an object")
    missing_hashes = sorted(REQUIRED_ARTIFACTS - set(recorded_hashes))
    if missing_hashes:
        raise ValueError(f"manifest lacks hashes for: {', '.join(missing_hashes)}")
    for name in sorted(REQUIRED_ARTIFACTS):
        expected = str(recorded_hashes[name])
        _require_sha256(expected, f"artifact_sha256[{name}]")
        actual = _sha256_file(output_dir / name)
        if actual != expected:
            raise ValueError(f"artifact hash mismatch: {name}")

    health_rows = _csv_rows(output_dir / "health_features.csv")
    if len(health_rows) != associated:
        raise ValueError("health_features row count disagrees with manifest")
    for row in health_rows:
        timestamp = int(row["timestamp_ns"])
        source = int(row["max_source_timestamp_ns"])
        if source > timestamp:
            raise ValueError("causal provenance violation: feature source timestamp is in the future")

    prediction_rows = _csv_rows(output_dir / "predictions.csv")
    if len(prediction_rows) != associated:
        raise ValueError("prediction row count disagrees with manifest")
    eligible_rows = [row for row in prediction_rows if int(row["eligible"]) == 1]
    if len(eligible_rows) != eligible:
        raise ValueError("eligible prediction count disagrees with manifest")
    observed_positive = sum(int(row["future_failure"]) for row in eligible_rows)
    if observed_positive != positive:
        raise ValueError("positive future-failure count disagrees with manifest")
    if len(eligible_rows) - observed_positive != negative:
        raise ValueError("negative future-failure count disagrees with manifest")

    metrics = _load_json(output_dir / "metrics.json")
    missing_methods = sorted(REQUIRED_METHODS - set(metrics))
    if missing_methods:
        raise ValueError(f"metrics missing required methods: {', '.join(missing_methods)}")
    for method in sorted(REQUIRED_METHODS):
        payload = metrics[method]
        if not isinstance(payload, dict):
            raise ValueError(f"metrics for {method} must be an object")
        _require_probability_metric(payload.get("auroc"), f"{method}.auroc")
        _require_probability_metric(payload.get("auprc"), f"{method}.auprc")
        if int(payload.get("eligible_samples", -1)) != eligible:
            raise ValueError(f"{method} eligible_samples disagrees with manifest")
        if int(payload.get("positive_windows", -1)) != positive:
            raise ValueError(f"{method} positive_windows disagrees with manifest")
        if int(payload.get("negative_windows", -1)) != negative:
            raise ValueError(f"{method} negative_windows disagrees with manifest")

    for method in ("logistic_raw", "logistic_platt"):
        if "calibration" not in metrics[method]:
            raise ValueError(f"{method} must include calibration diagnostics")

    estimator_manifest = Path(str(manifest.get("estimator_manifest", "")))
    if not estimator_manifest.is_file():
        raise ValueError("referenced estimator manifest does not exist")

    git_state = manifest.get("git")
    if not isinstance(git_state, dict) or git_state.get("commit_sha") in (None, "", "unknown"):
        raise ValueError("git commit SHA must be captured")

    report = {
        "schema_version": 1,
        "validation": "SHIELD_VIO_PUBLIC_DATASET_SMOKE_GATE_V1",
        "status": "pass",
        "evidence_level": manifest["evidence_level"],
        "confirmatory": manifest["confirmatory"],
        "dataset": manifest["dataset"],
        "sequence": sequence,
        "associated_samples": associated,
        "eligible_samples": eligible,
        "positive_windows": positive,
        "negative_windows": negative,
        "failure_events": events,
        "validated_artifact_count": len(REQUIRED_ARTIFACTS),
        "causal_provenance_checked": True,
        "artifact_hashes_checked": True,
        "claim_boundary_checked": True,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-dataset", default="EuRoC_MAV")
    parser.add_argument("--expected-sequence")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = validate_smoke(
        args.output_dir,
        expected_dataset=args.expected_dataset,
        expected_sequence=args.expected_sequence,
    )
    destination = args.report or args.output_dir / "smoke_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
