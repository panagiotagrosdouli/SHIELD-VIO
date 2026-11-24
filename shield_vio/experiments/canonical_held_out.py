"""Evaluate sealed canonical V2 probability targets without reconstructing censoring."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from shield_vio.calibration_metrics.metrics import summarize_calibration
from shield_vio.datasets.splits import load_paper_split
from shield_vio.evaluation.prediction_metrics import auprc, auroc
from shield_vio.experiments.frozen_predictor import FrozenPredictor


def evaluate_canonical(
    root: Path,
    bundle: Path,
    split_config: Path,
    expected_sha256: str,
    *,
    dataset: str,
    sequence: str,
    estimator: str,
    split: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Return non-confirmatory probability metrics and aligned predictions."""
    if split not in ("test", "shifted_test"):
        raise ValueError("held-out partition required")
    load_paper_split(split_config).assert_run_membership(dataset, sequence, split)
    predictor = FrozenPredictor(bundle, split_config, expected_sha256)
    manifest = json.loads((root / "canonical_manifest.json").read_text())
    if (
        manifest.get("schema_version") != "SHIELD_VIO_PUBLIC_CANONICAL_V2"
        or manifest.get("failure_definition_id") != "SHIELD_VIO_FAILURE_V2"
        or manifest.get("ground_truth_is_predictor") is not False
        or (manifest.get("dataset"), manifest.get("sequence"), manifest.get("estimator"))
        != (dataset, sequence, estimator)
    ):
        raise ValueError("canonical manifest identity or schema mismatch")
    names = ("prediction_dataset.csv", "prediction_targets.csv", "failure_events.csv")
    for name in names:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != manifest.get("artifact_sha256", {}).get(name):
            raise ValueError(f"canonical artifact digest mismatch: {name}")

    def read(name: str) -> tuple[tuple[str, ...], list[dict]]:
        with (root / name).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = tuple(reader.fieldnames or ())
            if len(set(fields)) != len(fields):
                raise ValueError("duplicate CSV columns")
            return fields, list(reader)

    fields, rows = read(names[0])
    target_fields, targets = read(names[1])
    if fields != ("timestamp_ns", *predictor.feature_names):
        raise ValueError("ordered feature schema mismatch")
    schema_hash = hashlib.sha256("\n".join(fields[1:]).encode()).hexdigest()
    if schema_hash != manifest.get("feature_schema_sha256"):
        raise ValueError("canonical feature schema digest mismatch")
    ts = np.asarray([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    target_ts = np.asarray([int(row["timestamp_ns"]) for row in targets], dtype=np.int64)
    if (
        len(ts) < 2
        or len(ts) != manifest.get("samples")
        or np.any(np.diff(ts) <= 0)
        or not np.array_equal(ts, target_ts)
    ):
        raise ValueError("canonical timestamp alignment mismatch")
    horizon = predictor.horizon_seconds
    label_col = f"failure_within_{horizon}s"
    eligible_col = f"eligible_{horizon}s"
    if not {label_col, eligible_col} <= set(target_fields):
        raise ValueError("canonical target horizon missing")

    def binary(column: str) -> np.ndarray:
        values = [row[column] for row in targets]
        if any(value not in ("0", "1") for value in values):
            raise ValueError("canonical targets must be binary")
        return np.asarray([value == "1" for value in values])

    labels, eligible = binary(label_col), binary(eligible_col)
    features = np.asarray([[float(row[name]) for name in fields[1:]] for row in rows])
    raw, calibrated, alerts = predictor.predict(features, fields[1:])
    y = labels[eligible]

    def metrics(scores: np.ndarray) -> dict:
        p = scores[eligible]
        defined = len(np.unique(y)) == 2
        return {
            "auroc": auroc(y, p) if defined else None,
            "auprc": auprc(y, p) if defined else None,
            "calibration": asdict(summarize_calibration(p, y)) if len(y) else None,
        }

    report = {
        "dataset": dataset, "sequence": sequence, "estimator": estimator,
        "split": split, "confirmatory": False,
        "evidence_level": "CANONICAL_HELD_OUT_DEVELOPMENT",
        "development_seal_sha256": expected_sha256,
        "canonical_manifest_sha256": hashlib.sha256(
            (root / "canonical_manifest.json").read_bytes()
        ).hexdigest(),
        "horizon_seconds": horizon, "threshold": predictor.threshold,
        "eligible_rows": int(eligible.sum()), "positive_rows": int(y.sum()),
        "censored_rows": int((~eligible).sum()),
        "raw": metrics(raw), "calibrated": metrics(calibrated),
        "event_metrics": None,
        "claim_boundary": "Probability metrics only; event exposure and source provenance "
        "require separate validation.",
    }
    return report, {
        "timestamp_ns": ts, "label": labels, "eligible": eligible,
        "raw_probability": raw, "calibrated_probability": calibrated, "alert": alerts,
    }
