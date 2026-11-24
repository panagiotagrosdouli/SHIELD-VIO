from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shield_vio.experiments.canonical_held_out import evaluate_canonical
from test_development_seal import bundle  # noqa: F401, F811
from test_split_safe_development import _write_run


@pytest.fixture
def canonical(tmp_path: Path) -> Path:
    root = _write_run(tmp_path / "heldout", "test_seq")
    manifest_path = root / "canonical_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Censor an interior positive observation, not just the sequence tail.
    target = root / "prediction_targets.csv"
    target.write_text(target.read_text().replace("3000000000,1,1", "3000000000,1,0"))
    manifest.update({
        "schema_version": "SHIELD_VIO_PUBLIC_CANONICAL_V2",
        "failure_definition_id": "SHIELD_VIO_FAILURE_V2",
        "ground_truth_is_predictor": False,
        "samples": 8,
        "feature_schema_sha256": hashlib.sha256(
            b"estimator.nis\nestimator.nis__missing"
        ).hexdigest(),
        "artifact_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("prediction_dataset.csv", "prediction_targets.csv", "failure_events.csv")
        },
    })
    manifest_path.write_text(json.dumps(manifest))
    return root


def _run(canonical, bundle, split="test"):  # noqa: F811
    root, config, digest = bundle
    return evaluate_canonical(
        canonical, root, config, digest,
        dataset="euroc", sequence="test_seq", estimator="internal_eskf", split=split,
    )


def test_preserves_interior_censoring(canonical, bundle):  # noqa: F811
    report, predictions = _run(canonical, bundle)
    assert report["eligible_rows"] == 4
    assert report["positive_rows"] == 1
    assert report["censored_rows"] == 4
    assert not predictions["eligible"][3]
    assert report["event_metrics"] is None
    assert not report["confirmatory"]


def test_rejects_tampered_targets(canonical, bundle):  # noqa: F811
    path = canonical / "prediction_targets.csv"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        _run(canonical, bundle)


def test_rejects_development_split(canonical, bundle):  # noqa: F811
    with pytest.raises(ValueError, match="held-out partition required"):
        _run(canonical, bundle, split="train")
