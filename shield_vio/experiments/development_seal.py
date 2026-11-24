"""Verify a frozen development bundle before any held-out evaluation."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

SEALED_FILES = (
    "model.json",
    "calibration.json",
    "threshold.json",
    "development_manifest.json",
)


def verify_development_seal(
    root: str | Path, split_config: str | Path, expected_sha256: str
) -> dict:
    """Verify integrity and split provenance against an independently recorded digest.

    The expected digest must be recorded before opening held-out data.
    This is an integrity check, not authentication or evidence of performance.
    """
    root = Path(root)

    def read(name: str) -> dict:
        value = json.loads((root / name).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object: {name}")
        return value

    seal = read("development_seal.json")
    if (
        seal.get("schema_version") != "SHIELD_VIO_DEVELOPMENT_SEAL_V1"
        or seal.get("sealed_files") != list(SEALED_FILES)
        or seal.get("test_partition_loaded") is not False
    ):
        raise ValueError("invalid development seal metadata")
    digest = hashlib.sha256()
    for name in SEALED_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, expected_sha256) or actual != seal.get("sha256"):
        raise ValueError("development seal digest mismatch")
    manifest = read("development_manifest.json")
    if (
        manifest.get("schema_version") != "SHIELD_VIO_PUBLIC_DEVELOPMENT_BUNDLE_V1"
        or manifest.get("confirmatory") is not False
        or manifest.get("test_partition_loaded") is not False
        or set(manifest.get("partitions", {})) != {"train", "calibration", "validation"}
    ):
        raise ValueError("invalid development manifest")
    split_digest = hashlib.sha256(Path(split_config).read_bytes()).hexdigest()
    if split_digest != manifest.get("split_config_sha256"):
        raise ValueError("split configuration digest mismatch")
    model = read("model.json")
    calibration = read("calibration.json")
    threshold = read("threshold.json")
    if (
        model.get("fit_split") != "train"
        or calibration.get("fit_split") != "calibration"
        or threshold.get("selected_on") != "validation"
        or model.get("feature_names") != manifest.get("feature_schema")
        or threshold.get("horizon_seconds") != manifest.get("horizon_seconds")
    ):
        raise ValueError("inconsistent development provenance")
    return manifest
