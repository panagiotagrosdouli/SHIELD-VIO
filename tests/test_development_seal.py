from __future__ import annotations

import json
from pathlib import Path

import pytest

from shield_vio.experiments.development_seal import verify_development_seal
from shield_vio.experiments.split_safe_development import fit_development_bundle

from test_split_safe_development import _write_run, _write_split


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    split = tmp_path / "splits.yaml"
    _write_split(split)
    artifacts = tmp_path / "artifacts"
    for sequence in ("train_seq", "cal_seq", "val_seq"):
        _write_run(artifacts, sequence)
    output = tmp_path / "bundle"
    report = fit_development_bundle(
        artifacts, split, output, logistic_iterations=200
    )
    return output, split, report["development_seal_sha256"]


def test_verify_valid_bundle(bundle: tuple[Path, Path, str]) -> None:
    root, split, digest = bundle
    manifest = verify_development_seal(root, split, digest)
    assert manifest["test_partition_loaded"] is False


@pytest.mark.parametrize(
    "name", ["model.json", "calibration.json", "threshold.json", "development_manifest.json"]
)
def test_modified_payload_rejected(
    bundle: tuple[Path, Path, str], name: str
) -> None:
    root, split, digest = bundle
    path = root / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_development_seal(root, split, digest)


def test_modified_split_rejected(bundle: tuple[Path, Path, str]) -> None:
    root, split, digest = bundle
    split.write_bytes(split.read_bytes() + b" ")
    with pytest.raises(ValueError, match="split configuration digest mismatch"):
        verify_development_seal(root, split, digest)


def test_wrong_independent_digest_rejected(bundle: tuple[Path, Path, str]) -> None:
    root, split, _ = bundle
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_development_seal(root, split, "0" * 64)


def test_invalid_seal_metadata_rejected(bundle: tuple[Path, Path, str]) -> None:
    root, split, digest = bundle
    path = root / "development_seal.json"
    seal = json.loads(path.read_text(encoding="utf-8"))
    seal["test_partition_loaded"] = True
    path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid development seal metadata"):
        verify_development_seal(root, split, digest)
