from __future__ import annotations

import zipfile
from pathlib import Path

import yaml

from scripts.acquire_euroc_development import _extract_source, _reader_path
from scripts.convert_euroc_rosbag import _sha256, _source_bytes
from shield_vio.datasets.splits import load_paper_split


def test_ros2_archive_detection_and_tree_hash_are_deterministic(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("bag/metadata.yaml", "rosbag2_bagfile_information: {}\n")
        handle.writestr("bag/data_0.db3", b"fixture sqlite bytes")

    first = tmp_path / "first"
    second = tmp_path / "second"
    _extract_source(archive, first)
    _extract_source(archive, second)

    first_reader = _reader_path(first)
    second_reader = _reader_path(second)
    assert first_reader.name == "bag"
    assert second_reader.name == "bag"
    assert _sha256(first_reader) == _sha256(second_reader)
    assert _source_bytes(first_reader) == _source_bytes(second_reader)
    assert len(_sha256(first_reader)) == 64


def test_development_source_registry_contains_no_test_sequences() -> None:
    root = Path(__file__).parents[1]
    registry = yaml.safe_load(
        (root / "configs/paper/euroc_development_sources.yaml").read_text(
            encoding="utf-8"
        )
    )
    split = load_paper_split(root / "configs/paper/public_dataset_splits.yaml")
    test_sequences = {
        identity.sequence for identity in split.partition("test")
    } | {
        identity.sequence for identity in split.partition("shifted_test")
    }

    entries = registry["sequences"]
    assert entries
    assert not (set(entries) & test_sequences)
    for sequence, entry in entries.items():
        assert entry["split"] in {"train", "calibration", "validation"}
        split.assert_run_membership("euroc", sequence, entry["split"])
        assert entry["source_kind"] == "openvins_google_drive_rosbag2"
        assert entry["google_drive_file_id"]


def test_archive_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")

    try:
        _extract_source(archive, tmp_path / "output")
    except ValueError as exc:
        assert "escapes destination" in str(exc)
    else:
        raise AssertionError("unsafe archive path was not rejected")
