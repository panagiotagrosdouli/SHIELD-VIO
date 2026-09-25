#!/usr/bin/env python3
"""Acquire one registered EuRoC development sequence and convert it to minimal ASL."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import yaml

from scripts.convert_euroc_rosbag import _sha256, _source_bytes, convert_rosbag
from shield_vio.datasets.splits import load_paper_split


def _mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_google_drive(file_id: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for development dataset acquisition; "
            "install shield-vio[dataset]"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    result = gdown.download(id=file_id, output=str(destination), quiet=False)
    if result is None or not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError(f"Google Drive acquisition failed for file id {file_id}")


def _safe_destination(root: Path, member_name: str) -> Path:
    candidate = (root / member_name).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"archive path escapes destination: {member_name}")
    return candidate


def _extract_source(archive: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                _safe_destination(destination, member.filename)
            handle.extractall(destination)
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                _safe_destination(destination, member.name)
            handle.extractall(destination)
        return

    prefix = archive.read_bytes()[:32]
    if prefix.startswith(b"#ROSBAG V2.0"):
        shutil.copy2(archive, destination / "sequence.bag")
        return

    raise ValueError(
        "unsupported Google Drive source packaging; expected ZIP, TAR, or ROS1 bag. "
        f"first_bytes={prefix!r}"
    )


def _reader_path(extracted: Path) -> Path:
    metadata = sorted(extracted.rglob("metadata.yaml"))
    if len(metadata) == 1:
        return metadata[0].parent
    if len(metadata) > 1:
        raise ValueError("multiple ROS2 metadata.yaml files found in development source")
    bags = sorted(extracted.rglob("*.bag"))
    if len(bags) == 1:
        return bags[0]
    if len(bags) > 1:
        raise ValueError("multiple ROS1 bag files found in development source")
    raise ValueError("could not locate ROS1 bag or ROS2 metadata.yaml after extraction")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        with destination.open("wb") as stream:
            shutil.copyfileobj(response, stream)


def acquire(
    sequence: str,
    output_root: Path,
    work_dir: Path,
    *,
    registry_path: Path,
    split_config: Path,
) -> dict[str, Any]:
    registry = _mapping(registry_path)
    sequences = registry.get("sequences")
    if not isinstance(sequences, dict) or sequence not in sequences:
        raise ValueError(f"sequence is not registered for development acquisition: {sequence}")
    entry = sequences[sequence]
    if not isinstance(entry, dict):
        raise ValueError(f"invalid source registry entry: {sequence}")
    split_role = str(entry["split"])
    if split_role not in {"train", "calibration", "validation"}:
        raise ValueError("development acquisition cannot target test/shifted_test")

    split = load_paper_split(split_config)
    split.assert_run_membership("euroc", sequence, split_role)

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(entry["google_drive_file_id"])
    source_url = f"https://drive.google.com/file/d/{file_id}/view"
    download_path = work_dir / "source.download"
    _download_google_drive(file_id, download_path)
    download_sha256 = _file_sha256(download_path)
    expected = entry.get("expected_download_sha256")
    if expected is None:
        checksum_status = "observed_unfrozen"
    else:
        normalized = str(expected).strip().lower()
        if download_sha256 != normalized:
            raise ValueError(
                f"development source checksum mismatch: expected {normalized}, "
                f"got {download_sha256}"
            )
        checksum_status = "predeclared_verified"

    extracted = work_dir / "extracted"
    _extract_source(download_path, extracted)
    reader_path = _reader_path(extracted)
    reader_sha256 = _sha256(reader_path)

    revision = str(registry["openvins_revision"])
    openvins = work_dir / "openvins-pinned"
    if openvins.exists():
        shutil.rmtree(openvins)
    openvins.mkdir(parents=True)
    base = f"https://raw.githubusercontent.com/rpng/open_vins/{revision}"
    ground_truth = openvins / str(entry["ground_truth_file"])
    imucam = openvins / "kalibr_imucam_chain.yaml"
    imu = openvins / "kalibr_imu_chain.yaml"
    _download(
        f"{base}/ov_data/euroc_mav/{entry['ground_truth_file']}",
        ground_truth,
    )
    _download(f"{base}/config/euroc_mav/kalibr_imucam_chain.yaml", imucam)
    _download(f"{base}/config/euroc_mav/kalibr_imu_chain.yaml", imu)

    conversion = convert_rosbag(
        reader_path,
        output_root,
        sequence=sequence,
        expected_bag_sha256=reader_sha256,
        source_url=source_url,
        openvins_ground_truth=ground_truth,
        openvins_imucam_calibration=imucam,
        openvins_imu_calibration=imu,
        openvins_revision=revision,
        evidence_scope="PUBLIC_DATASET_DEVELOPMENT_ACQUISITION",
    )

    receipt = {
        "schema_version": "SHIELD_VIO_EUROC_DEVELOPMENT_ACQUISITION_V1",
        "dataset": "euroc",
        "sequence": sequence,
        "split": split_role,
        "evidence_level": "PUBLIC_DATASET_DEVELOPMENT_ACQUISITION",
        "confirmatory": False,
        "confirmatory_identity": False,
        "checksum_status": checksum_status,
        "source": {
            "kind": str(entry["source_kind"]),
            "google_drive_file_id": file_id,
            "url": source_url,
            "download_sha256": download_sha256,
            "download_bytes": int(download_path.stat().st_size),
            "expected_download_sha256": expected,
            "reader_sha256": reader_sha256,
            "reader_bytes": _source_bytes(reader_path),
        },
        "openvins": {
            "revision": revision,
            "ground_truth_file": str(entry["ground_truth_file"]),
            "ground_truth_sha256": _file_sha256(ground_truth),
            "imucam_sha256": _file_sha256(imucam),
            "imu_sha256": _file_sha256(imu),
        },
        "registry": {
            "path": str(registry_path),
            "sha256": _file_sha256(registry_path),
            "split_config": str(split_config),
            "split_config_sha256": _file_sha256(split_config),
        },
        "conversion_manifest_sha256": _file_sha256(
            output_root / "portable_acquisition_manifest.json"
        ),
        "conversion_summary": conversion["conversion"],
        "claim_boundary": (
            "Development-only acquisition from a pinned OpenVINS-listed source. "
            "An observed_unfrozen download hash must be committed to the registry and "
            "re-verified before this source is treated as frozen development provenance. "
            "This artifact is never confirmatory dataset identity."
        ),
    }
    (output_root / "development_acquisition_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/paper/euroc_development_sources.yaml"),
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=Path("configs/paper/public_dataset_splits.yaml"),
    )
    args = parser.parse_args()
    receipt = acquire(
        args.sequence,
        args.output,
        args.work_dir,
        registry_path=args.registry,
        split_config=args.split_config,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
