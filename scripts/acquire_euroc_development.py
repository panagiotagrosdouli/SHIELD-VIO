#!/usr/bin/env python3
"""Acquire one registered EuRoC development sequence and build a minimal ASL tree."""

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

from scripts.convert_euroc_rosbag import (
    _sha256,
    _source_bytes,
    build_asl_calibrations,
    convert_openvins_ground_truth,
    convert_rosbag,
)
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


def _files_sha256(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
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

    with archive.open("rb") as stream:
        prefix = stream.read(32)
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


def _openvins_inputs(
    work_dir: Path,
    *,
    revision: str,
    ground_truth_file: str,
) -> tuple[Path, Path, Path]:
    openvins = work_dir / "openvins-pinned"
    if openvins.exists():
        shutil.rmtree(openvins)
    openvins.mkdir(parents=True)
    base = f"https://raw.githubusercontent.com/rpng/open_vins/{revision}"
    ground_truth = openvins / ground_truth_file
    imucam = openvins / "kalibr_imucam_chain.yaml"
    imu = openvins / "kalibr_imu_chain.yaml"
    _download(f"{base}/ov_data/euroc_mav/{ground_truth_file}", ground_truth)
    _download(f"{base}/config/euroc_mav/kalibr_imucam_chain.yaml", imucam)
    _download(f"{base}/config/euroc_mav/kalibr_imu_chain.yaml", imu)
    return ground_truth, imucam, imu


def _member_relative_to_sequence(member_name: str, sequence: str) -> str | None:
    normalized = member_name.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    try:
        index = parts.index(sequence)
    except ValueError:
        return None
    relative = "/".join(parts[index + 1 :])
    return relative or None


def _remote_identity(info: dict[str, Any]) -> dict[str, Any]:
    etag = info.get("ETag", info.get("etag"))
    size = info.get("size", info.get("ContentLength"))
    return {
        "etag": None if etag is None else str(etag).strip('"'),
        "archive_bytes": None if size is None else int(size),
    }


def _selective_huggingface_asl(
    url: str,
    output_root: Path,
    *,
    sequence: str,
    ground_truth: Path,
    imucam: Path,
    imu: Path,
    openvins_revision: str,
    expected_subset_sha256: str | None,
) -> dict[str, Any]:
    """Extract cam0+imu0 for one sequence using HTTP range requests only."""

    try:
        import fsspec
    except ImportError as exc:
        raise RuntimeError(
            "fsspec[http] is required for selective aggregate extraction; "
            "install shield-vio[dataset]"
        ) from exc

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    extracted_source_files: list[Path] = []
    member_count = 0
    camera_frames = 0
    remote_info: dict[str, Any] = {}

    opener = fsspec.open(
        url,
        mode="rb",
        block_size=16 * 1024 * 1024,
        cache_type="readahead",
    )
    with opener as remote:
        if not bool(getattr(remote, "seekable", lambda: False)()):
            raise RuntimeError("aggregate HTTP source is not seekable; range extraction required")
        try:
            remote_info = _remote_identity(remote.fs.info(remote.path))
        except Exception:
            remote_info = {"etag": None, "archive_bytes": None}

        with zipfile.ZipFile(remote) as archive:
            selected: list[tuple[zipfile.ZipInfo, str]] = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                relative = _member_relative_to_sequence(info.filename, sequence)
                if relative is None:
                    continue
                keep = (
                    relative == "mav0/cam0/data.csv"
                    or relative == "mav0/imu0/data.csv"
                    or (
                        relative.startswith("mav0/cam0/data/")
                        and relative.lower().endswith(".png")
                    )
                )
                if keep:
                    selected.append((info, relative))

            required = {"mav0/cam0/data.csv", "mav0/imu0/data.csv"}
            present = {relative for _, relative in selected}
            missing = sorted(required - present)
            if missing:
                raise ValueError(
                    f"aggregate archive lacks required {sequence} members: {', '.join(missing)}"
                )
            camera_frames = sum(
                1
                for _, relative in selected
                if relative.startswith("mav0/cam0/data/")
                and relative.lower().endswith(".png")
            )
            if camera_frames < 100:
                raise ValueError(
                    f"aggregate archive has unexpectedly few {sequence} cam0 frames: "
                    f"{camera_frames}"
                )

            for info, relative in selected:
                destination = _safe_destination(output_root, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                extracted_source_files.append(destination)
                member_count += 1

    subset_sha256 = _files_sha256(output_root, extracted_source_files)
    checksum_status = "observed_unfrozen"
    if expected_subset_sha256 is not None:
        normalized = str(expected_subset_sha256).strip().lower()
        if subset_sha256 != normalized:
            raise ValueError(
                f"development subset checksum mismatch: expected {normalized}, "
                f"got {subset_sha256}"
            )
        checksum_status = "predeclared_verified"

    camera_calibration, imu_calibration = build_asl_calibrations(imucam, imu)
    camera_yaml = output_root / "mav0/cam0/sensor.yaml"
    imu_yaml = output_root / "mav0/imu0/sensor.yaml"
    camera_yaml.parent.mkdir(parents=True, exist_ok=True)
    imu_yaml.parent.mkdir(parents=True, exist_ok=True)
    camera_yaml.write_text(
        yaml.safe_dump(camera_calibration, sort_keys=False),
        encoding="utf-8",
    )
    imu_yaml.write_text(
        yaml.safe_dump(imu_calibration, sort_keys=False),
        encoding="utf-8",
    )
    ground_truth_count = convert_openvins_ground_truth(
        ground_truth,
        output_root / "mav0/state_groundtruth_estimate0/data.csv",
    )

    imu_csv = output_root / "mav0/imu0/data.csv"
    with imu_csv.open("r", encoding="utf-8") as stream:
        imu_samples = sum(
            1 for line in stream if line.strip() and not line.lstrip().startswith("#")
        )

    conversion = {
        "sequence": sequence,
        "camera_frames": camera_frames,
        "imu_samples": imu_samples,
        "ground_truth_samples": ground_truth_count,
        "source_member_count": member_count,
        "source_subset_sha256": subset_sha256,
        "source_subset_uncompressed_bytes": int(
            sum(path.stat().st_size for path in extracted_source_files)
        ),
    }
    manifest = {
        "schema_version": "SHIELD_VIO_EUROC_PORTABLE_ASL_SUBSET_V1",
        "dataset": "EuRoC_MAV",
        "sequence": sequence,
        "evidence_scope": "PUBLIC_DATASET_DEVELOPMENT_ACQUISITION",
        "confirmatory_identity": False,
        "source": {
            "kind": "huggingface_aggregate_zip_range",
            "url": url,
            **remote_info,
            "subset_sha256": subset_sha256,
            "expected_subset_sha256": expected_subset_sha256,
            "checksum_status": checksum_status,
        },
        "openvins": {
            "revision": openvins_revision,
            "ground_truth_sha256": _file_sha256(ground_truth),
            "imucam_sha256": _file_sha256(imucam),
            "imu_sha256": _file_sha256(imu),
        },
        "conversion": conversion,
        "claim_boundary": (
            "Development-only selective extraction from a public aggregate mirror. "
            "Only cam0 and imu0 source files for the registered sequence are fetched; "
            "ground truth/calibration are pinned OpenVINS inputs. This is not "
            "confirmatory dataset identity."
        ),
    }
    (output_root / "portable_acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


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

    revision = str(registry["openvins_revision"])
    ground_truth, imucam, imu = _openvins_inputs(
        work_dir,
        revision=revision,
        ground_truth_file=str(entry["ground_truth_file"]),
    )

    file_id = str(entry["google_drive_file_id"])
    drive_url = f"https://drive.google.com/file/d/{file_id}/view"
    download_path = work_dir / "source.download"
    drive_error: str | None = None
    conversion: dict[str, Any]
    source: dict[str, Any]
    checksum_status: str

    try:
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
        conversion = convert_rosbag(
            reader_path,
            output_root,
            sequence=sequence,
            expected_bag_sha256=reader_sha256,
            source_url=drive_url,
            openvins_ground_truth=ground_truth,
            openvins_imucam_calibration=imucam,
            openvins_imu_calibration=imu,
            openvins_revision=revision,
            evidence_scope="PUBLIC_DATASET_DEVELOPMENT_ACQUISITION",
        )
        source = {
            "kind": str(entry["source_kind"]),
            "google_drive_file_id": file_id,
            "url": drive_url,
            "download_sha256": download_sha256,
            "download_bytes": int(download_path.stat().st_size),
            "expected_download_sha256": expected,
            "reader_sha256": reader_sha256,
            "reader_bytes": _source_bytes(reader_path),
        }
    except Exception as exc:
        drive_error = f"{type(exc).__name__}: {exc}"
        if download_path.exists():
            download_path.unlink()
        fallback = entry.get("fallback")
        if not isinstance(fallback, dict):
            raise
        fallback_url = str(fallback["url"])
        conversion = _selective_huggingface_asl(
            fallback_url,
            output_root,
            sequence=sequence,
            ground_truth=ground_truth,
            imucam=imucam,
            imu=imu,
            openvins_revision=revision,
            expected_subset_sha256=fallback.get("expected_subset_sha256"),
        )
        fallback_source = conversion["source"]
        checksum_status = str(fallback_source["checksum_status"])
        source = {
            "kind": str(fallback["source_kind"]),
            "url": fallback_url,
            "etag": fallback_source.get("etag"),
            "archive_bytes": fallback_source.get("archive_bytes"),
            "subset_sha256": fallback_source["subset_sha256"],
            "expected_subset_sha256": fallback_source.get("expected_subset_sha256"),
            "primary_source_error": drive_error,
        }

    receipt = {
        "schema_version": "SHIELD_VIO_EUROC_DEVELOPMENT_ACQUISITION_V1",
        "dataset": "euroc",
        "sequence": sequence,
        "split": split_role,
        "evidence_level": "PUBLIC_DATASET_DEVELOPMENT_ACQUISITION",
        "confirmatory": False,
        "confirmatory_identity": False,
        "checksum_status": checksum_status,
        "source": source,
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
            "Development-only acquisition. An observed_unfrozen source/subset hash must "
            "be committed to the registry and re-verified before the development source "
            "is treated as frozen provenance. This artifact is never confirmatory dataset "
            "identity."
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
