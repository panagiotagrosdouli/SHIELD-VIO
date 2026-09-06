"""Deterministic identities and provenance for public-dataset benchmark runs."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RunIdentity:
    dataset: str
    sequence: str
    estimator_id: str
    condition: str = "clean"
    severity: str = "none"
    seed: int = 0
    config_version: str = "PHASE_C_V1"

    def run_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(paths: Iterable[str | Path]) -> dict[str, object]:
    """Hash named metadata/index files; this is not an official archive checksum."""
    entries = []
    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({"path": path.as_posix(), "sha256": sha256_file(path)})
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {"algorithm": "sha256", "scope": "listed local metadata/index files; not an official dataset archive checksum", "files": entries, "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def environment_provenance() -> dict[str, object]:
    packages: dict[str, str] = {}
    for name in ("numpy", "matplotlib", "PyYAML", "Pillow", "opencv-python-headless"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "implementation": platform.python_implementation(), "packages": packages,
    }
