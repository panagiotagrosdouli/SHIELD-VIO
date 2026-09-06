"""Resolve publication benchmark experiments without frame-level splitting."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from shield_vio.datasets.provenance import RunIdentity
from shield_vio.datasets.splits import load_paper_split


@dataclass(frozen=True)
class BenchmarkRun:
    dataset: str
    sequence: str
    estimator: str
    degradation: str
    severity: str
    seed: int
    split: str
    run_id: str
    status: str = "NOT_AVAILABLE"


def resolve_benchmark_matrix(config_path: str | Path, split_path: str | Path) -> tuple[BenchmarkRun, ...]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("runs"), list):
        raise ValueError("benchmark config requires a runs list")
    split = load_paper_split(split_path)
    result: list[BenchmarkRun] = []
    seen: set[str] = set()
    for row in raw["runs"]:
        if not isinstance(row, dict):
            raise ValueError("benchmark run must be a mapping")
        dataset = str(row["dataset"])
        sequence = str(row["sequence"])
        estimator = str(row.get("estimator", "internal_eskf"))
        degradation = str(row.get("degradation", "clean"))
        severity = str(row.get("severity", "none"))
        seed = int(row.get("seed", 0))
        partition = str(row["split"])
        split.assert_run_membership(dataset, sequence, partition)
        identity = RunIdentity(dataset, sequence, estimator, degradation, severity, seed)
        run_id = identity.run_id()
        if run_id in seen:
            raise ValueError(f"duplicate logical experiment: {run_id}")
        seen.add(run_id)
        result.append(BenchmarkRun(dataset, sequence, estimator, degradation, severity, seed, partition, run_id))
    return tuple(result)


def write_benchmark_matrix(runs: tuple[BenchmarkRun, ...], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(runs[0]).keys()) if runs else ["dataset", "sequence", "estimator", "degradation", "severity", "seed", "split", "run_id", "status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))


def write_split_manifest(runs: tuple[BenchmarkRun, ...], split_path: str | Path, output: str | Path) -> None:
    import hashlib

    split_bytes = Path(split_path).read_bytes()
    payload = {
        "schema_version": 1,
        "evidence_level": "PUBLIC_DATASET_BENCHMARK_INFRASTRUCTURE",
        "confirmatory": False,
        "split_config_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "number_of_runs": len(runs),
        "runs": [asdict(run) for run in runs],
    }
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
