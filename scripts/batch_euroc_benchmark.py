"""Evaluate multiple EuRoC sequences and write JSON/CSV benchmark artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from shield_vio.evaluation.euroc_batch import (
    evaluate_euroc_batch,
    write_batch_csv,
    write_batch_json,
)


def _load_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(",", maxsplit=1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError(f"invalid manifest row {line_number}: expected sequence_root,estimate")
        entries.append((parts[0].strip(), parts[1].strip()))
    if not entries:
        raise ValueError("manifest contains no benchmark entries")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV-like sequence_root,estimate manifest")
    parser.add_argument("--json", dest="json_path", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--csv", dest="csv_path", type=Path, default=Path("benchmark.csv"))
    parser.add_argument("--timestamp-unit", default="seconds")
    parser.add_argument("--max-gap", type=float, default=0.02)
    parser.add_argument("--rpe-delta", type=int, default=1)
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument(
        "--fail-on-sequence-error",
        action="store_true",
        help="Exit non-zero when one or more sequence evaluations fail",
    )
    args = parser.parse_args()

    report = evaluate_euroc_batch(
        _load_manifest(args.manifest),
        timestamp_unit=args.timestamp_unit,
        max_gap=args.max_gap,
        rpe_delta=args.rpe_delta,
        align=not args.no_align,
    )
    write_batch_json(report, args.json_path)
    write_batch_csv(report, args.csv_path)
    print(
        f"evaluated {len(report.results)} sequences: "
        f"{report.success_count} succeeded, {report.failure_count} failed"
    )
    if args.fail_on_sequence_error and report.failure_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
