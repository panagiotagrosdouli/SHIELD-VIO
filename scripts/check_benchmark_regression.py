"""Fail CI when a candidate EuRoC benchmark regresses beyond configured limits."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from shield_vio.evaluation.regression_guard import (
    compare_benchmark_reports,
    load_benchmark_report,
    write_guard_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline batch benchmark JSON")
    parser.add_argument("candidate", help="Candidate batch benchmark JSON")
    parser.add_argument("--max-relative-increase", type=float, default=0.05)
    parser.add_argument("--max-absolute-increase", type=float, default=0.01)
    parser.add_argument("--output", help="Optional JSON result artifact")
    args = parser.parse_args()

    result = compare_benchmark_reports(
        load_benchmark_report(args.baseline),
        load_benchmark_report(args.candidate),
        max_relative_increase=args.max_relative_increase,
        max_absolute_increase=args.max_absolute_increase,
    )
    if args.output:
        write_guard_result(result, args.output)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
