#!/usr/bin/env python3
"""Run the built-in ESKF over a EuRoC sequence and export experiment artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.experiments.euroc_runner import run_euroc_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_root", type=Path, help="Path containing the EuRoC mav0 directory")
    parser.add_argument("--output", type=Path, required=True, help="Artifact output directory")
    parser.add_argument("--camera", default="cam0", help="EuRoC camera stream used for frame timestamps")
    args = parser.parse_args()

    summary = run_euroc_sequence(args.sequence_root, args.output, camera=args.camera)
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
