#!/usr/bin/env python3
"""Import pinned OpenVINS total-state exports into SHIELD-VIO run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.experiments.openvins_adapter import import_openvins_total_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--deviation", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter-config",
        type=Path,
        default=Path("configs/paper/openvins_adapter_v1.yaml"),
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    manifest = import_openvins_total_state(
        args.estimate,
        args.deviation,
        args.sequence_root,
        args.output,
        adapter_config=args.adapter_config,
        evaluate=not args.skip_evaluation,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
