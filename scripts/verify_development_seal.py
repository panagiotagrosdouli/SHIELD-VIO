#!/usr/bin/env python3
"""Verify a development bundle without loading held-out data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shield_vio.experiments.development_seal import verify_development_seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    manifest = verify_development_seal(
        args.bundle, args.split_config, args.expected_sha256
    )
    print(json.dumps({"verified": True, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
