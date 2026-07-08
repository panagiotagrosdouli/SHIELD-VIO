#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pytest pyyaml
export PYTHONPATH="$PWD/src"
python scripts/demo_health_monitor.py
pytest -q
