# SHIELD-VIO

Failure-aware visual–inertial estimation with runtime protection.

SHIELD-VIO studies how a visual–inertial estimator can detect degradation before localization failure becomes safety-critical. The framework combines feature tracking, IMU preintegration, error-state filtering, consistency diagnostics, failure prediction, domain-shift monitoring, and supervisory recovery actions.

## Method

The current implementation includes:

- Shi–Tomasi and Lucas–Kanade feature tracking
- bias-aware IMU preintegration
- a 15-state error-state EKF
- NIS and NEES consistency diagnostics
- deterministic visual and inertial degradations
- rule-based and logistic failure detectors
- calibration and conformal evaluation
- a stateful navigation shield

## Reproduce

```bash
git clone https://github.com/panagiotagrosdouli/SHIELD-VIO.git
cd SHIELD-VIO
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/run_all.py
pytest -q
```

## Scope

The current evidence is based mainly on numerical checks, unit validation, and synthetic multi-seed experiments. Public-dataset execution, robust relocalization, ROS 2 integration, simulation, and hardware validation remain incomplete.

The supervisory shield is research logic, not a formally verified controller or a standalone safety mechanism.

[Greek documentation](README_GR.md)