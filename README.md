<div align="center">

# SHIELD-VIO

## Estimator Introspection, Calibrated Failure Prediction, and Protective Navigation

A reproducible research framework for detecting visual–inertial degradation before localization failure becomes safety-critical.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](.) [![Status](https://img.shields.io/badge/status-research%20prototype-orange)](.) [![Evidence](https://img.shields.io/badge/evidence-synthetic%20validation-purple)](.)

**English** · [Ελληνικά](README_GR.md)

</div>

<p align="center"><img src="assets/readme/shield_vio_pipeline_v2.svg" alt="SHIELD-VIO protective estimation pipeline" width="100%" /></p>

<p align="center"><em>Conceptual research figure. It is not production VIO, hardware validation, formal certification, or a safety guarantee.</em></p>

## Abstract

SHIELD-VIO studies how a visual–inertial estimator can recognize that its own state estimate is becoming unreliable, quantify near-term failure risk, detect domain shift, and protect downstream navigation through an explicit supervisory shield. The framework connects feature tracking, bias-aware IMU preintegration, a 15-state error-state EKF, consistency diagnostics, failure detectors, calibration, conformal bounds, and stateful recovery actions.

The central design principle is that trajectory accuracy alone is insufficient for safety-oriented autonomy. The estimator must also expose health, uncertainty, innovation consistency, failure probability, shift state, and the reason for any protective action.

## Research question

> How can a visual–inertial estimator identify loss of reliability early enough to protect navigation, trigger recovery, and maintain meaningful confidence under sensor degradation and domain shift?

## Architecture

```text
camera frames → feature tracking ┐
                                 ├→ 15-state ESKF
IMU stream → preintegration ─────┘
  → health and consistency diagnostics
  → failure prediction and calibration
  → conformal bounds and shift detection
  → stateful navigation shield
  → normal / slow / hold / relocalize / halt / emergency stop
```

## Scientific formulation

```math
x=\{p_{WI},v_{WI},q_{WI},b_a,b_g\},
\qquad
\delta x=[\delta p,\delta v,\delta\theta,\delta b_a,\delta b_g]^T.
```

Consistency is monitored through

```math
\mathrm{NIS}=\nu^TS^{-1}\nu,
\qquad
\mathrm{NEES}=e^TP^{-1}e
```

when ground truth is available. An uncalibrated diagnostic score is never treated as a probability.

## Implemented research components

- Shi–Tomasi and pyramidal Lucas–Kanade tracking;
- persistent tracks, forward–backward rejection, and visual-health diagnostics;
- bias-aware IMU preintegration and covariance propagation;
- 15-state ESKF with Joseph-form updates;
- deterministic visual and IMU degradation injection;
- rule-based and logistic failure detectors;
- Brier, NLL, ECE, and conformal evaluation;
- rolling domain-shift states;
- stateful shield with hysteresis, dwell behavior, stale-sensor handling, and emergency override.

## Shield states

```text
NORMAL → WARNING → SLOW_DOWN → HOLD_POSITION
       → RELOCALIZE_REQUESTED → HALT → EMERGENCY_STOP
```

## Reproduce

```bash
git clone https://github.com/panagiotagrosdouli/SHIELD-VIO.git
cd SHIELD-VIO
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/run_all.py
ruff check shield_vio scripts tests
black --check .
pytest -q
```

Repeated scenarios:

```bash
python scripts/run_scenario_suite.py --num-seeds 20 --output results/scenario_suite
```

## Evidence boundaries

| Layer | Evidence |
|---|---|
| Feature frontend | Research Prototype |
| IMU preintegration | Analytical Unit Validation |
| 15-state ESKF | Numerical Invariant Validation |
| Failure prediction and calibration | Experimental |
| Domain-shift detection | Experimental |
| Navigation shield | Closed-loop Unit Validation |
| Multi-seed degradations | Synthetic Validation |
| EuRoC / TUM-VI | Pending Dataset Execution |
| ROS 2 / hardware | Validation Required |

## Limitations

- A production-quality integrated VIO backend is not complete.
- Public-dataset execution remains pending.
- The logistic detector is not benchmarked on real failure data.
- Conformal coverage depends on calibration assumptions and may degrade under severe shift.
- Loop closure, robust relocalization, ROS 2, simulation, and hardware execution remain incomplete.
- The shield is supervisory research logic, not a formally verified controller.

## Responsible use

SHIELD-VIO must not be used as the sole safety mechanism of a real robot or vehicle without independent validation, fail-safe control, and domain-specific certification.
