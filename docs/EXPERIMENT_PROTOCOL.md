# Experimental Protocol

## Objective

Evaluate whether SHIELD-VIO can detect visual-inertial degradation before localization failure, infer plausible degradation causes, and select recovery actions that improve estimator robustness.

## Baselines

1. VIO backend without SHIELD-VIO monitoring.
2. Threshold-only health monitor.
3. SHIELD-VIO with Bayesian diagnosis but no recovery.
4. SHIELD-VIO with diagnosis-conditioned recovery.

## Datasets

Planned datasets:

- EuRoC MAV
- TUM-VI
- Hilti SLAM Challenge
- UZH FPV

## Fault Injection Conditions

| Fault | Control Variable | Expected Effect |
|---|---:|---|
| motion blur | blur kernel size | reduced visual sharpness and feature stability |
| low texture | texture blending severity | feature collapse |
| feature dropout | dropout probability | sparse tracking support |
| IMU bias | bias magnitude | inertial inconsistency and drift |
| high reprojection error | synthetic residual inflation | estimator inconsistency |

## Metrics

- Absolute Trajectory Error
- Relative Pose Error
- Navigation Health Index
- time-to-warning before tracking loss
- diagnosis posterior calibration
- recovery action count
- recovery success rate
- false alarm rate

## Ablations

1. Remove temporal health filter.
2. Remove causal graph risk propagation.
3. Remove uncertainty decomposition.
4. Replace Bayesian diagnosis with dominant-threshold rule.
5. Disable recovery actions.

## Reporting Requirements

Every experiment should report:

- dataset and sequence
- backend configuration
- fault type and severity
- random seed
- health curves
- diagnosis posterior curves
- recovery timeline
- ATE/RPE before and after degradation
- failure and recovery events

## Current Status

This protocol is implemented as a research plan and API skeleton. Full dataset replay, OpenVINS execution, and statistically meaningful results are planned work.
