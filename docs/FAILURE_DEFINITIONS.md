# Failure definitions

SHIELD-VIO separates **degradation labels** from **estimator failure labels**. A degradation event describes an injected sensor condition. It is not automatically considered estimator failure.

The executable labeler in `shield_vio/evaluation/failure_labels.py` evaluates observable quantities against versioned thresholds.

| Failure kind | Default criterion | Unit |
|---|---:|---|
| Position error | `position_error > 1.0` | m |
| Relative pose error | `relative_pose_error > 0.5` | m |
| Covariance instability | `trace(P) > 2.0` | state-dependent squared SI units |
| Innovation inconsistency | `NIS > 11.345` | dimensionless |
| Tracking loss | invalid tracking or fewer than 12 features | count |
| Bias instability | combined bias norm above `0.5` | sensor-dependent SI units |
| Unsafe navigation | obstacle clearance below `0.25` | m |

Thresholds are experiment configuration, not universal physical constants. Every experiment must preserve the exact threshold set in its manifest. Dataset studies must define thresholds before inspecting test-sequence outcomes.

## Horizon labels

A detector predicting failure within a future horizon must derive its target from these sample-level labels by marking a sample positive when any failure occurs in `(t, t + horizon]`. Training, calibration, and test episodes must remain disjoint.

## Evidence classification

- Label implementation: **Implemented**
- Threshold validation on synthetic scenarios: **Synthetic Validation**
- Public-dataset threshold validation: **Pending Dataset Execution**
- Hardware safety threshold validation: **Hardware Validation Required**
