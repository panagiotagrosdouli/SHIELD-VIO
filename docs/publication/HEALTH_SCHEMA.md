# SHIELD-VIO Canonical Health Schema

## Scope

Phase B defines `SHIELD_VIO_HEALTH_V1`, a typed health representation for causal impending-failure prediction. It structurally separates deployable health signals from evaluation-only diagnostics. The existing `shield_vio/features/health_vector.py` remains supported; the new `shield_vio/health/` package provides the publication-facing schema and role firewall.

## Structure

`HealthSample` contains timestamp and optional sequence/estimator identifiers plus four typed groups:

- `visual`: online visual/tracking diagnostics;
- `inertial`: online IMU and bias diagnostics;
- `estimator`: online estimator diagnostics;
- `evaluation`: explicitly non-deployable ground-truth diagnostics.

Every signal is represented as `HealthValue(value, valid, source_timestamp_ns)`. Missing values remain explicit and are not silently imputed. A missing value has `valid=False`, `value=None`, and no source timestamp.

## Feature roles

The registry uses machine-readable roles:

- `DEPLOYABLE_FEATURE` — allowed in the predictor matrix;
- `TARGET` — prediction label only;
- `GROUPING_METADATA` — dataset/sequence/condition/seed information;
- `EXPERIMENT_ORACLE` — injected degradation metadata or other experiment-only information;
- `EVALUATION_ONLY` — ground truth, NEES, trajectory error, and similar diagnostics.

`flatten_deployable()` emits only `DEPLOYABLE_FEATURE` fields and a deterministic `__missing` indicator for each optional value. Evaluation-only fields cannot enter through the normal flattening API.

## Causality

Every valid `HealthValue` records its maximum source timestamp. Construction fails if any source timestamp is later than the `HealthSample.timestamp_ns`.

Temporal features are built by `build_trailing_features()` using timestamp windows `[t-W, t]` only. Default supported windows are 0.25, 0.5, 1.0, and 2.0 seconds. The implementation uses timestamps rather than sample indices and therefore supports irregular streams, dropped packets, and missing updates.

For each signal/window the temporal layer provides trailing mean, standard deviation, minimum, maximum, causal slope, rate of change, persistence duration, missing fraction, and time since last valid observation. No centered window, future interpolation, or future smoothing is permitted.

## Ground-truth separation

`EvaluationDiagnostics` contains ground-truth position error, ground-truth rotation error, and NEES. These fields are never returned by `flatten_deployable()`. Ground truth remains permitted only for offline label construction and evaluation.

## Backend neutrality

All publication-facing health fields are optional. An external estimator may provide only the signals it actually exposes; unavailable values are marked missing rather than fabricated. The registry records source, units, ground-truth requirement, and backend dependency for every canonical field.

## Provenance

Schema version: `SHIELD_VIO_HEALTH_V1`.

Feature-level provenance is defined by `FeatureSpec`: name, category, unit, role, source, ground-truth requirement, backend dependency, and missing-value semantics. Row-level provenance is carried by prediction timestamp and per-value source timestamp.

## Scientific claim boundary

This schema supports the statement: **“Causal prediction-dataset infrastructure implemented and tested.”** It does not establish that early failure prediction is accurate, calibrated on public data, estimator-agnostic, or safety-improving.
