# Observable Failure Definition

## Design principles

Failure labels describe estimator or navigation behavior, not the experimental intervention. Degradation family, severity, seed, dataset name, and injected event boundaries may be used for stratified analysis but are forbidden as deployable predictor inputs and cannot make a frame a failure by themselves.

The primary label is chosen to avoid circular evaluation. Covariance and NIS are candidate predictor signals and baselines; therefore they are not part of the primary trajectory-behavior failure label. They are retained as secondary consistency-failure definitions for sensitivity analysis.

## Time and event notation

Let `t_i` be a strictly increasing estimator-output timestamp. Each raw criterion produces a boolean exceedance `c_k(t_i)`. A criterion becomes a persistent failure at the first time `t_onset` for which its exceedance has remained continuously true for at least its declared persistence duration. Short gaps up to one sample interval may be bridged only if the configuration declares the gap tolerance.

Failures separated by less than the event merge gap belong to the same event. The default merge gap is 1.0 s. The event ends only after all triggering criteria have remained false for the recovery-confirmation duration.

## Primary estimation-failure definition

An estimation failure begins when any of the following observable criteria reaches persistence:

| Criterion | Default threshold | Persistence | Required source |
|---|---:|---:|---|
| SE(3)-aligned position error | `> 1.0 m` | `0.5 s` | Offline ground truth and estimator pose |
| Orientation geodesic error | `> 15 deg` | `0.5 s` | Offline ground truth and estimator attitude |
| Translational relative-pose error over 1 s | `> 0.5 m` | `0.5 s` | Offline ground truth and estimator pose |
| Rotational relative-pose error over 1 s | `> 10 deg` | `0.5 s` | Offline ground truth and estimator attitude |
| Invalid/non-finite pose or covariance | Any occurrence | Immediate | Estimator output validity |
| Output starvation | No valid pose for `> 0.5 s` | `0.5 s` | Output timestamps |
| Terminal tracking loss | Backend declares terminal/lost | Immediate | Tracking state |
| Visual-update starvation while motion is observed | No accepted update for `> 0.5 s` | `0.5 s` | Update events and IMU motion gate |
| Estimator reset or unrecovered relocalization | Declared event | Immediate | Backend event log |

### V2 motion-gate and applicability freeze

The current confirmatory configuration is `configs/paper/failure_primary_v2.yaml` (`SHIELD_VIO_FAILURE_V2`). V2 was frozen before confirmatory test inspection after the V1 real-data availability audit exposed two underspecified cases.

For `visual_update_starvation_while_motion`, motion is detected causally from the trailing 0.25 s IMU window. Motion is present when either gyroscope-norm RMS exceeds 0.05 rad/s or RMS deviation of accelerometer norm from 9.81 m/s² exceeds 0.30 m/s². These are predeclared engineering thresholds, not values selected on EuRoC test performance, and they are included in the sensitivity grid.

Backend-declared criteria distinguish missing instrumentation from non-applicability. A criterion may be `NOT_APPLICABLE` only when the run explicitly declares that the backend does not implement the corresponding semantic state or stream and the frozen configuration permits that policy. Missing fields, ambiguous support, or absent instrumentation remain `NOT_OBSERVABLE` and block primary-label construction. `NOT_APPLICABLE` criteria are omitted from the union rather than silently set healthy.

Ground truth is privileged and is used only by offline label construction and evaluation. It must never enter the health feature table, runtime detector, calibrator, shift detector, or shield policy.

The main paper reports the union of these primary criteria and also reports criterion-specific performance. An onset caused only by ground-truth error remains evaluable even if deployable health diagnostics appear nominal; this is necessary to measure silent failure.

## Secondary consistency-failure definition

For sensitivity analysis, not the primary H1/H2 endpoint, a consistency failure may be declared when one of these conditions persists:

| Criterion | Threshold rule | Persistence |
|---|---|---:|
| NIS inconsistency | Above the predeclared chi-square quantile for the innovation dimension | 0.5 s |
| Rolling NIS exceedance | More than 80% of updates exceed the quantile in a 1 s window | 0.5 s |
| Covariance numerical invalidity | Non-finite, non-symmetric beyond tolerance, or materially non-PSD | Immediate |
| Covariance runaway | Trace above a training-independent engineering limit | 0.5 s |
| Tracking collapse | Feature count below backend-specific minimum or tracking state lost | 0.5 s |
| Bias instability | Bias magnitude above a sensor-specific admissible bound | 0.5 s |

When a required signal is unavailable from a backend, the default state is `NOT_OBSERVABLE`; it is not silently treated as healthy. `NOT_APPLICABLE` is reserved for an explicitly declared unsupported backend semantic under the frozen V2 applicability policy.

## Navigation-critical failure

A navigation-critical failure occurs when the localization error or invalidity causes, or would cause in the counterfactual no-shield replay, at least one of these outcomes:

- a control command is issued while localization exceeds its admissible error;
- estimated corridor clearance is positive while ground-truth clearance is below the safety margin;
- the robot crosses a declared boundary;
- controller tracking error exceeds its safety limit;
- the safe reaction deadline is missed;
- recovery becomes infeasible before intervention;
- distance traveled under invalid localization exceeds the declared limit.

Each scenario configuration must declare the admissible localization error, safety margin, controller-error limit, reaction deadline, recovery deadline, and maximum invalid-localization distance.

## Future-horizon target

For horizon `tau`, the binary target at time `t_i` is

`y_tau(t_i) = 1` if a first failure onset exists in `(t_i, t_i + tau]`, otherwise `0`.

Rules:

1. Samples at or after an active failure onset are excluded from early-warning discrimination unless a separately named ongoing-failure task is evaluated.
2. The current time is excluded so a detector is not rewarded for recognizing a failure that has already begun.
3. If observation ends before `t_i + tau`, the sample is censored and excluded unless the run ended because of a recorded terminal failure.
4. Targets are computed after features, but the target builder receives only timestamps and offline event onsets; no target field may enter feature generation.
5. Horizons are evaluated independently at 0.5, 1.0, 2.0, 3.0, and 5.0 s.

## Warning, reaction, and recovery timing

For an event with onset `t_f`:

- detection time `t_d` is the first threshold crossing after the previous event/refractory period;
- lead time is `t_f - t_d` and is positive only for an early detection;
- the warning deadline is `t_f - T_reaction`;
- an intervention is timely when `t_d <= t_f - T_reaction`;
- recovery succeeds when the estimator returns to valid operation before `t_f + T_recovery` and remains valid for the recovery-confirmation duration.

Defaults for the first benchmark are `T_reaction = 0.5 s`, `T_recovery = 5.0 s`, and recovery confirmation `1.0 s`. These defaults require sensitivity analysis and are not universal safety limits.

## Event-level scoring

- One or more alerts in the event warning window count as one detected event.
- Additional alerts inside the same warning window do not increase true positives.
- An alert outside all warning windows is a false alarm.
- False alarms are normalized by valid non-warning monitoring time in minutes.
- A failure without a timely alert is a missed event, even if an alert occurs after onset.
- Event precision, recall, F1, missed-event rate, and lead-time distribution are primary.
- Frame-level metrics are secondary and report positive/negative window counts.

## Required configuration fields

Every experiment manifest must record:

- label schema and version;
- each active criterion and threshold;
- persistence and gap tolerance;
- event merge gap and recovery confirmation;
- prediction horizon;
- warning/reaction and recovery deadlines;
- trajectory alignment and association policy;
- privileged input fields used only for labels;
- excluded/censored sample counts by reason;
- number of events by criterion and union label.

## Relationship to current code

`shield_vio/evaluation/failure_definition.py` implements versioned criterion thresholds, timestamp-aware persistence, event merging/recovery timing, oracle-field rejection, and future-horizon targets. `shield_vio/evaluation/primary_observables.py` exports the real-run observable quantities and distinguishes observability from applicability. The older `shield_vio/evaluation/failure_labels.py` remains a sample-level diagnostic helper and is not the paper’s definitive event-label path.

Primary composite labels must use the V2 observable table and must censor samples for which any applicable criterion is unavailable. They must not substitute the older smoke-only position-error label or the sample-level diagnostic helper.

## Sensitivity definitions

The confirmatory definition above is frozen before final test inspection. Sensitivity reports vary position error (`0.5, 1.0, 2.0 m`), orientation error (`10, 15, 25 deg`), persistence (`0.25, 0.5, 1.0 s`), RPE interval, event merge gap, and motion gating. Conclusions must be reported as fragile if their direction changes under plausible label settings.

