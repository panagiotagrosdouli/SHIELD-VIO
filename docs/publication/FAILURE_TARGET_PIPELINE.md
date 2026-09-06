# SHIELD-VIO Failure-Target Pipeline

## Definition

The Phase B target is

`y_tau(t) = 1{T_failure in (t, t + tau]}`

where `T_failure` is the first qualifying observable failure onset and the feature state at `t` may use only information available at or before `t`.

The frozen primary configuration is `configs/paper/failure_primary_v1.yaml` with schema version `SHIELD_VIO_FAILURE_V1`. Primary horizons are 0.5, 1.0, 2.0, 3.0, and 5.0 seconds.

## Pipeline

Observable estimator/navigation quantities
→ frozen criterion threshold evaluation
→ criterion-specific timestamp-aware persistence
→ union of primary criteria
→ recovery confirmation and event merge timing
→ deterministic event IDs/onsets/offsets
→ independent future-horizon targets.

The current time is excluded from the future interval. A failure that has already become active at `t` therefore does not count as an impending-failure positive at `t`. Active-failure samples are ineligible for the early-warning task, and incomplete run tails are censored when `t + tau` exceeds the available observation interval.

## Primary criteria

The checked-in configuration mirrors `FAILURE_DEFINITION.md`: aligned position error, orientation geodesic error, 1 s translational and rotational relative-pose error, invalid/non-finite state, output starvation, terminal tracking loss, visual-update starvation while motion is observed, and estimator reset/unrecovered relocalization. Ground-truth-dependent criteria are explicitly marked as such and are restricted to the offline label path.

Covariance/NIS consistency criteria are not part of this primary definition because they are candidate deployable predictors/baselines. They belong to separately versioned sensitivity definitions.

## Oracle firewall

`evaluate_failure_criteria()` accepts only criterion names declared by the frozen definition. Fields containing degradation/corruption/severity/seed/injected/oracle semantics are rejected. Degradation metadata may still exist in experiment grouping tables but cannot define a failure event.

## Determinism

For identical timestamps, observable criterion arrays, and configuration, event IDs and target arrays are deterministic. Event IDs include the failure-definition schema version and event onset timestamp.

## Claim boundary

Implemented evidence language: **“Observable failure-event construction implemented and unit tested.”**

Not permitted from Phase B alone: **“Early failure prediction demonstrated”**, **“failure prediction validated”**, or any numerical performance claim.
