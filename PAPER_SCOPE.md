# SHIELD-VIO Paper Scope

## Frozen paper direction

**Working title:** *SHIELD-VIO: Calibrated Early Failure Prediction and Protective Control for Visual–Inertial Odometry under Sensor Degradation and Domain Shift*

**Central research question:** Can an estimator-aware, calibrated failure-prediction layer detect visual–inertial localization failure early enough to improve downstream navigation safety under sensor degradation and domain shift?

SHIELD-VIO is an estimator-introspection and navigation-protection framework. It is not presented as a new state-of-the-art VIO estimator. The internal ESKF is an experimental backend used for controlled integration and debugging; established VIO backends are required before making estimator-agnostic claims.

## Primary claim

The confirmatory paper will test one coherent claim:

> A causal, multi-signal health representation combining visual, inertial, innovation, covariance, consistency, missingness, and temporal diagnostics predicts impending VIO failure earlier and more reliably than predeclared single-signal heuristics; held-out calibration and explicit shift handling make those predictions more useful to a stateful protective navigation policy.

The claim is represented by one causal chain:

`sensor degradation -> estimator-health change -> future failure risk -> calibrated uncertainty -> protective action -> downstream outcome`

The contribution is therefore indivisible into four connected components:

1. a backend-neutral estimator-health representation;
2. causal prediction of failure within a future horizon;
3. held-out probability calibration plus independently evaluated shift awareness;
4. a stateful protective policy evaluated in a closed navigation loop.

## Prediction target

For timestamp `t` and horizon `tau`, the deployable target is

`P(first observable failure onset in (t, t + tau] | health data with timestamp <= t)`.

Primary horizons are 0.5, 1.0, 2.0, 3.0, and 5.0 seconds. The current frame is excluded from the future interval. Preprocessing, rolling features, missing-value treatment, model fitting, calibration, conformal fitting, and threshold selection must use only the permitted split and information available by `t`.

## Confirmatory hypotheses and endpoints

| ID | Hypothesis | Primary endpoint | Key secondary endpoints |
|---|---|---|---|
| H1 | Estimator-health signals warn before observable failure. | Median event lead time at a predeclared false-alarm constraint | Event recall, false alarms/min, AUROC, AUPRC |
| H2 | The full multi-signal representation outperforms every individual signal family. | Paired sequence-condition difference in AUPRC | Lead time, missed-event rate, precision at operational recall |
| H3 | Held-out calibration improves probabilistic reliability and decisions. | Paired difference in Brier score | NLL, ECE, slope/intercept, conformal coverage, policy utility |
| H4 | Explicit shift awareness limits overconfidence under unseen conditions. | Change in false-negative rate and calibration error under shift | Selective risk, coverage, shift delay, confidence under error |
| H5 | A stateful calibrated shield improves downstream safety-utility trade-offs. | Unsafe localization-dependent exposure at matched mission completion | Avoided unsafe actions, unnecessary interventions, recovery success, utility |

AUPRC is reported alongside AUROC and is the primary discrimination metric because failure windows are expected to be imbalanced. Event-level results are primary; frame-level results are descriptive because adjacent windows are correlated.

## Experimental unit and inference

The experimental unit is a complete `dataset x sequence x estimator x degradation condition x seed` run. Frames are never treated as independent experimental replicates. The test statistic is computed per run and aggregated using sequence-grouped or condition-grouped paired bootstrap intervals. Comparisons use the same sequences, seeds, labels, horizons, and degradation events.

The primary operating threshold is chosen on validation data to minimize missed events subject to at most 0.2 false alarms per minute. This value is frozen before confirmatory test inspection; alternatives are sensitivity analyses.

## Evidence tiers

| Tier | Meaning | Permitted claim language |
|---|---|---|
| Analytical/unit | Numerical invariants or fixture behavior | “passes unit/analytical validation” |
| Synthetic | Generated trajectories and degradations | “demonstrated in the declared synthetic setting” |
| Public-dataset pipeline smoke | Real sensor files execute end to end, without confirmatory split size | “executed on named public sequence”; no hypothesis confirmation |
| Public-dataset evaluation | Frozen sequence splits and stored run artifacts | Dataset- and protocol-bounded empirical claim |
| Closed-loop simulation | Estimator errors affect a controller and safety boundaries | Simulator-bounded protective-utility claim |
| Physical robot | Physically executed, recorded experiment | Hardware-bounded claim only |

## Repository audit at revision `f46355a`

| Area | Existing implementation | Current evidence | Paper-critical gap |
|---|---|---|---|
| Internal estimator | ESKF propagation/update, IMU runner, visual rotation, stereo-PnP frontend | Unit and mocked sequence tests | No established external estimator adapter; public trajectory quality not stored on `main` |
| EuRoC data | Stream, calibration, ground-truth, and runner support | Filesystem fixtures; workflows can download MH_01 | No committed completed paper experiment or strict multi-sequence benchmark artifact |
| TUM-VI data | Layout discovery adapter | Mocked filesystem fixture only | No executable sequence runner or public-data result |
| Health signals | Covariance trace/condition number/NIS plus separate visual update diagnostics | Unit tests and synthetic outputs | No unified timestamped health table, causal history builder, missingness schema, or feature provenance |
| Failure labels | Observable sample-level rules | Unit tests | No persistence, onset/event construction, horizon target, or primary-vs-sensitivity label separation |
| Detectors | Rule score and dependency-light logistic regression | Controlled-array tests | No benchmark orchestration, tree/temporal baselines, event thresholds, or serialized models |
| Calibration | Brier, NLL, ECE/MCE; split-conformal scalar interval | Controlled-array tests | No Platt/isotonic/temperature implementation, calibration split enforcement, slope/intercept, or reliability artifact from public data |
| Domain shift | Rolling standardized-distance state machine | Controlled-array test | No fitted reference artifact, public shifted evaluation, delay metric, or demonstrated effect on risk/policy |
| Shield | Stateful decision logic with dwell, stale-sensor override, and recovery request | Unit and synthetic tests | Simplified state/action set; no cost sweep or link to evaluated calibrated predictions |
| Closed loop | Point-navigation state and shield-controlled progress | Unit test | No obstacles, corridor boundaries, estimator-error coupling, mission benchmark, or paired policy comparison |
| Statistics | Descriptive summaries and binary counts | Unit tests | Normal-theory CI is not sequence bootstrap; AUROC/AUPRC, event metrics, paired tests, and effect sizes incomplete |
| Reproducibility | CI, deterministic synthetic pipeline, partial manifests | 122 tests passed locally file-by-file; Ruff passed | No top-level paper command, typed complete manifest, dataset checksums registry, or leakage guard suite |
| Manuscript | Short abstract/contribution/future-work notes | Documentation only | No complete paper narrative or generated-table placeholders |

## Explicit non-claims

The project does not claim formal safety guarantees, certified autonomy, production readiness, state-of-the-art trajectory accuracy, universal calibration, real-time operation, ROS 2 validation, hardware validation, or improved mission safety before the corresponding executable evidence exists.

In particular:

- a diagnostic score is not called a probability unless a calibration method was fitted on a disjoint calibration split;
- injected degradation metadata is never a failure label or deployable feature;
- ground truth is permitted for offline label construction and evaluation only;
- an oracle may appear only as a non-deployable upper bound;
- a single EuRoC pipeline smoke run cannot confirm H1-H5;
- shift detection is not described as repairing calibration unless an evaluated policy demonstrates that result;
- public-data sensor-health summaries are not equivalent to estimator failure-prediction evidence;
- the internal ESKF is not used to claim superiority over mature VIO estimators.

## Paper-readiness gates

The paper narrative may move from placeholders to numerical claims only after all of the following hold:

1. EuRoC and TUM-VI have checksum-identified public sequence executions.
2. Train, calibration, validation, test, and shifted-test sequences are disjoint.
3. At least 20 independent sequence-degradation evaluation conditions exist.
4. Failure definitions, persistence, horizons, warning deadlines, and recovery deadlines are frozen.
5. Three basic health heuristics, logistic regression, a tree method, and the proposed multi-signal method share identical labels and splits.
6. Calibration is fitted only on calibration data and thresholds only on validation data.
7. Event-level metrics, grouped confidence intervals, paired effects, and exact sample counts are generated from stored predictions.
8. The full closed-loop policy is compared with no shield and four simpler protection policies.
9. Required ablations and sensitivity analyses execute from frozen configurations.
10. Every reported number resolves to a manifest, raw prediction table, configuration hash, and Git commit.

