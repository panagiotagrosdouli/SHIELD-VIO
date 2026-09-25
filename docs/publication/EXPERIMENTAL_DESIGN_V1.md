# SHIELD-VIO Experimental Design v1

## Purpose

This document turns the literature-defined novelty boundary into a falsifiable publication experiment. Each claimed contribution must correspond to a frozen comparison, a stored artifact, and run-level uncertainty.

Central question:

> Given only estimator-health information available up to time t, can SHIELD-VIO estimate the held-out probability that a persistent observable VIO failure will begin within a specified future horizon, remain useful under predeclared distribution shifts, and improve downstream safety–utility outcomes when its risk estimate drives a stateful protective policy?

This formulation does not claim novelty for VIO health monitoring, generic failure prediction, short-horizon risk assessment, degradation benchmarking, or fallback switching in isolation. Those capabilities have direct prior art, including SUPER, VIO integrity monitoring, SM/VIO, introspective SLAM, and failure-centric VIO benchmarks \cite{gaus2025super,wang2019pdrviointegrity,joshi2023smvio,naveed2022introspectiveslam,zhu2026failurebenchmark}.

## 1. Confirmatory contribution claims

### C1 — Persistent future-event prediction

SHIELD-VIO predicts whether a versioned, persistent observable localization-failure event will begin in the interval (t, t + tau], using only causal health history available by t.

Primary horizons: 0.5, 1.0, 2.0, 3.0, and 5.0 seconds.

The key distinction from current-state monitoring is explicit failure-onset semantics, persistence, censoring, recovery/event handling, and multiple fixed warning horizons.

### C2 — Split-safe probability calibration

A detector score is called a probability only after a calibration mapping is fitted on complete calibration sequences that are disjoint from detector training, operating-threshold selection, and final test sequences.

Reliability is evaluated with Brier score, NLL, ECE/MCE, calibration slope/intercept, reliability source data, and empirical conformal coverage where applicable.

### C3 — Reliability under distribution shift

The paper measures how discrimination, probability calibration, uncertainty coverage, and confidence-under-error change under predeclared unseen conditions. Shift detection and shift response are evaluated separately.

### C4 — Protective decision utility

Identical prediction streams are replayed through fixed policy variants. The paper measures both protective benefit and utility cost: unsafe localization-dependent exposure, completion, delay, intervention rate, recovery, and unnecessary stopping.

The contribution is empirical supervisory protection, not a formal safety guarantee.

## 2. Primary scientific contrasts

All methods use the same run units, labels, horizons, degradation events, and false-alarm constraint.

### Contrast A — SHIELD-VIO versus conventional reactive health signals

Compare against covariance trace, largest covariance eigenvalue, innovation NIS, feature count / tracking health, rolling normalized innovation, and a frozen moving-average multi-health score.

Purpose: determine whether causal multi-signal history adds predictive value beyond conventional current-health indicators.

### Contrast B — SHIELD-VIO versus the closest-prior-art short-horizon risk comparator

Preferred comparator: the official SUPER implementation/configuration if executable source and compatible outputs are available \cite{gaus2025super}.

Fallback comparator: a clearly named SUPER_STYLE_RISK reproduction implementing only the publicly specified class of inputs/logic that can be reproduced faithfully: estimator uncertainty, residual magnitude, geometric conditioning when available, and short-horizon temporal trends. A reproduction must never be presented as the original SUPER implementation.

Purpose: test whether explicit persistent-event targets, held-out probability calibration, shift evaluation, and downstream decision analysis add value beyond a short-horizon VIO risk signal.

### Contrast C — Calibrated versus uncalibrated risk

For the same base detector and test runs, compare raw risk with the frozen primary calibrator.

Purpose: isolate probability calibration from discrimination.

### Contrast D — Calibrated risk versus calibrated plus stateful protection

Replay the same calibrated prediction table through fixed policy variants.

Purpose: isolate the value and cost of statefulness, hysteresis, shift response, and recovery.

## 3. Datasets and split discipline

### EuRoC MAV

Use the frozen complete-sequence split:

- train: MH_01_easy, MH_02_easy, V1_01_easy;
- calibration: MH_03_medium, V1_02_medium;
- validation: V2_01_easy, V2_02_medium;
- test: MH_04_difficult, MH_05_difficult, V1_03_difficult, V2_03_difficult.

All degraded variants inherit the parent sequence split.

### TUM-VI

The official export inventory, sequence IDs, usable ground-truth intervals, checksums, and split registry must be frozen before confirmatory TUM-VI model fitting or test evaluation.

Until the registry is frozen, TUM-VI executions are smoke or exploratory evidence only.

### Leakage rules

- preprocessing/model fitting: training sequences only;
- probability/conformal calibration: calibration sequences only;
- model variant, threshold, history length, and policy parameter selection: validation sequences only;
- confirmatory reporting: sealed test and shifted-test sequences only;
- no complete physical sequence may cross roles through another degradation, seed, or estimator;
- degradation metadata and ground truth are forbidden runtime features;
- future samples are forbidden from feature construction;
- test prevalence cannot alter class weights or thresholds.

## 4. Estimator coverage

### Required backend A — internal ESKF

The internal estimator provides a controlled integration path and rich health signals. Its results support protocol validation and estimator-specific conclusions.

### Target backend B — OpenVINS

OpenVINS is the preferred established estimator adapter \cite{geneva2020openvins}. The adapter must emit the standardized state/health contract and mark unavailable signals explicitly.

A cross-estimator claim is permitted only after both backends execute the same registered public sequences and the reduced-signal experiment is reported.

If OpenVINS integration is not completed, estimator generalization remains untested and the paper must restrict its conclusions accordingly.

## 5. Experimental unit and benchmark matrix

Independent experimental unit:

dataset × sequence × estimator × degradation_condition × seed

Prediction horizon is a repeated analysis dimension, not an independent physical run when one run produces targets for several horizons.

Frames and overlapping windows are never independent replicates.

The confirmatory benchmark must contain at least 20 independent test sequence-degradation run units and retain conditions that produce no failure.

For stochastic transformations, event schedules and seeds are paired across methods.

## 6. Degradation families

Development examples defined by frozen configuration include darkness, contrast reduction, additive image noise, accelerometer noise, gyroscope noise, and mild packet loss.

Validation-only examples include overexposure, feature dropout, and bias drift.

Shifted-test examples include occlusion, frame dropout, saturation, axis failure, combined degradations, cross-dataset transfer, and estimator shift.

Exact severity values, durations, event schedules, and family assignments must be frozen before test execution.

## 7. Failure-event contract

The primary failure definition is versioned and independent of degradation metadata.

For each timestamp:

1. compute permitted observable failure criteria offline;
2. apply the frozen persistence duration;
3. construct unique onset/offset event IDs;
4. apply recovery/merge semantics;
5. create horizon labels for (t, t + tau];
6. censor samples whose future interval cannot be observed.

Active-failure timestamps are not relabeled as new future onsets. Sensitivity definitions are stored separately.

## 8. Predictor set

### Heuristic / conventional

- H-COV-TRACE
- H-COV-EIG
- H-NIS
- H-NORM-INNOV
- H-FEATURE
- H-TRACK
- H-MA

### Learned / risk

- P-LOG: logistic regression;
- P-GBT: compact gradient-boosted tree baseline;
- P-SUPER when official execution is possible;
- P-SUPER-STYLE otherwise, explicitly labeled reproduction;
- P-PROPOSED: compact interpretable multi-signal temporal predictor.

A temporal neural baseline is secondary and included only if it can be trained under the same split/tuning budget without displacing the core comparisons.

All methods export a scalar score with declared direction. Only calibrated outputs may be called probabilities.

## 9. Primary feature representation

The proposed model receives causal summaries of available signal families:

- covariance magnitude, conditioning, eigenstructure, and growth;
- innovations / NIS and rolling exceedance statistics;
- feature, correspondence, inlier, survival, and update-starvation diagnostics;
- IMU gaps/loss/saturation and inertial summaries where available;
- estimator tracking/reset/reinitialization state where available;
- causal slopes, variability, recent extrema, and missingness indicators.

Every feature row records source timestamp range, unit, availability, and schema version.

The strongest test of temporal value is the current-only ablation using the same model class.

## 10. Calibration design

Primary calibration method: Platt scaling unless validation evidence frozen before test supports a different predeclared primary choice.

Secondary methods: raw score, isotonic regression, temperature scaling when logits are meaningful, and split-conformal bounds/nonconformity analysis.

Calibration is fitted only on complete calibration sequences. Test reliability is never used to select a calibrator.

Primary calibration endpoint: paired per-run Brier-score difference.

Required secondary diagnostics: NLL, fixed-bin ECE/MCE, adaptive ECE, calibration slope/intercept, reliability source table, and empirical conformal coverage/width where applicable.

## 11. Operating point

For methods requiring a binary action threshold, select the threshold only on validation sequences.

Primary operating constraint:

false alarms <= 0.2 per eligible minute

Within that constraint, minimize missed failure events; ties use earlier median warning.

The threshold and validation counts are serialized before test execution.

A matched-false-alarm analysis compares methods at a common operational budget.

## 12. H1 — Future failure warning

Primary comparison: P-PROPOSED versus the strongest validation-selected deployable single-signal baseline, P-SUPER or P-SUPER-STYLE, P-LOG, and P-GBT.

Co-primary descriptive outcomes:

- event recall at the frozen false-alarm constraint;
- median lead time among timely detected events;
- AUPRC.

No single metric hides a trade-off. If a method has longer lead time but substantially worse recall or false alarms, report the trade-off.

Additional outcomes: AUROC, precision, F1, missed events, false alarms/min, warning-time distribution, performance by horizon, and failure-family stratification.

## 13. H2 — Multi-signal and temporal contribution

Primary contrast: all causal signals versus strongest validation-selected single family.

Primary endpoint: paired per-run AUPRC difference.

Required ablations: current-only; visual only; IMU only; consistency/NIS only; covariance only; visual + consistency; visual + inertial; leave-one-family-out; reduced black-box signal set.

Lead time is specifically inspected for the current-only ablation because a temporal model should demonstrate advance warning rather than merely classify already-bad frames.

## 14. H3 — Calibrated risk and domain shift

### In-domain calibration

Compare raw versus calibrated risk on sealed in-domain test runs.

Primary endpoint: paired Brier-score difference.

### Shift degradation

Evaluate the frozen predictor/calibrator unchanged on unseen degradation family, unseen severity, cross-dataset transfer, and estimator shift when a second estimator is available.

Report change from in-domain to shift in Brier, NLL/ECE, AUPRC, false-negative rate, confidence on errors, and conformal coverage.

Shift-state detection metrics are reported separately from predictive metrics.

## 15. H4 — Protective utility

Replay identical predictions through:

1. A-NONE: no shield;
2. A-COV: reactive covariance threshold;
3. A-RISK: SUPER/SUPER-style risk threshold where available;
4. A-RAW: raw proposed detector threshold;
5. A-CAL: calibrated detector threshold;
6. A-HYST: calibrated detector plus hysteresis/dwell;
7. A-FULL: calibrated plus shift-aware plus recovery policy.

Estimated state drives the controller. Ground truth scores outcomes only.

Primary safety–utility view: unsafe localization-dependent exposure against mission completion, not safety in isolation.

Secondary outcomes: unnecessary interventions, intervention precision/delay, recovery success, halt rate, boundary violations, mission duration, path overhead, and state-transition count.

Several frozen cost vectors and a Pareto analysis are reported; no single hand-picked utility function determines the conclusion.

## 16. Statistical analysis

The unit of inference is one scalar endpoint per complete run unit.

Primary uncertainty is paired grouped bootstrap over matched run keys.

Minimum report:

- paired unit count;
- dropped/unavailable units;
- mean and median paired effect;
- 95% interval;
- metric direction;
- bootstrap seed.

Frames never enter the bootstrap as independent observations.

Exploratory inference may use paired permutation/Wilcoxon tests on run summaries, McNemar on paired event decisions, and Holm correction over declared secondary families.

P-values are secondary to effect sizes and uncertainty intervals.

## 17. Required paper tables

1. Dataset / split / evidence inventory.
2. Main future-failure prediction: covariance, NIS, feature, logistic, GBT, SUPER/SUPER-style, proposed.
3. Horizon analysis at 0.5 / 1 / 2 / 3 / 5 s.
4. Calibration: raw, Platt, isotonic, temperature if applicable, conformal.
5. Domain shift: ID versus each shifted condition.
6. Protective policy: no shield, covariance, risk, raw, calibrated, hysteresis, full.
7. Ablations / estimator transfer / runtime.

Every table includes exact run/event denominators and evidence tier.

## 18. Required core figures

1. system/causal pipeline;
2. health → calibrated risk → failure-onset timeline;
3. PR curves for main methods;
4. lead-time distribution at matched false-alarm budget;
5. reliability diagram raw versus calibrated;
6. performance versus prediction horizon;
7. in-domain versus shifted calibration;
8. safety–utility Pareto curve;
9. closed-loop intervention/recovery example;
10. ablation effects with paired intervals.

Every figure has machine-readable source data and an evidence-tier label.

## 19. Falsification / qualification rules

The paper must weaken or remove a contribution claim when:

- useful warning and discrimination do not improve relative to the strongest relevant comparator;
- calibration does not improve proper scoring/reliability on held-out runs;
- apparent shift robustness comes only from conservative stopping with severe completion loss;
- estimator transfer fails and only the internal ESKF is evaluated;
- results depend strongly on one failure threshold without surviving declared sensitivity;
- paired intervals are too wide for the intended directional statement;
- the SUPER-style reproduction is too approximate for a direct method comparison.

Negative or mixed findings remain reportable; test runs are not retuned after inspection.

## 20. Execution order

1. Merge/validate publication-grade run-level aggregation.
2. Produce one real EuRoC PUBLIC_DATASET_SMOKE.
3. Produce one real TUM-VI PUBLIC_DATASET_SMOKE.
4. Freeze TUM-VI inventory/splits/checksums.
5. Implement/export the complete primary failure observables.
6. Implement P-GBT and the SUPER/SUPER-style comparator interface.
7. Complete split-safe detector/calibrator/threshold orchestration.
8. Run EuRoC development partitions; freeze model/config choices.
9. Seal test configuration and run EuRoC confirmatory matrix.
10. Run TUM-VI and cross-dataset shift evaluation.
11. Integrate OpenVINS and repeat the reduced-signal/transfer subset.
12. Run closed-loop policy replay and ablations.
13. Generate all tables/figures from run-level artifacts.
14. Freeze manuscript claims only after artifact-to-claim audit.

## 21. Minimum versus stronger paper package

### Minimum defensible package

- real EuRoC + TUM-VI execution;
- sequence-disjoint train/calibration/validation/test protocol;
- heuristics + logistic + GBT + SUPER/SUPER-style + proposed;
- multi-horizon persistent-event targets;
- held-out calibration;
- unseen-condition evaluation;
- run-level paired bootstrap;
- closed-loop policy comparison.

### Stronger package

Everything above plus OpenVINS transfer, official SUPER execution rather than reproduction, cross-estimator detector transfer, richer shift/conformal analysis, and extensive runtime/reduced-signal studies.

The manuscript must state which package was actually completed.
