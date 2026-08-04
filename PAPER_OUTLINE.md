# Paper Outline and Artifact Placeholders

## Working title

*SHIELD-VIO: Calibrated Early Failure Prediction and Protective Control for Visual–Inertial Odometry under Sensor Degradation and Domain Shift*

No numerical result is entered manually. Placeholders are populated by reporting scripts from checksum-verified aggregate artifacts.

## 1. Abstract

State the localization-failure problem, future-horizon formulation, estimator-health approach, public datasets/estimators actually executed, calibration/shift/policy evaluation, primary held-out results, and bounded conclusion.

Placeholders: `{{N_DATASETS}}`, `{{N_SEQUENCES}}`, `{{N_FAILURE_EVENTS}}`, `{{METRIC_MAIN_AUPRC}}`, `{{METRIC_MEDIAN_LEAD_TIME}}`, `{{METRIC_BRIER_CHANGE}}`, `{{METRIC_UNSAFE_EXPOSURE_CHANGE}}`.

## 2. Introduction

- Motivate silent VIO degradation and downstream reaction time.
- Explain why current-frame confidence and trajectory accuracy do not answer the early-warning question.
- Present the causal sensor-to-action pipeline.
- State H1-H5 and the single primary contribution.
- Delimit estimator-introspection from inventing another VIO backend.

Artifact links: `{{FIGURE_SYSTEM_ARCHITECTURE}}`, `{{TABLE_CLAIM_EVIDENCE}}`.

## 3. Related Work

Organize by visual-inertial odometry; estimator consistency; failure detection in SLAM/localization; uncertainty calibration; out-of-distribution detection; selective prediction/abstention; runtime assurance/safety shields; failure-aware robotics; recovery/relocalization.

End with a citation-backed novelty matrix using the columns in `CLAIM_EVIDENCE_MATRIX.md`. Do not claim novelty until each comparison cell is sourced and checked.

Placeholder: `{{TABLE_NOVELTY_MATRIX}}`.

## 4. Problem Formulation

- Define timestamped standardized estimator output and health vector `h_t`.
- Define causal history `H_t`.
- Define observable persistent failure events.
- Define `P(failure in (t,t+tau] | H_t)` and censoring.
- Define calibration, shift state, protective action, and downstream cost.
- State information restrictions and privileged offline variables.

Artifact links: `FAILURE_DEFINITION.md`, feature schema, target schema.

## 5. SHIELD-VIO Method

- Estimator adapter contract and optional signal groups.
- Visual, inertial, innovation, covariance, consistency, derivative, and missingness features.
- Causal resampling/windowing.
- Interpretable primary detector and computational complexity.
- Reduced-input behavior for black-box backends.

Placeholders: `{{FIGURE_HEALTH_TIMELINE}}`, `{{TABLE_SIGNAL_AVAILABILITY}}`.

## 6. Failure Prediction and Calibration

- Training objective and imbalance handling.
- Multiple horizons and event aggregation.
- Raw score versus calibrated probability.
- Platt, isotonic, temperature, and conformal procedures.
- Validation-only operating point.

Placeholders: `{{FIGURE_PREDICTION_TIMELINE}}`, `{{FIGURE_RELIABILITY}}`, `{{FIGURE_PR_CURVES}}`, `{{FIGURE_LEAD_TIME}}`, `{{FIGURE_HORIZONS}}`, `{{TABLE_FAILURE_PREDICTION}}`, `{{TABLE_CALIBRATION}}`.

## 7. Domain-Shift Awareness

- Training reference distribution and rolling statistic.
- Shift-state transitions and detection delay.
- Response policies: inflation, threshold reduction, bound widening, abstention, conservative mode.
- Independent detection evaluation and effect on prediction/policy.

Placeholders: `{{FIGURE_SHIFT_CALIBRATION}}`, `{{TABLE_DOMAIN_SHIFT}}`.

## 8. Protective Navigation Policy

- Stateful policy, thresholds, hysteresis, dwell, cooldown, stale sensors, emergency override.
- Recovery request and confirmation.
- Cost model and policy variants.
- Separation between empirical risk reduction and formal guarantees.

Placeholder: `{{FIGURE_POLICY_STATE_MACHINE}}`.

## 9. Experimental Setup

- Datasets, official versions, checksums, sequences, sensor rates, and ground-truth coverage.
- Train/calibration/validation/test/shifted-test registry.
- Estimator backends and signal availability.
- Degradations, severities, events, and seeds.
- Failure thresholds, persistence, horizons, deadlines.
- Baselines, fitting budgets, calibration, thresholds.
- Grouped inference, exact sample counts, runtime hardware.

Placeholders: `{{TABLE_DATASETS_SPLITS}}`, `{{TABLE_RUNTIME}}`.

## 10. Public-Dataset Evaluation

- H1 discrimination and event timing.
- H2 multi-signal comparison.
- H3 reliability and threshold consequences.
- H4 cross-sequence, cross-dataset, unseen-family, unseen-severity, and estimator-shift results.
- Failure/false-alarm case studies.

Placeholders: `{{TABLE_FAILURE_PREDICTION}}`, `{{TABLE_CALIBRATION}}`, `{{TABLE_DOMAIN_SHIFT}}`, `{{FIGURE_FAILURE_CASES}}`.

## 11. Closed-Loop Evaluation

- Simulator and scenario validation.
- Paired policy comparison.
- Unsafe exposure, completion, recovery, interventions, and cost sweeps.
- Same detector accuracy yielding different outcomes through calibration/policy.

Placeholders: `{{TABLE_CLOSED_LOOP}}`, `{{FIGURE_SAFETY_UTILITY}}`, `{{FIGURE_CLOSED_LOOP_EXAMPLE}}`.

## 12. Ablation and Sensitivity Analysis

- Required component removals and positive feature subsets.
- Horizon/history/calibration-size/degradation/label/estimator/dataset sensitivities.
- Practical effect sizes, uncertainty, runtime, and missing-signal trade-offs.

Placeholders: `{{TABLE_ABLATIONS}}`, `{{FIGURE_ABLATIONS}}`, `{{TABLE_SENSITIVITY}}`.

## 13. Limitations

Discuss dataset/estimator coverage, dependence on observable failure definitions, partial TUM-VI ground truth where applicable, calibration under non-exchangeable shift, synthetic nature of degradations, simulator-to-robot gap, rare catastrophic events, missing internal diagnostics in black-box estimators, runtime/hardware limits, and absence of formal guarantees.

No limitation is deferred only to supplementary material.

## 14. Conclusion

Answer the central question only within the executed protocol. Separate supported findings from future external-estimator, simulator, ROS 2, and hardware work.

## Supplementary material

- full configurations and split registries;
- feature/label/manifest schemas;
- all per-sequence results and confidence intervals;
- calibration and selective-risk details;
- degradation validation;
- failure definitions and sensitivity;
- complete ablation/runtime results;
- additional case studies;
- reproduction and artifact appendix.

## Required table generation

| Table | Source artifact |
|---|---|
| Dataset and split summary | Dataset registry + label event index |
| Failure prediction | Per-run prediction metrics aggregate |
| Calibration | Held-out calibration metrics aggregate |
| Domain shift | Paired in-domain/shift aggregate |
| Closed-loop protection | Policy outcome aggregate |
| Ablations | Paired ablation join table |
| Runtime | Component timing and memory aggregate |

## Required figure generation

Every figure exports PDF or SVG, a preview raster, and machine-readable source data. Required figures are system architecture; health timeline; risk/calibration timeline; reliability; precision-recall; lead-time distribution; horizon performance; shift calibration degradation; ablation results; safety-utility trade-off; closed-loop example; and failure/false-alarm case studies.

## Placeholder integrity

The manuscript build fails if a numerical placeholder is unresolved, resolves to a failed/dirty/incomplete run, lacks an evidence label, or has no manifest/artifact hash. Text-only structural placeholders may remain in early drafts; invented numbers are prohibited.

