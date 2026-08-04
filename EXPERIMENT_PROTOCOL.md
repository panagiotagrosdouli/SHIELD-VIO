# Confirmatory Experiment Protocol

## Scope and preregistration status

This protocol evaluates the frozen claim in `PAPER_SCOPE.md`. It is prospective: test-set outcomes must not be inspected until dataset registries, failure definitions, feature schema, methods, calibration choices, thresholds, cost settings, and analysis code are versioned and the leakage suite passes.

Exploratory changes after test inspection require a new protocol version and must be reported as exploratory. Failed runs and exclusions are retained with reasons.

## Common run contract

Each method processes exactly the same standardized estimator output for a given run. A run is identified by:

`dataset / sequence / estimator / degradation_family / severity / seed / horizon`.

Common controls across methods:

- identical raw sequence and calibration;
- identical degradation transform and event schedule;
- identical observable failure labels and event IDs;
- identical causal feature timestamps;
- identical train/calibration/validation/test membership;
- identical prediction horizons;
- identical censoring and monitoring duration;
- identical random seeds where the method is stochastic;
- identical threshold-selection constraint;
- identical closed-loop scenario and policy cost setting.

## End-to-end protocol

1. Verify dataset indexes, timestamps, sensor rates, calibration, coordinate frames, and checksums.
2. Execute or import each estimator and convert output to the standardized schema.
3. Apply controlled degradations deterministically and rerun the estimator; do not transform already-computed health summaries as a substitute.
4. Construct observable failure criteria, persistent events, onset times, and censored future-horizon targets offline.
5. Build causal health features using only source timestamps not later than the prediction timestamp.
6. Fit preprocessing and detector parameters on training sequences.
7. Fit Platt, isotonic, temperature where applicable, and conformal components on calibration sequences.
8. Select method variant, score threshold, shift policy, and shield parameters on validation sequences.
9. Seal code/configuration hashes and run leakage checks.
10. Evaluate test and shifted-test sequences once.
11. Run paired grouped inference and sensitivity analyses without changing the primary endpoint.
12. Replay identical predictions through all protective policies in the closed-loop simulator.
13. Generate tables/figures only from machine-readable aggregate artifacts.

## H1: Early detection

**Methods:** covariance trace, largest covariance eigenvalue, feature count, track survival, NIS, tracking-state rule, moving-average rule, logistic regression, tree detector, proposed multi-signal detector.

**Primary analysis:** event-level median lead time for timely detected failures at the validation-selected operating point constrained to at most 0.2 false alarms/min. Event recall and the achieved false-alarm rate are reported beside lead time to prevent selective reporting among easy detections.

**Secondary analyses:** AUROC, AUPRC, precision, recall, F1, missed-event rate, time-to-failure curves, precision at predeclared recall, and frame-level counts. The trajectory-error oracle is shown only as a non-deployable upper bound.

**Success criterion:** the proposed detector’s paired lead-time interval favors earlier warning and its AUPRC interval favors higher discrimination against the strongest deployable single-signal baseline, without violating the false-alarm constraint. If endpoints disagree, the claim is qualified rather than collapsed into a single “better” statement.

## H2: Multi-signal advantage

Run the full method and all predeclared feature ablations using the same model class and tuning budget. Compare each ablation to the full representation per sequence-condition unit.

Required variants: visual only; IMU only; consistency only; covariance only; visual + consistency; visual + inertial; all signals; all signals without shift features; leave-one-family-out variants; no temporal history; strongest single signal.

**Primary endpoint:** paired AUPRC difference. **Secondary endpoints:** event recall, false alarms/min, lead time, calibration metrics, and runtime. Feature importance is descriptive and cannot substitute for ablation.

## H3: Calibration

Fit calibration mappings on the calibration split only. Raw scores and calibrated probabilities are both retained. Compare raw logistic output, Platt scaling, isotonic regression, temperature scaling where logits exist, split-conformal bounds, and the uncalibrated rule score.

**Primary endpoint:** paired Brier-score difference on held-out test runs. **Secondary endpoints:** NLL, fixed-bin ECE/MCE, adaptive ECE, reliability diagrams, calibration slope/intercept, empirical conformal coverage, interval width, and downstream policy utility.

The number of bins and adaptive-binning rule are frozen. Calibration results are stratified by dataset, degradation, estimator, and shift state. A low ECE alone is not sufficient because it is bin-dependent and can be misleading under imbalance.

## H4: Domain shift

Fit the shift reference using training health vectors only. Evaluate unseen degradation family, unseen severity, unseen sequence, cross-dataset transfer, visual-only training to combined degradation, and estimator shift.

Compare no shift handling with probability inflation, threshold reduction, conformal-bound widening, abstention, and conservative shield mode. The shift detector is evaluated independently using detection delay and state accuracy before its policy effect is evaluated.

**Primary endpoints:** change in false-negative rate and Brier score from in-domain to shifted conditions. **Secondary endpoints:** AUROC/AUPRC degradation, confidence under error, selective risk/coverage, shift-detection delay, conformal coverage, intervention timing, and unnecessary interventions.

Shift detection is not said to “repair calibration” unless a predeclared handling policy improves held-out metrics relative to no handling.

## H5: Protective utility

Replay identical estimator outputs and detector predictions through:

1. no shield;
2. covariance threshold;
3. raw detector threshold;
4. calibrated detector threshold;
5. calibrated detector plus hysteresis;
6. full SHIELD-VIO with shift awareness and recovery.

The simulator includes a ground-truth trajectory, estimated trajectory, controller/path follower, goal, obstacles or boundaries, degradation, intervention, recovery/relocalization, and mission outcome. Estimated state drives decisions; ground truth scores outcomes.

**Primary endpoint:** distance or time under invalid localization at matched mission-completion strata. **Secondary endpoints:** unsafe actions avoided, unnecessary interventions, intervention precision, lead time, halt/emergency-stop rates, recovery success, boundary violations, completion, delay, and path overhead.

Report a safety-utility Pareto curve and several frozen cost settings:

`J = C_failure*N_failure + C_late*N_late + C_false*N_unnecessary + C_delay*T_delay + C_stop*N_stop`.

No single arbitrary cost vector determines the conclusion.

## Imbalance and event dependence

- AUPRC and event metrics are primary to the same degree as AUROC.
- Failure windows are not independent observations.
- Event-level aggregation prevents long failures from dominating.
- False alarms are divided by eligible monitoring time, excluding active failure/refractory intervals.
- Positive and negative window counts, event counts, and sequence counts are printed in every aggregate table.
- Class weights or resampling are fitted on training data and declared per method.
- Thresholds are never retuned on the test prevalence.

## Statistical inference

Point estimates are accompanied by 95% confidence intervals and paired effect sizes.

- Use paired hierarchical bootstrap resampling at sequence first and condition/event within sequence where justified.
- Use paired permutation tests or Wilcoxon signed-rank tests on run-level summaries for robust exploratory inference.
- Use DeLong only for properly paired AUROC estimates with dependence assumptions addressed.
- Use McNemar for paired event decisions at a frozen operating point.
- Correct families of exploratory comparisons with Holm’s procedure.
- Report exact numbers of datasets, sequences, runs, events, eligible minutes, positive windows, and negative windows.
- Do not use frame-count-based normal-theory confidence intervals as the primary uncertainty estimate.
- P-values are supporting evidence, never the sole decision criterion.

## Exclusions and failures

Predeclared exclusion reasons are corrupt/missing source file, checksum mismatch, non-monotonic timestamp, invalid calibration, estimator process failure, or insufficient ground-truth coverage for the declared metric. Estimator divergence is an outcome, not an exclusion. A degradation condition that causes no failure remains in the analysis.

Every failed/partial run produces a manifest with error type, stage, stderr path, completed artifacts, and retry count. Exclusion counts are summarized by method and split.

## Runtime protocol

Measure wall time and peak memory for estimator adapter, health extraction, detector inference, calibration, shift update, and policy update separately. Report median, p95, and p99 latency on named hardware after warm-up. “Real-time” is prohibited unless total measured latency and input rate support that statement.

## Required outputs

Each method/horizon writes:

- per-timestamp predictions and bounds;
- per-event detections and lead times;
- per-run metrics with denominators;
- threshold and calibration artifacts;
- feature schema and preprocessing parameters;
- grouped-bootstrap source table and resample seed;
- policy transition and closed-loop outcome tables;
- complete manifest and hashes.

Aggregate scripts generate the seven required tables and twelve required figures. PDF/SVG figures and machine-readable source tables are generated together.

## First vertical-slice acceptance

Before the confirmatory experiment, one real EuRoC sequence must execute end to end and produce standardized health signals, a causal future-failure label, three heuristic scores, one learned score, a held-out calibration mapping, AUROC, AUPRC, Brier, ECE, false alarms/min, lead time, reliability diagram, prediction timeline, manifest, and leakage-test log.

If this smoke run uses temporal partitions of one physical sequence, it is labeled `PUBLIC_DATASET_SMOKE`, not H1-H5 evidence. The production pipeline must use the sequence splits in `DATASET_SPLITS.md`.

