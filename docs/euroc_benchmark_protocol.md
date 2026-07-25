# EuRoC Benchmark and Ablation Protocol

## Purpose

This document defines the first public-dataset evaluation protocol for SHIELD-VIO. Its purpose is to test whether estimator-health signals can predict visual–inertial localization failure early enough to support protective navigation decisions. It is an experimental protocol, not a claim of production readiness or formal safety.

## Primary research questions

1. Can SHIELD-VIO predict an operationally defined localization failure before the failure threshold is crossed?
2. Are predicted risks calibrated on held-out EuRoC sequences and degradation conditions?
3. Does the stateful shield reduce unsafe navigation exposure compared with an unshielded estimator?
4. Which signal families contribute materially to prediction, calibration, and intervention quality?
5. How sensitive are conclusions to sequence choice, degradation severity, random seed, and failure definition?

## Dataset scope

Use the EuRoC MAV dataset sequences below.

| Split | Sequences | Permitted use |
|---|---|---|
| Development | `MH_01_easy`, `MH_02_easy`, `V1_01_easy` | Debugging, feature engineering, and detector fitting |
| Calibration | `MH_03_medium`, `V1_02_medium` | Probability calibration, conformal calibration, and threshold selection |
| Test | `MH_04_difficult`, `MH_05_difficult`, `V1_03_difficult`, `V2_01_easy`, `V2_02_medium`, `V2_03_difficult` | Final reporting only |

The test split must not be used for feature selection, threshold tuning, model selection, or calibration. Any departure from this split must be recorded in the run manifest and reported as a separate exploratory experiment.

## Sensor and trajectory conventions

- Use the left monocular camera and synchronized IMU stream unless a run explicitly declares a stereo configuration.
- Preserve original timestamps and calibration files.
- Express trajectories in the EuRoC ground-truth reference frame after one documented alignment operation.
- Report the alignment method with every trajectory metric. The primary ATE result uses rigid SE(3) alignment; scale-correcting Sim(3) alignment may be reported only as a secondary diagnostic.
- Do not silently discard estimator resets, tracking interruptions, or shield interventions. Record them as events.

## Controlled degradation matrix

Evaluate the clean sequence and the following deterministic degradations. Every stochastic transformation must be repeated with at least 10 seeds on the test split.

### Visual degradations

- brightness reduction;
- overexposure;
- contrast loss;
- additive image noise;
- feature dropout;
- partial occlusion;
- frame dropout.

### Inertial degradations

- accelerometer noise increase;
- gyroscope noise increase;
- bias drift;
- scale-factor error;
- saturation;
- single-axis failure;
- packet loss.

### Combined degradations

At minimum, evaluate three combined conditions:

1. darkness plus gyroscope bias drift;
2. frame dropout plus IMU packet loss;
3. occlusion plus accelerometer saturation.

Each degradation must define low, medium, and high severity numerically in configuration. Severity values must remain fixed after the calibration phase.

## Operational failure definitions

Failure labels are prospective targets and must not be equated with injected degradation. A sample at time `t` is labelled positive when at least one of the following occurs within prediction horizon `H`:

1. translational error exceeds `e_p` for at least `tau_p` seconds;
2. rotational error exceeds `e_R` for at least `tau_R` seconds;
3. the estimator produces no valid pose for longer than `tau_gap`;
4. covariance trace exceeds `c_max` while trajectory error is also outside its accepted bound;
5. a reset, relocalization, or terminal tracking failure occurs.

The primary experiment uses prediction horizons of 0.5 s, 1.0 s, 2.0 s, and 3.0 s. Threshold values must be stored in configuration and included in every manifest.

## Compared methods

Report at least the following detector variants:

- no detector and no shield;
- transparent rule-based detector;
- logistic detector;
- logistic detector with probability calibration;
- calibrated detector with conformal risk bound;
- full detector with conformal bound and domain-shift state.

For closed-loop evaluation, compare:

- unshielded navigation;
- instantaneous threshold policy without persistence;
- stateful SHIELD-VIO policy with hysteresis and dwell time;
- oracle intervention using ground-truth future failure labels, reported only as an upper bound.

## Ablation matrix

Run leave-one-family-out ablations for:

- visual-health features;
- IMU-health features;
- innovation and NIS features;
- covariance and uncertainty features;
- temporal persistence features;
- calibration layer;
- conformal layer;
- domain-shift state;
- shield hysteresis;
- recovery actions.

Ablations must use the same data split, seeds, degradation schedule, and selected operating point as the corresponding full model.

## Metrics

### State-estimation performance

- absolute trajectory error RMSE and median;
- relative pose error for translation and rotation;
- valid-pose availability;
- reset count and time without a valid estimate;
- NIS and, where ground truth supports it, NEES coverage statistics.

### Failure prediction

- AUROC;
- area under the precision–recall curve;
- sensitivity and specificity at the selected operating point;
- false alarms per minute;
- missed-failure rate;
- median and lower-quartile warning time;
- event-level precision, recall, and F1 score.

Frame-level metrics must not replace event-level metrics because temporal correlation can substantially inflate apparent sample size.

### Calibration and uncertainty

- Brier score;
- negative log likelihood;
- expected and maximum calibration error;
- reliability diagrams;
- conformal empirical coverage and average bound width;
- calibration metrics stratified by clean, degraded, and shifted conditions.

### Closed-loop protection

- fraction of time exposed to failed localization while motion remains enabled;
- intervention count and duration;
- unnecessary intervention rate;
- emergency-stop count;
- recovery-request success rate;
- mission completion rate in simulation;
- path-length and completion-time overhead;
- collision or safety-boundary violation rate where the simulator supports it.

## Statistical analysis

- Treat sequence as the principal experimental unit.
- Report per-sequence values in addition to pooled summaries.
- For stochastic degradations, report mean, standard deviation, median, interquartile range, and 95% bootstrap confidence intervals.
- Use paired comparisons between methods on identical sequence–degradation–seed runs.
- Report effect sizes with confidence intervals; do not report p-values alone.
- Correct families of hypothesis tests when multiple ablations are tested against the same full model.
- Clearly distinguish confirmatory test-split analyses from exploratory development analyses.

## Model selection and operating points

All model fitting uses only the development split. Probability calibration, conformal calibration, and operating-threshold selection use only the calibration split. The primary operating point minimizes missed failures subject to a declared maximum false-alarm rate. Alternative operating points may be shown as sensitivity analyses but may not replace the predeclared primary result after inspecting the test split.

## Reproducibility requirements

Every run must record:

- Git commit;
- dataset sequence and checksum;
- camera and IMU calibration identifiers;
- complete experiment configuration;
- degradation type, severity, interval, and seed;
- detector and shield parameters;
- software dependency versions;
- host information;
- start and end timestamps;
- generated artifact paths;
- completion status and failure reason.

Aggregated tables and figures must be generated from machine-readable run artifacts. No manuscript value should be copied manually from terminal output.

## Minimum publication-quality evidence

The EuRoC milestone is complete only when all of the following exist:

1. clean-sequence trajectory results for every test sequence;
2. the complete declared degradation matrix;
3. detector discrimination and calibration metrics;
4. warning-time and event-level failure metrics;
5. the full ablation matrix;
6. confidence intervals and per-sequence tables;
7. closed-loop simulation results for unshielded and shielded policies;
8. scripts that regenerate every reported table and figure;
9. a limitations section documenting dataset, estimator, and generalization constraints.

## Claim boundaries

Results from this protocol may support claims about performance on the declared EuRoC experiments. They do not establish formal safety, hardware readiness, generalization to arbitrary robots or environments, or superiority to methods that were not evaluated under the same protocol. Synthetic and public-dataset evidence must remain visually and textually separated.