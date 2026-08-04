# Ablation and Sensitivity Plan

## Objective

Ablations determine which parts of the causal SHIELD-VIO pipeline materially affect prediction, calibration, shift behavior, and protective utility. They are not a collection of unrelated feature tests. Every ablation is evaluated on the same run units, labels, horizons, and degradation events as the full method.

## Common ablation design

- Start from the frozen full method.
- Change exactly the declared factor.
- Refit preprocessing and detector parameters on training data when the feature space changes.
- Refit calibration on the same calibration sequences when the score distribution changes.
- Retune the operating threshold on the same validation sequences under the same false-alarm constraint.
- Never inspect or retune for test/shifted-test outcomes.
- Preserve random seeds and estimator outputs so comparisons are paired.
- Report discrimination, event timing, calibration, shift, policy, and runtime effects where applicable.

The primary contrast is `full - ablated` per sequence-condition unit. A second fixed-threshold deployment analysis may isolate what happens if a component becomes unavailable at runtime, but it is not substituted for the refit primary analysis.

## Required ablations

| ID | Removed/replaced factor | Exact intervention | Primary question | Primary metrics |
|---|---|---|---|---|
| A1 | Visual features | Remove counts, survival, track age, flow error, outlier ratio, blur, brightness, contrast, frame-drop fields and their missingness/derivatives | Does vision health add information beyond estimator/inertial signals? | AUPRC, event recall, lead time |
| A2 | IMU features | Remove packet gap/loss, norm statistics, saturation, bias and inertial missingness/derivatives | Does inertial health improve combined-degradation warning? | AUPRC, shifted FNR, lead time |
| A3 | Consistency features | Remove innovations, NIS, exceedance rates and related temporal summaries | Is estimator consistency predictive beyond covariance and sensor quality? | AUPRC, event recall |
| A4 | Covariance features | Remove trace, log determinant, eigenvalue, condition number and growth | Does reported uncertainty add independent value? | AUPRC, Brier, policy utility |
| A5 | Temporal history | Use current timestamp features only; retain missingness | Is failure prediction more than current-frame quality classification? | Lead time, AUPRC by horizon |
| A6 | Probability calibration | Feed raw detector score to threshold/policy | Does calibration improve probabilistic reliability and decisions? | Brier, NLL, ECE, unsafe exposure |
| A7 | Conformal bound | Remove bound from decision and reporting; keep calibrated mean risk | Does risk-bound information improve shift/reaction behavior? | Coverage, FNR, intervention rate |
| A8 | Domain-shift awareness | Force `IN_DISTRIBUTION` and disable shift response | Does explicit shift handling help on unseen conditions? | Shifted FNR/Brier, selective risk, utility |
| A9 | Shield hysteresis | Apply instantaneous state thresholds with no dwell/release margin | Does statefulness prevent action chattering and unnecessary intervention? | Transition count, unnecessary intervention, utility |
| A10 | Learned detector | Replace full detector with predeclared multi-signal rule | Is learning required beyond transparent aggregation? | AUPRC, lead time, runtime |
| A11 | Multi-signal detector | Replace with strongest validation-selected single signal | Does combination beat the strongest conventional heuristic? | Paired AUPRC and lead time |
| A12 | Recovery policy | Replace recovery/relocalization with halt-only | Does active recovery improve completion without unsafe exposure? | Completion, recovery success, stop duration, utility |
| A13 | Full signal access | Restrict inputs to signals available from black-box VIO output | How much performance survives reduced estimator introspection? | AUPRC, calibration, runtime by backend |

## Structured feature-family variants

In addition to leave-one-family-out tests, the following positive subsets are evaluated: visual only, IMU only, consistency only, covariance only, visual + consistency, visual + inertial, all signals, and all signals without shift-derived features. This prevents a misleading conclusion where removing one correlated family appears harmless even though no individual family is sufficient.

## Calibration ablations

Apply raw score, Platt, isotonic, temperature scaling where applicable, and conformal bounds to the same base detector. Report calibration-set size and class count. Re-run with calibration subsets at 10%, 25%, 50%, 75%, and 100% of the available calibration sequences/windows using grouping that does not split events.

## Shift-response ablations

Hold the shift statistic fixed and compare no response, probability inflation, threshold reduction, bound widening, abstention, and conservative shield mode. Then hold the response fixed and compare shift statistics. This separates shift detection quality from response-policy quality.

## Policy ablations

Replay the same prediction table through all policy variants. Detector outputs cannot be recomputed differently for a favorable policy. Report action-state dwell distributions, transition counts, intervention reasons, late interventions, and recovery confirmation.

## Sensitivity studies

| Factor | Declared values/approach |
|---|---|
| Prediction horizon | 0.5, 1.0, 2.0, 3.0, 5.0 s |
| Health history | current-only, 0.5, 1.0, 2.0, 5.0 s |
| Failure thresholds | Values in `FAILURE_DEFINITION.md` |
| Persistence | 0.25, 0.5, 1.0 s |
| Detector threshold | Operational PR/false-alarm curve, not post-hoc test optimum |
| Hysteresis/dwell | none, short, primary, long settings defined in config |
| Calibration size | 10%, 25%, 50%, 75%, 100% grouped calibration data |
| Degradation strength | low, medium, high fixed transformations |
| Feature sampling | native, 20, 10, 5 Hz causal resampling |
| Estimator | internal ESKF plus each established adapter |
| Dataset | EuRoC, TUM-VI, each transfer direction |
| Missing groups | Each family missing alone; observed black-box availability patterns |
| Policy costs | Several predeclared failure/late/false/delay/stop ratios |

## Inference and multiplicity

The full-versus-strongest-single-signal A11 contrast is the primary ablation. Other ablations are ordered secondary analyses. Report paired median/mean effects and sequence-grouped 95% bootstrap intervals. Apply Holm correction within the family of twelve secondary performance comparisons; retain unadjusted effect sizes and intervals. Do not interpret non-significance as equivalence.

## Artifact requirements

Every ablation artifact records parent full-method configuration hash, changed factor, complete resulting configuration, training/calibration/threshold split IDs, model/calibrator hashes, prediction paths, per-run metrics, paired join keys, runtime, and status. `run_ablations.py` must fail when any paired full-method unit is missing.

## Interpretation rules

- A component is supported when its removal yields a practically meaningful paired deterioration with uncertainty consistent across plausible label settings.
- A component that improves AUROC but worsens AUPRC, lead time, or policy utility is described with that trade-off.
- A calibration method that improves ECE but worsens Brier/NLL or operational utility is not called uniformly better.
- A shift response that reduces failures through excessive halting must disclose the lost completion/utility.
- Runtime and missing-signal costs are part of the conclusion.

