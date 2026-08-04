# Baseline Matrix

## Fairness rules

All baselines receive only signals available at the prediction timestamp, use identical split membership and labels, and are evaluated at identical horizons. Preprocessing and fitting are split-scoped. Each method receives a comparable tuning budget on validation data. Missing signals are explicit. No baseline uses degradation metadata or ground truth at runtime.

The trajectory-error oracle is non-deployable and is excluded from rankings, significance claims, and statements about practical superiority.

## Confidence heuristics

| ID | Baseline | Input | Frozen score direction | Status at audit | Required work |
|---|---|---|---|---|---|
| H-COV-TRACE | Covariance trace | Pose/state covariance | Larger is riskier | Logged by internal backend | Add validation-only threshold and unavailable-signal handling |
| H-COV-EIG | Largest covariance eigenvalue | Pose/state covariance | Larger is riskier | Covariance available internally | Add standardized covariance block extraction |
| H-FEATURE | Feature-count threshold | Tracked/inlier feature count | Smaller is riskier | Visual update counts exported | Merge causally with health timestamps and tune threshold |
| H-SURVIVAL | Track survival threshold | Track survival ratio/age | Smaller is riskier | Partial tracker diagnostics | Export stable backend-neutral definition |
| H-NIS | Innovation NIS | Innovation residual/covariance | Larger is riskier | Internal NIS logged | Use correct innovation dimension and missing indicator |
| H-NORM-INNOV | Rolling normalized innovation | NIS/innovation history | Larger is riskier | Not benchmarked | Implement causal rolling mean/max and exceedance rate |
| H-TRACK | Tracking-state heuristic | Backend tracking enum | Lost/stale is riskier | Internal backend always reports `tracking` | Map real backend states and update starvation |
| H-MA | Moving-average health rule | Predeclared scalar health score | Larger is riskier | Not implemented as benchmark | Freeze components/window and validation threshold |
| H-ORACLE | Future trajectory error oracle | Ground truth | Earlier known onset | Offline labels possible | Implement only as upper bound; mark privileged |

## Learned and distributional baselines

| ID | Method | Role | Tuning controls | Status at audit |
|---|---|---|---|---|
| P-LOG | Logistic regression | Interpretable learned baseline | L2 strength, class weight, history summary | Dependency-light implementation; no split-aware orchestration |
| P-RF | Random forest | Nonlinear tabular baseline | Tree count/depth/min leaf | Missing |
| P-GBT | Gradient-boosted trees | Strong tabular baseline | Depth, learning rate, estimators | Missing |
| P-MLP | Small MLP | Compact nonlinear baseline | Width, depth, L2, early stopping | Missing; add only with fixed small budget |
| P-GRU/TCN | GRU or temporal convolution | Explicit temporal baseline | Window, hidden size, dropout | Missing; secondary unless materially beneficial |
| P-ONECLASS | One-class anomaly detector | Failure-scarce baseline | Contamination/nu, kernel | Missing |
| P-MAHA | Gaussian/Mahalanobis health distance | Interpretable distribution baseline | Covariance shrinkage, window | Rolling shift code provides a related unit-tested primitive |
| P-PROPOSED | Interpretable multi-signal temporal detector | Primary method | Frozen feature families and compact model class | Health signals exist separately; unified causal representation missing |

The primary proposed detector should remain interpretable and computationally realistic. A large neural architecture is out of scope unless it yields a material, paired improvement over the compact baselines and its calibration/runtime costs are reported.

## Calibration baselines

| ID | Method | Fit split | Applicability | Status at audit |
|---|---|---|---|---|
| C-RAW | Raw detector output | None | Every detector | Available for rule/logistic |
| C-PLATT | Platt logistic mapping | Calibration | Any scalar score | Missing |
| C-ISO | Isotonic regression | Calibration | Any scalar score; data hungry | Missing |
| C-TEMP | Temperature scaling | Calibration | Models with logits | Missing |
| C-CONF | Split-conformal risk interval/bound | Calibration | Scalar probability under exchangeability | Primitive implemented and unit-tested |
| C-RULE | Uncalibrated rule score | None | Heuristic reference | Implemented as a score; must not be called probability |

Calibration methods use the same calibration examples and are selected before test inspection. Metrics include Brier, NLL, ECE, MCE, adaptive ECE, slope/intercept, reliability source data, coverage, and bound width.

## Domain-shift baselines and policies

| ID | Shift method or response | Description | Status |
|---|---|---|---|
| S-NONE | No awareness | Apply in-domain predictor unchanged | Required reference |
| S-MAHA | Rolling Mahalanobis/standardized distance | Compare health to training reference | Simplified rolling standardized detector exists |
| S-MMD | Maximum mean discrepancy | Windowed two-sample statistic | Missing |
| S-ENERGY | Energy distance | Windowed distribution distance | Missing |
| S-CLASSIFIER | Classifier two-sample test | Distinguish training/current health | Missing |
| S-CONF-NC | Conformal nonconformity | Shift score from calibration residuals | Missing |
| R-INFLATE | Probability inflation | Conservative monotone mapping under shift | Missing evaluation |
| R-THRESHOLD | Threshold reduction | Intervene at lower predicted risk | Missing evaluation |
| R-WIDEN | Bound widening | Increase uncertainty interval | Missing evaluation |
| R-ABSTAIN | Selective abstention | Decline localization-dependent action | Missing evaluation |
| R-CONSERVATIVE | Conservative shield state | Enter restricted policy mode | State input exists; paired effect unsupported |

Shift detection and shift response are evaluated separately. A detector with good shift classification but harmful intervention behavior does not support H4.

## Protective-policy baselines

| ID | Policy | Memory/recovery | Expected role | Status at audit |
|---|---|---|---|---|
| A-NONE | No shield | None | Safety/utility reference | Can be represented by normal navigation |
| A-COV | Covariance threshold | Optional dwell | Conventional estimator confidence | Missing paired simulator orchestration |
| A-RAW | Raw detector threshold | None | Effect of discrimination without calibration | Missing paired simulator orchestration |
| A-CAL | Calibrated threshold | None | Effect of calibration alone | Missing calibrator and orchestration |
| A-HYST | Calibrated + hysteresis | Dwell/release | Effect of statefulness | Core stateful logic partly implemented |
| A-FULL | Shift-aware calibrated risk + recovery | Full state/recovery policy | Proposed system | Partial state machine; no full closed-loop evidence |
| A-ORACLE | Oracle intervention | Ideal timing | Non-deployable upper bound | Planned |

## Baseline reporting requirements

For every baseline report:

- exact feature names and units;
- signal availability and missing fraction by estimator;
- train/calibration/validation/test counts;
- tuned hyperparameters and selection objective;
- score direction and threshold;
- runtime and memory;
- frame- and event-level metrics;
- grouped confidence interval and paired effect versus the proposed method;
- failure cases and false alarms;
- evidence level and claim boundary.

## Minimum vertical-slice subset

The first public-data smoke slice implements and executes:

1. covariance-trace heuristic;
2. feature-count heuristic;
3. NIS heuristic;
4. logistic regression on the causal multi-signal health vector;
5. Platt calibration fitted on a held-out partition;
6. an explicit `PUBLIC_DATASET_SMOKE` manifest that prevents these results from being interpreted as confirmatory evidence.

