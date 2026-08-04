# Implementation Roadmap

## Audit conclusion

SHIELD-VIO already contains useful research primitives: a typed internal ESKF backend, EuRoC ingestion/calibration/ground-truth association, OpenCV visual rotation and stereo-PnP providers, observable sample labels, rule/logistic detectors, basic calibration metrics, a conformal primitive, a rolling shift state, stateful shield logic, deterministic synthetic experiments, CI, and 122 passing tests.

The repository is not yet paper-ready because those primitives do not form one executable public-data causal pipeline. The highest-priority gap is the bridge from timestamped estimator output to causal health history, persistent future-failure events, fair detector/calibration evaluation, and policy outcomes. Adding more estimator or visualization features before that bridge is complete would not strengthen the main claim.

## Phase 0: Claim and protocol freeze

**Deliverables:** the ten root research documents; evidence-tier vocabulary; primary endpoints; observable failure definition; EuRoC split revision; TUM-VI inventory rule; unsupported-claim register.

**Acceptance:** documents agree on target interval, split roles, primary failure criteria, operating constraint, experimental unit, evidence boundaries, baselines, and ablations. No paper number is invented.

## Phase 1: First public-data vertical slice

### 1.1 Standardized health table

- Merge estimator health and visual update diagnostics by timestamp.
- Add visual counts/ratios, covariance summaries/growth, NIS/missingness, update intervals/starvation, IMU health summaries, and causal slopes/variability.
- Store feature names, units, source timestamp range, and missingness.
- Ensure backend optionality and reduced-signal operation.

### 1.2 Persistent failure events and horizon targets

- Associate trajectory and ground truth without feeding ground truth to runtime code.
- Implement primary criteria, persistence, merge/recovery logic, censoring, and event IDs.
- Build targets for 0.5/1/2/3/5 s using `(t, t + tau]`.
- Export labels separately from deployable features.

### 1.3 Baseline and calibration smoke

- Implement covariance, feature-count, and NIS scores.
- Train logistic regression on causal multi-signal features.
- Implement Platt calibration and reliability source data.
- Compute AUROC, AUPRC, Brier, ECE, false alarms/min, and lead time.
- Generate one reliability diagram and one prediction timeline.

### 1.4 Manifest and leakage guards

- Expand the manifest to the required typed schema.
- Add tests for split overlap, future feature timestamps, preprocessing/calibration/test contamination, privileged fields, event splitting, and test threshold selection.
- Add a real EuRoC CI/workflow run that stores artifacts without committing large raw data.

**Acceptance:** at least one real EuRoC sequence executes the estimator-to-report pipeline and emits a `PUBLIC_DATASET_SMOKE` manifest. If a single sequence uses temporal development partitions, outputs are explicitly non-confirmatory. Unit/fixture tests cannot satisfy this gate.

## Phase 2: Strict multi-sequence EuRoC benchmark

- Encode the four-way EuRoC sequence split in machine-readable configuration.
- Apply sensor degradation before estimator execution with deterministic event schedules.
- Complete event/frame metrics, threshold selection, tree baseline, grouped bootstrap, paired effects, and artifact indexes.
- Execute all split sequences and at least 20 test sequence-degradation conditions.
- Produce the first six paper tables/figures from raw predictions.

**Acceptance:** H1-H3 can be evaluated on sealed EuRoC test data with complete denominators and paired uncertainty. This remains EuRoC-bounded evidence.

## Phase 3: TUM-VI and domain shift

- Inventory official TUM-VI EuRoC-export files and checksums.
- Implement real TUM-VI stream/calibration/ground-truth intervals and standardized runner.
- Freeze the TUM-VI split registry.
- Add cross-dataset, unseen-family, unseen-severity, and partial-ground-truth evaluation.
- Implement and compare shift statistics and response policies.

**Acceptance:** H4 has zero-adaptation and target-calibration results in both dataset directions, with selective-risk, coverage, confidence-under-error, and shift-delay artifacts.

## Phase 4: Estimator generalization

- Add an adapter for at least one mature external estimator, preferably OpenVINS first because it exposes useful internal diagnostics; add a second backend if execution resources permit.
- Pin external commit/container/configuration.
- Export standardized events and signal availability.
- Run reduced-input and cross-estimator transfer experiments.

**Acceptance:** any architecture-general statement names the executed estimators and includes reduced-signal results. Internal ESKF performance is not the basis of a superiority claim.

## Phase 5: Protective closed-loop utility

- Build a reproducible path-following simulator with obstacles/boundaries, estimated-state control, ground-truth scoring, degradation, recovery, and mission outcome.
- Replay identical predictions through all six policy variants.
- Implement dwell/cooldown/recovery confirmation, action costs, and cost sweeps.
- Measure unsafe exposure, intervention precision, completion, recovery, halts, and delay.

**Acceptance:** H5 is supported only if the full policy yields a favorable paired safety-utility trade-off rather than merely increasing stops. Evidence is labeled closed-loop simulation.

## Phase 6: Full ablation, sensitivity, and runtime study

- Run A1-A13 and all structured feature subsets.
- Run horizon/history/calibration-size/degradation/label/missing-signal sensitivities.
- Measure component latency/memory on named hardware.
- Generate paired source tables and corrected exploratory inference.

**Acceptance:** contribution statements survive plausible label/settings changes or are explicitly qualified.

## Phase 7: Paper artifact and release

- Implement `scripts/reproduce_paper.py` and smoke mode.
- Generate seven tables and twelve figures from machine-readable sources.
- Build complete manuscript, supplementary material, and artifact appendix.
- Validate every placeholder and claim against the evidence matrix.
- Archive configs, manifests, model/calibration artifacts, predictions, and source data with a release identifier.

**Acceptance:** a clean checkout plus documented public-data placement reproduces the complete artifact; unresolved numerical placeholders or unindexed outputs fail the build.

## Immediate ordered work items

1. Add typed health, event-label, split, and manifest schemas.
2. Implement causal window and future-target builders with leakage tests.
3. Implement rank-based AUROC/AUPRC, event metrics, and Platt calibration.
4. Build a vertical-slice CLI consuming real EuRoC runner artifacts.
5. Add deterministic fixture integration tests for the complete slice.
6. Add/execute the public EuRoC workflow and index its artifacts.
7. Only after the vertical slice passes, expand to strict multi-sequence data and additional methods.

## Dependency and risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| Internal ESKF produces poor or frequent divergence | Labels may be dominated by backend weakness | Treat as controlled backend; add established adapter; report estimator-stratified results |
| One sequence lacks enough failure/non-failure examples | Calibration/metrics unstable | Smoke label only; move confirmatory work to strict multi-sequence runs |
| Failure definition overlaps a baseline signal | Circular advantage | Primary labels use trajectory/output behavior; consistency definitions are secondary |
| TUM-VI partial ground truth | Biased coverage and censoring | Evaluate declared intervals, report coverage, use room sequences for full-GT primary results |
| Severe shift breaks exchangeability | Conformal undercoverage | Measure empirical coverage; avoid guarantee language; compare abstention/conservative responses |
| Correlated windows inflate confidence | Overstated evidence | Event-level primary metrics and sequence-grouped inference |
| Excessive shield conservatism appears “safe” | Low utility/mission failure | Report completion, delays, unnecessary intervention, and Pareto curves |
| Workflow data downloads are fragile/large | Incomplete public execution | Checksummed retrying downloads, resumable stages, retained compact artifacts |
| Optional estimator signals differ | Unfair comparisons | Signal availability table, missingness indicators, reduced-input ablations |

## Definition of done

The project is paper-ready when every main claim has an executable experiment and stored artifact, every number is generated from raw outputs, public-data and closed-loop evidence meet the frozen protocol, leakage tests pass, and the manuscript’s conclusion stays within the declared evidence boundaries.

