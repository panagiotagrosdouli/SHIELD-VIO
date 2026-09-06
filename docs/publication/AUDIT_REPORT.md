# SHIELD-VIO — Phase A Publication Readiness Audit

Audit target: `main` at `88fcf890000e8f7fcd39583829f484198104d9a0` (2026-08-11). This report distinguishes executable code from documentation and does not infer empirical results from plans, fixtures, or unit tests.

## Executive summary

**Publication-readiness score: 48/100.** The repository has progressed materially beyond the older audit embedded in `PAPER_SCOPE.md`: a causal timestamped health matrix, persistent failure events/future-horizon targets, Platt calibration, event metrics, split/leakage guards, a backend protocol, and a public-dataset vertical-slice smoke path now exist. However, the confirmatory paper chain remains incomplete. There is no stored multi-sequence EuRoC/TUM-VI confirmatory evaluation, no TUM-VI runner, no established external VIO integration, no tree/temporal baseline, no grouped paired bootstrap/effect-size implementation, no evaluated OOD response policy, and no estimator-error-coupled closed-loop mission benchmark.

The most important audit finding is **documentation drift**: several authoritative planning documents describe as missing components that are now implemented on `main`. This is not merely editorial; stale claim/evidence state can cause incorrect paper claims, wrong milestone ordering, and reviewer-visible inconsistency.

## Strongest current components

1. **Causal health-feature construction — IMPLEMENTED_AND_TESTED.** `shield_vio/features/health_vector.py` enforces strictly increasing timestamps, source-time provenance, backward-looking rolling windows, explicit missingness for NIS/visual updates, and rejects privileged feature names. `tests/test_paper_failure_prediction_core.py` exercises future-source rejection and causal visual joins.
2. **Failure-event and horizon-target primitives — IMPLEMENTED_AND_TESTED.** `shield_vio/evaluation/prediction_targets.py` implements persistence, onset/offset construction, active-state exclusion, `(t,t+tau]` future labels, and tail censoring.
3. **Event/discrimination evaluation — IMPLEMENTED_AND_TESTED.** `shield_vio/evaluation/prediction_metrics.py` implements AUROC, AUPRC, event recall/precision/F1, false alarms/min, lead times, and validation threshold selection.
4. **Split/leakage guards — IMPLEMENTED_PARTIALLY.** `shield_vio/evaluation/splits.py` plus `tests/test_paper_leakage_guards.py` prevent sequence overlap and stage misuse. The guards do not yet prove that every preprocessing/imputation/model-selection path is split-safe.
5. **Nonconfirmatory vertical slice — IMPLEMENTED_AND_TESTED.** `shield_vio/experiments/paper_vertical_slice.py` writes feature, prediction, metric, calibration, reliability, timeline, model, and manifest artifacts and explicitly labels them `PUBLIC_DATASET_SMOKE`/nonconfirmatory.

## Critical blockers

### P0 — invalidates scientific conclusions

- **P0-1: Confirmatory data separation/orchestration is not end-to-end enforced.** Split primitives exist, but the available vertical slice trains/calibrates/selects threshold on an analytic synthetic reference and evaluates a public sequence. It is explicitly a smoke study, not the frozen public-data train/calibration/validation/test protocol.
- **P0-2: Primary failure definition is not frozen in executable configuration.** `failure_labels.py` exposes multiple criteria including ground-truth position/RPE and navigation clearance, while the vertical slice constructs events only from aligned position error. Documentation requires primary-vs-sensitivity separation, recovery confirmation, and versioned thresholds.
- **P0-3: Grouped statistical inference is missing.** `statistics.py` uses normal-theory confidence intervals over supplied values; there is no sequence/condition-grouped paired bootstrap, paired effect-size pipeline, or protection against treating frame rows as replicates in aggregate inference.

### P1 — required for core publication

- Multi-sequence EuRoC confirmatory benchmark and stored evidence.
- Real TUM-VI runner/evaluation; current support is filesystem discovery only.
- At least one mature external VIO backend adapter and standardized health export.
- Fair common-protocol baselines: predeclared heuristics, logistic, tree method, and proposed multi-signal predictor; temporal baseline remains absent if retained as a claim/comparison.
- Full held-out calibration suite/metrics: Platt exists; isotonic/temperature, slope/intercept and complete held-out reliability pipeline are incomplete.
- OOD benchmark with fitted reference artifact, shift delay/false alarms, and measured effect on prediction/calibration/policy.
- Estimator-error-coupled closed-loop benchmark with no-shield and simpler-policy controls, unsafe exposure, mission completion, intervention cost and recovery outcomes.
- One-command paper reproduction and complete artifact index.

### P2 — important but not blocking initial experiments

- Richer health schema: visual survival/track age/flow/F-B consistency/reprojection/blur/brightness/exposure/contrast/entropy; IMU saturation/clipping/loss/gaps/scale anomalies; estimator rejected/missing updates and reset/reinitialization.
- Isotonic/temperature calibration and conformal-policy comparisons.
- Ablation automation and sensitivity sweeps.
- Typed manifest/schema validation and dataset checksum registry.

### P3 — polish / optional

- Manuscript automation, publication-quality table/figure styling, ROS 2/hardware validation, latency characterization.

## Claim-to-code audit

| Claim/capability | Implementation location | Test/evidence | Status | Gap |
|---|---|---|---|---|
| Causal health history | `features/health_vector.py` | `test_paper_failure_prediction_core.py` | IMPLEMENTED_AND_TESTED | Signal coverage/provenance schema incomplete |
| Persistent failure events | `evaluation/prediction_targets.py` | core paper tests | IMPLEMENTED_AND_TESTED | Primary definition/config versioning incomplete |
| Future horizon labels | `evaluation/prediction_targets.py` | core paper tests | IMPLEMENTED_AND_TESTED | Needs multi-horizon benchmark orchestration |
| Logistic predictor | `failure_detection/baselines.py` | research/core tests and vertical slice | IMPLEMENTED_AND_TESTED | No common serialized benchmark API |
| Heuristic baselines | rule detector + vertical-slice covariance/feature/NIS scores | controlled tests | IMPLEMENTED_PARTIALLY | Fair event-level benchmark across frozen splits missing |
| Tree baseline | none located | none | MISSING | Required baseline |
| Temporal learned baseline | none located | none | MISSING | Required if retained in baseline matrix |
| Platt calibration | `failure_detection/calibration.py` | core paper tests | IMPLEMENTED_AND_TESTED | Public held-out calibration evidence missing |
| Isotonic/temperature | documentation only | none | DOCUMENTED_ONLY | Implement or remove from confirmatory comparison |
| Calibration metrics | `calibration_metrics/metrics.py` | controlled tests | IMPLEMENTED_PARTIALLY | slope/intercept/adaptive ECE completeness |
| Domain shift | `domain_shift/detector.py` | controlled-array tests | IMPLEMENTED_PARTIALLY | fitting artifact + OOD benchmark + policy effect missing |
| Backend abstraction | `backends/base.py` | `test_estimator_backends.py` | IMPLEMENTED_AND_TESTED | No established external backend |
| EuRoC execution | dataset/calibration/runner modules + scripts | fixture/integration tests; CI recipes | IMPLEMENTED_PARTIALLY | no retained confirmatory multi-sequence evidence |
| TUM-VI | `datasets/adapters.py` discovery | mocked fixture | IMPLEMENTED_PARTIALLY | no runner/benchmark/result |
| Closed-loop shield | `navigation/closed_loop.py`, `safety/shield.py` | closed-loop unit tests | IMPLEMENTED_PARTIALLY | controller does not consume erroneous estimated state in a mission benchmark |
| Grouped inference | `evaluation/statistics.py` | unit tests | IMPLEMENTED_PARTIALLY | paired grouped bootstrap/effect sizes missing |
| Paper artifact generation | synthetic/vertical-slice scripts | pipeline tests | IMPLEMENTED_PARTIALLY | no complete confirmatory tables/figures/master command |

## Complete subsystem audit

The repository contains substantial implementations in core math/models, feature tracking, IMU/preintegration, ESKF estimation, stereo/two-view geometry, EuRoC ingestion/calibration/evaluation, failure prediction, calibration, domain shift, safety/navigation, simulation, visualization and regression testing. The publication-critical weakness is not absence of code in general; it is the absence of a frozen, common, multi-dataset experimental path joining these components with correct split discipline and stored evidence.

A notable repository/document mismatch is that requested paths such as `shield_vio/consistency`, `shield_vio/shield`, and `shield_vio/recovery` do not exist as named packages on current `main`; related functionality is distributed across estimation/uncertainty, `safety/shield.py`, and `navigation/closed_loop.py`. This should be documented rather than silently mapped in a paper.

## Health representation audit

A single **derived timestamped health feature matrix now exists**, so the older statement that this is wholly missing is stale. It is nevertheless narrower than the intended publication schema.

| Signal family | Current support | Causal/online | Ground truth needed | Missingness/history | External compatible | Paper-ready |
|---|---|---|---|---|---|---|
| Covariance trace/condition/growth | yes | yes | no | rolling history | backend contract partly | mostly |
| Innovation NIS | yes | yes | no | missing flag + rolling mean | optional backend issue | mostly |
| Feature detected/tracked | yes | yes | no | visual missing + rolling mean/slope | frontend-neutral if exported | partial |
| Correspondence/inlier counts/ratio | yes | yes | no | latest-past join | provider dependent | partial |
| Visual-update age/status | yes | yes | no | explicit | yes if exported | partial |
| Tracking state | yes | yes | no | encoded | yes | partial |
| Visual survival/track age/flow/F-B/reprojection/image quality | not in canonical matrix | potentially | no | no canonical handling | unknown | no |
| IMU statistics/bias evolution/saturation/loss/gaps | mostly absent from canonical matrix | potentially | no | no canonical handling | unknown | no |
| NEES | not deployable; would require ground truth | no for online deployment | yes | n/a | n/a | evaluation-only |
| rejected/missing updates/reset state | incomplete | potentially | no | incomplete | backend-specific | no |

## Leakage audit

### Potential leakage vectors

| File/function | Severity | Risk | Required correction |
|---|---:|---|---|
| `experiments/paper_vertical_slice.py::_aligned_position_errors` | controlled/expected | Ground truth constructs offline labels. Safe only if object separation is maintained. | Keep GT inaccessible to deployable feature/model path; add integration assertion on exported feature schema. |
| `failure_detection/baselines.py::LogisticFailureDetector.fit` | P0 if mis-orchestrated | Normalization is fitted inside `fit`; safe only when caller passes training rows. | Require split-aware training wrapper/manifest and tests that test/calibration rows cannot enter fit. |
| vertical-slice threshold helper | P1 | Smoke threshold is selected on synthetic validation and is not the frozen event-level 0.2 FA/min protocol. | Confirmatory runner must call event-level validation threshold selection only. |
| any future interpolation during public alignment | currently no deployable leak found | Health visual joins use only `<=t`; GT interpolation is label-only. | Preserve separation and add explicit provenance tests. |

No prohibited deployable feature was found in the canonical health matrix: it rejects names containing ground truth, failure/future-failure, degradation/severity, oracle, or split tokens. The causal builder was inspected for rolling windows, visual joining, missing-value treatment and source timestamps. Split tests were inspected for calibration/threshold misuse and sequence/event overlap. This is good evidence against several obvious leakage paths, but it is not yet a repository-wide proof for every future experiment runner.

## Failure definition audit

`failure_labels.py` defines sample-level criteria from position error, RPE, covariance, NIS, tracking/feature count, bias norm, and navigation clearance. Degradation metadata is not used. `prediction_targets.py` adds persistence and event/horizon construction. The vertical slice, however, uses **position error only** for its public smoke events. Therefore documentation and executable smoke behavior must be reported separately.

Answers: (1) there is no single frozen executable primary composite definition yet; (2) labels are independent of degradation metadata; (3) sample criteria and event construction both exist; (4) event onset is uniquely defined after persistence; (5) persistence is implemented; (6) primary vs sensitivity definitions are not fully encoded; (7) future labels can be generated without leaking future information into features, because future state is used only to construct offline targets.

## Dataset audit

| Capability | EuRoC | TUM-VI |
|---|---|---|
| adapter | yes | discovery only |
| runner | yes | no dedicated runner located |
| calibration parser | yes | no validated dedicated parser located |
| timestamp handling | yes in EuRoC path | not benchmark-validated |
| ground truth | yes | path discovery only |
| trajectory export | yes | no |
| health export | yes | no |
| failure prediction artifacts | smoke path can consume EuRoC run | no |
| checksum/version registry | incomplete | incomplete |
| multi-sequence confirmatory benchmark | incomplete | missing |
| real public integration evidence retained on main | insufficient | missing |

Filesystem fixtures are not counted as public-dataset validation.

## Estimator-backend audit

A minimal adapter abstraction **does exist** in `backends/base.py`: initialize, timestamped IMU, visual update, snapshot, and health. The health contract is currently limited to initialization/tracking, propagated IMU count, covariance trace/condition and NIS. This is insufficient for the intended multi-signal cross-backend paper claim. No mature external estimator integration was located.

Smallest publication interface: timestamped pose/validity plus optional standardized diagnostics with explicit availability/missingness: covariance summary, innovation/residual statistics, visual tracking/update counts, reset/relocalization events, latency, and provenance/backend version. Unsupported signals must be absent/marked missing, never fabricated.

## Baseline audit

Heuristic covariance, feature-count and NIS scores plus logistic regression exist. The rule detector is also implemented. No tree baseline or temporal learned baseline was located. Logistic is trainable and causal when fed the canonical feature matrix, but serialization is performed ad hoc by the vertical-slice script rather than by a common detector interface. Fair same-split, same-label, same-horizon event-wise comparison across public test sequences is not yet present.

## Calibration audit

Raw scores and logistic outputs exist. Platt calibration is now implemented and serializable; split-conformal code also exists. Isotonic and temperature scaling were not located. Brier/NLL/ECE/MCE are implemented in calibration metrics; calibration slope/intercept as evaluation diagnostics and a complete confirmatory reliability artifact pipeline remain incomplete. The vertical slice correctly separates synthetic train/calibration/validation domains, but that does not establish public-data held-out calibration.

## Domain-shift audit

`RollingShiftDetector` uses supplied reference mean/std, rolling buffers, z-distance thresholds (2/3/5), four states, and a confidence multiplier. It lacks a standardized fitted reference artifact, persistence beyond the rolling window behavior, shift-delay/false-alarm benchmark, and public ID/OOD evaluation. Shift state can affect the navigation shield (confirmed/severe shift causes slowdown), but there is no evidence that this improves calibration, prediction, or downstream outcomes.

## Closed-loop audit

The shield has state, hysteresis/dwell behavior, slowdown/hold/relocalize/halt/emergency actions. The simple `NavigationState` moves directly toward a goal using shield speed scaling. The audited closed loop does **not** establish the central downstream claim: there are no obstacles/corridor safety boundaries and no demonstrated path where estimator error perturbs the controller's consumed state and creates unsafe exposure. Mission completion, unnecessary intervention and unsafe exposure are not yet a complete paired benchmark. Current tests are software-behavior tests, not paper evidence.

## Statistics audit

Implemented: AUROC, AUPRC, precision, recall, F1, event recall/precision/F1, false alarms/min, lead times, Brier/NLL/ECE/MCE (elsewhere). Missing/incomplete: sequence-grouped paired bootstrap, paired effect sizes, grouped confidence intervals, exact confirmatory aggregation and multiplicity handling. `summarize()` computes a normal-theory 95% interval; it must not be used on frames as if independent.

Expected experimental unit remains `dataset × sequence × estimator × degradation condition × seed`.

## Reproducibility audit

CI installs the dev environment, runs Ruff and pytest, generates deterministic synthetic evidence, performs a regression gate, and uploads short-lived artifacts. The vertical slice writes a useful manifest and machine-readable sources. Still missing are a typed complete manifest contract across all stages, persistent dataset checksum registry, external-estimator provenance, and `scripts/reproduce_paper.py`/master artifact index. The repository tree contains no `Dockerfile` on the audited `main`, despite the Phase-A checklist requesting inspection of one.

## Test audit

The tree contains broad unit/integration coverage for core math, EuRoC ingestion/runner/evaluation, vision, ESKF, trajectory metrics, synthetic demo, paper prediction primitives, leakage guards, vertical slice, closed-loop behavior and regression gates. CI executes `ruff check shield_vio scripts tests` and `pytest -q`. Black is configured as a dev dependency but CI does not run `black --check`; no static type checker is configured in `pyproject.toml`. Historical documentation saying “122 tests passed” refers to an older revision and is not treated as current execution evidence in this audit because the GitHub connector does not provide a clean local checkout/test execution environment.

Missing trust-critical tests: confirmatory split end-to-end enforcement, preprocessing/imputation fit isolation, external-backend contract integration, real TUM-VI smoke, grouped bootstrap correctness, OOD policy evaluation, estimator-error-coupled closed-loop replay, and full paper artifact regression.

## Executability audit

| Workflow | Current command/path | Inspection result |
|---|---|---|
| synthetic demo | `python scripts/run_synthetic_demo.py ...` | implemented; CI executes |
| synthetic suite | `scripts/run_scenario_suite.py`, `run_all.py` | implemented/partial orchestration |
| EuRoC processing | `scripts/run_euroc.py`, `run_euroc.py`, experiment runner | implemented when data supplied |
| TUM-VI processing | none located | missing |
| failure labels/events/horizons | library functions | implemented primitives |
| health features | `load_causal_health_features` | implemented |
| detector training | logistic API/vertical slice | implemented partial |
| calibration | Platt API/vertical slice | implemented partial |
| threshold selection | event selector + smoke sample selector | partial; confirmatory wiring missing |
| event/lead-time eval | prediction metrics | implemented |
| domain-shift eval | detector only | benchmark missing |
| closed-loop shield eval | unit/simple synthetic paths | publication benchmark missing |
| ablations | plan/docs | orchestration missing |
| paper tables | no complete confirmatory generator | missing |
| paper figures | generic/smoke figure scripts | partial |

No large dataset was downloaded during this audit.

## Reviewer rejection risks

A skeptical reviewer could reject the paper today because the central causal chain is only connected at prototype/smoke level, not by a frozen confirmatory benchmark. The strongest objections would be: (1) novelty is asserted as a joint pipeline but not yet demonstrated against a sufficiently complete literature/baseline matrix; (2) public evidence is incomplete and TUM-VI is not executable end-to-end; (3) an established external VIO backend is absent, weakening estimator-agnostic claims; (4) the primary failure definition is not fully frozen/versioned and the smoke path uses position error only; (5) tree/temporal baseline coverage is incomplete; (6) held-out calibration evidence is not yet public-data confirmatory; (7) OOD claims have no paired empirical support; (8) grouped paired inference is missing; and (9) the closed loop does not yet make localization error causally consequential to navigation safety/utility.

The most serious leakage objection is currently **risk of orchestration leakage**, not an identified future-data bug in the canonical feature builder. A reviewer should still demand proof that normalization, imputation, model selection, calibration and thresholds are fitted only on their declared sequence splits.

## Final assessment

**If submitted today, why would this paper be rejected?** Because the repository demonstrates many necessary primitives but not the complete confirmatory experiment required to support H1–H5. The paper would overreach if it converted unit/smoke evidence into claims of cross-dataset early warning, calibration under shift, estimator generality, or improved downstream safety-utility.

**Minimum changes that materially reduce rejection risk:** freeze/version the primary failure/event definition; implement the full sequence-disjoint public-data experiment registry and runner; add tree baseline and one established external VIO adapter; execute EuRoC and TUM-VI with stored raw predictions; fit calibration only on calibration sequences and thresholds only on validation sequences; implement paired sequence-grouped bootstrap/effect sizes; execute an unseen-condition shift benchmark; and build an estimator-error-coupled closed-loop paired policy benchmark. Then add one-command artifact generation.

## Phase-A concise summary

1. **Readiness:** 48/100.
2. **Five biggest blockers:** confirmatory split/orchestration; frozen primary failure definition; multi-dataset stored evidence; grouped statistics; meaningful external-backend/closed-loop validation.
3. **Strongest scientific component:** causal health/event/prediction primitives with explicit leakage guards.
4. **Weakest scientific component:** empirical downstream/OOD evidence.
5. **First implementation task:** freeze and version the executable primary failure-event specification and integrate it with horizon-target generation (the canonical causal health representation is already present).
6. **Next milestone:** sequence-disjoint train/calibration/validation/test/OOD benchmark orchestration using that frozen label contract.
