# SHIELD-VIO Publication Implementation Plan

This plan is ordered from scientific validity to empirical breadth. It deliberately does **not** start by rebuilding the canonical health matrix because `shield_vio/features/health_vector.py` already implements a causal timestamped representation with provenance and leakage tests. Milestone 1 therefore freezes the remaining P0 label contract first.

## Milestone 1 — Frozen failure events and horizon targets

**Objective:** create one versioned executable primary failure specification and separate sensitivity definitions.

**Files:** `shield_vio/evaluation/failure_labels.py`, `shield_vio/evaluation/prediction_targets.py`, new `configs/paper/failure_primary_v1.yaml`, tests.

**Tests:** deterministic event IDs/onsets/offsets; persistence; censoring; `(t,t+tau]`; degradation metadata independence; primary-vs-sensitivity separation.

**Command:** `pytest -q tests/test_paper_failure_prediction_core.py tests/test_paper_failure_definition.py`

**Artifacts:** versioned label config/schema and fixture label/event tables.

**Scientific acceptance:** given identical timestamped offline observations, the same primary event table and all frozen horizon labels are produced; no deployable feature or degradation metadata enters the label criterion.

**Prerequisites:** none.

## Milestone 2 — Canonical health schema completion

**Objective:** version the existing causal feature matrix and extend only publication-critical optional diagnostics.

**Files:** `shield_vio/features/health_vector.py` (or a small `shield_vio/health/` package), backend health contract, runner exports.

**Tests:** source timestamp <= prediction timestamp; backward-only rolling windows; explicit missingness; stable schema/order/units; optional backend signals never fabricated.

**Command:** `pytest -q tests/test_paper_failure_prediction_core.py tests/test_estimator_backends.py`

**Artifacts:** schema JSON/YAML, causal feature CSV fixture, provenance metadata.

**Scientific acceptance:** every feature at time `t` depends only on sources `<=t`; schema declares units, provenance and availability.

**Prerequisites:** none; may merge after Milestone 1.

## Milestone 3 — Dataset split enforcement

**Objective:** make train/calibration/validation/test/shifted-test boundaries unavoidable in orchestration.

**Files:** `shield_vio/evaluation/splits.py`, dataset registries, common experiment runner.

**Tests:** no physical sequence overlap; derived degradation inherits parent split; detector/preprocessing=train only; calibration=calibration only; threshold/policy tuning=validation only; test cannot enter any fit stage.

**Command:** `pytest -q tests/test_paper_leakage_guards.py`

**Artifacts:** sequence-level split manifest.

**Scientific acceptance:** intentional misuse of a test sequence in any fit/tuning stage fails before computation.

**Prerequisites:** Milestone 1.

## Milestone 4 — Benchmark baselines

**Objective:** place heuristics, logistic, tree, and proposed multi-signal method behind a common causal API.

**Files:** `shield_vio/failure_detection/baselines.py`, new detector interface/tree implementation, serialization helpers.

**Tests:** fit/score/serialize round trip; deterministic seeds; same feature/label protocol; no privileged columns.

**Command:** `pytest -q tests/test_paper_baselines.py`

**Artifacts:** model JSONs and raw score tables.

**Scientific acceptance:** all methods execute on identical run keys, horizons and labels; no method receives extra oracle information.

**Prerequisites:** Milestones 1–3.

## Milestone 5 — Calibration

**Objective:** complete held-out probability calibration and reliability evaluation.

**Files:** `failure_detection/calibration.py`, `calibration_metrics/metrics.py`, calibration runner.

**Tests:** Platt; isotonic when sample support is adequate; temperature only for logit-producing models; split enforcement; slope/intercept/reliability source.

**Command:** `pytest -q tests/test_paper_calibration.py tests/test_paper_leakage_guards.py`

**Artifacts:** raw scores, calibrated probabilities, calibrator JSON, reliability source/PDF/SVG, Brier/NLL/ECE/slope/intercept.

**Scientific acceptance:** calibrators fit exclusively on calibration sequences and are evaluated unchanged on held-out test/OOD sequences.

**Prerequisites:** Milestones 3–4.

## Milestone 6 — Event/lead-time evaluation and grouped statistics

**Objective:** make event outcomes and run-level paired inference publication-valid.

**Files:** `evaluation/prediction_metrics.py`, `evaluation/statistics.py`, aggregate runner.

**Tests:** event matching, false alarms/min duration, missed events, median lead time, validation threshold <=0.2 FA/min, paired grouped bootstrap with known synthetic effects.

**Command:** `pytest -q tests/test_paper_failure_prediction_core.py tests/test_grouped_statistics.py`

**Artifacts:** per-run metrics, bootstrap source table, paired differences, CIs/effect sizes.

**Scientific acceptance:** frames are never resampled as independent experimental units; paired comparisons use identical run keys.

**Prerequisites:** Milestones 1–5.

## Milestone 7 — EuRoC benchmark

**Objective:** execute the frozen public-data protocol across declared EuRoC sequences and degradations.

**Files:** common runner, `configs/datasets/euroc_paper_v1.yaml`, benchmark scripts.

**Tests:** fixture integration plus one explicit real-data smoke when dataset is supplied; artifact schema validation.

**Command:** proposed `python scripts/run_failure_benchmark.py --dataset-config configs/datasets/euroc_paper_v1.yaml --paper-config configs/paper/main.yaml`

**Artifacts:** standardized estimator outputs, labels/events, health features, raw/calibrated predictions, per-run metrics/manifests.

**Scientific acceptance:** every declared sequence-condition-seed run completes or records a failure manifest; test results are produced without fitting on test data.

**Prerequisites:** Milestones 1–6.

## Milestone 8 — TUM-VI benchmark

**Objective:** add real TUM-VI ingestion/runner/evaluation using the same artifact contract.

**Files:** TUM-VI parser/calibration/runner, config registry, tests.

**Tests:** timestamp/calibration/GT parsing; smoke execution; artifact parity with EuRoC.

**Command:** analogous benchmark command with TUM-VI config.

**Artifacts:** checksum-linked TUM-VI run manifests and predictions.

**Scientific acceptance:** real named TUM-VI sequences execute end-to-end; filesystem discovery alone does not satisfy this milestone.

**Prerequisites:** Milestones 1–7.

## Milestone 9 — External estimator adapter

**Objective:** test estimator-agnostic behavior with at least one established VIO system.

**Files:** `shield_vio/backends/` adapter, standardized import/export bridge, provenance config.

**Tests:** contract compliance, missing-signal semantics, timestamp monotonicity, version/config capture.

**Command:** backend-specific benchmark invocation through the common runner.

**Artifacts:** standardized state/health tables and adapter manifest.

**Scientific acceptance:** the same held-out prediction/evaluation protocol runs on internal ESKF and one established external estimator; reduced-signal variants are reported where diagnostics are unavailable.

**Prerequisites:** Milestones 2, 7/8.

## Milestone 10 — Domain-shift benchmark

**Objective:** evaluate shift detection and its downstream handling under unseen conditions.

**Files:** `domain_shift/`, reference fitting, OOD metrics/evaluation runner.

**Tests:** reference fit only on declared ID data; deterministic state transitions; shift delay and false-alarm metrics.

**Command:** benchmark runner with frozen shifted-test registry.

**Artifacts:** reference distribution artifact, shift timeline, delay/FA metrics, selective-risk/calibration tables.

**Scientific acceptance:** report with/without shift handling on unseen family/severity/sequence/cross-dataset conditions; no improvement claim without paired evidence.

**Prerequisites:** Milestones 3–9.

## Milestone 11 — Closed-loop benchmark

**Objective:** make localization quality causally affect navigation and compare protective policies fairly.

**Files:** `navigation/`, `safety/`, `simulation/`, replay/benchmark runner.

**Tests:** controller consumes estimated state; GT only scores outcomes; identical replay across policies; hysteresis/dwell; unsafe exposure/completion/intervention accounting.

**Command:** proposed `python scripts/run_closed_loop_benchmark.py --config configs/paper/closed_loop_v1.yaml`.

**Artifacts:** per-policy trajectories/actions, unsafe exposure, mission completion, interventions, delay, recovery, utility curves.

**Scientific acceptance:** no-shield, simple threshold policies, hysteresis and full calibrated policy are paired on identical scenarios and predictions.

**Prerequisites:** Milestones 5–10.

## Milestone 12 — Ablations

**Objective:** execute the frozen feature-family, calibration, shift-response and policy ablations.

**Files:** configs and orchestration only where possible.

**Tests:** each ablation changes only its declared factor; run keys remain paired.

**Command:** proposed `python scripts/run_ablations.py --config configs/paper/ablations_v1.yaml`.

**Artifacts:** raw paired ablation predictions/metrics and aggregate tables.

**Scientific acceptance:** all declared ablations execute without manual code edits and retain exact sample counts.

**Prerequisites:** Milestones 7–11.

## Milestone 13 — Paper artifact generation

**Objective:** regenerate all paper evidence from manifests and raw outputs.

**Files:** `scripts/reproduce_paper.py`, table/figure generators, `configs/paper/main.yaml`, manuscript placeholders.

**Tests:** clean smoke reproduction; artifact hash validation; manuscript numbers cannot resolve from partial/dirty/unverified runs.

**Command:** `python scripts/reproduce_paper.py --config configs/paper/main.yaml` and `--smoke`.

**Artifacts:** master manifest, tables, source CSV/JSON, PDF/SVG figures, manuscript-ready values.

**Scientific acceptance:** every reported number resolves to a completed manifest, raw prediction/label table, config hash and Git commit.

**Prerequisites:** Milestones 1–12.
