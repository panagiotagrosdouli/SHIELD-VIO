# SHIELD-VIO — Phase B Completion Report

## Scope

Phase B implements the scientific dataset-generation foundation for causal impending-failure prediction. It does **not** optimize a classifier, inspect confirmatory test performance, or support an early-failure performance claim.

Evidence language after this phase:

- **Causal prediction-dataset infrastructure implemented and tested.**
- **Observable failure-event construction implemented and unit tested.**
- **Deterministic synthetic pipeline integration implemented and tested.**
- **Early failure prediction performance is not yet validated.**

## Files created

- `configs/paper/failure_primary_v1.yaml`
- `configs/paper/failure_synthetic_validation_v1.yaml`
- `shield_vio/health/__init__.py`
- `shield_vio/health/schema.py`
- `shield_vio/health/temporal.py`
- `shield_vio/health/dataset.py`
- `shield_vio/evaluation/failure_definition.py`
- `shield_vio/experiments/phase_b_dataset.py`
- `scripts/build_phase_b_prediction_dataset.py`
- `tests/test_phase_b_health_schema.py`
- `tests/test_phase_b_prediction_dataset.py`
- `tests/test_paper_failure_definition.py`
- `tests/test_phase_b_synthetic_pipeline.py`
- `docs/publication/HEALTH_SCHEMA.md`
- `docs/publication/FAILURE_TARGET_PIPELINE.md`
- `docs/publication/PHASE_B_COMPLETION_REPORT.md`

## Files modified

No production estimator, public dataset adapter, detector model, calibration model, threshold, or closed-loop policy was changed. The existing CI workflow was temporarily used to execute `black --check .`; because the repository has broad pre-existing Black drift, the temporary check was removed rather than reformatting unrelated code.

## Scientific decisions implemented

1. Canonical health schema version: `SHIELD_VIO_HEALTH_V1`.
2. Frozen primary failure definition version: `SHIELD_VIO_FAILURE_V1`.
3. Primary horizons: 0.5, 1.0, 2.0, 3.0, and 5.0 s.
4. Future target semantics: `y_tau(t) = 1{T_failure in (t, t+tau]}`; the current instant is excluded.
5. Active-failure samples are ineligible for impending-failure discrimination; incomplete tails are censored by the existing target builder.
6. Primary criteria use the thresholds and persistence declared in `FAILURE_DEFINITION.md`; covariance/NIS are not promoted into the primary trajectory-behavior label.
7. Missing health signals remain explicit; unavailable signals are not silently imputed.
8. Ground-truth trajectory error and NEES are structurally evaluation-only.
9. Injected degradation metadata is separated as experiment metadata and cannot enter the normal predictor feature API.
10. Synthetic integration uses a separate `sensitivity` definition (`SHIELD_VIO_SYNTHETIC_FAILURE_VALIDATION_V1`) because the deterministic demo does not expose every primary criterion. This prevents unobserved primary criteria from being silently treated as healthy.

## Existing components reused

- `shield_vio/features/health_vector.py` remains the existing causal matrix implementation; Phase B does not duplicate or remove it.
- `shield_vio/evaluation/prediction_targets.py::build_persistent_failure_events` is reused for timestamp-aware criterion persistence.
- `shield_vio/evaluation/prediction_targets.py::future_failure_targets` is reused for `(t,t+tau]` labels, active-state exclusion, and tail censoring.
- `shield_vio/simulation/synthetic_vio.py` supplies the deterministic infrastructure-validation stream.
- Existing Ruff, pytest, synthetic benchmark, regression-gate, and README-evidence CI stages remain intact.

## Tests added

### Schema / firewall

`tests/test_phase_b_health_schema.py`

- explicit missingness;
- evaluation-only exclusion;
- future-source timestamp rejection;
- feature-role validation;
- irregular-timestamp temporal windows.

`tests/test_phase_b_prediction_dataset.py`

- structural separation of features, targets, groups, and experiment metadata;
- degradation metadata remains accessible for analysis but absent from the feature matrix;
- ground-truth error and NEES remain absent from the feature matrix.

### Causality

`test_future_modification_cannot_change_past_temporal_features` modifies only future observations and asserts that all earlier temporal feature rows remain exactly identical.

### Failure events / horizons

`tests/test_paper_failure_definition.py`

- checked-in versioned primary definition;
- deterministic event IDs/onsets;
- criterion thresholding;
- timestamp-aware persistence;
- transient violation rejection;
- current failure excluded from future-positive labeling;
- inclusive `t+tau` boundary;
- incomplete-tail ineligibility;
- experiment-oracle rejection;
- undeclared failure-label rejection.

### Reproducibility / integration

`tests/test_phase_b_synthetic_pipeline.py`

- deterministic synthetic execution;
- canonical health/target/prediction exports;
- diagnostic report and manifest;
- byte-equivalent repeated canonical CSV/JSON outputs where declared;
- explicit `SYNTHETIC_PIPELINE_VALIDATION` evidence label;
- prohibited fields absent from the exported predictor matrix.

## Leakage protections added

The feature-label firewall is typed, not only name-based. `FeatureRole` distinguishes:

- `DEPLOYABLE_FEATURE`
- `TARGET`
- `GROUPING_METADATA`
- `EXPERIMENT_ORACLE`
- `EVALUATION_ONLY`

`HealthSample.evaluation` is outside the normal deployable flattening path. `flatten_deployable()` enumerates only the machine-readable deployable registry and adds explicit missingness indicators. `PredictionDataset.features()` returns only this deployable matrix. Targets and experiment metadata are exposed through separate APIs.

Failure criterion evaluation accepts only fields declared in the loaded failure definition and rejects degradation/corruption/severity/seed/injected/oracle criterion names. Degradation metadata does not define a failure.

## Minimal diagnostic report

`build_synthetic_prediction_dataset()` emits `diagnostic_report.json` containing:

- sample count;
- duration;
- available feature columns;
- missingness fractions;
- failure-event count and IDs;
- failure onset times;
- positive rate per horizon;
- excluded/censored sample count per horizon;
- explicit confirmation that prohibited fields were excluded from the predictor matrix.

The synthetic integration also writes:

- `health_samples.csv`
- `prediction_targets.csv`
- `prediction_dataset.csv`
- `manifest.json`

## Publication gap / evidence status

- `GAP-001` — **materially resolved for the Phase-B infrastructure scope**: a checked-in versioned primary failure definition and deterministic event/horizon API now exist. Public benchmark evidence remains future work.
- `GAP-007` — **materially advanced**: a versioned backend-neutral typed schema, role registry, units/source metadata, explicit missingness, source timestamps, and causal temporal layer now exist. Broader real-estimator signal population remains incomplete.
- `GAP-018` — **partially advanced**: `HEALTH_SCHEMA.md`, `FAILURE_TARGET_PIPELINE.md`, and this evidence report describe current behavior. Older root-level planning documents may still contain stale historical audit language and should be normalized in the next documentation-only pass.
- Claim status: **infrastructure only**. H1–H5 remain unconfirmed.

## Remaining limitations

1. The canonical typed schema is not yet populated end-to-end by EuRoC/TUM-VI/external-estimator runners; Phase B synthetic integration populates only signals actually emitted by the deterministic synthetic demo.
2. The full primary failure definition is executable as a config/criterion/event contract, but some primary observables (orientation/RPE/starvation/reset/relocalization) still require standardized computation/export in later runner work.
3. Split-safe model preprocessing/training/calibration/threshold orchestration remains Phase C/D work.
4. No tree/proposed detector comparison, public held-out calibration experiment, grouped bootstrap inference, OOD benchmark, or estimator-error-coupled closed-loop benchmark is claimed here.
5. `black --check .` is not repository-clean: the validation run reported 94 files that would be reformatted, spanning substantial pre-existing code. A whole-repository formatting rewrite was deliberately not performed because it is unrelated to the Phase B scientific change set.
6. No public dataset download is required for these Phase B tests.

## Validation commands

Repository CI executes:

```bash
python -m pip install -e '.[dev]'
ruff check shield_vio scripts tests
pytest -q
python scripts/run_synthetic_demo.py --out results/synthetic_demo --seed 7
python scripts/build_synthetic_benchmark_report.py results/synthetic_demo --output results/benchmark-gate/candidate.json
python scripts/check_benchmark_regression.py tests/fixtures/benchmarks/synthetic_baseline.json results/benchmark-gate/candidate.json --max-relative-increase 0.05 --max-absolute-increase 0.01 --output results/benchmark-gate/regression-result.json
```

Phase B synthetic dataset generation command:

```bash
python scripts/run_synthetic_demo.py --out results/synthetic_demo --seed 7
python scripts/build_phase_b_prediction_dataset.py results/synthetic_demo --output results/phase_b_prediction_dataset --seed 7
```

Formatting audit executed separately in CI:

```bash
black --check .
```

## Validation results

Final scoped CI run for the Phase B branch:

- dependency installation: **PASS**;
- `ruff check shield_vio scripts tests`: **PASS**;
- `pytest -q`: **PASS**;
- deterministic synthetic benchmark: **PASS**;
- benchmark regression gate: **PASS**;
- synthetic README evidence generation: **PASS**.

Separate Black audit:

- `black --check .`: **FAIL — repository-wide pre-existing formatting drift**;
- Black reported 94 files that would be reformatted and 46 unchanged;
- the failure is recorded rather than hidden;
- no broad formatting refactor was performed.

## Required explicit answers

### Can a future observation influence a past feature?

**NO.** Supported by `test_future_modification_cannot_change_past_temporal_features` and source-timestamp validation in `HealthValue`/`TemporalFeatureTable`.

### Can degradation metadata enter the deployable feature matrix?

**NO through the canonical Phase B API.** `PredictionDataset.features()` consumes only `flatten_deployable()` output; degradation fields remain separate experiment metadata. The synthetic integration test verifies their absence from the exported matrix.

### Can ground-truth trajectory error or NEES enter deployable features?

**NO through the canonical Phase B API.** They are typed as `EvaluationDiagnostics` and are excluded from `flatten_deployable()` and `PredictionDataset.features()`.

### Are failure labels derived from degradation metadata?

**NO.** Failure criterion evaluation consumes only the frozen observable failure-definition fields and explicitly rejects degradation/oracle criterion names.

### Are we ready to claim early failure prediction performance?

**NO.** Phase B validates trustworthy dataset-generation infrastructure only. Performance claims require the later frozen public-data detector/calibration/evaluation experiments.

## Next recommended milestone

Proceed to end-to-end **sequence-level split enforcement and split-aware benchmark orchestration**: training-only preprocessing/detector fit, calibration-only calibrator fit, validation-only threshold/policy selection, and sealed test/shifted-test evaluation. This is the next P0 dependency before baseline or calibration performance can be scientifically interpreted.
