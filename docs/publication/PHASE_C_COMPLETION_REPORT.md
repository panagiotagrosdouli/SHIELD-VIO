# Phase C Completion Report

## Scope

Phase C implements publication-grade public-dataset benchmark infrastructure. It does **not** claim SHIELD-VIO superiority or confirmatory public-data performance.

## Dataset support status

### EuRoC

- Adapter: executable discovery/integrity validation for camera, IMU, calibration and ground truth.
- Runner: existing internal ESKF runner retained; common Phase C CLI dispatches to it.
- Ground truth: ATE/RPE via existing EuRoC evaluator; position-error labels are evaluation-only.
- Health pipeline: estimator artifacts map into Phase B `HealthSample` / `PredictionDataset` with ground truth excluded from deployable features.
- Smoke status: **DATA NOT AVAILABLE LOCALLY** in this connected implementation session. No new real-data smoke claim is fabricated.

### TUM-VI

- Adapter: executable validation for EuRoC/DSO-style export, including camera images, IMU, calibration and optional mocap.
- Runner: executable internal ESKF sensor-stream runner using the common synchronized export semantics.
- Ground truth: optional mocap is label/evaluation-only; it is deliberately not passed through the EuRoC evaluator.
- Health pipeline: same canonical Phase B prediction builder as EuRoC.
- Smoke status: **DATA NOT AVAILABLE LOCALLY**. Official paper sequence assignments remain unfrozen until the official export inventory and identities are recorded.

## Split enforcement

Unit of split: complete physical sequence. Frozen EuRoC mapping is version-controlled in `configs/paper/public_dataset_splits.yaml`. `PaperSplit` rejects overlap, unknown partitions, unregistered sequences, and any derived run whose requested partition differs from its parent sequence. No `train_test_split` use exists in the paper benchmark path.

Calibration cannot influence detector training; validation is the only threshold-selection partition; test and shifted-test remain sealed for later confirmatory work.

## Reproducibility

- Deterministic logical run IDs are SHA-256-derived from dataset, sequence, estimator, condition, severity, seed and config version.
- Dataset fingerprints hash the listed local sensor indexes, calibration files and available ground-truth file. They are explicitly not represented as official archive checksums.
- Smoke manifests record Git revision where available, Python/platform/package versions, command, working directory, timestamps, estimator identity/configuration, failure definition, horizons, dataset fingerprint and artifact paths.
- Benchmark matrix resolution writes a split-config SHA-256 and deterministic run list.

## Tests added

- frozen partition disjointness;
- duplicate physical-sequence rejection;
- inherited split enforcement;
- unknown-sequence rejection;
- EuRoC and TUM-VI fixture layout validation;
- nonmonotonic timestamp rejection;
- missing-image rejection;
- deterministic run identity and fingerprint sensitivity;
- benchmark matrix uniqueness;
- public-run to Phase B canonical prediction integration;
- ground-truth/degradation predictor exclusion assertion.

Fixture evidence is `FIXTURE_INTEGRATION_TEST`, never public-dataset validation.

## Public evidence actually produced

- Fixture evidence: parser/infrastructure tests only.
- Real public smoke: **DATA NOT AVAILABLE LOCALLY** during this session.
- Benchmark evidence: deterministic declared EuRoC clean benchmark matrix infrastructure; no empirical benchmark results are claimed.
- Confirmatory evidence: none.

## Scientific boundary

The common smoke builder uses `SHIELD_VIO_FAILURE_PUBLIC_SMOKE_V1`, a position-error-only **sensitivity/infrastructure** definition, because the current public runner does not yet expose every observable required by `SHIELD_VIO_FAILURE_V1`. It must not replace the primary confirmatory failure definition. This prevents an incomplete runner from silently treating unavailable primary criteria as healthy.

## Remaining blockers

1. Execute at least one real EuRoC and one real TUM-VI sequence locally and store `PUBLIC_DATASET_SMOKE` artifacts.
2. Freeze the official TUM-VI export inventory, exact ground-truth intervals, paper partitions and local checksum identities.
3. Extend public runner diagnostics so every criterion in `SHIELD_VIO_FAILURE_V1` is directly observable before confirmatory labels are generated.
4. Run the full declared multi-sequence matrix before calling evidence `PUBLIC_DATASET_BENCHMARK`.
5. Repository-wide Black formatting debt remains outside the scientific Phase C change set unless separately cleaned.

## Final Phase C questions

**Can frames from the same experimental run appear in both training and test partitions?** NO.

**Can the calibration split influence model training?** NO.

**Can test data influence threshold selection?** NO.

**Can injected degradation metadata enter model inputs?** NO.

**Can real public sequences execute through the same canonical prediction-data pipeline as synthetic sequences?** YES at the implementation/interface level, with named fixture integration evidence; real `PUBLIC_DATASET_SMOKE` evidence is not claimed because the datasets were not locally available in this session.

**Are we ready to claim SHIELD-VIO outperforms baselines?** NO. That belongs to the later baseline/model-comparison phase.
