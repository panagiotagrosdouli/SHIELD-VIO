# Public Dataset Protocol

## Scope

Phase C establishes public-data execution infrastructure; it does not establish model superiority. Evidence names are restricted to `FIXTURE_INTEGRATION_TEST`, `DATASET_VALIDATED`, `PUBLIC_DATASET_SMOKE`, and `PUBLIC_DATASET_BENCHMARK`. `PUBLIC_DATASET_CONFIRMATORY` is reserved for the later frozen analysis.

## EuRoC MAV

The existing EuRoC reader/runner remains authoritative for `mav0/cam0`, optional `cam1`, `mav0/imu0`, calibration YAML and `state_groundtruth_estimate0`. Clean execution may produce trajectory, raw estimator health and ATE/RPE. Ground truth is evaluation/label-only.

## TUM-VI

Phase C targets the official EuRoC/DSO-style export. The adapter discovers `mav0/cam0/data.csv`, `mav0/imu0/data.csv`, calibration YAML and optional `mav0/mocap0/data.csv`. The executable runner reuses the synchronized sensor-stream path because the export is layout-compatible. TUM-VI mocap is not passed to the EuRoC evaluator; coverage and association must be handled explicitly.

Exact TUM-VI paper sequence assignments remain unfrozen until the official export inventory and local reproducibility fingerprints are recorded. No filenames or official checksums are invented.

## Integrity and synchronization

Before execution, timestamp CSVs must exist, contain the required columns, and have strictly increasing integer nanosecond timestamps. Local dataset fingerprints hash the listed sensor indexes, calibration files and available ground-truth index. The fingerprint is explicitly a local reproducibility identity, not an official archive checksum.

Ground-truth interpolation is evaluation-only. Deployable health values retain source timestamps and Phase B rejects future-source dependencies.

## Canonical outputs

A complete real-data smoke run must eventually contain `trajectory.csv`, `health_samples.csv`, `failure_events.csv`, `prediction_dataset.csv`, `metrics.json`, and `manifest.json`. Dataset validation alone is not a smoke run. Fixtures validate parsers only and are never public-data evidence.

## Clean-run rule

Clean sequences are evaluated before deterministic degradation injection. A technically unusable sequence remains recorded with its failure reason; it is not silently removed. Degradation type, severity and onset remain experiment-oracle metadata and never enter the deployable feature matrix.
