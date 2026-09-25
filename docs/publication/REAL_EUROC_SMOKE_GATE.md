# Real EuRoC PUBLIC_DATASET_SMOKE Gate

## Purpose

This gate establishes that one real EuRoC sequence can execute end to end from downloaded sensor files through estimator artifacts, causal health features, offline failure-event labels, prediction scores, calibration diagnostics, figures, hashes, and a publication-scoped manifest.

Passing this gate is **pipeline evidence only**. It is not H1-H5 evidence and no numerical result from this smoke may be used as a confirmatory paper result.

## Executed path

The GitHub Actions workflow .github/workflows/real-euroc-paired.yml:

1. downloads the official EuRoC MH_01_easy archive from the ETH ASL dataset host;
2. runs the built-in ESKF in IMU-only mode;
3. runs the built-in ESKF with OpenCV relative-rotation visual updates;
4. generates a paired trajectory comparison;
5. builds causal health features from the OpenCV-rotation run;
6. constructs offline persistent failure events from aligned public ground truth;
7. trains the smoke detector only on an analytic synthetic health domain;
8. fits Platt calibration only on a separate analytic synthetic calibration sample;
9. selects the smoke threshold only on a separate analytic synthetic validation sample;
10. evaluates the real EuRoC sequence as the public smoke test;
11. validates all publication smoke artifacts fail-closed;
12. uploads the complete artifact bundle.

## Required evidence label

The vertical-slice manifest must contain PUBLIC_DATASET_SMOKE evidence level, confirmatory=false, status=complete, and an explicit claim boundary stating that the run does not confirm H1-H5.

A workflow success without these fields is not accepted as publication evidence.

## Required artifact bundle

The smoke gate requires health_features.csv, predictions.csv, metrics.json, model.json, calibration.json, reliability_source.csv, reliability_diagram.pdf/svg, prediction_timeline_source.csv, prediction_timeline.pdf/svg, experiment_manifest.json, and smoke_validation.json.

The estimator run directory is also retained, including trajectory, health, visual-update, metric, and estimator-manifest artifacts.

## Validation invariants

scripts/validate_public_smoke.py rejects the run when any of the following occurs:

1. the evidence tier is not PUBLIC_DATASET_SMOKE;
2. the artifact is not explicitly non-confirmatory;
3. the manifest is incomplete;
4. the real public sequence appears in train/calibration/validation;
5. the smoke does not contain associated and eligible samples;
6. either future-failure class is absent;
7. no persistent failure event is observed;
8. an expected artifact is absent;
9. an artifact SHA-256 differs from the manifest;
10. a causal feature uses a source timestamp later than its prediction timestamp;
11. feature/prediction row counts disagree with the manifest;
12. metric denominators disagree with prediction labels;
13. a required smoke baseline is absent;
14. AUROC/AUPRC is non-finite or outside [0,1];
15. raw and calibrated logistic predictions do not carry calibration diagnostics;
16. the referenced estimator manifest is missing;
17. the Git revision is not captured.

## What this smoke can establish

A passing run supports only these statements:

- real EuRoC camera/IMU/ground-truth files execute through the implemented runner;
- timestamped estimator health can be transformed into causal features;
- privileged ground truth is used only in the offline label/evaluation path;
- persistent future-event labels can be constructed for a real public sequence;
- heuristic and learned scores can be evaluated on real sensor-derived health data;
- calibration/reliability artifacts and prediction timelines can be generated;
- artifact provenance and hashes are sufficient for a reproducible smoke bundle.

## What this smoke cannot establish

It cannot establish:

- that SHIELD-VIO outperforms SUPER, heuristics, logistic regression, or any other baseline;
- that the learned detector generalizes from real train sequences;
- that the reported probabilities are calibrated on the target public-data distribution;
- that H1-H5 are supported;
- that the primary composite failure definition is fully available on real data;
- that the method generalizes to TUM-VI or another estimator;
- that domain-shift handling improves reliability;
- that the protective policy improves safety or mission utility;
- real-time, hardware, formal-safety, or state-of-the-art claims.

## Known smoke-specific limitations

The current vertical slice uses one EuRoC sequence (MH_01_easy), the OpenCV relative-rotation configuration rather than a mature production VIO backend, a position-error-only smoke failure definition, analytic synthetic training/calibration/validation health distributions, and one primary smoke horizon from configs/paper/vertical_slice_smoke.yaml.

These choices are acceptable for pipeline validation but intentionally insufficient for confirmatory evaluation.

## Gate-to-paper transition

After this gate passes:

1. preserve/download the workflow artifact;
2. record the validated manifest, commit SHA, workflow run ID, and smoke-validation report;
3. repeat the equivalent real-data smoke for TUM-VI;
4. freeze the complete TUM-VI registry/checksums;
5. ensure all primary failure observables are exported;
6. execute real train/calibration/validation/test partitions;
7. generate run-level metric tables compatible with the paired aggregation contract;
8. only then populate confirmatory manuscript tables.
