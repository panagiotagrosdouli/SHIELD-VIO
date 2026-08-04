# Paper Reproducibility Checklist

## Repository and environment

- [ ] Git commit SHA and dirty-tree state captured for every run.
- [ ] Python version, OS, CPU, RAM, accelerator, and relevant driver versions captured.
- [ ] Complete installed-package list and project version captured.
- [ ] Exact command, working directory, start/end time, exit code, and stdout/stderr paths captured.
- [ ] Configuration is schema-validated, immutable for the run, and stored with a canonical hash.
- [ ] Random generators have explicit seeds and seeded libraries are listed.
- [ ] CI runs lint, unit tests, leakage tests, smoke reproduction, and artifact schema validation.
- [ ] Generated tooling/cache directories are excluded from source control.

## Data provenance

- [ ] Dataset release/export variant and official source recorded.
- [ ] License/citation obligations documented.
- [ ] Complete sequence index and calibration checksums recorded.
- [ ] Camera, IMU, and ground-truth rates measured and compared with declarations.
- [ ] Timestamps are finite, strictly increasing per stream, and units are explicit.
- [ ] Coordinate frames, transforms, quaternion order, gravity convention, and alignment are documented.
- [ ] Ground-truth coverage and excluded intervals are recorded.
- [ ] Raw public data are not silently modified or committed.

## Split integrity

- [ ] Train, calibration, validation, test, and shifted-test sequence lists are checked in.
- [ ] No physical sequence appears in multiple splits.
- [ ] All derived degradations/seeds inherit the parent sequence split.
- [ ] Windows from one failure event are not split across partitions.
- [ ] Cross-dataset target data are absent from source fitting unless explicitly declared.
- [ ] Test labels and metrics remain sealed until the protocol freeze.

## Degradations

- [ ] Family, numeric severity, start, duration, seed, and transform version recorded.
- [ ] Same event schedule is used for all methods.
- [ ] Original and transformed indexes/checksums or deterministic manifests are stored.
- [ ] Sensor transformation is applied before estimator execution.
- [ ] Injected metadata is excluded from failure labels and deployable features.
- [ ] Combined degradations are distinct declared conditions.
- [ ] Conditions that produce no failure are retained.

## Standardized estimator output

- [ ] Timestamp, pose, validity, frame, and latency exported.
- [ ] Velocity/covariance/innovation/residuals exported when available.
- [ ] Feature/tracking/update/reset/relocalization diagnostics exported when available.
- [ ] Missing signals use explicit indicators, never silently fabricated values.
- [ ] External estimator version, configuration, build flags, and adapter version captured.
- [ ] Estimator crashes/divergence create completed failure manifests rather than exclusions.

## Failure labels and targets

- [ ] Label schema/version and all thresholds recorded.
- [ ] Persistence, merge gap, recovery confirmation, and censoring implemented.
- [ ] Primary trajectory-behavior and secondary consistency labels are distinct.
- [ ] Ground truth is used only in offline label/evaluation stages.
- [ ] Future targets use `(t, t + tau]` and exclude the current failure state.
- [ ] Prediction horizons, warning deadline, and recovery deadline recorded.
- [ ] Event IDs/onsets/offsets and excluded sample reasons exported.
- [ ] Label sensitivity configurations execute without editing code.

## Causal feature construction

- [ ] Every feature row records its prediction timestamp and maximum source timestamp.
- [ ] Maximum source timestamp is not later than prediction timestamp.
- [ ] Rolling windows are backward-looking only.
- [ ] Resampling uses no future interpolation.
- [ ] Normalization and imputation fit on training data only.
- [ ] Missingness indicators accompany optional signals.
- [ ] Ground truth, future labels, degradation metadata, and split-sensitive aggregates are forbidden fields.
- [ ] Feature schema, units, order, and code version are stored.

## Training, calibration, and thresholding

- [ ] Detector fit receives training rows only.
- [ ] Model selection budget and objective are fixed on train/validation data.
- [ ] Calibration and conformal fitting receive calibration rows only.
- [ ] Operating thresholds and policy parameters receive validation rows only.
- [ ] Test data do not influence prevalence correction, normalization, calibration, or thresholds.
- [ ] Model, preprocessing, calibrator, and threshold artifacts are serialized and hashed separately.
- [ ] Class imbalance treatment is declared.

## Evaluation and statistics

- [ ] Same labels/horizons/runs are joined across methods.
- [ ] AUROC and AUPRC source scores are retained.
- [ ] Event and frame metrics are reported separately.
- [ ] False alarms/min uses eligible monitoring duration.
- [ ] Lead-time distribution includes missed events and timely-event definition.
- [ ] Brier, NLL, ECE/MCE, adaptive ECE, slope/intercept, coverage, and width are computed as applicable.
- [ ] Exact sequence/run/event/window counts and durations are reported.
- [ ] Confidence intervals resample sequence-level units, not frames as independent observations.
- [ ] Paired effects use identical run keys and report effect size.
- [ ] Exploratory multiplicity correction is recorded.

## Closed-loop protection

- [ ] Estimated state, not ground truth, drives the controller/policy.
- [ ] Ground truth scores boundaries and outcomes only.
- [ ] No-shield and all simpler policies replay identical predictions/scenarios.
- [ ] Entry/exit thresholds, hysteresis, dwell, cooldown, stale handling, override, and recovery confirmation recorded.
- [ ] Several cost settings and the full safety-utility curve are generated.
- [ ] Unsafe exposure, interventions, completion, recovery, halts, and delays are retained per run.
- [ ] Simulator evidence is not described as hardware validation or a formal guarantee.

## Figures, tables, and manuscript

- [ ] All figures have labels, units, sample size, uncertainty where appropriate, and consistent method names.
- [ ] PDF/SVG and machine-readable source data are emitted together.
- [ ] Tables are generated from aggregate artifacts, not hand-edited values.
- [ ] Manuscript numerical placeholders resolve automatically.
- [ ] Every number links to a configuration, manifest, prediction source, and commit.
- [ ] Evidence tier is visible for every result.
- [ ] Failed, partial, synthetic, public-data, simulator, and hardware evidence remain distinct.
- [ ] Claim–evidence matrix is updated before submission.

## One-command reproduction

- [ ] `python scripts/reproduce_paper.py --config configs/paper/main.yaml` verifies or executes the full pipeline.
- [ ] `--smoke` uses small declared inputs and never emits confirmatory claims.
- [ ] Stages can resume only after input/config/output hash validation.
- [ ] Cached estimator outputs record their producing command and checksum.
- [ ] Final master manifest indexes all raw, processed, model, calibration, prediction, closed-loop, figure, and table artifacts.
- [ ] A clean checkout can reproduce the paper artifact using the documented dataset placement and command.

## Current audit status at `f46355a`

- [x] Repository cloned and complete tree inspected.
- [x] 122 existing tests in 33 files passed when executed file-by-file in the audit environment.
- [x] Ruff static checks passed.
- [x] Existing synthetic evidence is explicitly labeled.
- [x] EuRoC stream/calibration/runner primitives exist.
- [ ] Public multi-sequence failure-prediction experiment executed and indexed.
- [ ] TUM-VI real execution implemented.
- [ ] External established VIO adapter implemented.
- [ ] Persistent event/horizon target and full leakage suite implemented.
- [ ] Full calibration, shift, grouped statistics, and closed-loop benchmark implemented.
- [ ] Paper reproduction command and complete manuscript implemented.

