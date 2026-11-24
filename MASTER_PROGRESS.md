# SHIELD-VIO Master Progress

**Snapshot:** 2026-11-24 (UTC)  
**Audited base:** `main` includes synthetic runner merge `e760a0421034253ee1cca6c9b65ff5b76ea0699d`  
**Purpose:** Track implementation and evidence without treating planned work, unit tests, synthetic runs, or an in-progress workflow as real benchmark validation.

## Status key

- **GREEN** — implemented and verified against its stated acceptance criteria.
- **YELLOW** — partially implemented, under execution, or awaiting required validation.
- **RED** — an observed failure or blocker prevents acceptance.
- **WHITE** — not started or no implementation/evidence found in the current audit.

## Current repository evidence

| Evidence | State at snapshot |
|---|---|
| Main CI evidence | **GREEN on candidate heads:** CI runs [45386831620](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45386831620), [45387125611](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45387125611), and [45387445677](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45387445677), [45388990757](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45388990757) passed. The last candidate also passed the real EuRoC workflow before squash-merge `d1e1306`. |
| Real EuRoC paired benchmark | **RED for estimator accuracy / GREEN for execution:** run [45387445711](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45387445711) uploaded artifact `shield-vio-real-euroc-mh01-paired` (ID `14916768372`; SHA-256 `86b0a696d2e352c6b587f259ad119b9ed81ed745fd2bc5b112a3a43bf95062dd`). The new audit reports an 80.63 m GT path and ATE/path ratios of 860.41 (IMU-only) and 950.48 (rotation update); see evidence below. |
| PR #105 | **Integrated:** merged to `main` at `7559dd5`; its CI run [45380153039](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45380153039) passed. It adds canonical held-out probability evaluation and explicitly limits the result to non-confirmatory probability metrics. |
| PR #104 | **Integrated:** squash-merged as `0946e069`; CI [45382473614](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45382473614) and pinned OpenVINS `MH_01_easy` replay [45382473608](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45382473608) passed. Artifact `14915278229` (SHA-256 `366b13ee50c001bbda6999eaba9668e9a41ef94e6cbd67c06554e4978c7330cb`) records the headless patch digest, 138 s replay and 0.09214 m ATE RMSE. This is one non-confirmatory sequence smoke; primary failure labels have 0 events and 0 positive windows at all tested horizons, so detector discrimination is undefined. |
| Synthetic seed provenance | PR #108 merged as `584874f`; CI [45385915303](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45385915303) passed. Reports now read the actual seed, with seed-42 and invalid-type regression coverage. |
| Synthetic experiment runner | PR #112 merged as `e760a04`; full CI [45388990757](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45388990757) passed. Runner now has typed configuration, validated numeric bounds, a reproducibility manifest, summary/report/plots, synthetic-only claim boundary, and artifact tests. |
| OpenVINS runtime regression coverage | PR #110 merged as `44dec65`; CI [45386831620](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45386831620) passed. PR #111 merged as `2ecf1b6`; CI [45387125611](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45387125611) passed, including ROS setup-before-nounset guard. |
| Existing artifacts | `docs/publication/evidence/euroc_mh01_smoke_v1.json`; inspect its provenance and scope before using it as evidence. |
| Repository shape | Existing Python estimator, experiment, evaluation, dataset, and test packages; no ROS 2 package, Dockerfile, or complete manuscript source was present in the audited main tree. |


## Earlier EuRoC ESKF diagnostic — run 45380153306

The `mh01-paired` job completed successfully, including checksum-verified download, estimator runs, independent upstream `evo` evaluation, public-data smoke gate, and artifact upload. Successful execution is not successful estimator validation:

- IMU-only ESKF: ATE RMSE **69,371.62 m**, translational RPE RMSE **71.68 m** (one-sample delta).
- ESKF plus monocular relative-rotation updates: ATE RMSE **76,633.96 m**, translational RPE RMSE **80.26 m**; this is 10.47% worse ATE and 11.98% worse RPE than the baseline.
- The smoke predictor had **10 eligible windows, all positive and 0 negative**, so AUROC/AUPRC are undefined. The calibrated logistic event metric missed its sole event; raw logistic detected it with 0.45 s lead. The gate correctly marks the result `PUBLIC_DATASET_SMOKE`, `confirmatory: false`, `SINGLE_CLASS_POSITIVE`.
- Stationary-IMU gravity alignment was rejected because its observed initialization window violated stationarity prerequisites; no corrected result was reported.
- The source receipt is for a mirrored ROS bag and pinned OpenVINS calibration/GT exports, and explicitly says confirmatory identity with the current ETH archive is not established.

**Disposition:** preserve the artifact as a failure/diagnostic result. Do not tune or relabel it as a benchmark success. Investigate unit/frame conventions, IMU propagation and bias initialization against independent known-motion checks, then rerun the same training-split sequence with a predeclared diagnostic protocol before proceeding to other sequences.

## OpenVINS headless replay findings — run 45382473608

The previously observed process exit 134 and `LibraryUnloadException` for `libimage_transport_plugins.so` were not reproduced after patching pinned `ROS1Visualizer.cpp` to avoid constructing its three unused visualization publishers. The pinned serial estimator built and replayed checksum-verified EuRoC `MH_01_easy` for 138 s without passing ground truth to the estimator.

- Trajectory export: 2,761 associated samples; ATE RMSE **0.09214 m**; translational RPE RMSE **0.01628 m** at one-sample delta.
- Primary observables are ready for label construction; 2,741 samples have complete observability and 42 are censored. The frozen primary failure definition yields **0 events** and **0 positive windows** for horizons 0.5–5 s, including 0/2,701 positive windows at 2 s. Discrimination is undefined.
- The run remains `PUBLIC_DATASET_DEVELOPMENT`, `confirmatory: false`; performance and prediction quality are not established.

**Disposition:** the headless transport fix is runtime-verified for this pinned serial run. Generalization to other OpenVINS modes/versions is untested. Preserve the artifact and do not describe this sequence as predictor validation.

## EuRoC accuracy audit — run 45387445711

The paired workflow reran both ESKF configurations, independent `evo` evaluation, V2 label construction, and the public smoke gate. It uploaded artifact `14916768372` (SHA-256 `86b0a696d2e352c6b587f259ad119b9ed81ed745fd2bc5b112a3a43bf95062dd`).

- EuRoC `MH_01_easy` ground-truth path length: **80.63 m**.
- IMU-only ESKF: ATE RMSE **69,371.62 m**, ratio **860.41×** path length; translational RPE **71.68 m**.
- ESKF plus monocular relative-rotation updates: ATE RMSE **76,633.96 m**, ratio **950.48×**; translational RPE **80.26 m**.
- Both receive a coarse catastrophic-drift warning. This is a diagnostic only, not a scientific pass/fail threshold. The available ground truth starts 1.075 s after the estimator stream; over the first 4.0 s after GT coverage begins, the IMU-only estimate travels 200.5 m while GT travels 0.319 m, and estimated speed reaches 83.1 m/s. This documents rapid divergence, but does not identify its cause.
- Smoke gate: 10 eligible windows, 10 positive, 0 negative. The gate passes pipeline integrity but discrimination remains undefined; evidence is `PUBLIC_DATASET_SMOKE`, `confirmatory: false`.

**Code-path finding (hypothesis, not yet confirmed):** the ESKF starts with identity attitude and zero biases unless stationary-IMU gravity alignment is explicitly requested; propagation applies `R @ accel + gravity`. The IMU-only run has no visual updates to estimate biases, and its stationary 100-sample initialization window was rejected. This setup can leak gravity into world-frame acceleration if the initial attitude is not aligned. The output supports rapid divergence, but the exact cause still requires raw IMU/frame checks and an independent initialization experiment.\n\n**Disposition:** retain the severe drift as a RED estimator-accuracy finding. Do not tune on held-out sequences; diagnose frame, unit, initialization, and propagation conventions on predeclared training data before scaling the benchmark.

## Workstream status and acceptance gates

| Workstream | Status | Current implementation / evidence | Remaining acceptance criteria |
|---|---|---|---|
| Repository, PR, and CI audit | YELLOW | PRs #104, #105, #107–#111 are integrated; stale #5, #87, #90, #96 and #100 were closed as obsolete/superseded. PR #8 remains open, 326 commits behind, and needs adaptation review. No open issues were found in the audit. | Review remaining inactive branches and workflows, inspect experimental artifacts, and keep this record current. |
| CI/CD and regression suite | YELLOW | Ruff, pytest, deterministic synthetic regression, and the real EuRoC smoke passed on PR #109 candidate; main CI run for the latest merged commit is not yet recorded. Earlier CLI-escaping failure was diagnosed and fixed before the passing run. | Run full required checks on the final release candidate, inspect failures instead of suppressing them, and confirm all required workflows pass on the release commit. |
| VIO estimation backend | YELLOW | Research implementations exist for feature tracking, IMU preintegration, a 15-state ESKF, calibration/stereo geometry, trajectory output, and loop closure, with numerical/unit tests. README correctly calls these research prototypes. | Independently validate coordinate/quaternion conventions, Jacobians and covariance consistency; exercise complete calibrated camera–IMU synchronization, initialization/recovery, stereo and monocular pipelines; document limitations with measured evidence. |
| EuRoC benchmark | RED | One checksum-verified `MH_01_easy` paired smoke completed. Both ESKF variants diverge catastrophically by 860–950× path length; OpenVINS has a separate successful 138 s smoke. All are non-confirmatory and only one sequence is tested. | Diagnose ESKF failure on development data; verify source identity; execute all 11 registered sequences under the frozen split; inspect outcomes and runtime; report grouped uncertainty. |
| TUM-VI benchmark | WHITE | Dataset adapter/stream infrastructure is mentioned in existing documentation; no complete official sequence/checksum registry or executed benchmark evidence found in this audit. | Freeze official inventory, calibration/fisheye handling, partial-ground-truth rules, and shifted-test protocol before evaluating; run and report required sequences with censored/missing GT handled explicitly. |
| OpenVINS comparison | YELLOW | Pinned headless serial replay and V2 artifact pipeline pass for `MH_01_easy`; PR #104 records the fix. Artifact reports 0.09214 m ATE RMSE, but primary failure labels contain no events, so prediction discrimination is undefined. | Compare matched estimator inputs/conventions and assess additional EuRoC sequences; validate trajectory/covariance exports and cross-mode/version behavior. Keep failed runs visible. |
| Failure prediction models | YELLOW | Causal health features, failure targets, split/leakage guards, rules/logistic infrastructure, and frozen predictor/canonical evaluation code exist. | Implement and fairly compare required tree and temporal models where data volume supports them; evaluate all horizons; prove sequence/preprocessing isolation and seal models before held-out scoring. |
| Calibration and uncertainty | YELLOW | Calibration metrics, Platt/frozen predictor elements, conformal bounds and canonical held-out probability evaluation exist. | Verify independent calibration/test partitions; report Brier, NLL, ECE/MCE, reliability, risk–coverage and OOD outcomes; implement/justify temperature and isotonic comparisons; preserve the 0.2 false-alarm/min operating point where protocol requires it. |
| Synthetic experiment reporting | GREEN for reproducible diagnostic outputs | PR #112 adds typed scenario configuration, deterministic run summaries, provenance manifest, report, plots, bounds checking, and artifact tests; CI run [45388990757](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45388990757) passed. Output remains synthetic-only evidence. | Use paired, predefined-severity scenarios with seeds and compare nominal/degraded metrics; do not use synthetic results as real estimator validation. |
| Navigation shield | YELLOW | Stateful warning/slowdown/hold/halt/relocalization and recovery policy components have unit/closed-loop synthetic evidence. | Couple decisions to estimator failure outcomes; quantify risk reduction, false interventions and recovery success on matched sequences; validate state-machine transitions and conservative fallbacks; avoid unsupported formal guarantees. |
| Ablations | WHITE | No verified full ablation matrix or matched results located. | Retrain each specified ablation with the same development protocol and compare on identical held-out sequences, conditions and seeds. |
| ROS 2 and simulation | WHITE | No ROS 2 package or Dockerfile was present in the audited main tree. | Implement and test camera/IMU subscriptions, odometry/risk/diagnostic/shield publishers, launch/config/RViz, playback, supported simulator scenarios, ROS integration tests, and pinned reproducible container instructions. |
| Statistical analysis | YELLOW | Grouped statistics and metric aggregation modules/tests exist. | Verify physical-sequence aggregation, grouped bootstrap CIs, paired tests/effect sizes, multiplicity correction, sensitivity analysis, and explicit missing/censored handling on actual benchmark outputs. |
| Scientific manuscript | YELLOW | Paper outline, scope, abstract, contribution notes, related-work notes, and BibTeX exist. | Complete English manuscript and figures/tables from observed evidence; verify every citation and claim; compile LaTeX/PDF; state limitations and unresolved results. |
| Release and reproducibility | WHITE | Research prototype documentation, configs, workflows, and partial reproducibility notes exist. | Satisfy benchmark/CI/paper gates, then version a release or clearly labeled prerelease with environment, manifests, data instructions, changelog, citation and license review. |

## Immediate execution queue

1. Diagnose the catastrophic ESKF drift using a predeclared development-only protocol for units, frames, initialization, and propagation.
2. Extend EuRoC execution across the frozen 11-sequence registry after the estimator is stable; preserve all failed outcomes.
3. Freeze TUM-VI inventory/checksums and shifted-test rules before evaluating any sequence.
4. Implement and fairly compare the remaining prediction, calibration, robustness, and shield variants on valid development/test partitions.
5. Build ROS 2 integration, matched ablations, sequence-level statistics, the manuscript, and release artifacts only from observed evidence.
6. Record CI on each final candidate and keep benchmark status synchronized with artifacts.

## Decision rule

A workstream moves to **GREEN** only when implementation, tests or experimental artifacts, provenance, and its stated acceptance criteria agree. A successful CI run does not verify real-dataset performance, OpenVINS runtime behavior, cross-dataset generalization, safety benefit, or publication claims. No benchmark value, citation, significance result, or formal safety claim may be inferred from this tracker.
