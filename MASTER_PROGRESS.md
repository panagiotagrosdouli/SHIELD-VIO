# SHIELD-VIO Master Progress

**Snapshot:** 2026-11-24 (UTC)  
**Audited base:** `main` at `44b53f8e852a6541166a2188d0022a561661220b`  
**Purpose:** Track implementation and evidence without treating planned work, unit tests, synthetic runs, or an in-progress workflow as real benchmark validation.

## Status key

- **GREEN** — implemented and verified against its stated acceptance criteria.
- **YELLOW** — partially implemented, under execution, or awaiting required validation.
- **RED** — an observed failure or blocker prevents acceptance.
- **WHITE** — not started or no implementation/evidence found in the current audit.

## Current repository evidence

| Evidence | State at snapshot |
|---|---|
| Main CI for `44b53f8` | **GREEN:** GitHub Actions run [45382962876](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45382962876) completed successfully. |
| Real EuRoC paired benchmark for `7559dd5` | **YELLOW:** run [45380153306](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45380153306) completed and uploaded artifact `shield-vio-real-euroc-mh01-paired` (artifact ID `14912809427`; SHA-256 `9772d64f57e9973fa47d93b8725059693b2df025aa65dcb9b0651fc243677179`). The pipeline passed, but the estimator result is unusable and smoke prediction is single-class; see findings below. |
| PR #105 | **Integrated:** merged to `main` at `7559dd5`; its CI run [45380153039](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45380153039) passed. It adds canonical held-out probability evaluation and explicitly limits the result to non-confirmatory probability metrics. |
| PR #104 | **Runtime verified:** draft `fix/openvins-headless-transport`, head `a36d59e`; PR CI and pinned OpenVINS `MH_01_easy` replay pass in run [45382473608](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45382473608). The headless source patch omits unused visualization publishers. Artifact `14915278229` (SHA-256 `366b13ee50c001bbda6999eaba9668e9a41ef94e6cbd67c06554e4978c7330cb`) records the patch digest, successful 138 s replay and 0.09214 m ATE RMSE. This is one non-confirmatory sequence smoke; primary failure labels have 0 events and 0 positive windows at all tested horizons, so detector discrimination is undefined. |
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

## OpenVINS headless replay findings — run 45382473608\n\nThe previously observed process exit 134 and `LibraryUnloadException` for `libimage_transport_plugins.so` were not reproduced after patching the pinned `ROS1Visualizer.cpp` to avoid constructing its three unused visualization publishers. The pinned serial estimator built and replayed checksum-verified EuRoC `MH_01_easy` for 138 s without passing ground truth to the estimator. The workflow and post-steps completed successfully.\n\n- Trajectory export: 2,761 associated samples; ATE RMSE **0.09214 m**; translational RPE RMSE **0.01628 m** at one-sample delta.\n- Primary observables are ready for label construction; 2,741 samples have complete observability and 42 are censored. The frozen primary failure definition yields **0 events** and **0 positive windows** for horizons 0.5–5 s, including 0/2,701 positive windows at 2 s. Discrimination is undefined.\n- The run remains `PUBLIC_DATASET_DEVELOPMENT`, `confirmatory: false`; performance and prediction quality are not established.\n\n**Disposition:** the headless transport fix is runtime-verified for this pinned serial run. It resolves the reported teardown exception in this execution, but generalization to other OpenVINS modes/versions is untested. Preserve the artifact and do not describe this sequence as predictor validation.\n\n## Workstream status and acceptance gates

| Workstream | Status | Current implementation / evidence | Remaining acceptance criteria |
|---|---|---|---|
| Repository, PR, and CI audit | YELLOW | Main CI passes; PR #105 and PR #107 merged. PR #104 headless patch and runtime are verified; PR #100 and #96 remain open and need disposition review. | Review remaining open PRs and relevant issues, branch history, workflow logs and experiment artifacts; resolve remaining integration work; keep this record current. |
| CI/CD and regression suite | YELLOW | Latest main CI passes; CI failures existed on earlier commits and were addressed before the current base. | Run full required checks on the final candidate, inspect failures instead of suppressing them, and confirm all required workflows pass on the release commit. |
| VIO estimation backend | YELLOW | Research implementations exist for feature tracking, IMU preintegration, a 15-state ESKF, calibration/stereo geometry, trajectory output, and loop closure, with numerical/unit tests. README correctly calls these research prototypes. | Independently validate coordinate/quaternion conventions, Jacobians and covariance consistency; exercise complete calibrated camera–IMU synchronization, initialization/recovery, stereo and monocular pipelines; document limitations with measured evidence. |
| EuRoC benchmark | YELLOW | One checksum-verified mirrored `MH_01_easy` run completed on real sensor/GT data with CI and uploaded artifacts. This source receipt explicitly says it does not establish identity with the current ETH archive and is non-confirmatory. | Diagnose the catastrophic trajectory drift before scaling; verify official-source identity and frozen split; execute all 11 registered sequences; inspect per-sequence trajectories, diagnostics, failures and runtime; publish reproducible metrics/artifacts and grouped uncertainty. |
| TUM-VI benchmark | WHITE | Dataset adapter/stream infrastructure is mentioned in existing documentation; no complete official sequence/checksum registry or executed benchmark evidence found in this audit. | Freeze official inventory, calibration/fisheye handling, partial-ground-truth rules, and shifted-test protocol before evaluating; run and report required sequences with censored/missing GT handled explicitly. |
| OpenVINS comparison | YELLOW | Pinned headless serial replay and V2 artifact pipeline pass for `MH_01_easy`; PR #104 records the fix. Artifact reports 0.09214 m ATE RMSE, but primary failure labels contain no events, so prediction discrimination is undefined. | Compare matched estimator inputs/conventions and assess additional EuRoC sequences; validate trajectory/covariance exports and cross-mode/version behavior. Keep failed runs visible. |
| Failure prediction models | YELLOW | Causal health features, failure targets, split/leakage guards, rules/logistic infrastructure, and frozen predictor/canonical evaluation code exist. | Implement and fairly compare required tree and temporal models where data volume supports them; evaluate all horizons; prove sequence/preprocessing isolation and seal models before held-out scoring. |
| Calibration and uncertainty | YELLOW | Calibration metrics, Platt/frozen predictor elements, conformal bounds and canonical held-out probability evaluation exist. | Verify independent calibration/test partitions; report Brier, NLL, ECE/MCE, reliability, risk–coverage and OOD outcomes; implement/justify temperature and isotonic comparisons; preserve the 0.2 false-alarm/min operating point where protocol requires it. |
| Degradation and robustness | YELLOW | Synthetic degradation and scenario evaluation infrastructure exists. | Run paired, deterministic, predefined-severity experiments for each required visual, IMU, timing, packet-loss and calibration perturbation; report nominal/degraded results without test-driven tuning. |
| Navigation shield | YELLOW | Stateful warning/slowdown/hold/halt/relocalization and recovery policy components have unit/closed-loop synthetic evidence. | Couple decisions to estimator failure outcomes; quantify risk reduction, false interventions and recovery success on matched sequences; validate state-machine transitions and conservative fallbacks; avoid unsupported formal guarantees. |
| Ablations | WHITE | No verified full ablation matrix or matched results located. | Retrain each specified ablation with the same development protocol and compare on identical held-out sequences, conditions and seeds. |
| ROS 2 and simulation | WHITE | No ROS 2 package or Dockerfile was present in the audited main tree. | Implement and test camera/IMU subscriptions, odometry/risk/diagnostic/shield publishers, launch/config/RViz, playback, supported simulator scenarios, ROS integration tests, and pinned reproducible container instructions. |
| Statistical analysis | YELLOW | Grouped statistics and metric aggregation modules/tests exist. | Verify physical-sequence aggregation, grouped bootstrap CIs, paired tests/effect sizes, multiplicity correction, sensitivity analysis, and explicit missing/censored handling on actual benchmark outputs. |
| Scientific manuscript | YELLOW | Paper outline, scope, abstract, contribution notes, related-work notes, and BibTeX exist. | Complete English manuscript and figures/tables from observed evidence; verify every citation and claim; compile LaTeX/PDF; state limitations and unresolved results. |
| Release and reproducibility | WHITE | Research prototype documentation, configs, workflows, and partial reproducibility notes exist. | Satisfy benchmark/CI/paper gates, then version a release or clearly labeled prerelease with environment, manifests, data instructions, changelog, citation and license review. |

## Immediate execution queue

1. Observe completion of main EuRoC paired run `45380153306`; inspect job logs and uploaded artifacts before changing benchmark status.
2. Resolve PR #104 against current main, reproduce the OpenVINS failure in its pinned workflow, and require a successful runtime job before describing the exception as fixed.
3. Reconcile older status/audit documents with the current code and evidence, retaining frozen protocols and historical audit dates.
4. Prioritize the smallest reproducible multi-sequence EuRoC vertical slice, then scale through the frozen sequence registry; do not tune on test sequences.
5. Freeze TUM-VI sequence identities/checksums and shifted-test rules before inspecting performance.
6. Build ROS 2, ablations, complete statistical analysis and the final paper only against executable interfaces and observed evidence.

## Decision rule

A workstream moves to **GREEN** only when implementation, tests or experimental artifacts, provenance, and its stated acceptance criteria agree. A successful CI run does not verify real-dataset performance, OpenVINS runtime behavior, cross-dataset generalization, safety benefit, or publication claims. No benchmark value, citation, significance result, or formal safety claim may be inferred from this tracker.
