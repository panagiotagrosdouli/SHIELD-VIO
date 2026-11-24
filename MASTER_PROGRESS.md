# SHIELD-VIO Master Progress

**Snapshot:** 2026-11-24 (UTC)  
**Audited base:** `main` includes stereo attitude-Jacobian correction merge `f1cc342be060f227145dc0b3641f0c7e51138603`  
**Purpose:** Track implementation and evidence without treating planned work, unit tests, synthetic runs, or an in-progress workflow as real benchmark validation.

## Status key

- **GREEN** — implemented and verified against its stated acceptance criteria.
- **YELLOW** — partially implemented, under execution, or awaiting required validation.
- **RED** — an observed failure or blocker prevents acceptance.
- **WHITE** — not started or no implementation/evidence found in the current audit.

## Current repository evidence

| Evidence | State at snapshot |
|---|---|
| Main CI evidence | **GREEN on candidate heads:** CI run [45421082731](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45421082731) and paired benchmark [45421082348](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45421082348) passed for PR #124. Earlier evidence: CI runs [45386831620](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45386831620), [45387125611](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45387125611), and [45387445677](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45387445677), [45388990757](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45388990757) passed. The last candidate also passed the real EuRoC workflow before squash-merge `d1e1306`. |
| Real EuRoC paired benchmark | **RED for estimator accuracy / GREEN for execution:** latest paired run [45479238539](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45479238539) uploaded artifact `shield-vio-real-euroc-mh01-paired` (ID `14961872095`; SHA-256 `d3ec851fef4f2f3e96e5f044f1639f940a26139b48979965038e2b613f3063a5`). After stereo attitude-Jacobian correction, ATE/RPE are 798.95/1.198 m; NIS median is 81.55 and 96.13% exceed the nominal 95% reference. The estimate remains extremely divergent. Earlier drift evidence is below. |
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

**Code-path finding (hypothesis, not yet confirmed):** the ESKF starts with identity attitude and zero biases unless stationary-IMU gravity alignment is explicitly requested; propagation applies `R @ accel + gravity`. The IMU-only run has no visual updates to estimate biases, and its stationary 100-sample initialization window was rejected. This setup can leak gravity into world-frame acceleration if the initial attitude is not aligned. The output supports rapid divergence, but the exact cause still requires raw IMU/frame checks and an independent initialization experiment.

**Disposition:** retain the severe drift as a RED estimator-accuracy finding. Do not tune on held-out sequences; diagnose frame, unit, initialization, and propagation conventions on predeclared training data before scaling the benchmark.

## Workstream status and acceptance gates

| Workstream | Status | Current implementation / evidence | Remaining acceptance criteria |
|---|---|---|---|
| Repository, PR, and CI audit | YELLOW | PRs #104–#117 and #119–#125 are integrated; #118 was closed unmerged after the then-current converter failed to export cam1. PR #123 CI [45418359701](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45418359701), development replay [45418359699](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45418359699), and paired EuRoC [45418360673](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45418360673) passed. | Review inactive branches/workflows and audit remaining artifacts; keep this record current. |
| CI/CD and regression suite | YELLOW | Main CI run [45396057574](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45396057574) passed after PR #113 merge. Tracker commit CI [45396135430](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45396135430) also passed Ruff, pytest, deterministic synthetic regression, and end-to-end checks. PR #115 candidate CI [45398814742](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45398814742) and paired EuRoC run [45398814585](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45398814585) passed; main push CI run [45400097960](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45400097960) passed on the tracker commit. | Run full required checks on the final release candidate, inspect failures instead of suppressing them, and confirm all required workflows pass on the release commit. |
| VIO estimation backend | YELLOW | Research implementations exist for feature tracking, IMU preintegration, a 15-state ESKF, calibration/stereo geometry, trajectory output, and loop closure, with numerical/unit tests. README correctly calls these research prototypes. | Independently validate coordinate/quaternion conventions, Jacobians and covariance consistency; exercise complete calibrated camera–IMU synchronization, initialization/recovery, stereo and monocular pipelines; document limitations with measured evidence. |
| EuRoC benchmark | RED | One checksum-pinned mirrored `MH_01_easy` stereo paired smoke completed. Monocular and stereo ESKF variants diverge catastrophically; stereo reduced ATE by about 98.94% versus rotation-only but still has 809.79 m ATE (~10× GT path). PR #124 records median innovation NIS 80.41, extreme trajectory speed, and ill-conditioned covariance. OpenVINS has a separate successful 138 s smoke. All are non-confirmatory and only one sequence is tested. | Diagnose ESKF failure on development data; verify source identity; execute all 11 registered sequences under the frozen split; inspect outcomes and runtime; report grouped uncertainty. |
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

1. Diagnose remaining stereo drift, including camera distortion handling, visual measurement covariance/correlation, and predeclared known-motion checks.
2. Extend EuRoC execution across the frozen 11-sequence registry after the estimator is stable; preserve all failed outcomes.
3. Freeze TUM-VI inventory/checksums and shifted-test rules before evaluating any sequence.
4. Implement and fairly compare the remaining prediction, calibration, robustness, and shield variants on valid development/test partitions.
5. Build ROS 2 integration, matched ablations, sequence-level statistics, the manuscript, and release artifacts only from observed evidence.
6. Record CI on each final candidate and keep benchmark status synchronized with artifacts.

## Decision rule

A workstream moves to **GREEN** only when implementation, tests or experimental artifacts, provenance, and its stated acceptance criteria agree. A successful CI run does not verify real-dataset performance, OpenVINS runtime behavior, cross-dataset generalization, safety benefit, or publication claims. No benchmark value, citation, significance result, or formal safety claim may be inferred from this tracker.

## Stationary IMU directional-dispersion candidate (2026-11-24)

**YELLOW — merged as PR #113 at `ff45791637710120b4a89666c292ed7ea8715f46`; implementation, regression tests, CI, and development replay have been verified.** The vector RMS dispersion check rejects constant-magnitude directional variation, and the near-zero mean guard prevents invalid gravity normalization. CI [45395274245](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45395274245) passed. Development MH_02_easy run [45395274172](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45395274172) and paired MH_01_easy run [45395274116](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45395274116) completed. In MH_01_easy the 100-sample startup window was rejected as nonstationary; no gravity-aligned trajectory was generated. The default estimator still has catastrophic drift (ATE 69,371.62 m against an 80.63 m GT path); this guard does not resolve it. Keep EuRoC estimator accuracy RED and continue development-only diagnosis.

## Live GitHub verification — 2026-11-24

- **PR #114 merged:** squash commit `9d846d4fdeb847d7b9d727a1d4014c7439d53f8f`. CI run [45393123639](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45393123639) passed Ruff, pytest, deterministic synthetic benchmark, and end-to-end benchmark regression gate. The change rejects non-binary/non-finite calibration labels before integer conversion. This validates the integrity fix; it does not change the frozen evaluation protocol.
- **PR #113 merged:** squash commit `ff45791637710120b4a89666c292ed7ea8715f46` after rebasing its implementation onto `main` commit `81f24ec`. CI [45395274245](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45395274245) passed Ruff, pytest, deterministic synthetic benchmark, and end-to-end regression. Development replay [45395274172](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45395274172) completed; artifact `14920162832` (SHA-256 `e25bf7297a845132df6350a685d1842a6c46c332d315fc284d35969264d0627e`) reports MH_02_easy ATE RMSE 33,570.27 m and RPE translation RMSE 44.98 m, with 10/10 positive windows at all horizons.
- The rebased real paired EuRoC run [45395274116](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45395274116) completed successfully and uploaded artifact `14919604871` (SHA-256 `b4bbe5d9d9fa8cf6b510cb462b6450740d165bf7404f4ca44025797141ea7c14`). For MH_01_easy, IMU-only ATE RMSE is 69,371.62 m and monocular rotation ATE is 76,633.96 m (GT path 80.63 m); the 100-sample gravity initialization is rejected as nonstationary. This validates pipeline behavior, not estimator accuracy or a fix for divergence. The smoke labels have 10 eligible, all-positive windows at each horizon; AUROC/AUPRC remain undefined.
- **ESKF accuracy remains RED.** Source inspection confirms the default backend starts with identity attitude and zero biases and propagates (R a + g); the development workflow's first-window stationary initializer rejects this sequence. This is consistent with gravity leakage as a hypothesis, but does not establish the root cause. Continue frame/unit/initialization/propagation diagnosis on development data; do not claim the stationarity guard resolves the drift.


## Position-aligned attitude diagnostic — run 45398814585

PR #115 was squash-merged as `bd958b9a2861f9c98888f680276255ff255beaa8` after candidate CI [45398814742](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45398814742) and the real paired EuRoC run [45398814585](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45398814585) passed. The paired artifact is `14922008885` (SHA-256 `5dc6827065ff87f35ecc7fbb6fa1b83ddaa7fd5f589482d5602dbc3fc21e88b8`). The audit paired 3,639 samples and applied one global position-derived rotation to each estimated attitude sequence:

| Run | Median attitude error | First 5 s median | 95th percentile |
|---|---:|---:|---:|
| IMU-only | 123.28° | 121.04° | 173.88° |
| OpenCV rotation | 157.96° | 103.05° | 176.44° |

The report is explicitly non-confirmatory and ground truth is used only post-hoc. Large residual attitude error remains after a single position-derived global rotation; this is a diagnostic association, not proof that attitude initialization caused trajectory divergence. ESKF accuracy remains **RED**. Next development-only checks should inspect timestamped raw IMU/ground-truth frame conventions and short-window attitude propagation with independently verified initialization before changing estimator behavior.
Supplemental post-hoc analysis of the same artifact compares relative rotations from the first paired sample, which removes one constant world-frame attitude offset. First-5-s interval median error is 11.66° for IMU-only and 8.46° for OpenCV rotation; over the full sequence, medians are 113.85° and 108.17° (95th percentiles 171.05° and 164.94°). The first paired estimator orientation is identity; the ground-truth body-down direction is tilted 65.42° from the estimator's assumed body-down direction. This supports testing gravity/frame initialization as a candidate cause, but remains post-hoc and depends on matching the EuRoC body/IMU frame convention. The initial absolute quaternion discrepancy includes arbitrary world-frame yaw and should not be interpreted as a pure tilt error.

**Alignment caveat:** the earlier position-aligned attitude residual fits one rotation using the entire trajectory. Given the enormous translational divergence, that rotation is not a trustworthy estimate of the true fixed world-frame transform; treat those absolute residuals as descriptive only. Relative-rotation errors are the better diagnostic signal here, and raw IMU/ground-truth frame consistency still needs independent verification before any estimator change.


## Raw IMU, initialization, and gyro-bias diagnostics — PRs #116–#122 (2026-11-24)

- **PR #116** merged as `092ad080b9bdb5725fb418ef076c2bfa14b63a5a`. The post-hoc raw IMU/GT audit found the direct quaternion direction and positive gravity convention best match a smoothed position-derived acceleration reference (RMSE 0.7150 m/s², versus 9.7467 and 11.3165 m/s² for the compared alternatives). This is convention evidence, not independent sensor calibration.
- **PR #117** merged as `79d230e2bd35679c968f02d4854f3bdf3514c32b`. Startup windows of 10, 20, 50, 100, 200, and 400 IMU samples were all rejected as nonstationary by the existing guard; no threshold was loosened.
- **PR #118** was closed unmerged. Its stereo-PnP paired run [45404947860](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45404947860) stopped because that version of the portable converter exported cam0 and IMU only. PR #123 later showed the pinned checksum-verified mirror itself contains cam1; the converter had simply discarded it. PR #118 produced no stereo estimator result.
- **PR #119** merged as `ee1e53202fda00686244686a9c362ffef5a02fdc`. GT attitude initialization alone reduced ATE from 68,322.9 m to 52,401.3 m (23%), leaving catastrophic drift.
- **PR #120** merged as `83852d789b46370eb6eac20a57c7317b929404a6`. Direct body-rate convention RMSE was 0.05241 rad/s versus 0.28813 rad/s for reversed orientation. The post-hoc GT-derived mean gyro residual was [-0.00320, 0.02130, 0.07839] rad/s; differentiation/frame effects prevent interpreting it as calibrated sensor bias.
- **PR #121** merged as `e28dffbf6d4c61f9292e8d35fb242f81ebde33fb`. Oracle GT attitude plus GT-derived bias reduced ATE to 2,088.1 m and RPE to 0.2256 m/sample, but remained far from useful navigation accuracy. This leaks GT and is only an upper-bound diagnostic.
- **PR #122** merged as `fe603e06c38eabb11d85e3af4d5a66e58f616126`. CI, development replay, and paired MH_01_easy workflow all passed. Artifact `14931240767`, SHA-256 `93e393dd2e99dadbe59ae2d0f9df21967da572b44228ebfcb82af77b19ccf9e0`. With OpenCV rotation updates, default prior gave ATE/RPE 76,633.96/80.2610 m. Initial gyro-bias standard deviations 0.05, 0.1, and 0.2 rad/s gave ATE/RPE 78,986.09/82.7250, 81,830.25/85.5831, and 76,534.06/81.0385 m, respectively. The largest prior changed ATE by only -0.13% while worsening RPE by about 0.97%; none repaired the failure. Estimated median bias vectors ([0.00195, 0.00548, 0.02730], [0.00867, -0.00889, 0.03969], and [0.01807, -0.02597, 0.05392] rad/s) do not converge to the post-hoc GT-derived residual. This is a non-confirmatory sensitivity result; GT was used only for trajectory evaluation.

**Updated diagnosis:** frame direction, acceleration units, and gravity sign are broadly consistent with the audit reference. Attitude-only oracle initialization helps modestly; the gyro-bias prior sweep does not. Oracle attitude plus bias greatly reduces drift but remains catastrophically inaccurate. A likely remaining structural limitation is that the current backend is not full metric visual-inertial odometry: monocular updates provide rotation constraints, not translation/scale. Stereo metric updates now reduce errors substantially, but accuracy remains **RED**; the cause of the remaining drift needs calibrated residual/covariance review and repeat runs on additional stereo sequences.


## Stereo-PnP replay enabled — PR #123 (2026-11-24)

PR #123 merged as `79befa5aa152264429e385bf71ea773b2d73caf1`. It updates the rosbag converter to preserve cam1 images/calibration, associates right-camera images by timestamp with a 10 ms cap, conditionally runs stereo-PnP, and fixes the OpenCV `undistortPoints` call to pass camera intrinsics. Unit CI [45418359701](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45418359701), development run [45418359699](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45418359699), and paired EuRoC run [45418360673](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45418360673) passed.

- The converter recovered 3,682 cam0 and 3,682 cam1 frames from the checksum-pinned 2,673,818,914-byte smoke bag. Both camera CSV timestamp streams match. The official legacy ZIP and ROS-bag URLs were unavailable in the runner, so source identity remains the existing third-party mirror plus pinned OpenVINS exports; no confirmatory identity is claimed.
- Stereo-PnP generated 3,510 visual updates across 3,682 frames. At 3,638 associated GT samples, ATE/RPE were 809.79 m / 1.20884 m per sample. The paired monocular-rotation baseline was 76,633.96 m / 80.2610 m per sample. These correspond to 98.94% lower ATE and 98.49% lower RPE, but ATE remains about 10.0 times the 80.63 m ground-truth path length.
- The paired artifact is `14934190181`, SHA-256 `1b4a40df0078f75f246e7da6f471295b11cb6bbb9c986abfd851d895304b2406`. It remains public-data smoke evidence, non-confirmatory.

**Disposition:** preserving stereo metric updates is a promising estimator improvement, not a pass. Keep accuracy **RED** until calibration, update residuals/covariance, outlier rejection, and consistency are reviewed and the improvement is repeated across the registered stereo sequences with a frozen official data identity.

**Next actions:** inspect stereo visual update status/innovation and residual distributions; verify cam0/cam1 calibration transforms and their convention against the official archive; compare open-loop and stereo-assisted drift by time segment; then run the frozen multi-sequence EuRoC split and report failures. Continue to keep the current estimator accuracy gate RED.



## Stereo-PnP health audit — PR #124 (2026-11-24)

PR #124 was squash-merged as `816605a11819a14e5e302c34d2837528f1369b89`. CI [45421082731](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45421082731) and paired EuRoC run [45421082348](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45421082348) passed. The run artifact is `14936091337`, SHA-256 `5deb36736f76f3ecc5f2a7a945b959701131ca73ca77337abf4b13118891692d`.

- Stereo-PnP emitted 3,510 accepted updates from 3,682 camera frames; median correspondences/inliers were 259/248.
- Six-DOF innovation NIS median was 80.41 and p95 was 629.62; 95.95% exceeded the nominal 95% chi-square reference. These rates are descriptive only: sequential updates and frontend selection violate the simple independent-sample interpretation.
- Covariance symmetry error stayed near machine precision and observed minimum eigenvalues were positive, but the covariance condition number had median 29,414.7 (p95 48,347.2).
- The estimated step-speed median was 16.47 m/s (p95 48.14 m/s), maximum position norm 2,830 m, ATE 809.79 m, and one-sample RPE 1.209 m. This is not an accuracy fix despite the large reduction relative to monocular rotation-only replay.

**Disposition:** keep estimator accuracy **RED**. Treat high NIS as a strong residual/model-review lead, not a formal consistency verdict. Verify stereo calibration and residual construction, inspect time-segment drift and rejection behavior, then repeat on the frozen multi-sequence stereo split with confirmed source identity. The vertical-slice gate passing in the same run remains pipeline smoke only: all 10 eligible target windows were positive, so discrimination remains undefined.


## Stereo attitude-Jacobian frame correction — PR #125 (2026-11-24)

PR #125 merged as `f1cc342be060f227145dc0b3641f0c7e51138603`. CI [45479238528](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45479238528), development replay [45479238717](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45479238717), and paired EuRoC run [45479238539](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45479238539) passed. Artifact `14961872095` (SHA-256 `d3ec851fef4f2f3e96e5f044f1639f940a26139b48979965038e2b613f3063a5`).

- Source review found that stereo rotation residuals `log(R_measured R_predicted.T)` are expressed in the world frame, while the ESKF injects right-multiplicative attitude errors in body coordinates. PR #125 maps these through the inverse SO(3) right Jacobian and predicted rotation.
- On MH_01_easy, ATE/RPE changed from 809.79/1.20884 m to 798.95/1.19809 m (about 1.34%/0.89% lower). Median NIS rose from 80.41 to 81.55; 96.13% of updates still exceed the nominal 95% reference.
- This fixes a real linearization-frame defect, but the small metric change and persistently extreme NIS show it does not explain most of the drift. Accuracy remains **RED**.

**Next source-inspection lead:** stereo calibration objects expose camera distortion coefficients, but the current stereo frontend receives only camera matrices and uses raw feature pixels with zero/unspecified distortion in triangulation and PnP. Verify the dataset image/calibration convention, then compare distortion-aware undistortion against the current behavior before treating it as a confirmed cause.
