# Public Benchmark Status

Status vocabulary: `NOT_AVAILABLE`, `DATASET_VALIDATED`, `SMOKE_COMPLETE`, `BENCHMARK_COMPLETE`, `FAILED`.

## EuRoC declared clean matrix

| Sequence | Split | Status |
|---|---|---|
| MH_01_easy | train | SMOKE_COMPLETE* |
| MH_02_easy | train | NOT_AVAILABLE |
| V1_01_easy | train | NOT_AVAILABLE |
| MH_03_medium | calibration | NOT_AVAILABLE |
| V1_02_medium | calibration | NOT_AVAILABLE |
| V2_01_easy | validation | NOT_AVAILABLE |
| V2_02_medium | validation | NOT_AVAILABLE |
| MH_04_difficult | test | NOT_AVAILABLE |
| MH_05_difficult | test | NOT_AVAILABLE |
| V1_03_difficult | test | NOT_AVAILABLE |
| V2_03_difficult | test | NOT_AVAILABLE |

`NOT_AVAILABLE` means no qualifying sequence execution is recorded here; it is not a statement that the public dataset cannot be obtained. Historical Phase C status is retained below.

## TUM-VI

Official paper sequence registry: `NOT_AVAILABLE` pending frozen official export inventory/checksum identities. Adapter and executable stream runner exist, but fixture evidence must not be renamed public validation.

## MH_01_easy smoke evidence (2026-11-24)

`SMOKE_COMPLETE*` records execution only. The [paired EuRoC GitHub Actions run](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45380153306) completed and uploaded artifact `shield-vio-real-euroc-mh01-paired` (artifact ID `14912809427`, SHA-256 `9772d64f57e9973fa47d93b8725059693b2df025aa65dcb9b0651fc243677179`).

- IMU-only ESKF: ATE RMSE 69,371.62 m; translational RPE RMSE 71.68 m at one-sample delta.
- ESKF plus monocular relative-rotation updates: ATE RMSE 76,633.96 m; translational RPE RMSE 80.26 m. This was worse than the IMU-only run on this sequence.
- The public-data failure-prediction smoke had 10 eligible windows, all positive, so AUROC and AUPRC are undefined. It is a pipeline smoke, not evidence of predictive performance.
- The stationary-IMU gravity initialization was rejected because the selected input window did not meet its stationarity checks.
- The checksum-verified source receipt uses a mirrored ROS bag with pinned OpenVINS calibration and ground truth; it explicitly does not establish identity with the current ETH archive.

`SMOKE_COMPLETE*` does not mean the estimator passed an accuracy gate. Retain this run as a diagnostic failure; do not use it to claim benchmark completion, confirmatory performance, model discrimination, or generalization.

## OpenVINS headless smoke evidence (2026-11-24)

Pinned OpenVINS revision `69488123ed9362dd44b6f28e7f4680abbff1442b` ran serially on `MH_01_easy` in [workflow run 45382473608](https://github.com/panagiotagrosdouli/SHIELD-VIO/actions/runs/45382473608) after the headless visualization-publisher patch. The runtime and artifact pipeline passed; the reported `LibraryUnloadException` did not recur. Artifact `14915278229` has SHA-256 `366b13ee50c001bbda6999eaba9668e9a41ef94e6cbd67c06554e4978c7330cb`.

- The estimator received no ground truth. The receipt records a 138 s run and a verified bag digest.
- 2,761 trajectory samples were associated; ATE RMSE was 0.09214 m and translational RPE RMSE was 0.01628 m at a one-sample delta.
- Primary observability audit passed for label construction, with 2,741 fully observable samples and 42 censored. The frozen primary failure definition produced 0 events and 0 positive target windows at all horizons (0/2,701 at 2 s), so predictor discrimination is undefined.
- This is one non-confirmatory development smoke, not an all-sequence comparison or an OpenVINS accuracy/generalization claim.

## Evidence boundary

The two one-sequence `PUBLIC_DATASET_SMOKE` executions above are smoke evidence only. No all-sequence `PUBLIC_DATASET_BENCHMARK` or `PUBLIC_DATASET_CONFIRMATORY` result is claimed. The remaining ten EuRoC matrix sequences and the TUM-VI sequence registry remain unavailable in this status. Historical Phase C note: at that implementation snapshot, no local real-dataset execution had been performed.
