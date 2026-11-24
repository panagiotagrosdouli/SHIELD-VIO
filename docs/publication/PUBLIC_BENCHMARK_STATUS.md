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

## Evidence boundary

Only the one-sequence `PUBLIC_DATASET_SMOKE` execution above is recorded. No all-sequence `PUBLIC_DATASET_BENCHMARK` or `PUBLIC_DATASET_CONFIRMATORY` result is claimed. The remaining ten EuRoC matrix sequences and the TUM-VI sequence registry remain unavailable in this status. Historical Phase C note: at that implementation snapshot, no local real-dataset execution had been performed.
