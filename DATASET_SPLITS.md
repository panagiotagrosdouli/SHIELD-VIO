# Dataset Splits and Data Governance

## Non-negotiable rules

1. Splits are defined by complete sequence, never by frame or rolling window.
2. A physical sequence cannot appear in more than one of train, calibration, validation, test, or shifted test, including under a different degradation or estimator.
3. All degradation variants and seeds derived from a sequence inherit that sequence’s split.
4. Normalization and missing-value models fit on training sequences only.
5. Detector parameters fit on training sequences only.
6. Probability and conformal calibration fit on calibration sequences only.
7. Operating thresholds and policy costs are selected on validation sequences only.
8. Confirmatory metrics are computed once on test and shifted-test sequences after the protocol is frozen.
9. Cross-dataset transfer treats the entire target dataset as unavailable to training unless the experiment explicitly names a target-dataset calibration condition.
10. Dataset, sensor, calibration, and file-index checksums are stored before execution.

## Split registry schema

The machine-readable registry will contain one row per physical sequence:

| Field | Meaning |
|---|---|
| `dataset` | Stable dataset identifier and release/export variant |
| `sequence` | Canonical sequence identifier |
| `split` | `train`, `calibration`, `validation`, `test`, or `shifted_test` |
| `environment_family` | Dataset-provided environment category |
| `ground_truth_coverage` | Full, partial, or none, with evaluated intervals |
| `camera_variant` | Resolution, stereo/mono stream, photometric variant |
| `sensor_rates_hz` | Declared and measured camera/IMU/ground-truth rates |
| `index_checksum` | Hash of timestamp/file indexes |
| `calibration_checksum` | Hash of sensor calibration files |
| `source` | Official dataset page or archive URL |
| `license_note` | Redistribution and citation requirements |

The split loader rejects duplicates, unknown split names, checksum mismatches, and any derived run whose parent sequence has a different split.

## EuRoC MAV confirmatory split

The existing `configs/euroc/benchmark_v1.yaml` has development, calibration, and test groups but no independent validation group. The paper split below supersedes that three-way design and must be encoded in `configs/datasets/euroc_paper_v1.yaml` before confirmatory execution.

| Split | Sequences | Purpose |
|---|---|---|
| Train | `MH_01_easy`, `MH_02_easy`, `V1_01_easy` | Feature preprocessing and detector fitting |
| Calibration | `MH_03_medium`, `V1_02_medium` | Probability and conformal calibration only |
| Validation | `V2_01_easy`, `V2_02_medium` | Detector/policy thresholds, history length, and predeclared model choice |
| Test | `MH_04_difficult`, `MH_05_difficult`, `V1_03_difficult`, `V2_03_difficult` | One-time confirmatory in-dataset evaluation |

All nominal and degraded versions of a sequence remain in the same split. Development may use only train, calibration, and validation groups; test labels remain sealed until the pipeline and analysis plan are frozen.

## TUM-VI split construction

TUM-VI provides room, corridor, magistrale, outdoors, and slides sequence families, with different ground-truth coverage. The paper uses the official EuRoC/DSO export at one frozen resolution. Exact archive filenames must be imported from the official index and checksum-verified before sequence assignments are frozen; filenames are not guessed in this document.

The TUM-VI registry will be constructed by this deterministic rule:

1. inventory all official exported sequences and ground-truth intervals;
2. exclude calibration recordings from evaluation sequences;
3. stratify by environment family and usable ground-truth duration;
4. sort canonical sequence IDs within each family;
5. allocate complete sequences to train/calibration/validation/test using a checked-in versioned mapping;
6. ensure room sequences with full ground truth occur in every available split and report partial-coverage sequences separately;
7. reserve the slides family, plus at least one outdoor sequence, as a shifted test when those families are absent from training;
8. publish the registry, official index checksum, and exact ground-truth intervals before model fitting.

Until that registry exists, TUM-VI support is **planned** and no TUM-VI empirical claim is permitted. The present `discover_tumvi_sequence` adapter has mocked layout coverage only.

## Experiment families

### In-domain

Fit, calibrate, validate, and test on disjoint sequences from the same dataset using the frozen registry. Degradation families may repeat across splits, but their random seeds and event schedules are deterministic and recorded.

### Cross-sequence

Use the EuRoC split above. The test sequences are never used to choose feature groups, detector family, calibration mapping, thresholds, or shield parameters.

### Cross-dataset

Two directional experiments are defined:

- EuRoC train/calibration/validation -> TUM-VI test, with no TUM-VI adaptation;
- TUM-VI train/calibration/validation -> EuRoC test, with no EuRoC adaptation.

A separately named `target_calibration` experiment may fit only a calibration mapping on target calibration sequences while keeping detector weights fixed. It cannot replace the zero-adaptation result.

### Unseen degradation family

Partition degradation families, not individual events:

- training families: darkness, contrast reduction, additive image noise, accelerometer noise, gyroscope noise, mild packet loss;
- validation families: overexposure, feature dropout, bias drift;
- shifted-test families: occlusion, frame dropout, saturation, axis failure, and combined degradations.

The final family allocation is configuration-controlled. If a family is used in training, none of its severity variants count as “unseen family.”

### Unseen severity

Train on low and medium severity, tune on separate medium-severity runs, and evaluate high severity on test sequences. Severity values are fixed numerically before test execution.

### Estimator shift

Detector fitting on the internal ESKF and testing on an established backend is reported separately from sensor/domain shift. Missing signal groups are represented explicitly; unavailable values are never imputed from ground truth or another estimator.

## Degradation-event allocation

For every parent sequence and degradation condition:

- use identical event start times, durations, and seeds across methods;
- prevent event overlap in single-family experiments;
- require adequate nominal context before the first event and after recovery;
- store original and degraded stream checksums or deterministic transformation manifests;
- retain injected metadata only in the experimental-condition table;
- build failure labels solely from estimator/navigation outputs;
- count the sequence-condition-seed run, not individual frames, as an independent unit.

## Minimum evaluation size

The minimum paper experiment contains at least 20 independent test sequence-degradation conditions, at least two public datasets, multiple horizons, and enough failure and non-failure events to report both event recall and false alarms per minute. If a declared condition produces no observable failure, it remains a valid negative condition and is not removed after inspection.

## Dataset execution states

| State | Required evidence |
|---|---|
| `REGISTERED` | Official source, license/citation note, sequence mapping, checksums |
| `VERIFIED` | Expected streams, monotonic timestamps, rate and calibration checks pass |
| `ESTIMATED` | Standardized estimator outputs and runtime metadata exist |
| `LABELED` | Observable failure events and horizon targets exist |
| `EVALUATED` | Predictions, metrics, and manifest are complete |
| `CONFIRMATORY_READY` | Split seal, configuration freeze, and leakage tests pass |

The first EuRoC vertical slice may be labeled `PUBLIC_DATASET_SMOKE` after one real sequence executes end to end. It does not become confirmatory evidence until the independent split and minimum-size conditions above are met.

