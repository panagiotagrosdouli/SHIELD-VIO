# Split Protocol

## Experimental unit

The paper path splits complete physical sequences, never frames, rolling windows, or individual degradation events. Every derived condition and seed inherits its parent sequence partition.

The frozen partitions are `train`, `calibration`, `validation`, `test`, and `shifted_test`. `PaperSplit` rejects duplicate physical sequences and rejects a derived run whose requested split differs from the registered parent split.

## EuRoC V1 mapping

- Train: `MH_01_easy`, `MH_02_easy`, `V1_01_easy`
- Calibration: `MH_03_medium`, `V1_02_medium`
- Validation: `V2_01_easy`, `V2_02_medium`
- Test: `MH_04_difficult`, `MH_05_difficult`, `V1_03_difficult`, `V2_03_difficult`
- Shifted test: none assigned by the EuRoC-only mapping

This mapping is copied from `DATASET_SPLITS.md`; it was not selected from observed Phase C performance.

## Isolation rules

Training fits detector parameters, normalization and imputation. Calibration fits probability/conformal mappings only. Validation selects operating thresholds and predeclared model choice. Test and shifted-test cannot influence fitting or threshold selection.

Random frame splitting is invalid because adjacent VIO states, temporal features and degradation events are strongly dependent. The benchmark matrix is therefore resolved from registered sequence identities only; an unknown sequence is rejected instead of being randomly assigned.

## TUM-VI

TUM-VI assignments are intentionally absent until the official export inventory, ground-truth intervals and reproducibility identities are frozen. Until then, TUM-VI can have fixture/integration and local smoke execution, but not a claimed confirmatory partition.
