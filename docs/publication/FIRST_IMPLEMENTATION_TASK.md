# SHIELD-VIO — First Implementation Task

## Goal

**Freeze and version the executable primary failure-event definition and its future-horizon targets.**

This is the first task because the originally preferred canonical timestamped causal health representation is already implemented on current `main` in `shield_vio/features/health_vector.py` and has causal/leakage tests. The remaining P0 defect is that the repository has multiple sample-level failure criteria and an event builder, while the public vertical slice uses position error alone; there is no single versioned primary-vs-sensitivity contract that every later benchmark must consume.

This task is small enough for one focused PR, requires no dataset download, and is prerequisite infrastructure for split enforcement, baseline fairness, calibration, event metrics and public-data experiments.

## Exact code changes

1. Add `configs/paper/failure_primary_v1.yaml` containing the frozen primary criterion set, numeric thresholds, persistence, merge/recovery behavior if used, censoring rules, horizons `0.5, 1.0, 2.0, 3.0, 5.0`, warning deadline semantics, and schema/version identifier.
2. Add a typed configuration object in `shield_vio/evaluation/failure_labels.py` or a small adjacent module that loads/validates this config without adding dependencies.
3. Add one public function that converts timestamped **offline observations** into criterion flags and a versioned event table. Ground-truth-derived quantities may be accepted only by this offline label layer and must never be returned as deployable features.
4. Reuse `build_persistent_failure_events()` and `future_failure_targets()` rather than reimplementing event/horizon logic.
5. Add explicit event IDs and excluded/censored-sample reasons to the returned/exportable structure if not already represented.
6. Add separate sensitivity config(s) only as examples/tests; do not silently mix them with the primary definition.
7. Update the vertical-slice label construction to call the frozen label API while preserving `PUBLIC_DATASET_SMOKE` and nonconfirmatory status. Do not change detector architecture or generate new scientific claims.

## Proposed API

```python
@dataclass(frozen=True)
class FailureDefinition:
    version: str
    persistence_seconds: float
    horizons_seconds: tuple[float, ...]
    # frozen primary thresholds / enabled criteria

@dataclass(frozen=True)
class FailureEventTable:
    definition_version: str
    event_ids: tuple[str, ...]
    onsets_ns: np.ndarray
    offsets_ns: np.ndarray
    active_mask: np.ndarray

@dataclass(frozen=True)
class HorizonTargetSet:
    definition_version: str
    targets: dict[float, FutureFailureTargets]


def build_failure_events_and_targets(
    timestamps_ns: np.ndarray,
    observations: OfflineFailureObservations,
    definition: FailureDefinition,
) -> tuple[FailureEventTable, HorizonTargetSet]:
    ...
```

The exact class names may vary, but the API must make the definition version and offline/deployable boundary explicit.

## Tests

Create `tests/test_paper_failure_definition.py` with at least:

- deterministic event construction from a hand-built timestamp sequence;
- persistence changes onset exactly as declared;
- current active failure is ineligible for prediction;
- future target is exactly `(t, t + tau]`;
- incomplete tail is censored;
- event IDs are stable for identical input/config;
- primary and sensitivity configs produce distinguishable version IDs;
- degradation family/severity/onset metadata cannot be accepted as a primary criterion;
- no field produced for deployable features contains ground truth/failure/future/degradation/oracle information;
- vertical-slice fixture still writes a complete nonconfirmatory artifact set.

## Backward compatibility

Preserve `FailureThresholds`, `FailureObservation`, `label_failure`, `build_persistent_failure_events`, and `future_failure_targets` during this PR unless a correctness defect requires otherwise. The new API should compose these primitives and give later experiment code a single frozen entrypoint. Existing synthetic and EuRoC runner behavior must remain unchanged.

## Acceptance criteria

The PR is accepted only when:

1. one checked-in primary failure-definition version exists;
2. the same inputs/config deterministically produce the same event IDs/onsets/offsets and all five horizon targets;
3. degradation metadata is structurally excluded from the label criterion;
4. ground truth remains confined to offline label/evaluation construction;
5. active failure and incomplete tails are correctly excluded from deployable prediction targets;
6. sensitivity definitions cannot silently replace the primary version;
7. existing tests plus the new failure-definition tests pass;
8. no public dataset download is required.

## Commands to validate

```bash
ruff check shield_vio scripts tests
pytest -q tests/test_paper_failure_prediction_core.py \
  tests/test_paper_leakage_guards.py \
  tests/test_paper_vertical_slice.py \
  tests/test_paper_failure_definition.py
pytest -q
```

## What not to do in this PR

Do not add deep learning, refactor the estimator, add an external backend, download EuRoC/TUM-VI, tune thresholds on test data, or redesign the health feature matrix. Those belong to later milestones.
