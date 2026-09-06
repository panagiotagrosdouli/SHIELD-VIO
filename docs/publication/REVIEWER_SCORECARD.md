# SHIELD-VIO Reviewer Scorecard

Scored against the repository as it exists now, not the planned system. Scale: 1 = poor/unsupported, 3 = promising but incomplete, 5 = strong publication-grade evidence.

| Dimension | Current score | Reason | Score required for submission | Actions required |
|---|---:|---|---:|---|
| Novelty | 3 | The joint estimator-introspection → future-risk → calibration/shift → protective-control framing is coherent, but the novelty matrix/literature comparison is not yet completed and individual ingredients are established ideas. | 4 | Structured literature review; define novelty as the joint causal/evaluated pipeline; avoid claiming novelty for individual diagnostics/calibration/state machines. |
| Technical correctness | 4 | Causal source timestamps, privileged-feature rejection, persistent events, horizon targets, split guards, event metrics and Platt calibration show careful implementation. | 4 | Freeze the primary failure contract; extend leakage tests across full orchestration; validate external-backend semantics. |
| Experimental rigor | 2 | Strong protocol documents and smoke tooling exist, but no complete frozen confirmatory experiment across public splits is stored. | 4 | Execute sequence-disjoint train/calibration/validation/test/OOD protocol with identical run keys and stored raw predictions. |
| Dataset adequacy | 2 | EuRoC infrastructure is substantial; TUM-VI is discovery-only and public multi-sequence confirmatory evidence is absent. | 4 | Complete EuRoC benchmark plus real TUM-VI execution; retain checksums and exact sequence counts. |
| Baselines | 2 | Heuristics and logistic exist; tree baseline is absent and the proposed predictor is not yet a clearly benchmarked method distinct from references. | 4 | Common API; covariance/feature/NIS/tracking/moving-average/logistic/tree/proposed comparisons on identical splits/horizons. |
| Calibration methodology | 3 | Platt and conformal primitives plus Brier/NLL/ECE exist; held-out public-data calibration and slope/intercept/isotonic coverage are incomplete. | 4 | Calibration-only fit, held-out reliability evaluation, validation-only threshold, raw-vs-calibrated paired comparison. |
| OOD methodology | 2 | Rolling z-score shift state machine exists, but no fitted reference artifact or unseen-condition benchmark demonstrates benefit. | 4 | Freeze ID reference, evaluate unseen family/severity/sequence/cross-dataset shift, report delay/false alarms/calibration/selective risk and policy effect. |
| Statistical rigor | 2 | Event metrics are good primitives, but current aggregate statistics use normal-theory CI and lack grouped paired bootstrap/effect sizes. | 4 | Implement run-unit paired grouped bootstrap, effect sizes, exact denominators and paired comparisons; never infer from frames as replicates. |
| Closed-loop relevance | 2 | Stateful shield and speed/hold/halt/relocalization logic exist, but simple point navigation does not yet establish localization-error-caused unsafe behavior or a safety-utility trade-off. | 4 | Estimated-state-driven controller, boundaries/obstacles, identical replay across policies, unsafe exposure/completion/intervention/recovery metrics. |
| Reproducibility | 3 | CI, deterministic synthetic regression, manifests and machine-readable smoke artifacts are strong foundations. | 4 | Complete typed manifest/checksum registry, persistent raw evidence, one-command paper reproduction and master artifact index. |
| Clarity of claims | 3 | Scope documents explicitly prohibit many overclaims, but they are stale relative to current code and can misstate what is implemented versus empirically validated. | 4 | Reconcile claim/evidence docs with current main while preserving conservative evidence tiers. |

## Current recommendation

**Weak Reject**

The project is scientifically promising and its software foundations are stronger than a typical early research prototype. Nevertheless, the core paper claim is empirical: early, calibrated failure prediction must improve a downstream outcome under degradation/domain shift. That complete claim is not yet supported by frozen multi-dataset, grouped-statistical, external-backend and meaningful closed-loop evidence. The correct next step is not broader architectural work; it is to freeze the remaining scientific contracts and execute the common protocol.
