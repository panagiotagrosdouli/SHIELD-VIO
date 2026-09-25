# SHIELD-VIO Literature Matrix

This matrix links planned manuscript claims to representative prior work and identifies the comparison that must be established before novelty language is used. It is a living research artifact, not a declaration that the cited set is exhaustive.

| Area | Representative references | What prior work establishes | SHIELD-VIO comparison question | Status before submission |
|---|---|---|---|---|
| Filter-based VIO | \cite{mourikis2007msckf,bloesch2017rovio,geneva2020openvins} | Mature filtering formulations and open VIO research platforms | Does the health/failure layer work without depending on one internal estimator? | External-estimator experiment required |
| Optimization-based VIO / VI-SLAM | \cite{leutenegger2015okvis,qin2018vinsmono,campos2021orbslam3} | High-quality tightly coupled VIO/SLAM systems | Is SHIELD-VIO clearly presented as a supervisory layer rather than a competing estimator? | Narrative requirement |
| Public VIO datasets | \cite{burri2016euroc,schubert2018tumvi} | Reproducible camera/IMU sequences and ground truth | Are sequence splits, checksums, ground-truth intervals, and experimental units frozen before test evaluation? | Real EuRoC/TUM-VI artifacts required |
| Introspective perception | \cite{daftry2016introspective} | Learned prediction of perception-system failures | Does SHIELD-VIO add a causal future-horizon, estimator-health formulation and calibrated risk? | Direct related-work comparison required |
| Introspective SLAM | \cite{rabiee2021ivslam} | Learned context-dependent visual error model integrated into SLAM | Does SHIELD-VIO remain estimator-agnostic and exploit multi-signal temporal health rather than only visual context? | Direct related-work comparison required |
| Localization monitoring | \cite{eder2022localizationmonitor} | ML-based monitoring of localization correctness | Is the proposed event definition and lead-time evaluation substantively different from current-state correctness monitoring? | Direct comparison of target definitions required |
| Localization integrity / confidence | \cite{carson2022integrity,miller2026lensdoubt} | Integrity/uncertainty estimates for visual place recognition | Which ideas transfer to VIO failure prediction, and which are VPR-specific? | Include in final literature review |
| Probability calibration | \cite{guo2017calibration} | Post-hoc calibration can materially change probability reliability | Are detector fitting, calibration, threshold selection, and testing strictly partitioned? | Held-out public-data calibration required |
| Conformal uncertainty | \cite{angelopoulos2023conformal} | Distribution-free coverage under stated assumptions | How does empirical coverage change under temporal dependence and domain shift? | Report coverage and assumption limits |
| OOD detection | \cite{hendrycks2017ood} | Confidence can provide a baseline signal for OOD/error detection | Is health-space shift detection evaluated separately from failure discrimination? | OOD benchmark required |
| Selective prediction / abstention | \cite{geifman2019selectivenet} | Risk-coverage trade-offs provide an abstention framework | Does abstention or conservative mode improve declared mission utility rather than merely reduce exposure by stopping? | Closed-loop comparison required |
| Runtime assurance | \cite{seto1998simplex,hsu2024safetyfilter} | Runtime monitoring/intervention can separate performance and protection layers | Is SHIELD-VIO appropriately framed as empirical supervisory protection rather than formal safety assurance? | Claim-language requirement |

## Closest-prior-art conclusion

The closest prior art materially narrows the novelty boundary.

### What is already established

- **Generic localization-failure prediction is not new.** Time-to-failure and predictive monitoring have been studied for scan-matching and particle-filter localization \cite{tsuchiya2020ttf,eder2022localizationmonitor,knitt2025predictivemonitoring}.
- **VIO integrity monitoring and fallback switching are not new.** Existing work monitors VIO integrity and can switch to a fallback estimator when health degrades \cite{wang2019pdrviointegrity,joshi2023smvio}.
- **Future tracking-failure avoidance in visual SLAM is not new.** Introspective-SLAM predicts the safety of candidate future navigation steps and replans to avoid tracking failure \cite{naveed2022introspectiveslam}.
- **Failure-centric VIO stress testing is not new.** Recent benchmarking evaluates multiple VIO systems under degradation, miscalibration and occlusion \cite{zhu2026failurebenchmark}.
- **Uncertainty-aware VIO robustness is not new.** Online statistical measurement-reliability learning has been proposed to adapt visual measurement weights \cite{choi2025statistical}.
- **Short-horizon VIO risk prediction plus stop/relocalization is now direct prior art.** SUPER combines propagated uncertainty, residuals, geometric conditioning and temporal trends, predicts near-future trajectory degradation, and uses the risk signal for stop/relocalization \cite{gaus2025super}.
- **Calibration/conformal mechanisms inside SLAM under domain variation are not new in the broad sense.** Recent work applies conformal calibration to SLAM factor weighting \cite{chen2026cf2slam}.

Therefore SHIELD-VIO must not claim novelty merely for a VIO health score, short-horizon degradation prediction, risk monitoring, degradation benchmarking, or a stop/relocalization reaction.

### Strongest remaining differentiation to test

The current literature audit has not identified one work that demonstrates the following **joint protocol**:

1. define **persistent failure-onset events** and causal future targets at several fixed horizons (e.g. 0.5/1/2/3/5 s), with censoring and explicit recovery/event semantics;
2. predict those events from a **backend-neutral multi-signal health history** spanning covariance, innovations/NIS, visual tracking/update diagnostics, inertial/estimator health, temporal derivatives and explicit missingness;
3. fit detector, probability calibrator, operating threshold and final test on **disjoint complete-sequence train/calibration/validation/test partitions**, with an explicit leakage firewall;
4. report **held-out probability reliability** (Brier/NLL/ECE, calibration slope/intercept and reliability data), rather than treating a heuristic risk score as a calibrated probability;
5. evaluate how discrimination, calibration/coverage and confidence change under **predeclared unseen degradation family/severity and cross-dataset shift**;
6. feed the same prediction outputs into **stateful protective-policy variants** and report downstream safety--utility outcomes (unsafe exposure, mission completion, intervention cost/delay, recovery), not only detector recall/FPR;
7. perform inference over complete sequence-condition-seed experimental units with **paired grouped uncertainty**, never treating adjacent frames as independent replicates.

This combination is the candidate SHIELD-VIO contribution. Every element must be demonstrated experimentally before it appears as a supported contribution in the manuscript.

### Relationship to SUPER

SUPER is the strongest direct comparator and should be discussed explicitly, not hidden in a broad related-work paragraph. It already predicts trajectory degradation 50 frames ahead and demonstrates a stop/relocalization policy \cite{gaus2025super}. The SHIELD-VIO paper should therefore test a different question:

> **Given causal estimator health up to time t, what is the held-out calibrated probability that a persistent observable localization-failure event will begin within a specified horizon, how reliable is that probability under domain shift, and what is the paired downstream utility of acting on it?**

The intended distinction is not “risk versus no risk.” It is **event semantics + split-safe probability calibration + shift evaluation + decision utility**.

### Candidate contribution wording

Subject to a final exhaustive search and successful experiments:

> **SHIELD-VIO presents a leakage-resistant evaluation and protection framework for visual--inertial localization that casts degradation as multi-horizon prediction of persistent future failure events from causal estimator-health histories, calibrates the resulting risks on disjoint sequence-level data, evaluates reliability under predeclared distribution shifts, and measures how stateful risk-triggered actions affect downstream safety--utility outcomes.**

Avoid “first” in the abstract until the literature search is frozen. A defensible phrasing is “we are not aware of prior work that jointly evaluates…” followed by the exact combination above.

### What would make the novelty substantially stronger

- Execute the same health/prediction contract on the internal ESKF **and at least one mature external estimator** such as OpenVINS.
- Include SUPER or a faithful SUPER-style sensitivity/risk baseline where implementation access permits.
- Compare a simple reactive health threshold, a SUPER-like short-horizon risk score, raw learned risk, calibrated risk, and calibrated+stateful protection on identical run units.
- Preserve full train/calibration/validation/test separation and publish the run-level source tables and paired bootstrap intervals.

## Literature gaps that still require search before novelty freeze

1. Papers that explicitly calibrate probabilities of **future VIO failure onset**, rather than uncertainty of pose/measurements.
2. Multi-horizon survival/hazard/event forecasting specifically for VIO/SLAM.
3. Work combining localization-risk calibration with domain-shift detection and policy switching.
4. Closed-loop localization-failure prediction papers reporting both safety and mission-utility endpoints.
5. 2025--2026 papers that may cite or extend SUPER, SM/VIO, introspective SLAM, or localization-integrity monitoring.

## Novelty freeze rule

No “first,” “novel,” or broad priority statement is accepted until the remaining searches above are documented with explicit inclusion/exclusion notes. The final contribution statement must be restricted to the datasets, estimators, horizons, degradations and policy variants actually executed.
