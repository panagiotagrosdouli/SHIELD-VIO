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

The current search rules out several broad novelty claims:

- **Do not claim:** “first to predict localization failure.” Time-to-failure and predictive localization monitoring already exist for scan-matching/particle-filter localization \cite{tsuchiya2020ttf,eder2022localizationmonitor,knitt2025predictivemonitoring}.
- **Do not claim:** “first VIO health monitor” or “first to switch when VIO fails.” VIO integrity monitoring and health-triggered estimator switching already exist \cite{wang2019pdrviointegrity,joshi2023smvio}.
- **Do not claim:** “first to avoid SLAM tracking failure using introspection.” Introspective-SLAM explicitly evaluates future navigation steps and plans around tracking failure \cite{naveed2022introspectiveslam}.
- **Do not claim:** “first failure-centric VIO benchmark.” A 2026 benchmark now stress-tests multiple VIO paradigms under degradation/miscalibration/occlusion \cite{zhu2026failurebenchmark}.
- **Do not claim:** “first calibrated or conformal SLAM under domain shift.” Calibration/conformal mechanisms are already being applied inside SLAM \cite{chen2026cf2slam}.

### Candidate defensible contribution

Subject to completion of the remaining literature search and the experiments, the strongest candidate contribution is:

> **A leakage-resistant, estimator-facing framework that formulates VIO degradation as calibrated prediction of persistent future localization-failure events from causal multi-signal health history, explicitly evaluates warning lead time and probability reliability under domain shift, and couples the predicted risk to stateful protective navigation actions under a common public-dataset/closed-loop protocol.**

The likely differentiator is therefore the **combination and evaluation contract**: VIO-specific causal health signals + fixed future horizons + persistent event labels + held-out probability calibration + domain-shift evaluation + protective-policy consequences + run-level paired inference. Each individual ingredient has prior art.

A stronger paper can additionally test whether this framework transfers from the internal ESKF to an established estimator such as OpenVINS. If successful, an architecture-general claim can be made narrowly over the executed estimators; without that experiment, the paper should remain VIO-backend-specific.

## Literature gaps that must still be searched before freezing novelty

1. VIO-specific **future** failure forecasting from estimator internals, not only current confidence or failure detection.
2. Innovation/NIS/covariance consistency monitoring used explicitly as a learned or thresholded precursor to localization failure.
3. Failure detection and recovery in visual-inertial SLAM, especially systems that trigger relocalization/reinitialization or navigation changes.
4. Localization integrity/risk estimation for autonomous navigation beyond visual place recognition.
5. Robotics work combining calibrated uncertainty with runtime policy switching under localization degradation.
6. Recent 2024--2026 papers on uncertainty-aware VIO/SLAM, failure prediction, and safety-aware localization.

## Novelty freeze rule

No “first” or broad novelty statement is accepted until every gap above has been searched with explicit inclusion/exclusion notes. The final contribution statement should name the exact combination that remains unsupported by prior work and should be restricted to the datasets, estimators, horizons, degradations, and protective policies actually evaluated.
