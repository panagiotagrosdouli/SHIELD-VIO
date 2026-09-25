# Related Work Draft

This document is a citation-backed working draft for Section 3 of the SHIELD-VIO paper. It is deliberately conservative: it identifies adjacent research directions and the comparison axes that must be tested, but it does not assert novelty until the literature matrix is complete and the corresponding SHIELD-VIO experiments have been executed.

## Visual-inertial estimation and benchmark context

Modern visual-inertial odometry (VIO) spans filtering and optimization-based formulations. The Multi-State Constraint Kalman Filter (MSCKF) established an influential filtering architecture for vision-aided inertial navigation \cite{mourikis2007msckf}. Keyframe-based nonlinear optimization is represented by OKVIS \cite{leutenegger2015okvis}, while ROVIO tightly integrates direct photometric residuals in an iterated EKF \cite{bloesch2017rovio}. VINS-Mono provides a complete monocular visual-inertial state estimator with relocalization and global pose-graph functionality \cite{qin2018vinsmono}. OpenVINS provides an open research platform for visual-inertial estimation and is a natural candidate for testing whether estimator-health prediction transfers beyond SHIELD-VIO's internal ESKF \cite{geneva2020openvins}. ORB-SLAM3 represents a mature visual, visual-inertial, and multimap SLAM system with broad sensor support \cite{campos2021orbslam3}.

SHIELD-VIO is not positioned as a replacement VIO estimator. Its scientific question begins after, or alongside, state estimation: whether causal estimator-health history can predict an observable persistent localization failure early enough to support a calibrated protective response. This distinction should remain explicit throughout the manuscript.

The public-data evaluation is anchored in EuRoC MAV and TUM VI. EuRoC provides synchronized stereo imagery, inertial measurements, calibration, and accurate motion ground truth across machine-hall and Vicon-room sequences \cite{burri2016euroc}. TUM VI expands the visual-inertial benchmark setting with diverse indoor and outdoor sequences, photometric calibration, synchronized cameras and IMU, and sequence-dependent ground-truth availability \cite{schubert2018tumvi}. These datasets support reproducible VIO evaluation, but they do not by themselves validate failure prediction, probability calibration, domain-shift handling, or closed-loop protection; those require the additional experimental protocol defined by SHIELD-VIO.

## Introspection, failure prediction, and localization integrity

A central antecedent is introspective perception: Daftry et al. frame perception systems as systems that should predict when their own outputs are likely to fail, demonstrating learned failure prediction for vision-based MAV operation \cite{daftry2016introspective}. IV-SLAM advances a related idea inside visual SLAM by learning context-dependent visual feature noise and using the learned model to improve feature selection and estimation robustness \cite{rabiee2021ivslam}. Localization monitoring has also been studied explicitly; Eder et al. use particle-filter structure and machine learning to detect incorrect robot localization \cite{eder2022localizationmonitor}. In visual place recognition, Carson et al. learn integrity measures from artifacts of the localization system to identify inaccurate localization outputs \cite{carson2022integrity}. More recently, Miller et al. study training-free uncertainty measures for visual place recognition under changing environments and viewpoints \cite{miller2026lensdoubt}.

These works motivate self-assessment but operate at different levels of the localization stack and with different targets. The SHIELD-VIO target is a future-horizon event formulation: from only information available by time (t), predict the risk that a versioned, observable, persistent VIO failure begins in ((t,t+\tau]). The intended comparison is therefore not simply “confidence estimation versus no confidence estimation.” It must test whether multi-signal temporal estimator health provides earlier and better-calibrated warning than covariance, innovation, tracking, feature-count, and learned baseline alternatives under identical sequence-level splits.

## Closest VIO risk-assessment prior art

Recent work makes the novelty boundary substantially narrower. SUPER derives a backend-agnostic VIO/SLAM risk indicator from propagated uncertainty, residuals, geometric conditioning, and short-horizon temporal trends; it reports prediction of trajectory degradation 50 frames ahead and demonstrates a stop/relocalization response \cite{gaus2025super}. Statistical uncertainty learning for VIO also adapts measurement reliability online from sensor data and optimization results to improve estimator robustness \cite{choi2025statistical}. Earlier VIO integrity monitoring and SM/VIO show that health monitoring and fallback switching are themselves established ideas \cite{wang2019pdrviointegrity,joshi2023smvio}.

Consequently, SHIELD-VIO should not be positioned as the first system to assess VIO risk, forecast degradation, or trigger a fallback. Its candidate distinction is the experimental and decision formulation: persistent future failure-onset events at multiple fixed horizons, causal multi-signal health histories, sequence-disjoint probability calibration and threshold selection, explicit evaluation of calibration under unseen shifts, and paired downstream safety--utility evaluation of stateful protective actions. This distinction must be demonstrated empirically and compared directly with simpler reactive/risk-monitoring alternatives.

## Probability calibration and conformal uncertainty

Discrimination and probability quality are different objectives. Guo et al. demonstrate that modern predictive models can be miscalibrated and show the effectiveness of simple post-hoc calibration such as temperature scaling \cite{guo2017calibration}. For SHIELD-VIO, the relevant implication is methodological: a detector score must not automatically be interpreted as a probability of future failure. Calibration must be fitted on a partition disjoint from detector fitting and operating-threshold selection, then evaluated on held-out data using proper scoring rules and reliability diagnostics.

Conformal prediction offers a model-agnostic framework for uncertainty sets or bounds under explicit exchangeability assumptions \cite{angelopoulos2023conformal}. SHIELD-VIO should therefore report conformal behavior as an empirical uncertainty mechanism with stated assumptions, especially under domain shift, rather than use distribution-free language outside the conditions in which the guarantee applies.

## Out-of-distribution detection and selective prediction

Out-of-distribution (OOD) detection addresses failures caused by mismatch between training and deployment distributions. Hendrycks and Gimpel provide an early baseline for detecting misclassified and OOD examples from model confidence \cite{hendrycks2017ood}. Selective prediction extends the operational response by allowing a model to reject or abstain on uncertain inputs; SelectiveNet is a representative approach that optimizes a risk-coverage trade-off \cite{geifman2019selectivenet}.

SHIELD-VIO uses these ideas at a different interface: the distribution being monitored is the causal estimator-health process, and the response can change the downstream navigation policy rather than merely reject a classification. The paper should separately evaluate (i) whether a shift statistic detects unseen conditions, (ii) whether failure prediction/calibration degrade under those conditions, and (iii) whether a shift-aware response improves the declared safety-utility endpoints. Conflating these three questions would make the H4 claim difficult to interpret.

## Runtime assurance and protective intervention

The Simplex architecture formalized the separation between a high-performance controller and a simpler high-assurance fallback selected by runtime decision logic \cite{seto1998simplex}. Contemporary safety-filter literature provides a broader view of runtime intervention mechanisms for autonomous systems and emphasizes the modular relationship between performance control, monitoring, and safety enforcement \cite{hsu2024safetyfilter}.

SHIELD-VIO is related in architectural spirit but currently makes a narrower empirical claim. Its navigation shield is a supervisory research policy driven by predicted localization risk, calibration state, shift state, hysteresis, and recovery logic. Unless formal reachability, barrier, or invariant guarantees are added and proved, the manuscript should describe measured risk reduction and safety-utility trade-offs rather than call the shield formally safe or guaranteed.

## Working novelty position

The defensible novelty target is the **joint causal pipeline**, not any one primitive in isolation:

1. standardized, timestamped, backend-neutral estimator-health signals;
2. future-horizon prediction of persistent observable VIO failure using only causal history;
3. held-out probability calibration and explicit uncertainty handling;
4. evaluation under unseen sensor/domain conditions;
5. stateful protective action evaluated with downstream mission outcomes; and
6. sequence/run-level paired inference with leakage-resistant public-data splits.

Each component has substantial prior art. The paper can claim a contribution only for the integration and experimental evidence that is actually demonstrated. A final novelty sentence must wait until the literature matrix is complete, especially for VIO-specific failure forecasting, localization-integrity monitoring, estimator consistency monitoring, and failure-aware navigation/recovery.

## Citation and claim rules

- Cite original method/dataset papers rather than only surveys when describing a specific system.
- Use surveys/reviews to frame broad research areas, not as evidence that no prior method exists.
- Do not use “first,” “novel,” “state of the art,” “safe,” or “guaranteed” until the corresponding literature/evidence gate is satisfied.
- Distinguish current-failure detection, confidence estimation, and **future failure prediction**.
- Distinguish estimator robustness from a separate supervisory predictor.
- Distinguish OOD detection quality from downstream protective-policy utility.
- Keep public-dataset evidence, simulation evidence, and hardware evidence explicitly separated.
