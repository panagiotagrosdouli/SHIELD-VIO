# PhD Proposal

## Title

**SHIELD-VIO: Self-Healing Intelligent Estimation and Localization for Degradation-aware Visual-Inertial Odometry**

## Abstract

Visual-Inertial Odometry (VIO) is a core capability for autonomous UAV navigation in GPS-denied environments. Despite strong progress in accuracy and efficiency, most VIO systems remain vulnerable to silent degradation caused by motion blur, low texture, illumination changes, dynamic scenes, sensor inconsistency, and estimator divergence. This proposal introduces SHIELD-VIO, a self-healing VIO framework that augments state estimation with continuous health monitoring, probabilistic failure diagnosis, and online recovery actions. The central hypothesis is that VIO robustness can be improved by treating degradation as a closed-loop estimation and control problem rather than a terminal failure condition. The research will develop a Navigation Health Index, interpretable diagnosis models, recovery policies, and ROS2-compatible integration with existing VIO backends. Evaluation will use standard VIO datasets, controlled fault injection, and staged UAV experiments.

## Motivation

Autonomous UAVs depend on reliable localization. In many real environments, GPS is unavailable or unreliable, and VIO becomes the primary navigation source. However, VIO can fail gradually before catastrophic tracking loss. A system may accumulate drift, lose feature diversity, reject visual updates, or become inconsistent without explicitly reporting that localization is unsafe.

For deployment, the important question is not only whether a VIO pipeline is accurate on clean benchmark sequences. It is whether the robot can detect that the estimator is becoming unreliable, identify the likely degradation source, and select a recovery strategy before mission safety is compromised.

## Research Gap

Existing VIO systems typically provide state estimates, covariance approximations, tracking status, and diagnostic logs. These are insufficient for autonomous self-recovery because:

1. Failure signals are not fused into a structured health representation.
2. Tracking loss is often treated as binary, late-stage information.
3. Degradation cause is rarely inferred explicitly.
4. Recovery is usually manual, heuristic, or external to the estimator.
5. Benchmarking focuses on accuracy, not recoverability or failure lead time.

SHIELD-VIO addresses this gap by making estimator health, diagnosis, and recovery explicit components of the VIO architecture.

## Research Questions

RQ1: Can VIO degradation be represented as a continuous health vector that predicts failures earlier than binary tracking status?

RQ2: Can onboard health signals infer a useful posterior over degradation causes such as camera, environment, motion, sensor, and algorithmic failure?

RQ3: Can diagnosis-conditioned recovery actions improve estimator uptime, accuracy, and consistency compared with static VIO settings?

RQ4: Can learned recovery policies outperform rule-based policies under domain shift while preserving safety constraints?

RQ5: Can a Navigation Health Index provide a calibrated and interpretable signal for downstream planning and mission-level autonomy?

## Methodology

### Work Package 1: Health Monitoring

Develop detectors for image entropy, motion blur, feature collapse, IMU consistency, reprojection error, innovation residuals, timestamp consistency, dynamic-scene likelihood, and drift growth. Normalize detector outputs into a health vector `h(t)`.

### Work Package 2: Diagnosis

Implement a Bayesian diagnosis baseline:

```text
p(c | h(t)) = η p(c) Π_i p(h_i(t) | c)
```

Then develop temporal learned models that use recent health history to classify degradation causes.

### Work Package 3: Navigation Health Index

Design and calibrate an NHI in `[0, 100]` that combines perception health, sensor consistency, estimator uncertainty, and operational risk.

### Work Package 4: Recovery Policies

Implement a hierarchy of recovery strategies:

- rule-based policy
- contextual bandit
- Bayesian optimization policy
- constrained learned policy

The policy selects actions such as feature-budget increase, robust-loss activation, covariance inflation, frontend reset, relocalization, hover request, or fallback inertial propagation.

### Work Package 5: ROS2 and VIO Integration

Integrate the health monitor with one or more existing VIO systems. Candidate backends include OpenVINS, VINS-Fusion, ORB-SLAM3, and Kimera-VIO. ROS2 nodes will publish health vectors, diagnosis posteriors, recovery commands, and NHI.

### Work Package 6: Evaluation

Evaluate on EuRoC MAV, TUM-VI, UZH-FPV, Hilti SLAM Challenge, and Newer College datasets. Add controlled degradation through synthetic blur, feature masking, timestamp offsets, IMU perturbations, exposure changes, and dynamic-object injection.

## Expected Contributions

1. A closed-loop self-healing architecture for VIO degradation detection, diagnosis, and recovery.
2. A Navigation Health Index for interpretable estimator trustworthiness.
3. A failure taxonomy and benchmark protocol for VIO degradation.
4. Probabilistic and learned diagnosis models using onboard signals.
5. Recovery policies that adapt estimator behavior under degradation.

## Evaluation Plan

Metrics:

- Absolute Trajectory Error (ATE)
- Relative Pose Error (RPE)
- tracking uptime
- failure-prediction lead time
- diagnosis accuracy and calibration
- recovery success rate
- recovery latency
- compute overhead
- expected calibration error

Baselines:

- nominal VIO backend
- VIO with logging only
- VIO with health monitoring but no recovery
- VIO with rule-based recovery
- VIO with learned recovery policy

## Three-Year PhD Timeline

### Year 1

- Formalize health representation and failure taxonomy.
- Implement detector library and NHI prototype.
- Build reproducible dataset degradation pipeline.
- Submit initial workshop or RA-L short paper on VIO health monitoring.

### Year 2

- Integrate with ROS2 and at least one VIO backend.
- Implement Bayesian and learned diagnosis models.
- Conduct benchmark study on EuRoC, TUM-VI, and UZH-FPV.
- Submit ICRA/IROS paper on diagnosis-aware VIO robustness.

### Year 3

- Develop adaptive recovery policies.
- Validate in simulation and staged UAV experiments.
- Complete full ablation and field-style evaluation.
- Submit RA-L/ICRA or RSS paper on closed-loop self-healing VIO.
- Consolidate thesis and public research repository.

## Publication Targets

- IEEE International Conference on Robotics and Automation (ICRA)
- IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
- Robotics: Science and Systems (RSS)
- IEEE Robotics and Automation Letters (RA-L)
- Science Robotics, if the system reaches strong field validation and conceptual maturity

## Risk Mitigation

- If learned policies are not reliable, the project remains valuable through interpretable diagnosis and rule-based recovery.
- If full UAV deployment is delayed, controlled dataset and simulator degradation can still support rigorous evaluation.
- If backend integration is difficult, SHIELD-VIO can initially operate as a backend-agnostic monitoring layer.
