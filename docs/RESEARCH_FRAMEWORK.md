# SHIELD-VIO Research Framework

**SHIELD-VIO** stands for **Self-Healing Intelligent Estimation and Localization for Degradation-aware Visual-Inertial Odometry**. The repository is currently a research prototype and architectural skeleton. It does not claim completed benchmark results; instead, it defines an implementable framework for detecting, diagnosing, and recovering from VIO degradation in autonomous UAV navigation.

## 1. Formal Problem Definition

Consider a UAV operating in a GPS-denied environment with a visual-inertial sensor suite. At time `t`, the platform receives camera measurements `z_I(t)` and IMU measurements `z_U(t)`. A VIO estimator produces a belief over the robot state:

```text
p(x_t | z_1:t) ≈ N(x_hat_t, P_t)
```

where `x_t` is the navigation state, `x_hat_t` is the estimated state, and `P_t` is the estimator covariance or an uncertainty surrogate.

Classical VIO optimizes state estimation under the implicit assumption that the sensing and estimation pipeline remains operational. SHIELD-VIO extends the problem by estimating a health state, diagnosing degradation causes, and selecting recovery actions online.

The objective is to maximize navigation reliability under sensor, motion, and environmental degradation:

```text
maximize   E[ uptime, accuracy, consistency, recoverability ]
subject to real-time compute, onboard sensing, and safety constraints.
```

## 2. VIO State Notation

A standard visual-inertial state is represented as:

```text
x_t = { R_WB(t), p_WB(t), v_WB(t), b_g(t), b_a(t), θ_ext, θ_cam }
```

where:

- `R_WB ∈ SO(3)` is body orientation in the world frame.
- `p_WB ∈ R^3` is body position.
- `v_WB ∈ R^3` is body velocity.
- `b_g`, `b_a` are gyroscope and accelerometer biases.
- `θ_ext` contains camera-IMU extrinsics.
- `θ_cam` contains camera calibration parameters.

A VIO backend may be implemented as an EKF, fixed-lag smoother, or factor graph. SHIELD-VIO is backend-agnostic and consumes estimator health signals rather than requiring a specific optimizer.

## 3. Health Vector

At each time step, detectors produce a normalized health vector:

```text
h(t) = [h_entropy, h_blur, h_features, h_imu, h_reproj, h_innov, h_sync, h_drift, ...]^T
```

Each component satisfies:

```text
h_i(t) ∈ [0, 1]
```

where `1` means nominal and `0` means severely degraded. The prototype currently implements:

- image entropy health
- motion blur health
- feature collapse health
- IMU consistency health
- reprojection error health

Future work will add dynamic-scene detection, timestamp desynchronization, innovation residual monitoring, feature-track lifetime, and drift-rate estimation.

## 4. Diagnosis Posterior

Let the latent degradation cause be:

```text
c ∈ C = {camera, environment, motion, sensor, algorithm, nominal}
```

SHIELD-VIO estimates:

```text
p(c | h(t)) = η p(c) Π_i p(h_i(t) | c)
```

where `η` is the normalization constant. The initial prototype implements an interpretable Bayesian diagnosis table. The planned learned alternative is a small temporal model:

```text
p(c_t | h_1:t) = f_θ(h_t, h_{t-1}, ..., h_{t-k})
```

The Bayesian baseline is retained because interpretability is important for field robotics and failure analysis.

## 5. Recovery Action Space

The recovery action space is defined as:

```text
A = { no_action, reduce_speed, increase_feature_budget, relax_ransac,
      tighten_ransac, reset_visual_frontend, reinitialize_bias,
      inflate_covariance, request_hover, relocalize, partial_reset,
      switch_camera_mode, increase_exposure, enable_robust_loss,
      reject_visual_update, keyframe_insertion, fallback_inertial,
      emergency_land }
```

The prototype implements a rule-based policy over a subset of these actions. Future work will compare rule-based control against contextual bandits, Bayesian optimization, and reinforcement-learning policies.

## 6. Navigation Health Index

The Navigation Health Index is a calibrated scalar:

```text
NHI(t) ∈ [0, 100]
```

It is computed as a weighted fusion of normalized health terms:

```text
NHI(t) = 100 · clip( Σ_i w_i h_i(t) - λ_r risk(t), 0, 1 )
```

where `w_i ≥ 0`, `Σ_i w_i = 1`, and `risk(t)` is an optional penalty derived from uncertainty growth, diagnosis severity, or safety-critical operating context.

A value near 100 means the system is operating nominally. A value near 0 means the estimator is not trustworthy for autonomous navigation.

## 7. Uncertainty Decomposition

SHIELD-VIO separates uncertainty into:

```text
U_total = U_state + U_sensor + U_perception + U_model + U_policy
```

- `U_state`: pose, velocity, and bias uncertainty from the estimator.
- `U_sensor`: IMU noise, bias instability, camera exposure, timestamp quality.
- `U_perception`: texture, blur, dynamic objects, feature geometry.
- `U_model`: mismatch between estimator assumptions and real conditions.
- `U_policy`: uncertainty about whether a selected recovery action will improve performance.

This decomposition allows the system to distinguish estimator uncertainty from perceptual degradation.

## 8. Closed-Loop Self-Healing Algorithm

```text
Algorithm: SHIELD-VIO Online Self-Healing Loop
Input: image I_t, IMU packet u_t, VIO state belief (x_hat_t, P_t)
Output: state estimate, NHI, diagnosis posterior, recovery action

1. Run the nominal VIO frontend/backend update.
2. Extract detector inputs: image statistics, feature tracks, IMU residuals, reprojection residuals.
3. Compute health vector h(t).
4. Estimate diagnosis posterior p(c | h(t)).
5. Compute NHI(t).
6. Select recovery action a_t = π(h(t), p(c | h(t)), NHI(t)).
7. Apply allowed estimator or platform-level reconfiguration.
8. Log h(t), p(c | h(t)), a_t, NHI(t), and trajectory metrics.
```

## 9. Failure Taxonomy

| Class | Examples | Observable symptoms |
|---|---|---|
| Camera degradation | blur, exposure failure, lens contamination | low entropy, high blur, low features |
| Environmental degradation | low texture, darkness, repetitive structure | feature collapse, poor geometry |
| Motion-induced degradation | aggressive rotation, high acceleration | blur, IMU/vision disagreement |
| Sensor degradation | IMU bias jump, saturation, timestamp errors | high IMU residual, inconsistent propagation |
| Algorithmic degradation | wrong data association, optimizer divergence | reprojection growth, covariance inflation |
| Nominal | healthy operation | high NHI, stable residuals |

## 10. Experimental Hypotheses

H1: A continuous health vector predicts VIO failure earlier than binary tracking status.

H2: Bayesian diagnosis from onboard health signals can identify dominant degradation categories with useful accuracy.

H3: Recovery actions conditioned on diagnosis improve tracking uptime and reduce catastrophic failures compared with static VIO parameters.

H4: Learned policies improve cross-domain adaptation after sufficient failure data is collected, but interpretable rule-based policies are stronger initial baselines.

## 11. Ablation Design

The intended ablations are:

1. Nominal VIO without health monitoring.
2. Health monitoring only, no recovery.
3. Health monitoring with rule-based recovery.
4. Bayesian diagnosis versus learned diagnosis.
5. Individual detector removal.
6. NHI with and without uncertainty terms.
7. Recovery policy comparison: rule table, contextual bandit, Bayesian optimization.

Primary metrics include ATE, RPE, failure prediction lead time, uptime, recovery latency, recovery success rate, calibration error, and compute overhead.

## 12. Limitations and Risks

- Health scores may be dataset-specific unless calibrated across domains.
- Recovery actions can destabilize an estimator if applied too aggressively.
- Diagnosis labels are difficult to obtain without controlled fault injection.
- A high NHI does not guarantee metric accuracy; it estimates operational trustworthiness.
- Real UAV validation requires strict safety constraints and staged testing.
- Learned recovery policies must not be deployed without fallback safety logic.

## 13. Implementation Status

Implemented in the current prototype:

- normalized detector interfaces
- entropy, blur, feature, IMU, and reprojection monitors
- Bayesian diagnosis baseline
- rule-based recovery policy
- Navigation Health Index
- synthetic demo loop
- unit-test skeleton

Planned work:

- ROS2 nodes and message definitions
- integration with OpenVINS, VINS-Fusion, or ORB-SLAM3
- dataset benchmark runner
- real-time C++/ROS2 implementation
- closed-loop UAV experiments
