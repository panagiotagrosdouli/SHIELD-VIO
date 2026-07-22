# Changelog

All notable changes to SHIELD-VIO are documented in this file.

The project follows semantic versioning where practical. Because SHIELD-VIO is a research prototype, minor releases may still revise experimental interfaces, configuration schemas, and generated artifacts. Release notes must distinguish implemented capability from validation evidence.

## [Unreleased]

### Planned

- Public-sequence execution on EuRoC MAV and TUM-VI.
- Reliability, lead-time, ablation, and sensitivity studies.
- Stronger relocalization and active-perception actions.
- ROS 2 bag replay and simulator validation.
- Hardware experiments only after dataset and simulator evidence is stable.

## [0.2.0] - 2026-07-22

### Added

- Shi–Tomasi and pyramidal Lucas–Kanade visual feature tracking with health diagnostics.
- Bias-aware IMU preintegration and a 15-state error-state EKF research implementation.
- Interpretable and logistic failure-prediction baselines.
- Calibration metrics, split-conformal bounds, and rolling domain-shift states.
- Stateful navigation shielding with warning, slowdown, hold, relocalization, halt, and emergency-stop actions.
- Deterministic synthetic scenarios, repeated-seed experiments, manifests, and README evidence generation.
- Local-layout adapters for EuRoC MAV, TUM-VI, and generic camera/IMU folders.
- Research-oriented documentation, evidence taxonomy, contribution guidance, and claim boundaries.

### Validation scope

- Unit and numerical-invariant validation.
- Deterministic synthetic experiments and repeated synthetic scenarios.
- Mocked filesystem validation for public-dataset adapters.

### Not yet validated

- End-to-end production-quality VIO.
- Public dataset metrics on real EuRoC or TUM-VI sequences.
- Real-world probability calibration.
- ROS 2, simulator, or hardware execution.
- Formal safety guarantees or state-of-the-art performance.

## [0.1.0] - 2026

### Added

- Initial safety-aware visual-inertial research toolkit.
- Early uncertainty monitoring, degradation simulation, and supervisory shielding components.

[Unreleased]: https://github.com/panagiotagrosdouli/SHIELD-VIO/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/panagiotagrosdouli/SHIELD-VIO/releases/tag/v0.2.0
[0.1.0]: https://github.com/panagiotagrosdouli/SHIELD-VIO/releases/tag/v0.1.0
