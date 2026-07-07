# Publication-Style Contributions

1. **A closed-loop self-healing VIO architecture.** SHIELD-VIO formulates visual-inertial robustness as an online detect-diagnose-recover loop rather than a post-hoc failure logging problem.

2. **A normalized multi-signal VIO health representation.** The framework defines a health vector composed of perceptual, inertial, and estimator-consistency signals, enabling continuous degradation monitoring before complete tracking loss.

3. **An interpretable diagnosis baseline for VIO degradation.** A Bayesian diagnosis module estimates a posterior over degradation causes such as camera, environment, motion, sensor, algorithmic, and nominal operation.

4. **A Navigation Health Index for autonomy interfaces.** The proposed NHI compresses health and risk signals into a calibrated `[0, 100]` scalar intended for planners, safety monitors, and human operators.

5. **A modular prototype and evaluation roadmap.** The repository provides executable Python components, unit tests, configuration, and a staged plan for ROS2 integration, VIO backend coupling, dataset benchmarking, and PhD-level publication development.
