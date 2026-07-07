#!/usr/bin/env python3
"""Run a synthetic SHIELD-VIO health monitoring demo.

The demo simulates progressive visual-inertial degradation and prints detector
scores, diagnosis probabilities, selected recovery action, and NHI.
"""

from __future__ import annotations

from self_healing_vio.detection import (
    FeatureCollapseDetector,
    IMUConsistencyDetector,
    ImageEntropyDetector,
    MotionBlurDetector,
    ReprojectionErrorMonitor,
)
from self_healing_vio.diagnosis import BayesianFailureDiagnosis
from self_healing_vio.health import NavigationHealthIndex
from self_healing_vio.recovery import RuleBasedRecoveryPolicy
from self_healing_vio.utils import SyntheticDegradationScenario


def main() -> None:
    scenario = SyntheticDegradationScenario()
    detectors = {
        "image_entropy": ImageEntropyDetector(),
        "motion_blur": MotionBlurDetector(),
        "feature_collapse": FeatureCollapseDetector(),
        "imu_consistency": IMUConsistencyDetector(),
        "reprojection_error": ReprojectionErrorMonitor(),
    }
    diagnosis = BayesianFailureDiagnosis()
    nhi_model = NavigationHealthIndex(weights={name: 1.0 for name in detectors})
    policy = RuleBasedRecoveryPolicy()

    print("SHIELD-VIO synthetic health monitor demo")
    print("step | NHI  | action                 | dominant cause | detector scores")
    print("-----+------+------------------------+----------------+----------------")

    total_steps = 12
    for step in range(total_steps):
        sample = scenario.sample(step, total_steps)
        results = [
            detectors["image_entropy"].evaluate(sample["image"]),
            detectors["motion_blur"].evaluate(sample["image"]),
            detectors["feature_collapse"].evaluate(sample["feature_count"]),
            detectors["imu_consistency"].evaluate(sample["imu_residual"]),
            detectors["reprojection_error"].evaluate(sample["reprojection_error"]),
        ]
        scores = {result.name: result.score for result in results}
        posterior = diagnosis.infer(scores)
        nhi = nhi_model.compute(scores)
        action = policy.select_action(scores, posterior, nhi)
        dominant = max(posterior, key=posterior.get)
        score_text = ", ".join(f"{name}={value:.2f}" for name, value in scores.items())
        print(f"{step:>4} | {nhi:>4.1f} | {action.value:<22} | {dominant:<14} | {score_text}")
        print("     diagnosis:", ", ".join(f"{k}={v:.2f}" for k, v in posterior.items()))


if __name__ == "__main__":
    main()
