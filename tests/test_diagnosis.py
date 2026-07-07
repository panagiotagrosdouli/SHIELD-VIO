from self_healing_vio.diagnosis import BayesianFailureDiagnosis


def test_diagnosis_probabilities_sum_to_one():
    posterior = BayesianFailureDiagnosis().infer(
        {"image_entropy": 0.5, "motion_blur": 0.2, "imu_consistency": 0.8}
    )
    assert abs(sum(posterior.values()) - 1.0) < 1e-9
    assert all(0.0 <= probability <= 1.0 for probability in posterior.values())
