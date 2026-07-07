from self_healing_vio.health import NavigationHealthIndex


def test_navigation_health_index_range():
    nhi = NavigationHealthIndex().compute({"image_entropy": 0.0, "motion_blur": 0.5, "imu_consistency": 1.0})
    assert 0.0 <= nhi <= 100.0
