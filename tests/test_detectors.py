import numpy as np

from self_healing_vio.detection import (
    FeatureCollapseDetector,
    IMUConsistencyDetector,
    ImageEntropyDetector,
    MotionBlurDetector,
    ReprojectionErrorMonitor,
)


def test_detector_output_ranges():
    image = np.random.default_rng(0).integers(0, 255, size=(64, 64), dtype=np.uint8)
    results = [
        ImageEntropyDetector().evaluate(image),
        MotionBlurDetector().evaluate(image),
        FeatureCollapseDetector().evaluate(120),
        IMUConsistencyDetector().evaluate(0.2),
        ReprojectionErrorMonitor().evaluate(2.5),
    ]
    for result in results:
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.raw_value, float)
