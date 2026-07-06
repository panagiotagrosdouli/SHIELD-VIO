from dataclasses import dataclass


@dataclass
class VioHealthSignals:
    tracking_ok: bool
    tracked_features: int
    mean_reprojection_error: float
    image_brightness: float
    blur_score: float
    imu_consistency: float


@dataclass
class VioHealthEstimate:
    score: float
    level: str
    failure_probability: float


class VioHealthMonitor:
    def __init__(self, warning_threshold: float = 0.55, critical_threshold: float = 0.30) -> None:
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def estimate(self, signals: VioHealthSignals) -> VioHealthEstimate:
        feature_score = min(max(signals.tracked_features, 0) / 200.0, 1.0)
        reprojection_score = max(0.0, 1.0 - signals.mean_reprojection_error / 8.0)
        brightness_score = min(max(signals.image_brightness, 0.0), 1.0)
        blur_score = min(max(signals.blur_score, 0.0), 1.0)
        imu_score = min(max(signals.imu_consistency, 0.0), 1.0)
        tracking_score = 1.0 if signals.tracking_ok else 0.0

        score = (
            0.20 * feature_score
            + 0.20 * reprojection_score
            + 0.15 * brightness_score
            + 0.15 * blur_score
            + 0.20 * imu_score
            + 0.10 * tracking_score
        )
        failure_probability = 1.0 - score

        if score < self.critical_threshold:
            level = 'critical'
        elif score < self.warning_threshold:
            level = 'warning'
        else:
            level = 'healthy'

        return VioHealthEstimate(
            score=score,
            level=level,
            failure_probability=failure_probability,
        )
