from dataclasses import dataclass

from self_healing_vio.temporal_buffer import TemporalBuffer, compute_trend


@dataclass
class FailureFeatures:
    tracked_features: int
    mean_reprojection_error: float
    image_brightness: float
    blur_score: float
    imu_consistency: float
    health_score: float


@dataclass
class FailurePrediction:
    probability: float
    confidence: float
    predicted_time_to_failure_frames: int


class FailurePredictor:
    def __init__(self, window_size: int = 5) -> None:
        self.history: TemporalBuffer[FailureFeatures] = TemporalBuffer(window_size)

    def update(self, features: FailureFeatures) -> FailurePrediction:
        self.history.append(features)
        return self.predict()

    def predict(self) -> FailurePrediction:
        values = self.history.values()
        if not values:
            return FailurePrediction(
                probability=0.0,
                confidence=0.0,
                predicted_time_to_failure_frames=-1,
            )

        latest = values[-1]
        health_trend = compute_trend(feature.health_score for feature in values)
        feature_trend = compute_trend(float(feature.tracked_features) for feature in values)

        visual_risk = 1.0 - min(max(latest.health_score, 0.0), 1.0)
        feature_loss_risk = max(0.0, -feature_trend.delta / 200.0)
        health_drop_risk = max(0.0, -health_trend.delta)
        reprojection_risk = min(max(latest.mean_reprojection_error / 8.0, 0.0), 1.0)
        appearance_risk = 0.5 * (1.0 - latest.image_brightness) + 0.5 * (1.0 - latest.blur_score)
        imu_risk = 1.0 - min(max(latest.imu_consistency, 0.0), 1.0)

        probability = (
            0.30 * visual_risk
            + 0.20 * reprojection_risk
            + 0.15 * appearance_risk
            + 0.15 * imu_risk
            + 0.10 * feature_loss_risk
            + 0.10 * health_drop_risk
        )
        probability = min(max(probability, 0.0), 1.0)
        confidence = min(len(values) / self.history.max_size, 1.0)

        if probability < 0.4:
            time_to_failure = -1
        elif probability < 0.7:
            time_to_failure = 10
        else:
            time_to_failure = 5

        return FailurePrediction(
            probability=probability,
            confidence=confidence,
            predicted_time_to_failure_frames=time_to_failure,
        )
