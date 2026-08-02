from __future__ import annotations

import cv2
import numpy as np
import pytest

from shield_vio.vision.stereo_pnp import MetricPnPResult, StereoLandmarks, StereoPnPFrontend


def _camera_matrix() -> np.ndarray:
    return np.array([[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]])


def _render(points: np.ndarray, camera_matrix: np.ndarray, transform: np.ndarray) -> np.ndarray:
    image = np.zeros((240, 320), dtype=np.uint8)
    points_camera = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    pixels = (camera_matrix @ points_camera.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    for index, point in enumerate(pixels):
        x, y = np.rint(point).astype(int)
        if 8 <= x < 312 and 8 <= y < 232:
            cv2.circle(image, (x, y), 3, 255, -1)
            cv2.line(image, (x - 4, y), (x + 4, y), 120 + index % 100, 1)
            cv2.line(image, (x, y - 4), (x, y + 4), 180, 1)
    return image


def _scene() -> np.ndarray:
    points = []
    for z in (2.0, 2.5, 3.0, 3.5):
        for x in np.linspace(-0.55, 0.55, 8):
            for y in np.linspace(-0.35, 0.35, 5):
                points.append([x, y, z])
    return np.asarray(points, dtype=float)


def test_metric_pnp_recovers_known_motion(monkeypatch: pytest.MonkeyPatch) -> None:
    camera = _camera_matrix()
    baseline = np.array([-0.11, 0.0, 0.0])
    frontend = StereoPnPFrontend(
        camera,
        camera,
        np.eye(3),
        baseline,
        max_features=1200,
        ratio_threshold=0.9,
        max_epipolar_error_px=1.5,
        min_pnp_correspondences=8,
    )
    points = _scene()
    left_transform = np.eye(4)
    right_transform = np.eye(4)
    right_transform[:3, 3] = baseline
    angle = np.deg2rad(2.0)
    current_transform = np.eye(4)
    current_transform[:3, :3] = np.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    current_transform[:3, 3] = np.array([0.04, -0.01, 0.02])

    descriptors = np.tile(np.arange(32, dtype=np.uint8), (len(points), 1))
    descriptors ^= np.arange(len(points), dtype=np.uint8)[:, None]
    previous_pixels = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera, None)[0].reshape(-1, 2)
    landmarks = StereoLandmarks(points, previous_pixels, descriptors, len(points))
    current_points = (current_transform[:3, :3] @ points.T).T + current_transform[:3, 3]
    current_pixels = cv2.projectPoints(current_points, np.zeros(3), np.zeros(3), camera, None)[0].reshape(-1, 2)
    keypoints = [cv2.KeyPoint(float(x), float(y), 8) for x, y in current_pixels]

    class _ORB:
        def detectAndCompute(self, image: np.ndarray, mask: object) -> tuple[list[cv2.KeyPoint], np.ndarray]:
            return keypoints, descriptors.copy()

    monkeypatch.setattr(frontend, "_orb", _ORB())
    result = frontend.estimate_motion(landmarks, np.zeros((240, 320), np.uint8))

    assert result is not None
    np.testing.assert_allclose(result.rotation_current_previous, current_transform[:3, :3], atol=1e-5)
    np.testing.assert_allclose(result.translation_current_previous_m, current_transform[:3, 3], atol=1e-5)
    assert result.inlier_count == len(points)
    assert result.reprojection_rmse_px < 1e-5


def test_pnp_returns_none_with_insufficient_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    camera = _camera_matrix()
    frontend = StereoPnPFrontend(camera, camera, np.eye(3), np.array([-0.11, 0.0, 0.0]))
    landmarks = StereoLandmarks(
        np.ones((4, 3)),
        np.ones((4, 2)),
        np.zeros((4, 32), dtype=np.uint8),
        4,
    )

    class _ORB:
        def detectAndCompute(self, image: np.ndarray, mask: object) -> tuple[list[cv2.KeyPoint], np.ndarray]:
            return [cv2.KeyPoint(float(i), float(i), 8) for i in range(4)], np.zeros((4, 32), np.uint8)

    monkeypatch.setattr(frontend, "_orb", _ORB())
    assert frontend.estimate_motion(landmarks, np.zeros((20, 20), np.uint8)) is None


def test_result_validation() -> None:
    with pytest.raises(ValueError, match="inlier_mask"):
        MetricPnPResult(np.eye(3), np.zeros(3), np.ones(2, bool), 3, 0.1)
    with pytest.raises(ValueError, match="stereo baseline"):
        StereoPnPFrontend(_camera_matrix(), _camera_matrix(), np.eye(3), np.zeros(3))
