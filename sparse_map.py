"""Persistent metric landmark map and map-based camera localization.

The existing stereo frontend reconstructs metric points in a single camera
frame.  This module turns those transient reconstructions into a persistent
world-frame map and localizes later frames against that map with descriptor
matching and PnP-RANSAC.  Transform names follow the convention
``transform_world_camera`` (``T_WC``): points are mapped from camera to world
coordinates by ``p_W = T_WC p_C``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from shield_vio.vision.stereo_pnp import StereoLandmarks


@dataclass(frozen=True)
class MapLandmark:
    """One persistent world-frame landmark with a binary descriptor."""

    landmark_id: int
    position_world_m: np.ndarray
    descriptor: np.ndarray
    first_keyframe_id: int
    last_keyframe_id: int
    observation_count: int = 1

    def __post_init__(self) -> None:
        position = np.asarray(self.position_world_m, dtype=float)
        descriptor = np.asarray(self.descriptor, dtype=np.uint8)
        if self.landmark_id < 0:
            raise ValueError("landmark_id must be non-negative")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_world_m must be a finite 3-vector")
        if descriptor.ndim != 1 or descriptor.size == 0:
            raise ValueError("descriptor must be a non-empty binary vector")
        if self.first_keyframe_id < 0 or self.last_keyframe_id < self.first_keyframe_id:
            raise ValueError("invalid landmark keyframe interval")
        if self.observation_count < 1:
            raise ValueError("observation_count must be positive")
        object.__setattr__(self, "position_world_m", position.copy())
        object.__setattr__(self, "descriptor", descriptor.copy())


@dataclass(frozen=True)
class Keyframe:
    """A camera pose and the landmarks created by one stereo keyframe."""

    keyframe_id: int
    timestamp_ns: int
    transform_world_camera: np.ndarray
    landmark_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.keyframe_id < 0 or self.timestamp_ns < 0:
            raise ValueError("keyframe id and timestamp must be non-negative")
        transform = _validated_transform(self.transform_world_camera)
        landmark_ids = tuple(int(value) for value in self.landmark_ids)
        if len(set(landmark_ids)) != len(landmark_ids) or any(value < 0 for value in landmark_ids):
            raise ValueError("landmark_ids must be unique and non-negative")
        object.__setattr__(self, "transform_world_camera", transform)
        object.__setattr__(self, "landmark_ids", landmark_ids)


@dataclass(frozen=True)
class MapLocalizationResult:
    """Robust world-frame pose estimate from 2D-to-map correspondences."""

    transform_world_camera: np.ndarray
    matched_landmark_ids: tuple[int, ...]
    inlier_landmark_ids: tuple[int, ...]
    matched_feature_indices: tuple[int, ...]
    inlier_feature_indices: tuple[int, ...]
    match_count: int
    inlier_count: int
    inlier_ratio: float
    reprojection_rmse_px: float

    def __post_init__(self) -> None:
        transform = _validated_transform(self.transform_world_camera)
        matches = tuple(int(value) for value in self.matched_landmark_ids)
        inliers = tuple(int(value) for value in self.inlier_landmark_ids)
        matched_features = tuple(int(value) for value in self.matched_feature_indices)
        inlier_features = tuple(int(value) for value in self.inlier_feature_indices)
        if any(value < 0 for value in matches + inliers + matched_features + inlier_features):
            raise ValueError("landmark ids and feature indices must be non-negative")
        if self.match_count != len(matches) or self.match_count != len(matched_features):
            raise ValueError("match_count must match landmark and feature associations")
        if len(set(matches)) != len(matches) or len(set(matched_features)) != len(matches):
            raise ValueError("matched landmark and feature ids must be unique")
        if self.inlier_count != len(inliers) or self.inlier_count != len(inlier_features):
            raise ValueError("inlier_count must match landmark and feature associations")
        if len(set(inliers)) != len(inliers) or len(set(inlier_features)) != len(inliers):
            raise ValueError("inlier landmark and feature ids must be unique")
        matched_pairs = set(zip(matches, matched_features, strict=True))
        inlier_pairs = set(zip(inliers, inlier_features, strict=True))
        if not inlier_pairs.issubset(matched_pairs):
            raise ValueError("inlier associations must be a subset of matched associations")
        expected_ratio = self.inlier_count / self.match_count if self.match_count else 0.0
        if not np.isfinite(self.inlier_ratio) or not np.isclose(
            self.inlier_ratio, expected_ratio
        ):
            raise ValueError("inlier_ratio is inconsistent with the counts")
        if self.reprojection_rmse_px < 0 or not np.isfinite(self.reprojection_rmse_px):
            raise ValueError("reprojection_rmse_px must be finite and non-negative")
        object.__setattr__(self, "transform_world_camera", transform)
        object.__setattr__(self, "matched_landmark_ids", matches)
        object.__setattr__(self, "inlier_landmark_ids", inliers)
        object.__setattr__(self, "matched_feature_indices", matched_features)
        object.__setattr__(self, "inlier_feature_indices", inlier_features)


class SparseLandmarkMap:
    """Persistent stereo landmark map with global descriptor-based PnP.

    New stereo keyframes create world-frame landmarks.  Later frames can be
    localized against all stored landmarks.  The map deliberately keeps data
    association and pose estimation explicit; bundle adjustment and landmark
    fusion are separate future stages rather than hidden side effects.
    """

    def __init__(self) -> None:
        self._keyframes: dict[int, Keyframe] = {}
        self._landmarks: dict[int, MapLandmark] = {}
        self._next_landmark_id = 0
        self._descriptor_size: int | None = None

    @property
    def keyframes(self) -> tuple[Keyframe, ...]:
        return tuple(self._keyframes[key] for key in sorted(self._keyframes))

    @property
    def landmarks(self) -> tuple[MapLandmark, ...]:
        return tuple(self._landmarks[key] for key in sorted(self._landmarks))

    def add_stereo_keyframe(
        self,
        keyframe_id: int,
        timestamp_ns: int,
        transform_world_camera: np.ndarray,
        stereo_landmarks: StereoLandmarks,
        *,
        observed_landmarks: dict[int, int] | None = None,
    ) -> Keyframe:
        """Insert a stereo keyframe and reuse geometrically observed landmarks.

        ``observed_landmarks`` maps current stereo feature indices to existing
        landmark ids.  Unassociated stereo features create new landmarks.  This
        prevents every keyframe from duplicating all successfully reobserved
        map points.
        """

        if keyframe_id < 0 or timestamp_ns < 0:
            raise ValueError("keyframe id and timestamp must be non-negative")
        if keyframe_id in self._keyframes:
            raise ValueError(f"duplicate keyframe id: {keyframe_id}")
        if self._keyframes and keyframe_id <= max(self._keyframes):
            raise ValueError("keyframe ids must increase monotonically")
        if self._keyframes and timestamp_ns <= self.keyframes[-1].timestamp_ns:
            raise ValueError("keyframe timestamps must increase monotonically")
        if not isinstance(stereo_landmarks, StereoLandmarks):
            raise TypeError("stereo_landmarks must be a StereoLandmarks instance")
        if len(stereo_landmarks.points_left_m) == 0:
            raise ValueError("a map keyframe requires at least one stereo landmark")
        transform = _validated_transform(transform_world_camera)
        descriptor_size = stereo_landmarks.descriptors.shape[1]
        if self._descriptor_size is None:
            self._descriptor_size = descriptor_size
        elif descriptor_size != self._descriptor_size:
            raise ValueError("descriptor dimension does not match the map")

        associations = dict(observed_landmarks or {})
        feature_count = len(stereo_landmarks.points_left_m)
        if any(index < 0 or index >= feature_count for index in associations):
            raise ValueError("observed feature index is outside the stereo keyframe")
        if len(set(associations.values())) != len(associations):
            raise ValueError("an existing landmark may be associated only once per keyframe")
        missing = set(associations.values()) - set(self._landmarks)
        if missing:
            raise ValueError(f"observations reference unknown landmarks: {sorted(missing)}")

        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        world_points = (rotation @ stereo_landmarks.points_left_m.T).T + translation
        landmark_ids: list[int] = []
        for feature_index, (point, descriptor) in enumerate(
            zip(world_points, stereo_landmarks.descriptors, strict=True)
        ):
            if feature_index in associations:
                landmark_id = associations[feature_index]
                landmark = self._landmarks[landmark_id]
                self._landmarks[landmark_id] = replace(
                    landmark,
                    last_keyframe_id=keyframe_id,
                    observation_count=landmark.observation_count + 1,
                )
                landmark_ids.append(landmark_id)
                continue
            landmark_id = self._next_landmark_id
            self._next_landmark_id += 1
            self._landmarks[landmark_id] = MapLandmark(
                landmark_id=landmark_id,
                position_world_m=point,
                descriptor=descriptor,
                first_keyframe_id=keyframe_id,
                last_keyframe_id=keyframe_id,
            )
            landmark_ids.append(landmark_id)

        keyframe = Keyframe(
            keyframe_id=keyframe_id,
            timestamp_ns=timestamp_ns,
            transform_world_camera=transform,
            landmark_ids=tuple(landmark_ids),
        )
        self._keyframes[keyframe_id] = keyframe
        return keyframe

    def localize(
        self,
        image_points_px: np.ndarray,
        descriptors: np.ndarray,
        camera_matrix: np.ndarray,
        *,
        distortion_coefficients: np.ndarray | None = None,
        min_correspondences: int = 6,
        min_inliers: int = 6,
        ratio_test: float = 0.75,
        max_descriptor_distance: float = 64.0,
        reprojection_error_px: float = 3.0,
        confidence: float = 0.999,
        iterations: int = 200,
    ) -> MapLocalizationResult | None:
        """Estimate ``T_WC`` from current pixels and persistent map points.

        Matching uses a Hamming-distance ratio test and unique map-landmark
        assignments.  PnP-RANSAC rejects geometric outliers; ``None`` denotes
        insufficient support or a failed/weak geometric estimate.
        """

        points, query_descriptors = _validated_features(image_points_px, descriptors)
        intrinsic = _validated_camera_matrix(camera_matrix)
        distortion = _validated_distortion(distortion_coefficients)
        if min_correspondences < 4 or min_inliers < 4:
            raise ValueError("PnP correspondence thresholds must be at least four")
        if not 0.0 < ratio_test < 1.0:
            raise ValueError("ratio_test must be between zero and one")
        if max_descriptor_distance <= 0 or reprojection_error_px <= 0:
            raise ValueError("matching and reprojection thresholds must be positive")
        if not 0.0 < confidence < 1.0 or iterations < 1:
            raise ValueError("invalid PnP RANSAC parameters")
        if self._descriptor_size is None or len(self._landmarks) < min_correspondences:
            return None
        if query_descriptors.shape[1] != self._descriptor_size:
            raise ValueError("descriptor dimension does not match the map")

        landmark_ids = tuple(sorted(self._landmarks))
        map_descriptors = np.vstack(
            [self._landmarks[landmark_id].descriptor for landmark_id in landmark_ids]
        )
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        nearest = matcher.knnMatch(query_descriptors, map_descriptors, k=2)
        accepted = []
        for neighbours in nearest:
            if len(neighbours) != 2:
                continue
            best, second = neighbours
            if best.distance > max_descriptor_distance:
                continue
            if best.distance >= ratio_test * second.distance:
                continue
            accepted.append(best)
        accepted.sort(key=lambda match: (match.distance, match.queryIdx, match.trainIdx))

        unique_matches = []
        used_map_indices: set[int] = set()
        for match in accepted:
            if match.trainIdx in used_map_indices:
                continue
            used_map_indices.add(match.trainIdx)
            unique_matches.append(match)
        if len(unique_matches) < min_correspondences:
            return None

        matched_ids = tuple(landmark_ids[match.trainIdx] for match in unique_matches)
        matched_features = tuple(match.queryIdx for match in unique_matches)
        object_points = np.asarray(
            [self._landmarks[landmark_id].position_world_m for landmark_id in matched_ids],
            dtype=np.float64,
        )
        matched_pixels = np.asarray(
            [points[match.queryIdx] for match in unique_matches], dtype=np.float64
        )
        success, rotation_vector, translation_world_to_camera, inlier_indices = (
            cv2.solvePnPRansac(
                object_points,
                matched_pixels,
                intrinsic,
                distortion,
                flags=cv2.SOLVEPNP_EPNP,
                iterationsCount=int(iterations),
                reprojectionError=float(reprojection_error_px),
                confidence=float(confidence),
            )
        )
        if not success or inlier_indices is None:
            return None
        inlier_indices = inlier_indices.reshape(-1)
        if len(inlier_indices) < min_inliers:
            return None

        inlier_objects = object_points[inlier_indices]
        inlier_pixels = matched_pixels[inlier_indices]
        refined, rotation_vector, translation_world_to_camera = cv2.solvePnP(
            inlier_objects,
            inlier_pixels,
            intrinsic,
            distortion,
            rotation_vector,
            translation_world_to_camera,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not refined:
            return None

        rotation_world_to_camera, _ = cv2.Rodrigues(rotation_vector)
        transform_camera_world = np.eye(4)
        transform_camera_world[:3, :3] = rotation_world_to_camera
        transform_camera_world[:3, 3] = translation_world_to_camera.reshape(3)
        transform_world_camera = np.linalg.inv(transform_camera_world)

        depths = (
            rotation_world_to_camera @ inlier_objects.T
            + translation_world_to_camera.reshape(3, 1)
        )[2]
        if float(np.mean(depths > 0.0)) < 0.9:
            return None

        projected, _ = cv2.projectPoints(
            inlier_objects,
            rotation_vector,
            translation_world_to_camera,
            intrinsic,
            distortion,
        )
        errors = projected.reshape(-1, 2) - inlier_pixels
        reprojection_rmse = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
        inlier_ids = tuple(matched_ids[index] for index in inlier_indices)
        inlier_features = tuple(matched_features[index] for index in inlier_indices)
        return MapLocalizationResult(
            transform_world_camera=transform_world_camera,
            matched_landmark_ids=matched_ids,
            inlier_landmark_ids=inlier_ids,
            matched_feature_indices=matched_features,
            inlier_feature_indices=inlier_features,
            match_count=len(matched_ids),
            inlier_count=len(inlier_ids),
            inlier_ratio=len(inlier_ids) / len(matched_ids),
            reprojection_rmse_px=reprojection_rmse,
        )

def _validated_features(
    image_points_px: np.ndarray, descriptors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(image_points_px, dtype=float)
    binary = np.asarray(descriptors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("image_points_px must be a finite Nx2 array")
    if binary.ndim != 2 or binary.shape[0] != len(points) or binary.shape[1] == 0:
        raise ValueError("descriptors must be a non-empty NxD array matching the pixels")
    return points.copy(), binary.copy()


def _validated_camera_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    intrinsic = np.asarray(camera_matrix, dtype=float)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.isclose(intrinsic[2, 2], 1.0):
        raise ValueError("camera_matrix must contain positive focal lengths and K[2,2] = 1")
    return intrinsic.copy()


def _validated_distortion(distortion: np.ndarray | None) -> np.ndarray | None:
    if distortion is None:
        return None
    coefficients = np.asarray(distortion, dtype=float).reshape(-1)
    if coefficients.size not in {4, 5, 8, 12, 14} or not np.all(np.isfinite(coefficients)):
        raise ValueError("invalid distortion coefficients")
    return coefficients.copy()


def _validated_transform(transform: np.ndarray) -> np.ndarray:
    pose = np.asarray(transform, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite 4x4 transform")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("pose must have a homogeneous bottom row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("pose rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("pose rotation determinant must be +1")
    return pose.copy()
