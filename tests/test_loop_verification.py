from __future__ import annotations

import numpy as np
import pytest

import shield_vio.loop_verification as module
from shield_vio.vision.two_view_geometry import RelativePoseMeasurement


def _measurement(*, degenerate: bool = False, epipolar: float = 0.2) -> RelativePoseMeasurement:
    mask = np.ones(30, dtype=bool)
    return RelativePoseMeasurement(
        rotation=np.eye(3),
        translation_direction=np.array([1.0, 0.0, 0.0]),
        inlier_mask=mask,
        correspondence_count=30,
        inlier_count=30,
        inlier_ratio=1.0,
        median_parallax_px=4.0,
        median_epipolar_error_px=epipolar,
        is_degenerate=degenerate,
    )


def test_accepts_verified_geometry_and_builds_metric_loop_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "estimate_relative_pose", lambda *args, **kwargs: _measurement())
    result = module.verify_loop_candidate(
        100,
        5,
        np.zeros((30, 2)),
        np.zeros((30, 2)),
        np.eye(3),
        translation_scale_m=2.5,
    )
    assert result.accepted
    assert result.reason == "accepted"
    assert result.edge is not None
    assert result.edge.kind == "loop"
    assert result.edge.source_id == 5
    assert result.edge.target_id == 100
    assert result.edge.transform[:3, 3] == pytest.approx([2.5, 0.0, 0.0])


def test_rejects_degenerate_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "estimate_relative_pose",
        lambda *args, **kwargs: _measurement(degenerate=True),
    )
    result = module.verify_loop_candidate(
        100,
        5,
        np.zeros((30, 2)),
        np.zeros((30, 2)),
        np.eye(3),
        translation_scale_m=1.0,
    )
    assert not result.accepted
    assert result.reason == "degenerate_geometry"
    assert result.edge is None


def test_rejects_excessive_epipolar_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "estimate_relative_pose",
        lambda *args, **kwargs: _measurement(epipolar=2.0),
    )
    result = module.verify_loop_candidate(
        100,
        5,
        np.zeros((30, 2)),
        np.zeros((30, 2)),
        np.eye(3),
        translation_scale_m=1.0,
        max_median_epipolar_error_px=1.0,
    )
    assert not result.accepted
    assert result.reason == "epipolar_error"
    assert result.edge is None


def test_rejects_invalid_scale_and_self_loop() -> None:
    with pytest.raises(ValueError, match="different frames"):
        module.verify_loop_candidate(
            5,
            5,
            np.zeros((8, 2)),
            np.zeros((8, 2)),
            np.eye(3),
            translation_scale_m=1.0,
        )
    with pytest.raises(ValueError, match="translation_scale"):
        module.verify_loop_candidate(
            6,
            5,
            np.zeros((8, 2)),
            np.zeros((8, 2)),
            np.eye(3),
            translation_scale_m=0.0,
        )
