from __future__ import annotations

import numpy as np
import pytest

from shield_vio.backends import ESKFBackend, EstimatorBackend
from shield_vio.estimation.state import STATE_DIM


def test_eskf_backend_satisfies_protocol() -> None:
    backend = ESKFBackend()

    assert isinstance(backend, EstimatorBackend)
    assert backend.name == "eskf"


def test_eskf_backend_requires_initialization() -> None:
    backend = ESKFBackend()

    with pytest.raises(RuntimeError, match="initialized"):
        backend.snapshot()


def test_eskf_backend_propagates_imu_and_returns_copied_state() -> None:
    backend = ESKFBackend(gravity=np.zeros(3))
    backend.initialize(1_000_000_000)

    state = backend.process_imu(
        1_100_000_000,
        linear_acceleration_m_s2=np.array([1.0, 0.0, 0.0]),
        angular_velocity_rad_s=np.zeros(3),
    )

    assert state.timestamp_ns == 1_100_000_000
    assert state.position == pytest.approx(np.array([0.005, 0.0, 0.0]))
    assert state.velocity == pytest.approx(np.array([0.1, 0.0, 0.0]))
    assert state.covariance.shape == (STATE_DIM, STATE_DIM)

    state.position[0] = 99.0
    assert backend.snapshot().position[0] != 99.0

    health = backend.health()
    assert health.initialized is True
    assert health.tracking_status == "tracking"
    assert health.propagated_imu_samples == 1
    assert health.covariance_trace > 0.0
    assert health.innovation_nis is None


def test_eskf_backend_rejects_non_monotonic_imu_timestamps() -> None:
    backend = ESKFBackend()
    backend.initialize(10)

    with pytest.raises(ValueError, match="strictly increasing"):
        backend.process_imu(10, np.zeros(3), np.zeros(3))


def test_eskf_backend_exposes_visual_innovation_health() -> None:
    backend = ESKFBackend()
    backend.initialize(1_000)
    measurement_matrix = np.zeros((1, STATE_DIM))
    measurement_matrix[0, 0] = 1.0

    backend.process_visual_update(
        1_000,
        residual=np.array([0.25]),
        measurement_matrix=measurement_matrix,
        measurement_covariance=np.array([[0.5]]),
    )

    health = backend.health()
    assert health.innovation_nis is not None
    assert health.innovation_nis >= 0.0
