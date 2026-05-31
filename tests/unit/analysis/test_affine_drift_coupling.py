"""Tests for canonical/double-pendulum coupling to AffineDrift surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.analysis.affine_drift_coupling import (
    couple_trace_to_affine_drift,
    extract_double_pendulum_kinematics,
    read_affine_drift_coupling,
    write_affine_drift_coupling,
)
from src.shared.python.simulation_backends.protocol import Trace

pytestmark = pytest.mark.unit


class _DiagonalProvider:
    """Deterministic two-DOF provider with position-dependent inertia."""

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        return np.diag([2.0 + float(q[0]), 4.0 + float(q[1])])

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.array(
            [
                10.0 + 2.0 * float(q[0]) + float(v[0]),
                -3.0 + float(q[1]) - 2.0 * float(v[1]),
            ],
            dtype=float,
        )


def test_extracts_double_pendulum_trace_columns_directly() -> None:
    trace = Trace(
        t=np.array([0.0, 0.1]),
        q=np.array([[0.1, 0.2], [0.3, 0.4]]),
        v=np.array([[1.0, 2.0], [3.0, 4.0]]),
        dt=0.1,
        backend="ode",
    )

    q, v = extract_double_pendulum_kinematics(trace)

    assert np.array_equal(q, trace.q)
    assert np.array_equal(v, trace.v)


def test_extracts_last_internal_joints_from_canonical_v2_layout() -> None:
    q = np.zeros((2, 9), dtype=float)
    v = np.zeros((2, 8), dtype=float)
    q[:, -2:] = [[0.1, 0.2], [0.3, 0.4]]
    v[:, -2:] = [[1.0, 2.0], [3.0, 4.0]]
    trace = Trace(t=np.array([0.0, 0.1]), q=q, v=v, dt=0.1, backend="estimator")

    q_dp, v_dp = extract_double_pendulum_kinematics(trace)

    assert np.array_equal(q_dp, q[:, -2:])
    assert np.array_equal(v_dp, v[:, -2:])


def test_coupling_samples_affine_drift_and_control_terms() -> None:
    trace = Trace(
        t=np.array([0.0, 0.1]),
        q=np.array([[0.0, 0.0], [0.5, 1.0]]),
        v=np.array([[1.0, -1.0], [2.0, 3.0]]),
        u=np.array([[2.0, 8.0], [5.0, 10.0]]),
        dt=0.1,
        backend="ode",
    )
    provider = _DiagonalProvider()

    result = couple_trace_to_affine_drift(provider, trace)

    expected_drift_0 = np.array([-11.0 / 2.0, 1.0 / 4.0])
    expected_control_0 = np.array([1.0, 2.0])
    assert np.allclose(result.drift_acceleration[0], expected_drift_0)
    assert np.allclose(result.control_acceleration[0], expected_control_0)
    assert np.allclose(result.total_acceleration[0], [-4.5, 2.25])
    assert np.allclose(result.affine_drift[0], [1.0, -1.0, -5.5, 0.25])
    assert np.allclose(
        result.affine_control_matrix[0],
        [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0], [0.0, 0.25]],
    )
    assert result.source_backend == "ode"


def test_coupling_uses_zero_torque_when_trace_has_no_controls() -> None:
    trace = Trace(
        t=np.array([0.0]),
        q=np.array([[0.0, 0.0]]),
        v=np.array([[1.0, -1.0]]),
        dt=0.1,
        backend="passive",
    )

    result = couple_trace_to_affine_drift(_DiagonalProvider(), trace)

    assert np.array_equal(result.tau, np.zeros((1, 2)))
    assert np.array_equal(result.control_acceleration, np.zeros((1, 2)))
    assert np.allclose(result.total_acceleration, result.drift_acceleration)


def test_coupling_persists_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    trace = Trace(
        t=np.array([0.0, 0.1]),
        q=np.array([[0.0, 0.0], [0.5, 1.0]]),
        v=np.array([[1.0, -1.0], [2.0, 3.0]]),
        u=np.array([[2.0, 8.0], [5.0, 10.0]]),
        dt=0.1,
        backend="ode",
    )
    path = tmp_path / "affine_drift_coupling.h5"
    result = couple_trace_to_affine_drift(_DiagonalProvider(), trace)

    write_affine_drift_coupling(result, path)
    loaded = read_affine_drift_coupling(path)

    assert loaded.source_backend == result.source_backend
    assert loaded.schema_version == result.schema_version
    assert np.array_equal(loaded.t, result.t)
    assert np.array_equal(loaded.q, result.q)
    assert np.array_equal(loaded.v, result.v)
    assert np.array_equal(loaded.tau, result.tau)
    assert np.array_equal(loaded.drift_acceleration, result.drift_acceleration)
    assert np.array_equal(loaded.affine_control_matrix, result.affine_control_matrix)
