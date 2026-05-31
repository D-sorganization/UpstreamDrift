"""Unit tests for simulation_backends.wrench_extractor (CC-25, #6798).

TDD: these tests were written before the implementation and drove the design.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.simulation_backends.protocol import Trace
from src.shared.python.simulation_backends.wrench_extractor import (
    WrenchImpulses,
    compute_wrench_impulses,
    force_torque_from_wrench_array,
    trace_wrench_impulses,
    wrench_array_from_force_torque,
)

pytestmark = pytest.mark.unit

_RNG = np.random.default_rng(0)
_T = 10  # timesteps
_DT = 0.01  # step [s]


def _times() -> np.ndarray:
    return np.arange(_T, dtype=float) * _DT


def _base_trace(wrench: np.ndarray | None = None) -> Trace:
    t = _times()
    q = _RNG.standard_normal((_T, 2))
    v = _RNG.standard_normal((_T, 2))
    return Trace(t=t, q=q, v=v, dt=_DT, backend="ode", wrench=wrench)


# ---------------------------------------------------------------------------
# WrenchImpulses dataclass
# ---------------------------------------------------------------------------


class TestWrenchImpulses:
    def test_construction(self) -> None:
        lin = np.array([1.0, 2.0, 3.0])
        ang = np.array([0.1, 0.2, 0.3])
        imp = WrenchImpulses(linear_impulse=lin, angular_impulse=ang)
        assert np.array_equal(imp.linear_impulse, lin)
        assert np.array_equal(imp.angular_impulse, ang)

    def test_frozen(self) -> None:
        imp = WrenchImpulses(linear_impulse=np.zeros(3), angular_impulse=np.zeros(3))
        with pytest.raises((AttributeError, TypeError)):
            imp.linear_impulse = np.ones(3)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# wrench_array_from_force_torque
# ---------------------------------------------------------------------------


class TestWrenchArrayFromForceTorque:
    def test_packs_correctly(self) -> None:
        force = np.ones((_T, 3)) * 10.0
        torque = np.ones((_T, 3)) * 2.0
        w = wrench_array_from_force_torque(force, torque)
        assert w.shape == (_T, 6)
        assert np.all(w[:, :3] == 10.0)
        assert np.all(w[:, 3:] == 2.0)

    def test_invalid_force_shape(self) -> None:
        with pytest.raises(ValueError, match="force"):
            wrench_array_from_force_torque(np.ones((_T,)), np.ones((_T, 3)))

    def test_invalid_torque_shape(self) -> None:
        with pytest.raises(ValueError, match="torque"):
            wrench_array_from_force_torque(np.ones((_T, 3)), np.ones(3))

    def test_mismatched_timesteps(self) -> None:
        with pytest.raises(ValueError, match="same number of timesteps"):
            wrench_array_from_force_torque(np.ones((_T, 3)), np.ones((_T + 1, 3)))


# ---------------------------------------------------------------------------
# force_torque_from_wrench_array
# ---------------------------------------------------------------------------


class TestForceTorqueFromWrenchArray:
    def test_unpacks_correctly(self) -> None:
        w = np.hstack([np.ones((_T, 3)) * 5.0, np.ones((_T, 3)) * 1.5])
        force, torque = force_torque_from_wrench_array(w)
        assert force.shape == (_T, 3)
        assert torque.shape == (_T, 3)
        assert np.all(force == 5.0)
        assert np.all(torque == 1.5)

    def test_round_trip(self) -> None:
        force_orig = _RNG.standard_normal((_T, 3))
        torque_orig = _RNG.standard_normal((_T, 3))
        w = wrench_array_from_force_torque(force_orig, torque_orig)
        force_rt, torque_rt = force_torque_from_wrench_array(w)
        assert np.allclose(force_rt, force_orig)
        assert np.allclose(torque_rt, torque_orig)

    def test_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            force_torque_from_wrench_array(np.ones((_T, 3)))


# ---------------------------------------------------------------------------
# compute_wrench_impulses
# ---------------------------------------------------------------------------


class TestComputeWrenchImpulses:
    def test_constant_force_known_impulse(self) -> None:
        """Constant 10 N force over 0.09 s → impulse = 0.9 N·s each component."""
        force = np.ones((_T, 3)) * 10.0
        torque = np.zeros((_T, 3))
        w = wrench_array_from_force_torque(force, torque)
        t = _times()
        imp = compute_wrench_impulses(t, w)
        expected = 10.0 * (t[-1] - t[0])
        assert np.allclose(imp.linear_impulse, [expected] * 3, atol=1e-12)
        assert np.allclose(imp.angular_impulse, [0.0, 0.0, 0.0], atol=1e-12)

    def test_constant_torque_known_impulse(self) -> None:
        force = np.zeros((_T, 3))
        torque = np.ones((_T, 3)) * 3.0
        w = wrench_array_from_force_torque(force, torque)
        t = _times()
        imp = compute_wrench_impulses(t, w)
        expected = 3.0 * (t[-1] - t[0])
        assert np.allclose(imp.angular_impulse, [expected] * 3, atol=1e-12)
        assert np.allclose(imp.linear_impulse, [0.0, 0.0, 0.0], atol=1e-12)

    def test_time_length_mismatch(self) -> None:
        w = np.ones((_T, 6))
        with pytest.raises(ValueError, match="time length"):
            compute_wrench_impulses(np.arange(_T + 1, dtype=float) * _DT, w)

    def test_invalid_wrench_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            compute_wrench_impulses(_times(), np.ones((_T, 3)))


# ---------------------------------------------------------------------------
# trace_wrench_impulses
# ---------------------------------------------------------------------------


class TestTraceWrenchImpulses:
    def test_returns_none_when_no_wrench(self) -> None:
        trace = _base_trace(wrench=None)
        assert trace_wrench_impulses(trace) is None

    def test_returns_impulses_when_wrench_present(self) -> None:
        force = np.ones((_T, 3)) * 5.0
        torque = np.ones((_T, 3)) * 1.0
        w = wrench_array_from_force_torque(force, torque)
        trace = _base_trace(wrench=w)
        imp = trace_wrench_impulses(trace)
        assert imp is not None
        assert isinstance(imp, WrenchImpulses)
        t = _times()
        expected_lin = 5.0 * (t[-1] - t[0])
        assert np.allclose(imp.linear_impulse, [expected_lin] * 3, atol=1e-12)

    def test_impulses_match_direct_computation(self) -> None:
        w = _RNG.standard_normal((_T, 6))
        trace = _base_trace(wrench=w)
        via_trace = trace_wrench_impulses(trace)
        direct = compute_wrench_impulses(_times(), w)
        assert via_trace is not None
        assert np.allclose(via_trace.linear_impulse, direct.linear_impulse)
        assert np.allclose(via_trace.angular_impulse, direct.angular_impulse)
