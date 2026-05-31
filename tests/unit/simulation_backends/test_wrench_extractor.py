"""Unit tests for canonical GRF/wrench extraction helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from src.bunkershot3d.postproc.wrench_trace import WrenchTrace
from src.shared.python.simulation_backends.protocol import Trace
from src.shared.python.simulation_backends.wrench_extractor import (
    WrenchImpulses,
    compute_wrench_impulses,
    force_torque_from_wrench_array,
    static_support_wrench_trace,
    trace_with_wrench_trace,
    trace_wrench_impulses,
    wrench_array_from_force_torque,
    wrench_array_from_trace,
    wrench_trace_from_array,
    wrench_trace_from_force_torque,
)

pytestmark = pytest.mark.unit


def _time() -> np.ndarray:
    return np.linspace(0.0, 0.09, 10)


def _trace(wrench: np.ndarray | None = None) -> Trace:
    time = _time()
    q = np.zeros((time.size, 2))
    v = np.zeros_like(q)
    return Trace(
        t=time, q=q, v=v, dt=float(time[1] - time[0]), backend="test", wrench=wrench
    )


def test_wrench_trace_from_force_torque_reuses_existing_primitive() -> None:
    time = _time()
    force = np.full((time.size, 3), 5.0)
    torque = np.full((time.size, 3), 0.5)

    wrench_trace = wrench_trace_from_force_torque(time, force, torque)

    assert isinstance(wrench_trace, WrenchTrace)
    np.testing.assert_allclose(wrench_trace.force_world, force)
    np.testing.assert_allclose(wrench_trace.torque_world, torque)


def test_wrench_array_round_trip_preserves_force_torque_components() -> None:
    force: npt.NDArray[np.float64] = np.arange(30, dtype=float).reshape(10, 3)
    torque: npt.NDArray[np.float64] = force + 100.0

    wrench = wrench_array_from_force_torque(force, torque)
    force_rt, torque_rt = force_torque_from_wrench_array(wrench)

    assert wrench.shape == (10, 6)
    np.testing.assert_allclose(force_rt, force)
    np.testing.assert_allclose(torque_rt, torque)


def test_wrench_trace_array_round_trip_preserves_layout() -> None:
    time = _time()
    force = np.ones((time.size, 3))
    torque = np.ones((time.size, 3)) * 2.0
    source = wrench_trace_from_force_torque(time, force, torque)

    wrench = wrench_array_from_trace(source)
    result = wrench_trace_from_array(time, wrench)

    np.testing.assert_allclose(result.time, source.time)
    np.testing.assert_allclose(result.force_world, source.force_world)
    np.testing.assert_allclose(result.torque_world, source.torque_world)


def test_compute_wrench_impulses_delegates_to_wrench_trace_convention() -> None:
    time = _time()
    force = np.full((time.size, 3), [10.0, 0.0, 2.0])
    torque = np.full((time.size, 3), [0.0, 3.0, 0.0])
    wrench = wrench_array_from_force_torque(force, torque)

    impulses = compute_wrench_impulses(time, wrench)

    assert isinstance(impulses, WrenchImpulses)
    expected_duration = time[-1] - time[0]
    np.testing.assert_allclose(
        impulses.linear_impulse,
        [10.0 * expected_duration, 0.0, 2.0 * expected_duration],
    )
    np.testing.assert_allclose(
        impulses.angular_impulse, [0.0, 3.0 * expected_duration, 0.0]
    )


def test_trace_wrench_impulses_returns_none_without_wrench() -> None:
    assert trace_wrench_impulses(_trace(wrench=None)) is None


def test_trace_with_wrench_trace_persists_cc4_wrench_field() -> None:
    trace = _trace()
    time = trace.t
    wrench_trace = wrench_trace_from_force_torque(
        time,
        np.full((time.size, 3), [1.0, 2.0, 3.0]),
        np.full((time.size, 3), [4.0, 5.0, 6.0]),
    )

    result = trace_with_wrench_trace(trace, wrench_trace)

    assert result.wrench is not None
    np.testing.assert_allclose(result.wrench[:, :3], wrench_trace.force_world)
    np.testing.assert_allclose(result.wrench[:, 3:], wrench_trace.torque_world)
    impulses = trace_wrench_impulses(result)
    assert impulses is not None
    np.testing.assert_allclose(
        impulses.linear_impulse,
        [0.09, 0.18, 0.27],
    )


def test_static_support_known_case_matches_body_weight() -> None:
    time = _time()
    body_mass_kg = 80.0
    gravity = 9.81

    wrench_trace = static_support_wrench_trace(
        time,
        body_mass_kg=body_mass_kg,
        gravity_m_s2=gravity,
    )

    np.testing.assert_allclose(wrench_trace.force_world[:, 2], body_mass_kg * gravity)
    np.testing.assert_allclose(wrench_trace.force_world[:, :2], 0.0)
    np.testing.assert_allclose(wrench_trace.torque_world, 0.0)


@pytest.mark.parametrize(
    ("time", "force", "torque", "match"),
    [
        ([0.0, 0.0], np.zeros((2, 3)), np.zeros((2, 3)), "strictly increasing"),
        ([0.0, 1.0], np.zeros((3, 3)), np.zeros((2, 3)), "same timestep count"),
        ([0.0, 1.0], np.zeros((2, 2)), np.zeros((2, 3)), "force_world"),
        ([0.0, 1.0], np.zeros((2, 3)), np.zeros((2, 2)), "torque_world"),
    ],
)
def test_wrench_trace_validation(
    time: list[float],
    force: np.ndarray,
    torque: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        wrench_trace_from_force_torque(time, force, torque)


def test_trace_with_wrench_trace_requires_matching_time_grid() -> None:
    trace = _trace()
    wrench_trace = static_support_wrench_trace(
        trace.t + 1.0,
        body_mass_kg=80.0,
    )

    with pytest.raises(ValueError, match="trace.t"):
        trace_with_wrench_trace(trace, wrench_trace)
