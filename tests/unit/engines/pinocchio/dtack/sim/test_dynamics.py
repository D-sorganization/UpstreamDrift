"""Tests for src.engines.physics_engines.pinocchio.python.dtack.sim.dynamics."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def dynamics_engine_cls() -> Generator[Any, None, None]:
    """Provide DynamicsEngine class, mocking pinocchio if not installed."""
    try:
        import pinocchio  # noqa: F401

        from src.engines.physics_engines.pinocchio.python.dtack.sim.dynamics import (
            DynamicsEngine,
        )

        yield DynamicsEngine
    except ImportError:
        with patch.dict(sys.modules, {"pinocchio": MagicMock()}):
            from src.engines.physics_engines.pinocchio.python.dtack.sim.dynamics import (
                DynamicsEngine,
            )

            yield DynamicsEngine


def test_compute_zvcf_signature_has_no_tau(dynamics_engine_cls: Any) -> None:
    """Verify that compute_zvcf signature has no tau parameter."""
    sig = inspect.signature(dynamics_engine_cls.compute_zvcf)
    assert "tau" not in sig.parameters
    assert "q" in sig.parameters
    assert "dt" in sig.parameters
    assert "f_ext" in sig.parameters


def test_compute_zero_velocity_controlled_signature_has_tau(
    dynamics_engine_cls: Any,
) -> None:
    """Verify that compute_zero_velocity_controlled signature has tau parameter."""
    sig = inspect.signature(dynamics_engine_cls.compute_zero_velocity_controlled)
    assert "tau" in sig.parameters
    assert "q" in sig.parameters
    assert "dt" in sig.parameters
    assert "f_ext" in sig.parameters


def test_compute_zvcf_calls_forward_dynamics_with_zero_tau(
    dynamics_engine_cls: Any,
) -> None:
    """Verify compute_zvcf passes tau=0 to forward_dynamics."""
    mock_model = MagicMock()
    mock_model.nv = 2
    mock_data = MagicMock()
    engine = dynamics_engine_cls(mock_model, mock_data)
    engine.forward_dynamics = MagicMock(return_value=np.array([1.0, 2.0]))

    with patch(
        "src.engines.physics_engines.pinocchio.python.dtack.sim.dynamics.pin"
    ) as mock_pin:
        mock_pin.integrate.return_value = np.array([0.11, 0.22])
        q = np.array([0.1, 0.2])
        dt = 0.01

        q_next, v_next = engine.compute_zvcf(q, dt)

        engine.forward_dynamics.assert_called_once()
        call_args = engine.forward_dynamics.call_args
        assert np.allclose(call_args[0][0], q)
        assert np.allclose(call_args[0][1], [0.0, 0.0])
        assert np.allclose(call_args[0][2], [0.0, 0.0])
        assert np.allclose(v_next, [0.01, 0.02])
        assert np.allclose(q_next, [0.11, 0.22])


def test_compute_zero_velocity_controlled_calls_forward_dynamics_with_tau(
    dynamics_engine_cls: type,
) -> None:
    """Verify compute_zero_velocity_controlled passes caller tau to forward_dynamics."""
    mock_model = MagicMock()
    mock_model.nv = 2
    mock_data = MagicMock()
    engine = dynamics_engine_cls(mock_model, mock_data)
    engine.forward_dynamics = MagicMock(return_value=np.array([3.0, 4.0]))

    with patch(
        "src.engines.physics_engines.pinocchio.python.dtack.sim.dynamics.pin"
    ) as mock_pin:
        mock_pin.integrate.return_value = np.array([0.13, 0.24])
        q = np.array([0.1, 0.2])
        tau = np.array([5.0, 6.0])
        dt = 0.01

        q_next, v_next = engine.compute_zero_velocity_controlled(q, tau, dt)

        engine.forward_dynamics.assert_called_once()
        call_args = engine.forward_dynamics.call_args
        assert np.allclose(call_args[0][0], q)
        assert np.allclose(call_args[0][1], [0.0, 0.0])
        assert np.allclose(call_args[0][2], tau)
        assert np.allclose(v_next, [0.03, 0.04])
        assert np.allclose(q_next, [0.13, 0.24])


def test_compute_zvcf_and_controlled_on_pinocchio_model() -> None:
    """Verify compute_zvcf equals forward_dynamics(q, 0, 0) step on a tiny Pinocchio model."""
    pin = pytest.importorskip("pinocchio")
    if isinstance(pin, MagicMock):
        pytest.skip("Pinocchio is mocked; real pinocchio required for this test.")

    from src.engines.physics_engines.pinocchio.python.dtack.sim.dynamics import (
        DynamicsEngine,
    )

    model = pin.Model()
    joint_id = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "joint1")
    body_inertia = pin.Inertia.FromSphere(1.0, 0.1)
    model.appendBodyToJoint(joint_id, body_inertia, pin.SE3.Identity())
    data = model.createData()

    engine = DynamicsEngine(model, data)
    q = np.array([0.5])
    dt = 0.01

    # compute_zvcf: canonical (v=0, tau=0)
    q_zvcf, v_zvcf = engine.compute_zvcf(q, dt)

    # manual expected: forward_dynamics(q, 0, 0)
    a_expected_zero = engine.forward_dynamics(q, np.zeros(model.nv), np.zeros(model.nv))
    v_expected_zero = np.zeros(model.nv) + a_expected_zero * dt
    q_expected_zero = pin.integrate(model, q, v_expected_zero * dt)

    assert np.allclose(q_zvcf, q_expected_zero)
    assert np.allclose(v_zvcf, v_expected_zero)

    # compute_zero_velocity_controlled: controlled variant (v=0, tau=tau)
    tau = np.array([2.5])
    q_ctrl, v_ctrl = engine.compute_zero_velocity_controlled(q, tau, dt)
    a_expected_ctrl = engine.forward_dynamics(q, np.zeros(model.nv), tau)
    v_expected_ctrl = np.zeros(model.nv) + a_expected_ctrl * dt
    q_expected_ctrl = pin.integrate(model, q, v_expected_ctrl * dt)

    assert np.allclose(q_ctrl, q_expected_ctrl)
    assert np.allclose(v_ctrl, v_expected_ctrl)
    assert not np.allclose(q_ctrl, q_zvcf)
