"""Supplementary unit tests for the ODE and MuJoCo CPU backends.

These target the stepping / time-accessor / shape-validation branches left
uncovered by the primary backend tests:

ODE backend (no optional deps):
    * :meth:`ODEBackend.step` with an explicit ``dt`` and its non-positive /
      non-finite precondition;
    * :meth:`ODEBackend.get_time` after a step.

MuJoCo CPU backend (gated behind ``requires_mujoco`` + a ``has_mujoco`` skip):
    * constructor ``TypeError`` on a non-:class:`GolfModelParams` and
      ``ValueError`` on a non-positive ``dt``;
    * :meth:`MuJoCoBackend.step` rejecting a non-positive ``dt``;
    * :meth:`MuJoCoBackend.rollout` rejecting a non-positive ``dt`` (distinct
      from the horizon guard) and a wrong-shape control history;
    * :meth:`MuJoCoBackend.set_control` rejecting a too-short vector and the
      ``_as_state_vector`` length guard via ``mass_matrix``;
    * :meth:`MuJoCoBackend.reset` round-trip from an explicit :class:`SimState`
      and its ``TypeError`` on a non-:class:`SimState`;
    * :meth:`MuJoCoBackend.forward_dynamics` with an explicit control vector;
    * :meth:`MuJoCoBackend.get_time` after a step.

Where useful the MuJoCo result is cross-checked against the ODE backend using
the same loose trajectory tolerances already adopted by the validation harness.
All RNG is seeded (``np.random.default_rng(0)``).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.simulation_backends import (
    GolfModelParams,
    SimState,
    has_mujoco,
)
from src.shared.python.simulation_backends.ode_backend import ODEBackend

pytestmark = pytest.mark.unit

# Skip the MuJoCo-dependent tests cleanly when the wheel is absent.
_skip_no_mujoco = pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed")

_RNG = np.random.default_rng(0)


def _params() -> GolfModelParams:
    """Return the canonical default golf model parameters."""
    return GolfModelParams.default()


def _make_mujoco(dt: float = 0.01):
    """Construct a default MuJoCoBackend (import deferred so collection is safe)."""
    from src.shared.python.simulation_backends.mujoco_backend import MuJoCoBackend

    return MuJoCoBackend(_params(), dt=dt)


# --------------------------------------------------------------------------- #
# ODE backend: step / get_time
# --------------------------------------------------------------------------- #
def test_ode_step_with_explicit_dt_advances_time() -> None:
    """An explicit ``dt`` overrides the constructor default for one step."""
    backend = ODEBackend(_params(), dt=0.05)
    backend.reset(SimState(q=[0.3, -0.2], v=[0.0, 0.0]))
    assert backend.get_time() == pytest.approx(0.0)
    backend.step(0.02)
    assert backend.get_time() == pytest.approx(0.02)


def test_ode_step_defaults_to_constructor_dt() -> None:
    """``step()`` with no argument uses the constructor ``dt``."""
    backend = ODEBackend(_params(), dt=0.05)
    backend.reset(None)
    backend.step()
    assert backend.get_time() == pytest.approx(0.05)


@pytest.mark.parametrize("bad_dt", [0.0, -0.01, np.inf, np.nan])
def test_ode_step_rejects_bad_dt(bad_dt: float) -> None:
    """A non-positive or non-finite explicit step size is rejected."""
    backend = ODEBackend(_params())
    with pytest.raises(ValueError):
        backend.step(bad_dt)


def test_ode_get_time_tracks_multiple_steps() -> None:
    """``get_time`` accumulates the per-step sizes."""
    backend = ODEBackend(_params(), dt=0.01)
    backend.reset(None)
    for _ in range(3):
        backend.step(0.01)
    assert backend.get_time() == pytest.approx(0.03)


# --------------------------------------------------------------------------- #
# MuJoCo backend: constructor preconditions
# --------------------------------------------------------------------------- #
@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_mujoco_constructor_rejects_non_params() -> None:
    """A non-GolfModelParams ``params`` raises TypeError."""
    from src.shared.python.simulation_backends.mujoco_backend import MuJoCoBackend

    with pytest.raises(TypeError, match="GolfModelParams"):
        MuJoCoBackend("not-a-params", dt=0.01)  # type: ignore[arg-type]


@pytest.mark.requires_mujoco
@_skip_no_mujoco
@pytest.mark.parametrize("bad_dt", [0.0, -0.5])
def test_mujoco_constructor_rejects_non_positive_dt(bad_dt: float) -> None:
    """A non-positive timestep is a precondition violation (ValueError)."""
    from src.shared.python.simulation_backends.mujoco_backend import MuJoCoBackend

    with pytest.raises(ValueError, match="positive"):
        MuJoCoBackend(_params(), dt=bad_dt)


# --------------------------------------------------------------------------- #
# MuJoCo backend: step / rollout dt guards
# --------------------------------------------------------------------------- #
@pytest.mark.requires_mujoco
@_skip_no_mujoco
@pytest.mark.parametrize("bad_dt", [0.0, -0.01])
def test_mujoco_step_rejects_non_positive_dt(bad_dt: float) -> None:
    """An explicit non-positive step ``dt`` raises ValueError."""
    backend = _make_mujoco()
    with pytest.raises(ValueError, match="positive"):
        backend.step(bad_dt)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
@pytest.mark.parametrize("bad_dt", [0.0, -0.01])
def test_mujoco_rollout_rejects_non_positive_dt(bad_dt: float) -> None:
    """A non-positive rollout ``dt`` (with a valid horizon) raises ValueError."""
    backend = _make_mujoco()
    with pytest.raises(ValueError, match="positive"):
        backend.rollout(controls=None, horizon=5, dt=bad_dt)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_mujoco_rollout_rejects_wrong_control_width() -> None:
    """Controls with the wrong trailing width raise ValueError."""
    backend = _make_mujoco()
    bad = np.zeros((5, 3))  # nu is 2, not 3
    with pytest.raises(ValueError, match="controls"):
        backend.rollout(controls=bad, horizon=5, dt=0.01)


# --------------------------------------------------------------------------- #
# MuJoCo backend: control / state shape guards
# --------------------------------------------------------------------------- #
@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_mujoco_set_control_rejects_too_short_vector() -> None:
    """A control vector shorter than ``nu`` entries is rejected."""
    backend = _make_mujoco()
    with pytest.raises(ValueError, match="at least"):
        backend.set_control(np.array([1.0]))


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_mujoco_mass_matrix_rejects_wrong_length_q() -> None:
    """``mass_matrix`` validates the ``q`` length via ``_as_state_vector``."""
    backend = _make_mujoco()
    with pytest.raises(ValueError, match="shape"):
        backend.mass_matrix(np.array([0.0, 0.1, 0.2]))


# --------------------------------------------------------------------------- #
# MuJoCo backend: reset round-trip and type guard
# --------------------------------------------------------------------------- #
@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_mujoco_reset_from_simstate_round_trips() -> None:
    """``reset(SimState(...))`` then ``get_state`` returns the seeded q/v."""
    backend = _make_mujoco()
    q = np.array([0.3, -0.7])
    v = np.array([1.1, -2.2])
    backend.reset(SimState(q=q, v=v, time=4.0))
    state = backend.get_state()
    np.testing.assert_allclose(state.q, q, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(state.v, v, atol=1e-12, rtol=0.0)
    # reset always zeroes the integrator clock (time argument is ignored).
    assert backend.get_time() == pytest.approx(0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_mujoco_reset_rejects_non_simstate() -> None:
    """A non-SimState, non-None reset target raises TypeError."""
    backend = _make_mujoco()
    with pytest.raises(TypeError, match="SimState"):
        backend.reset(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# MuJoCo backend: forward_dynamics with explicit control, get_time after step
# --------------------------------------------------------------------------- #
@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_mujoco_forward_dynamics_with_explicit_control() -> None:
    """``forward_dynamics`` with an explicit torque matches the ODE backend.

    The acceleration is an algebraic, single-state quantity, so the two
    independent derivations agree to a tight tolerance (the value used by the
    validation harness for bias forces).
    """
    mj = _make_mujoco()
    ode = ODEBackend(_params())
    q = np.array([0.25, -0.6])
    v = np.array([0.9, -1.1])
    u = np.array([2.5, -1.5])
    qacc_mj = mj.forward_dynamics(q, v, u)
    qacc_ode = ode.forward_dynamics(q, v, u)
    assert qacc_mj.shape == (2,)
    np.testing.assert_allclose(qacc_mj, qacc_ode, atol=1e-8, rtol=0.0)


@pytest.mark.requires_mujoco
@_skip_no_mujoco
def test_mujoco_get_time_after_step() -> None:
    """A single step advances the reported simulation time by ``dt``."""
    backend = _make_mujoco(dt=0.01)
    backend.reset(SimState(q=np.array([0.4, 0.1]), v=np.zeros(2)))
    assert backend.get_time() == pytest.approx(0.0)
    backend.step()
    assert backend.get_time() == pytest.approx(0.01, abs=1e-12)
