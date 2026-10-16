"""NSX-93 (#12722): the impedance and delayed-feedback layer inside the replay.

Oracles: the point mass has a closed-form delay margin
(:func:`analytic_pd_delay_margin`); the planar two-arm golfer is the shared
NSX-31 oracle.  Nothing here is calibrated to a golfer.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.feedback_comparison import ReplayPolicy
from src.shared.python.loop_nullspace.feedback_layer import (
    DelayedFeedback,
    FeedbackLayer,
    GainSchedule,
)
from src.shared.python.loop_nullspace.impedance_spec import (
    ImpedanceWindow,
    JointImpedance,
)
from src.shared.python.loop_nullspace.planar_model import PlanarGolferProvider
from src.shared.python.loop_nullspace.replay import (
    ReplayMode,
    ReplaySettings,
    ReplaySetup,
    replay,
)
from tests.unit.loop_nullspace._layer_fixtures import (
    PointMass,
    analytic_pd_delay_margin,
    point_mass_setup,
    zero_torque_allocation,
)
from tests.unit.loop_nullspace.test_replay import (
    N as N_PLANAR,
)
from tests.unit.loop_nullspace.test_replay import (
    S as S_PLANAR,
)
from tests.unit.loop_nullspace.test_replay import (
    T_END as T_PLANAR,
)
from tests.unit.loop_nullspace.test_replay import (
    _exact_allocation,
    _params,
    _reference,
    _setup,
    _state0,
    _tau,
)

pytestmark = pytest.mark.unit

M = PointMass.m
OMEGA_N, ZETA = 10.0, 0.5
KP, KD = M * OMEGA_N**2, 2.0 * ZETA * M * OMEGA_N
T_PM = 3.0


def _schedule(kp: float, kd: float, t_end: float = T_PM) -> GainSchedule:
    return GainSchedule(
        times_s=np.array([0.0, t_end]),
        kq=np.array([[[kp]]]),
        kv=np.array([[[kd]]]),
    )


def _point_mass_replay(layer: FeedbackLayer | None, *, t_end=T_PM, substeps=2):
    policy = (
        ReplayPolicy.INDEPENDENT_TIME_ONLY
        if layer is None
        else ReplayPolicy.STATE_FEEDBACK
    )
    return replay(
        zero_torque_allocation(t_end, 301, 1),
        PointMass(),
        policy=policy,
        setup=point_mass_setup(t_end),
        feedback=layer,
        settings=ReplaySettings(substeps=substeps, divergence_tolerance=1e6),
    )


# ------------------------------------------------------- negative control --
def test_zero_impedance_and_zero_gain_reproduce_the_open_loop_replay_exactly() -> None:
    """Tolerance 0: ``0.0 * finite == 0.0`` and ``a + 0.0 == a`` in IEEE-754."""
    open_loop = _point_mass_replay(None)
    zero = FeedbackLayer(
        n_actuated=1,
        impedance=JointImpedance.zero(1),
        feedback=DelayedFeedback(_schedule(0.0, 0.0), delay_s=0.03),
    )
    layered = _point_mass_replay(zero)
    assert layered.mode is ReplayMode.FEEDBACK
    np.testing.assert_array_equal(layered.q, open_loop.q)
    np.testing.assert_array_equal(layered.v, open_loop.v)
    assert layered.feedback_budget is not None
    assert layered.feedback_budget.energy_ratio == 0.0
    assert not np.any(layered.feedback_torque)


def test_zero_layer_reproduces_the_planar_golfer_open_loop_exactly() -> None:
    q0, v0 = _state0()
    allocation = _exact_allocation(16)
    provider = PlanarGolferProvider(_params())
    base = replay(
        allocation,
        provider,
        policy=ReplayPolicy.INDEPENDENT_TIME_ONLY,
        setup=_setup(),
        settings=ReplaySettings(substeps=2),
    )
    sol = _reference(_tau)
    from src.shared.python.loop_nullspace.replay import ReferenceTrajectory

    times = np.linspace(0.0, T_PLANAR, 31)
    st = sol.sol(times)
    ref = ReferenceTrajectory(times, st[:N_PLANAR].T, st[N_PLANAR:].T)
    layer = FeedbackLayer(
        n_actuated=N_PLANAR,
        impedance=JointImpedance.zero(N_PLANAR),
        feedback=None,
    )
    out = replay(
        allocation,
        provider,
        policy=ReplayPolicy.STATE_FEEDBACK,
        setup=ReplaySetup(actuation=S_PLANAR, q0=q0, v0=v0, reference=ref),
        feedback=layer,
        settings=ReplaySettings(substeps=2),
    )
    np.testing.assert_array_equal(out.q, base.q)
    np.testing.assert_array_equal(out.v, base.v)


# ---------------------------------------------------------- restoring sign --
def test_impedance_is_restoring_in_the_replay() -> None:
    """A displaced free mass held by a spring-damper returns to the reference."""
    imp = JointImpedance((ImpedanceWindow(0.0, T_PM, np.array([KP]), np.array([KD])),))
    out = _point_mass_replay(FeedbackLayer(1, imp, None))
    x = out.q[:, 0]
    assert abs(x[-1]) < 1e-3 * abs(x[0])  # damped (zeta 0.5): decayed in 3 s
    # the initial torque pushes against the +0.1 displacement
    assert out.feedback_torque[0, 0] == pytest.approx(-KP * 0.1)
    assert np.all(np.sign(out.feedback_torque[0]) == -np.sign(x[0]))


def test_a_declared_impedance_window_acts_only_inside_its_window() -> None:
    imp = JointImpedance((ImpedanceWindow(1.0, 2.0, np.array([KP]), np.array([KD])),))
    out = _point_mass_replay(FeedbackLayer(1, imp, None))
    inside = (out.times_s >= 1.0) & (out.times_s < 2.0)
    assert not np.any(out.feedback_torque[~inside])
    assert np.any(out.feedback_torque[inside])
    # free drift (v0 = 0) before the window: the displacement is untouched
    np.testing.assert_allclose(out.q[out.times_s < 1.0, 0], 0.1)


# ---------------------------------------------------------------- the delay --
def test_discrete_state_history_is_hermite_exact_for_cubics() -> None:
    from src.shared.python.loop_nullspace.feedback_layer import StateHistory

    hist = StateHistory(0.0, np.array([0.5]), np.array([0.0]))
    f = lambda t: 0.5 + 2.0 * t - 3.0 * t**2 + t**3  # noqa: E731
    df = lambda t: 2.0 - 6.0 * t + 3.0 * t**2  # noqa: E731
    for t in np.linspace(0.1, 1.0, 10):
        hist.append(float(t), np.array([f(t)]), np.array([df(t)]))
    q, v = hist.at(0.437)
    assert q[0] == pytest.approx(f(0.437), abs=1e-14)
    assert v[0] == pytest.approx(df(0.437), abs=1e-13)


def test_history_before_the_start_holds_the_initial_state() -> None:
    from src.shared.python.loop_nullspace.feedback_layer import StateHistory

    hist = StateHistory(0.0, np.array([0.3]), np.array([1.0]))
    hist.append(0.1, np.array([0.4]), np.array([1.0]))
    q, v = hist.at(-0.5)
    assert q[0] == 0.3 and v[0] == 1.0


def test_a_delay_shorter_than_the_integration_step_is_rejected() -> None:
    """RK4 stages need the state ``delay`` ago; it must already be integrated."""
    layer = FeedbackLayer(1, None, DelayedFeedback(_schedule(KP, KD), delay_s=1e-4))
    with pytest.raises(ValueError, match="delay"):
        _point_mass_replay(layer, substeps=1)  # step 0.01 s > 1e-4 s


@pytest.mark.parametrize(("factor", "stable"), [(0.7, True), (1.3, False)])
def test_a_delay_beyond_the_analytic_margin_reproduces_the_instability(
    factor: float, stable: bool
) -> None:
    """Stability switches at the closed-form margin (independent oracle).

    Margin 0.0711 s for ``omega_n = 10``, ``zeta = 0.5``.  The replay is a
    different computation (RK4 on the real plant with a stored-history delay)
    from the textbook frequency-domain margin, so agreement is not circular.
    """
    margin = analytic_pd_delay_margin(M, KP, KD)
    assert margin == pytest.approx(0.0711, abs=5e-4)
    layer = FeedbackLayer(
        1, None, DelayedFeedback(_schedule(KP, KD), delay_s=factor * margin)
    )
    out = _point_mass_replay(layer, t_end=T_PM, substeps=4)
    envelope_start = np.max(np.abs(out.q[out.times_s < 0.5, 0]))
    envelope_end = np.max(np.abs(out.q[out.times_s > T_PM - 0.5, 0]))
    if stable:
        assert envelope_end < 1e-2 * envelope_start
    else:
        assert envelope_end > 10.0 * envelope_start


# ------------------------------------------------------------------ budget --
def test_the_layer_budget_is_reported_on_its_own_line() -> None:
    imp = JointImpedance((ImpedanceWindow(0.0, T_PM, np.array([KP]), np.array([KD])),))
    out = _point_mass_replay(FeedbackLayer(1, imp, None))
    assert out.feedback_budget is not None
    np.testing.assert_array_equal(out.allocated_torque, np.zeros_like(out.q))
    np.testing.assert_array_equal(out.applied_torque, out.feedback_torque)
    assert out.feedback_budget.abs_work_feedback_j > 0.0
    assert out.feedback_budget.abs_work_allocated_j == 0.0


def test_layer_validation() -> None:
    sched = _schedule(1.0, 1.0)
    with pytest.raises(ValueError, match="delay_s"):
        DelayedFeedback(sched, delay_s=-1.0)
    with pytest.raises(ValueError, match="at least one"):
        FeedbackLayer(1, None, None)
    with pytest.raises(ValueError, match="n_actuated"):
        FeedbackLayer(2, JointImpedance.zero(1), None)
    with pytest.raises(ValueError, match="kq"):
        GainSchedule(np.array([0.0, 1.0]), np.zeros((2, 1, 1)), np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="times_s"):
        GainSchedule(np.array([1.0, 0.0]), np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
