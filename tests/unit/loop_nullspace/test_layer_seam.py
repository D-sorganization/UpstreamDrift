"""NSX-93 (#12722): the layer's seams with the tangent chain and the campaign.

* ``tangent_velocity_offset`` exposes the ``(P, O)`` of NSX-84 so actuated
  errors can be written on the tangent state.  Closed form for the pendulum
  (a unit-ish point mass on a circle, constant angular rate ``omega``): moving
  the angle by ``xi`` rotates the velocity, ``O = -(omega / L) q``.
* ``TorqueNoise.scale`` is the declared noise-inflation penalty applied to the
  campaign draw; ``None`` and a unit scale are the same trial exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace import paired_monte_carlo as pmc
from src.shared.python.loop_nullspace import tangent_linearization as tl
from src.shared.python.loop_nullspace.impedance_spec import (
    ImpedanceWindow,
    JointImpedance,
)
from src.shared.python.loop_nullspace.layer_linear import impedance_state_gain
from src.shared.python.loop_nullspace.replay_campaign import (
    TorqueNoise,
    replay_strategy,
)
from src.engines.feedback_comparison import ReplayPolicy
from tests.unit.loop_nullspace._tangent_fixtures import Pendulum, tangent_field
from tests.unit.loop_nullspace.test_replay import N
from tests.unit.loop_nullspace.test_replay_campaign import (
    FRAMES,
    _case,
    _outcomes,
    _spec,
)

pytestmark = pytest.mark.unit

EUCLID2 = tl.EuclideanManifold(2)
PROVIDERS = [tl.FiniteDifferenceDerivatives, tl.AssembledKKTDerivatives]


@pytest.mark.parametrize("cls", PROVIDERS)
def test_pendulum_velocity_offset_matches_the_closed_form(cls) -> None:
    pend = Pendulum()
    q, v = pend.state()
    prov = cls(pend, EUCLID2, basis_field=tangent_field)
    basis, offset = prov.tangent_velocity_offset(q, v)
    np.testing.assert_allclose(basis, tangent_field(q), atol=1e-12)
    expected = -(pend.omega / pend.length) * q.reshape(2, 1)
    np.testing.assert_allclose(offset, expected, atol=1e-6)
    # the offset keeps the perturbed velocity admissible: q . dv + v . dq = 0
    assert float(q @ offset[:, 0] + v @ basis[:, 0]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("cls", PROVIDERS)
def test_impedance_closes_the_pendulum_tangent_chain_in_closed_form(cls) -> None:
    """``F_cl = F - G K_imp`` with ``G[1] = t^T / m`` and ``K_imp = k P + d O``."""
    pend = Pendulum()
    q, v = pend.state()
    prov = cls(pend, EUCLID2, basis_field=tangent_field)
    derivs = prov.constrained_acceleration_derivatives(q, v, pend.load(0.0))
    ss = tl.tangent_state_space(
        derivs, q=q, manifold=EUCLID2, basis_field=tangent_field
    )
    basis, offset = prov.tangent_velocity_offset(q, v)
    k, d = 40.0, 6.0
    window = ImpedanceWindow(0.0, 1.0, k * np.eye(2), d * np.eye(2))
    k_imp = impedance_state_gain(
        JointImpedance((window,)), [0.0], np.eye(2), basis, offset
    )[0]
    # G acts on the two world forces: the tangent acceleration is t.f / m
    np.testing.assert_allclose(ss.G[1], (basis[:, 0] / pend.m), atol=1e-6)
    f_cl = ss.F - ss.G @ k_imp
    t_hat = basis[:, 0]
    assert f_cl[1, 0] == pytest.approx(
        ss.F[1, 0] - (k * (t_hat @ t_hat) + d * (t_hat @ offset[:, 0])) / pend.m,
        abs=1e-6,
    )
    assert f_cl[1, 1] == pytest.approx(
        ss.F[1, 1] - d * (t_hat @ t_hat) / pend.m, abs=1e-6
    )
    # the stiffness restores (negative feedback) and the damper dissipates
    assert f_cl[1, 0] < ss.F[1, 0] and f_cl[1, 1] < ss.F[1, 1]


def test_unit_and_absent_noise_scale_are_the_same_trial_and_zero_scale_is_silent() -> (
    None
):
    spec = _spec(0.05, n_trials=3)
    case = _case(ReplayPolicy.INDEPENDENT_TIME_ONLY)
    steps = FRAMES - 1

    def outcomes(scale):
        noise = TorqueNoise("joint_torque", np.eye(N), scale=scale)
        strat = replay_strategy(spec, "s", case, noise, _outcomes)
        return pmc.run_seed_set(spec, [strat], "paired").outcome_matrix("s")

    base = outcomes(None)
    np.testing.assert_array_equal(outcomes(np.ones((steps, N))), base)
    assert np.ptp(base, axis=0).max() > 0.0
    assert np.ptp(outcomes(np.zeros((steps, N))), axis=0).max() == 0.0
    # to first order the dispersion follows the scale: doubling doubles the spread
    ratio = np.std(outcomes(2.0 * np.ones((steps, N))), axis=0) / np.std(base, axis=0)
    np.testing.assert_allclose(ratio, 2.0, rtol=0.05)


def test_noise_scale_is_validated() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TorqueNoise("joint_torque", np.eye(N), scale=-np.ones((4, N)))
    with pytest.raises(ValueError, match="non-negative"):
        TorqueNoise("joint_torque", np.eye(N), scale=np.ones(N))
    spec = _spec(0.05, n_trials=2)
    with pytest.raises(ValueError, match="scale must be"):
        replay_strategy(
            spec,
            "s",
            _case(ReplayPolicy.INDEPENDENT_TIME_ONLY),
            TorqueNoise("joint_torque", np.eye(N), scale=np.ones((3, N))),
            _outcomes,
        )
