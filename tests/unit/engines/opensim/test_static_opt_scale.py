"""Full-body-scale static optimisation (SPN-10b, #12761).

SLSQP does not finish a 446-muscle, 51-DOF frame (200 s and a failed
iteration limit at p=2 on an unreachable demand; over 25 minutes in a real
study).  Above ``CONIC_MIN_MUSCLES`` the redistribution stage is therefore a
convex conic program (QP for p=2, power cone for p=3).  All data are INVENTED.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python import static_opt_core as core

pytestmark = pytest.mark.unit


def _problem(nm: int, nc: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    active = rng.uniform(100.0, 1000.0, nm)
    passive = rng.uniform(0.0, 20.0, nm)
    arms = rng.normal(scale=0.03, size=(nm, nc)) * (rng.random((nm, nc)) < 0.4)
    reach = arms.T @ (0.3 * active + passive)
    return active, passive, arms, reach


@pytest.mark.parametrize("exponent", [2, 3])
def test_conic_matches_slsqp_on_a_small_problem(exponent: int) -> None:
    active, passive, arms, reach = _problem(14, 4)
    gain = arms.T * active[None, :]
    a1, _ = core._min_reserve_activation(
        gain, reach + 5.0 - arms.T @ passive, np.ones(4)
    )
    slsqp, ok, _msg = core._redistribute(a1, gain, exponent)
    conic, ok_c, _msg_c = core._redistribute_conic(a1, gain, exponent)
    assert ok and ok_c
    np.testing.assert_allclose(conic, slsqp, atol=1e-5)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("exponent", [2, 3])
def test_full_body_scale_unreachable_demand_solves(exponent: int) -> None:
    nm = core.CONIC_MIN_MUSCLES + 300
    active, passive, arms, reach = _problem(nm, 45)
    tau = reach + np.random.default_rng(9).normal(scale=20.0, size=45)
    sol = core.solve_activations(
        active, passive, arms, tau, exponent=exponent, reserve_scale=np.ones(45)
    )
    assert sol.success, sol.message
    assert np.all((sol.activation >= 0.0) & (sol.activation <= 1.0))
    gain = arms.T * active[None, :]
    a1, _ = core._min_reserve_activation(gain, tau - arms.T @ passive, np.ones(45))
    # redistribution keeps the stage-1 muscle moment and cannot raise the cost
    np.testing.assert_allclose(gain @ sol.activation, gain @ a1, atol=1e-6, rtol=0.0)
    assert sol.cost <= core.activation_cost(a1, exponent)[0] + 1e-9
