"""Closure reactions split into drift and control, gravity once (Issue #14492).

At a fixed state the KKT system ``M a + h = tau + J_c^T lambda``,
``J_c a + Jdot_v = 0`` is linear in ``tau``, so the reaction splits like the
acceleration: ``lambda(q, v, tau) = lambda_drift(q, v) + lambda_control(q, tau)``.
The drift reaction (``tau = 0``) carries gravity and the velocity terms; the
control reaction carries neither. On the planar two-hand golfer these are the
wrist interaction forces of the MATLAB-era LW/RW plots.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.loop_nullspace.fixtures import (
    load_golden,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    PlanarGolferProvider,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend
from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    constrained_acceleration_at,
    drift_and_control_reactions,
)

pytestmark = pytest.mark.unit
ATOL = 1e-8


def _params() -> PlanarGolferParams:
    return PlanarGolferParams.from_golden(load_golden("planar_two_arm_full"))


@pytest.fixture(scope="module")
def planar():
    snapshot = planar_golfer_snapshot().snapshot
    rng = np.random.default_rng(14492)
    taus = [rng.uniform(-20.0, 20.0, snapshot.q.shape[0]) for _ in range(4)]
    return snapshot, PlanarGolferProvider(_params()), taus


def _kkt_reaction(M, load, J, Jdot_v):  # noqa: N803 - spec names
    n, m = M.shape[0], J.shape[0]
    kkt = np.block([[M, -J.T], [J, np.zeros((m, m))]])
    return np.linalg.solve(kkt, np.concatenate([load, -Jdot_v]))[n:]


def test_drift_plus_control_reaction_equals_total(planar) -> None:
    s, provider, taus = planar
    for tau in taus:
        drift, control = drift_and_control_reactions(provider, s.q, s.v, tau)
        _, total = constrained_acceleration_at(provider, s.q, s.v, tau)
        np.testing.assert_allclose(drift + control, total, atol=ATOL)
        _, unloaded = constrained_acceleration_at(provider, s.q, s.v, 0 * tau)
        np.testing.assert_allclose(drift, unloaded, atol=ATOL)


def test_control_reaction_matches_an_independent_kkt_solve(planar) -> None:
    s, provider, taus = planar
    M = provider.mass_matrix(s.q)  # noqa: N806 - spec name
    J = provider.closure_jacobian(s.q)  # noqa: N806 - spec name
    for tau in taus:
        _, control = drift_and_control_reactions(provider, s.q, s.v, tau)
        expected = _kkt_reaction(M, tau, J, np.zeros(J.shape[0]))
        np.testing.assert_allclose(control, expected, atol=ATOL)


def test_gravity_appears_only_in_the_drift_reaction(planar) -> None:
    s, provider, taus = planar
    weightless = PlanarGolferProvider(dataclasses.replace(_params(), gravity=0.0))
    zero_v = np.zeros_like(s.v)
    for tau in taus:
        drift_g, control_g = drift_and_control_reactions(provider, s.q, s.v, tau)
        drift_0, control_0 = drift_and_control_reactions(weightless, s.q, s.v, tau)
        np.testing.assert_allclose(control_g, control_0, atol=ATOL)
        assert np.abs(drift_g - drift_0).max() > 1e-3
        still, _ = drift_and_control_reactions(weightless, s.q, zero_v, tau)
        np.testing.assert_allclose(still, 0.0, atol=ATOL)


def test_control_reaction_ignores_velocity_and_is_linear(planar) -> None:
    s, provider, taus = planar
    zero_v = np.zeros_like(s.v)
    pairs = zip(taus, taus[1:], strict=False)
    for tau1, tau2 in pairs:
        at_rest = drift_and_control_reactions(provider, s.q, zero_v, tau1)[1]
        moving = drift_and_control_reactions(provider, s.q, s.v, tau1)[1]
        np.testing.assert_allclose(moving, at_rest, atol=ATOL)
        summed = drift_and_control_reactions(provider, s.q, s.v, tau1 + tau2)[1]
        parts = drift_and_control_reactions(provider, s.q, s.v, tau2)[1] + moving
        np.testing.assert_allclose(summed, parts, atol=ATOL)


def test_unconstrained_provider_is_rejected() -> None:
    backend = make_backend("ode", GolfModelParams.default())
    with pytest.raises(TypeError, match="ConstrainedEquationsOfMotionProvider"):
        drift_and_control_reactions(backend, np.zeros(2), np.zeros(2), np.ones(2))


def test_bad_torque_shape_is_rejected(planar) -> None:
    s, provider, _ = planar
    with pytest.raises(ValueError, match="tau"):
        drift_and_control_reactions(provider, s.q, s.v, np.ones(1))
