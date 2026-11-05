"""ZVCF is the control component; the gravity-only term is renamed (Issue #14494).

Decision (Dieter, 2026-11-03, epic #14497, reversing #11553): ZVCF is the
control component ``M(q)^-1 tau`` (torques on, velocity and gravity off),
matching the original MATLAB. The former gravity-only quantity
``-M(q)^-1 h(q, 0)`` keeps its value under the explicit name
``zero_velocity_zero_torque_acceleration``. On a closed loop ZVCF carries the
reaction change, ``M^-1 (tau + J^T (lambda - lambda_0))``; bare ``M^-1 tau``
is never applied to a constrained provider.
"""

from __future__ import annotations

import warnings

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
from src.shared.python.motion_matching.counterfactual import (
    AccelerationDecomposition,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend
from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    constrained_acceleration_at,
)
from src.shared.python.simulation_backends.ztcf_zvcf import (
    drift_and_control_split,
    zero_velocity_zero_torque_acceleration,
    ztcf_acceleration,
    zvcf_acceleration,
)

pytestmark = pytest.mark.unit
ATOL = 1e-9


def _samples():
    rng = np.random.default_rng(14494)
    for _ in range(4):
        yield rng.uniform(-1.5, 1.5, 2), rng.uniform(-3, 3, 2), rng.uniform(-9, 9, 2)


@pytest.fixture(scope="module")
def ode():
    return make_backend("ode", GolfModelParams.default())


@pytest.fixture(scope="module")
def planar():
    snapshot = planar_golfer_snapshot().snapshot
    params = PlanarGolferParams.from_golden(load_golden("planar_two_arm_full"))
    rng = np.random.default_rng(14494)
    taus = [rng.uniform(-20.0, 20.0, snapshot.q.shape[0]) for _ in range(3)]
    return snapshot, PlanarGolferProvider(params), taus


def test_zvcf_is_mass_inverse_torque_unconstrained(ode) -> None:
    for q, _, tau in _samples():
        np.testing.assert_allclose(
            zvcf_acceleration(ode, q, tau),
            np.linalg.solve(ode.mass_matrix(q), tau),
            atol=ATOL,
        )


def test_zvcf_equals_split_control_for_any_velocity(ode) -> None:
    for q, v, tau in _samples():
        np.testing.assert_allclose(
            zvcf_acceleration(ode, q, tau),
            drift_and_control_split(ode, q, v, tau)[1],
            atol=ATOL,
        )


def test_constrained_zvcf_carries_reaction_change(planar) -> None:
    s, provider, taus = planar
    for tau in taus:
        zvcf = zvcf_acceleration(provider, s.q, tau)
        total, _ = constrained_acceleration_at(provider, s.q, s.v, tau)
        np.testing.assert_allclose(
            ztcf_acceleration(provider, s.q, s.v) + zvcf, total, atol=1e-8
        )
        bare = np.linalg.solve(provider.mass_matrix(s.q), tau)
        assert not np.allclose(zvcf, bare, atol=1e-6)


def test_zero_velocity_zero_torque_keeps_former_value(ode) -> None:
    for q, _, _ in _samples():
        expected = -np.linalg.solve(ode.mass_matrix(q), ode.bias_forces(q, 0 * q))
        np.testing.assert_allclose(
            zero_velocity_zero_torque_acceleration(ode, q), expected, atol=ATOL
        )


def test_two_argument_zvcf_is_deprecated_alias(ode) -> None:
    q = np.array([0.4, -0.2])
    with pytest.warns(DeprecationWarning, match="zero_velocity_zero_torque"):
        legacy = zvcf_acceleration(ode, q)
    np.testing.assert_allclose(
        legacy, zero_velocity_zero_torque_acceleration(ode, q), atol=ATOL
    )


def test_three_argument_zvcf_does_not_warn(ode) -> None:
    q, _, tau = next(_samples())
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        zvcf_acceleration(ode, q, tau)


def test_decomposition_ztcf_plus_zvcf_is_total(ode) -> None:
    for q, v, tau in _samples():
        decomp = AccelerationDecomposition.from_dynamics(ode, q, v, tau)
        np.testing.assert_allclose(decomp.ztcf, ztcf_acceleration(ode, q, v))
        np.testing.assert_allclose(decomp.zvcf, zvcf_acceleration(ode, q, tau))
        np.testing.assert_allclose(
            decomp.ztcf + decomp.zvcf, ode.forward_dynamics(q, v, tau), atol=ATOL
        )
        np.testing.assert_allclose(
            decomp.zero_velocity_zero_torque,
            zero_velocity_zero_torque_acceleration(ode, q),
        )
