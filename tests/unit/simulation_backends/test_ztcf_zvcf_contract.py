"""Golden contract: drift + control = total, gravity counted once (Issue #14491).

Decision (Dieter, 2026-11-03, epic #14497): ZTCF is the drift
``M(q)^-1 (-C(q, v) v - g(q) - passive)`` with ``tau = 0``, and ZVCF is the
control component ``M(q)^-1 tau`` (torques on, velocity and gravity off),
matching the original MATLAB ``SCRIPT_ZVCF_GENERATOR.m``. At a fixed state::

    ztcf(q, v) + M(q)^-1 tau == forward(q, v, tau)

with gravity counted once. The identity is pointwise; integrated
counterfactual trajectories do not generally sum. The models here are
unconstrained: on a closed loop the control term also carries the reaction
change, ``M^-1 (tau + J^T (lambda - lambda_0))`` (see ``constrained_ztcf_zvcf``).

``forward`` is each backend's own ``forward_dynamics``, so the analytical
double pendulum (``ode``) and MuJoCo are checked against independent solves.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from src.shared.python.simulation_backends import (
    GolfModelParams,
    available_backends,
    has_mujoco,
    make_backend,
)
from src.shared.python.simulation_backends.ztcf_zvcf import (
    drift_and_control_split,
    zero_velocity_control_preserved_acceleration,
    ztcf_acceleration,
    zvcf_acceleration,
)

pytestmark = pytest.mark.unit

ATOL_ANALYTICAL = 1e-9
ATOL_MUJOCO = 1e-6

_BACKENDS = [
    pytest.param("ode", ATOL_ANALYTICAL, id="analytical"),
    pytest.param(
        "mujoco",
        ATOL_MUJOCO,
        id="mujoco",
        marks=[
            pytest.mark.requires_mujoco,
            pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed"),
        ],
    ),
]


def _backend(name: str, *, gravity: bool = True):
    if name not in available_backends():
        pytest.skip(f"{name} backend not registered")
    return make_backend(
        name, GolfModelParams.default().model_copy(update={"gravity_enabled": gravity})
    )


def _samples(n: int = 8) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(14491)
    for _ in range(n):
        yield (
            rng.uniform(-np.pi, np.pi, 2),
            rng.uniform(-5.0, 5.0, 2),
            rng.uniform(-10.0, 10.0, 2),
        )


def _control(provider, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return drift_and_control_split(provider, q, v, tau)[1]


@pytest.mark.parametrize(("name", "atol"), _BACKENDS)
def test_drift_plus_control_equals_forward_dynamics(name: str, atol: float) -> None:
    backend = _backend(name)
    for q, v, tau in _samples():
        control = _control(backend, q, v, tau)
        np.testing.assert_allclose(
            control, np.linalg.solve(backend.mass_matrix(q), tau), atol=atol
        )
        total = ztcf_acceleration(backend, q, v) + control
        np.testing.assert_allclose(
            total, backend.forward_dynamics(q, v, tau), atol=atol
        )


@pytest.mark.parametrize(("name", "atol"), _BACKENDS)
def test_control_preserved_diagnostic_double_counts_gravity(
    name: str, atol: float
) -> None:
    # The trap: M^-1 (tau - g) added to ZTCF counts gravity twice, and the
    # surplus is exactly -M^-1 g (bias at zero velocity is gravity here).
    backend = _backend(name)
    for q, v, tau in _samples():
        summed = ztcf_acceleration(backend, q, v)
        summed = summed + zero_velocity_control_preserved_acceleration(backend, q, tau)
        gravity = backend.bias_forces(q, np.zeros(2))
        surplus = summed - backend.forward_dynamics(q, v, tau)
        minus_m_inv_g = -np.linalg.solve(backend.mass_matrix(q), gravity)
        np.testing.assert_allclose(surplus, minus_m_inv_g, atol=atol)
        assert np.abs(minus_m_inv_g).max() > 1e-3


@pytest.mark.parametrize(("name", "atol"), _BACKENDS)
def test_control_is_independent_of_velocity_and_gravity(name: str, atol: float) -> None:
    with_g = _backend(name)
    without_g = _backend(name, gravity=False)
    for q, v, tau in _samples():
        reference = _control(with_g, q, np.zeros(2), tau)
        np.testing.assert_allclose(_control(with_g, q, v, tau), reference, atol=atol)
        np.testing.assert_allclose(_control(without_g, q, v, tau), reference, atol=atol)


@pytest.mark.parametrize(("name", "atol"), _BACKENDS)
def test_control_is_linear_in_torque(name: str, atol: float) -> None:
    backend = _backend(name)
    samples = list(_samples())
    for (q, v, tau1), (_, _, tau2) in zip(samples, samples[1:], strict=False):
        np.testing.assert_allclose(
            _control(backend, q, v, tau1 + tau2),
            _control(backend, q, v, tau1) + _control(backend, q, v, tau2),
            atol=atol,
        )


@pytest.mark.xfail(
    strict=True,
    raises=(TypeError, AssertionError),
    reason="#14494 restores zvcf_acceleration = M^-1 tau (control)",
)
def test_zvcf_is_the_control_component() -> None:
    backend = _backend("ode")
    for q, v, tau in _samples():
        np.testing.assert_allclose(
            ztcf_acceleration(backend, q, v) + zvcf_acceleration(backend, q, tau),
            backend.forward_dynamics(q, v, tau),
            atol=ATOL_ANALYTICAL,
        )
