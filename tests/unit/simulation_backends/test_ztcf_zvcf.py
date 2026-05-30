"""Unit tests for ZTCF / ZVCF counterfactual primitives (epic task M7).

These verify that the pointwise zero-torque / zero-velocity acceleration
decompositions, computed purely from ``DynamicsProvider`` primitives, match the
analytical ground truth (:class:`PendulumPhysicsEngine`) and -- when MuJoCo is
installed -- the MuJoCo CPU backend, establishing cross-backend agreement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
)
from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
    PendulumPhysicsEngine,
)
from src.shared.python.simulation_backends import (
    GolfModelParams,
    available_backends,
    has_mujoco,
    make_backend,
)
from src.shared.python.simulation_backends.ztcf_zvcf import (
    drift_and_control_split,
    evaluate_ztcf_along_trajectory,
    ztcf_acceleration,
    zvcf_acceleration,
)

pytestmark = pytest.mark.unit

# Tolerances: the analytical provider is bit-for-bit identical math, MuJoCo is a
# distinct implementation cross-checked to a looser bound.
ATOL_ANALYTICAL = 1e-9
ATOL_MUJOCO = 1e-7


class _AnalyticalProvider:
    """Minimal :class:`DynamicsProvider` built directly on the analytical EOM.

    Independent of any concrete backend, this lets the ZTCF/ZVCF math be
    validated even when the ``ode`` backend module is not yet present, and uses
    the *same* ``DoublePendulumDynamics`` that ``PendulumPhysicsEngine`` wraps.
    """

    def __init__(self, params: GolfModelParams) -> None:
        self._dyn = DoublePendulumDynamics(params.to_double_pendulum_parameters())

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return ``M(q)`` (depends only on the relative angle ``theta2``)."""
        return np.asarray(self._dyn.mass_matrix(float(q[1])), dtype=float)

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return Coriolis + gravity + damping at ``(q, v)``."""
        c1, c2 = self._dyn.coriolis_vector(float(q[1]), float(v[0]), float(v[1]))
        g1, g2 = self._dyn.gravity_vector(float(q[0]), float(q[1]))
        d1, d2 = self._dyn.damping_vector(float(v[0]), float(v[1]))
        return np.array([c1 + g1 + d1, c2 + g2 + d2])


def _seeded_states(n: int = 6) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``n`` reproducible ``(q, v)`` samples (seeded RNG)."""
    rng = np.random.default_rng(0)
    return [
        (rng.uniform(-np.pi, np.pi, 2), rng.uniform(-5.0, 5.0, 2)) for _ in range(n)
    ]


def _seeded_controls(n: int = 6) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``n`` reproducible ``(q, tau)`` samples (seeded RNG)."""
    rng = np.random.default_rng(1)
    return [
        (rng.uniform(-np.pi, np.pi, 2), rng.uniform(-10.0, 10.0, 2)) for _ in range(n)
    ]


# --------------------------------------------------------------------------- #
# Spec-mandated tests: against make_backend('ode', ...) and the engine.        #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    "ode" not in available_backends(), reason="ode backend not registered"
)
def test_ztcf_matches_engine_via_ode_backend() -> None:
    """ZTCF accel from the ODE provider matches ``compute_ztcf`` to 1e-9."""
    eng = PendulumPhysicsEngine()
    ode = make_backend("ode", GolfModelParams.default())
    for q, v in _seeded_states():
        expected = eng.compute_ztcf(np.asarray(q), np.asarray(v))
        actual = ztcf_acceleration(ode, q, v)
        assert np.allclose(actual, expected, atol=ATOL_ANALYTICAL)


@pytest.mark.skipif(
    "ode" not in available_backends(), reason="ode backend not registered"
)
def test_zvcf_matches_engine_via_ode_backend() -> None:
    """ZVCF accel from the ODE provider matches ``compute_zvcf`` to 1e-9."""
    eng = PendulumPhysicsEngine()
    ode = make_backend("ode", GolfModelParams.default())
    for q, tau in _seeded_controls():
        eng.control = np.asarray(tau)
        expected = eng.compute_zvcf(np.asarray(q))
        actual = zvcf_acceleration(ode, q, tau)
        assert np.allclose(actual, expected, atol=ATOL_ANALYTICAL)


# --------------------------------------------------------------------------- #
# Backend-independent correctness via the analytical provider.                 #
# --------------------------------------------------------------------------- #


def test_ztcf_matches_engine_analytical_provider() -> None:
    """ZTCF via the analytical provider equals ``compute_ztcf`` to 1e-9."""
    eng = PendulumPhysicsEngine()
    provider = _AnalyticalProvider(GolfModelParams.default())
    for q, v in _seeded_states():
        expected = eng.compute_ztcf(np.asarray(q), np.asarray(v))
        actual = ztcf_acceleration(provider, q, v)
        assert np.allclose(actual, expected, atol=ATOL_ANALYTICAL)


def test_zvcf_matches_engine_analytical_provider() -> None:
    """ZVCF via the analytical provider equals ``compute_zvcf`` to 1e-9."""
    eng = PendulumPhysicsEngine()
    provider = _AnalyticalProvider(GolfModelParams.default())
    for q, tau in _seeded_controls():
        eng.control = np.asarray(tau)
        expected = eng.compute_zvcf(np.asarray(q))
        actual = zvcf_acceleration(provider, q, tau)
        assert np.allclose(actual, expected, atol=ATOL_ANALYTICAL)


def test_drift_and_control_split_sums_to_total() -> None:
    """``drift + control`` reconstructs ``solve(M, tau - bias(q, v))``."""
    provider = _AnalyticalProvider(GolfModelParams.default())
    rng = np.random.default_rng(2)
    for _ in range(6):
        q = rng.uniform(-np.pi, np.pi, 2)
        v = rng.uniform(-5.0, 5.0, 2)
        tau = rng.uniform(-10.0, 10.0, 2)
        drift, control = drift_and_control_split(provider, q, v, tau)
        # drift must equal the standalone ZTCF acceleration.
        assert np.allclose(drift, ztcf_acceleration(provider, q, v), atol=1e-12)
        mass = provider.mass_matrix(q)
        bias = provider.bias_forces(q, v)
        total = np.linalg.solve(mass, tau - bias)
        assert np.allclose(drift + control, total, atol=1e-9)


def test_evaluate_ztcf_along_trajectory_is_pointwise() -> None:
    """Trajectory evaluation equals per-sample ZTCF (pointwise semantics)."""
    provider = _AnalyticalProvider(GolfModelParams.default())
    rng = np.random.default_rng(3)
    t = 12
    q_traj = rng.uniform(-np.pi, np.pi, (t, 2))
    v_traj = rng.uniform(-5.0, 5.0, (t, 2))
    out = evaluate_ztcf_along_trajectory(provider, q_traj, v_traj)
    assert out.shape == (t, 2)
    for i in range(t):
        assert np.allclose(out[i], ztcf_acceleration(provider, q_traj[i], v_traj[i]))


def test_zvcf_reduces_to_control_minus_gravity() -> None:
    """At ``v = 0`` the ZVCF bias is gravity only (Coriolis/damping vanish)."""
    dyn = DoublePendulumDynamics(
        GolfModelParams.default().to_double_pendulum_parameters()
    )
    provider = _AnalyticalProvider(GolfModelParams.default())
    rng = np.random.default_rng(4)
    for _ in range(5):
        q = rng.uniform(-np.pi, np.pi, 2)
        tau = rng.uniform(-10.0, 10.0, 2)
        g1, g2 = dyn.gravity_vector(float(q[0]), float(q[1]))
        mass = np.asarray(dyn.mass_matrix(float(q[1])), dtype=float)
        expected = np.linalg.solve(mass, tau - np.array([g1, g2]))
        assert np.allclose(zvcf_acceleration(provider, q, tau), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# Design-by-Contract precondition tests.                                       #
# --------------------------------------------------------------------------- #


def test_ztcf_rejects_mismatched_dims() -> None:
    """Mismatched ``q``/``v`` lengths raise ``ValueError``."""
    provider = _AnalyticalProvider(GolfModelParams.default())
    with pytest.raises(ValueError):
        ztcf_acceleration(provider, np.zeros(2), np.zeros(3))


def test_ztcf_rejects_non_finite() -> None:
    """Non-finite inputs raise ``ValueError``."""
    provider = _AnalyticalProvider(GolfModelParams.default())
    with pytest.raises(ValueError):
        ztcf_acceleration(provider, np.array([np.nan, 0.0]), np.zeros(2))


def test_zvcf_rejects_mismatched_dims() -> None:
    """Mismatched ``q``/``tau`` lengths raise ``ValueError``."""
    provider = _AnalyticalProvider(GolfModelParams.default())
    with pytest.raises(ValueError):
        zvcf_acceleration(provider, np.zeros(2), np.zeros(3))


def test_evaluate_trajectory_rejects_1d() -> None:
    """A 1-D trajectory raises ``ValueError`` (must be ``(T, n)``)."""
    provider = _AnalyticalProvider(GolfModelParams.default())
    with pytest.raises(ValueError):
        evaluate_ztcf_along_trajectory(provider, np.zeros(2), np.zeros(2))


# --------------------------------------------------------------------------- #
# MuJoCo cross-validation (skipped when MuJoCo is unavailable).                #
# --------------------------------------------------------------------------- #


@pytest.mark.requires_mujoco
@pytest.mark.skipif(not has_mujoco(), reason="mujoco not installed")
def test_ztcf_zvcf_mujoco_matches_analytical_engine() -> None:
    """MuJoCo-provider ZTCF/ZVCF agree with the analytical engine to 1e-7."""
    eng = PendulumPhysicsEngine()
    mj = make_backend("mujoco", GolfModelParams.default())
    for q, v in _seeded_states():
        expected = eng.compute_ztcf(np.asarray(q), np.asarray(v))
        assert np.allclose(ztcf_acceleration(mj, q, v), expected, atol=ATOL_MUJOCO)
    for q, tau in _seeded_controls():
        eng.control = np.asarray(tau)
        expected = eng.compute_zvcf(np.asarray(q))
        assert np.allclose(zvcf_acceleration(mj, q, tau), expected, atol=ATOL_MUJOCO)


# --------------------------------------------------------------------------- #
# Self-referential AGENT-NOTE presence guard (epic task M7.3).                 #
# --------------------------------------------------------------------------- #


def test_source_contains_agent_note() -> None:
    """The module must prominently document its pointwise semantics."""
    from src.shared.python.simulation_backends import ztcf_zvcf

    source = Path(ztcf_zvcf.__file__).read_text(encoding="utf-8")
    assert "AGENT-NOTE" in source
    assert "POINTWISE" in source.upper()
