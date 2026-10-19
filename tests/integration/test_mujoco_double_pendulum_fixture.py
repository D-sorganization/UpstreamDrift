"""Regression tests for MuJoCo double pendulum fixture phantom contact (#12989).

On tests/fixtures/models/double_pendulum.urdf, base_link was fixed to the
world but had a collision sphere overlapping upper_link. MuJoCo compiled
base_link into the worldbody and detected a permanent penetration (distance -0.02)
with upper_link, corrupting compute_contact_forces (~3 kN spurious GRF),
compute_drift_acceleration, and compute_inverse_dynamics.

These tests assert:
1. compute_contact_forces is zero on a model without ground.
2. compute_drift_acceleration matches -M^-1 * bias.
3. compute_inverse_dynamics matches M * qacc + bias.
4. MuJoCo and Pinocchio agree on drift, ID, and zero contact force when Pinocchio
   is available.
5. compute_drift_acceleration docstring explicitly documents contact handling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
    MuJoCoPhysicsEngine,
)
from src.shared.python.data_io.path_utils import get_repo_root
from src.shared.python.engine_core.engine_availability import require_engine

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

_URDF_PATH = get_repo_root() / "tests" / "fixtures" / "models" / "double_pendulum.urdf"
_Q = np.array([0.5, 0.3])
_V = np.array([0.2, -0.4])
_QACC = np.array([1.0, -1.0])


@pytest.fixture
def engine() -> MuJoCoPhysicsEngine:
    """Loaded MuJoCo engine with double_pendulum.urdf at non-trivial state."""
    eng = MuJoCoPhysicsEngine()
    eng.load_from_path(str(_URDF_PATH))
    eng.set_state(_Q, _V)
    return eng


def test_double_pendulum_no_phantom_contact_forces(
    engine: MuJoCoPhysicsEngine,
) -> None:
    """Fixture with no ground must yield zero contact forces (#12989)."""
    assert engine.data is not None
    assert engine.data.ncon == 0, (
        f"Expected ncon=0 on double_pendulum fixture, got {engine.data.ncon}"
    )
    contact_forces = engine.compute_contact_forces()
    np.testing.assert_allclose(
        contact_forces,
        np.zeros(3),
        atol=1e-9,
        err_msg="Expected zero contact force on fixture without ground",
    )


def test_double_pendulum_drift_acceleration_equals_unconstrained_bias(
    engine: MuJoCoPhysicsEngine,
) -> None:
    """Drift acceleration must equal -M^-1 * bias on ground-free model (#12989)."""
    drift = engine.compute_drift_acceleration()
    M = engine.compute_mass_matrix()
    bias = engine.compute_bias_forces()
    expected_drift = -np.linalg.solve(M, bias)

    np.testing.assert_allclose(
        drift,
        expected_drift,
        atol=1e-9,
        err_msg="MuJoCo drift acceleration corrupted by phantom contact forces",
    )
    # Check agreement with analytical reference values from issue #12989
    np.testing.assert_allclose(drift, [-4.92503515, -4.4189079], atol=1e-4)


def test_double_pendulum_inverse_dynamics_matches_analytical(
    engine: MuJoCoPhysicsEngine,
) -> None:
    """Inverse dynamics must equal M * qacc + bias on ground-free model (#12989)."""
    tau = engine.compute_inverse_dynamics(_QACC)
    M = engine.compute_mass_matrix()
    bias = engine.compute_bias_forces()
    expected_tau = M @ _QACC + bias

    np.testing.assert_allclose(
        tau,
        expected_tau,
        atol=1e-9,
        err_msg="MuJoCo inverse dynamics corrupted by phantom contact forces",
    )
    # Check agreement with analytical reference values from issue #12989
    np.testing.assert_allclose(tau, [8.96798933, 3.76418137], atol=1e-4)


@require_engine("pinocchio")
def test_double_pendulum_pinocchio_parity(engine: MuJoCoPhysicsEngine) -> None:
    """MuJoCo and Pinocchio agree on drift, ID, and zero contact force (#12989).

    ``require_engine`` rejects the ``MagicMock`` that ``tests/unit/conftest.py``
    installs for collection, which ``pytest.importorskip`` would accept.
    """
    import pinocchio

    pin_model = pinocchio.buildModelFromUrdf(str(_URDF_PATH))
    pin_data = pin_model.createData()

    pin_drift = pinocchio.aba(pin_model, pin_data, _Q, _V, np.zeros(pin_model.nv))
    pin_id = pinocchio.rnea(pin_model, pin_data, _Q, _V, _QACC)

    mj_drift = engine.compute_drift_acceleration()
    mj_id = engine.compute_inverse_dynamics(_QACC)
    mj_contact = engine.compute_contact_forces()

    np.testing.assert_allclose(mj_drift, pin_drift, atol=1e-8)
    np.testing.assert_allclose(mj_id, pin_id, atol=1e-8)
    np.testing.assert_allclose(mj_contact, np.zeros(3), atol=1e-8)


def test_drift_acceleration_docstring_states_contact_handling() -> None:
    """The drift docstring must state whether contact is included (#12989)."""
    doc = MuJoCoPhysicsEngine.compute_drift_acceleration.__doc__
    assert doc is not None
    doc_lower = doc.lower()
    assert "contact" in doc_lower, (
        "compute_drift_acceleration docstring must state contact handling"
    )
