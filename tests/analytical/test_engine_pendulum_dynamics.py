"""Real numeric 1-DOF pendulum dynamics parity for Drake and OpenSim (#7049).

The Drake and OpenSim engine unit tests were previously mock-only, with no
value-asserting dynamics. This module pins the closed-form 1-DOF pendulum
identities directly against each engine's analytic dynamics, gated by the
matching ``requires_*`` marker plus an explicit availability ``skipif`` so a
missing engine wheel is a clean skip.

For each engine and several joint angles ``theta`` we assert:

* **Gravity torque** matches the analytic ``m * g * L_com * sin(theta)``
  (the gravitational generalized force needed to hold the link static).
* **Mass matrix** ``M`` is symmetric positive-definite and equals the
  analytic rotational inertia about the pivot.
* **Newton-Euler round-trip** ``tau = M @ a + bias`` is reproduced by the
  engine's inverse-dynamics call to within ``1e-10`` (convention-independent
  internal-consistency identity).

The analytic ``L_com`` / inertia come from the checked-in model fixtures:

* Drake: ``tests/fixtures/models/simple_pendulum.urdf`` — point mass
  ``m = 1 kg`` at ``L = 1 m`` (so ``M ≈ m*L^2`` about the pivot).
* OpenSim: ``src/shared/models/opensim/examples/pendulum_1dof.osim`` — rod
  with COM at ``0.5 m``, ``Izz_com = 0.0833333`` (so ``M = Izz + m*L_com^2``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.core.constants import GRAVITY_M_S2
from src.shared.python.engine_core.engine_availability import (
    is_engine_available,
)

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]

DRAKE_URDF = REPO_ROOT / "tests" / "fixtures" / "models" / "simple_pendulum.urdf"
OPENSIM_OSIM = (
    REPO_ROOT
    / "src"
    / "shared"
    / "models"
    / "opensim"
    / "examples"
    / "pendulum_1dof.osim"
)

# Analytic model parameters (source of truth = the checked-in fixtures).
DRAKE_MASS_KG = 1.0
DRAKE_L_COM_M = 1.0  # point mass at the rod tip
DRAKE_INERTIA_ABOUT_PIVOT = DRAKE_MASS_KG * DRAKE_L_COM_M**2  # ≈ 1.0

OPENSIM_MASS_KG = 1.0
OPENSIM_L_COM_M = 0.5
OPENSIM_IZZ_COM = 0.0833333
OPENSIM_INERTIA_ABOUT_PIVOT = (
    OPENSIM_IZZ_COM + OPENSIM_MASS_KG * OPENSIM_L_COM_M**2
)  # ≈ 0.3333333

TEST_ANGLES_RAD = [0.0, 0.1, 0.5, 1.0, np.pi / 2]


def _gravity_torque_magnitude(mass: float, l_com: float, theta: float) -> float:
    """Analytic gravity-torque magnitude about the pivot (N·m).

    For a 1-DOF revolute pendulum the gravitational generalized force needed
    to hold the link static is ``m * g * L_com * sin(theta)`` (magnitude),
    independent of whether the engine measures ``theta`` from the hanging-down
    or horizontal reference.
    """
    return abs(mass * GRAVITY_M_S2 * l_com * np.sin(theta))


# --- Drake ---------------------------------------------------------------


def _make_drake_engine() -> object:
    from src.engines.physics_engines.drake.python.drake_physics_engine import (
        DrakePhysicsEngine,
    )

    engine = DrakePhysicsEngine()
    engine.load_from_path(str(DRAKE_URDF))
    return engine


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake not installed")
def test_drake_mass_matrix_positive_definite() -> None:
    """Drake pendulum mass matrix is SPD and matches m*L^2 about the pivot."""
    engine = _make_drake_engine()
    engine.set_state(np.array([0.3]), np.array([0.0]))
    engine.forward()
    m_mat = np.atleast_2d(engine.compute_mass_matrix())

    assert m_mat.shape == (1, 1)
    eigvals = np.linalg.eigvalsh(0.5 * (m_mat + m_mat.T))
    assert np.all(eigvals > 0.0), f"M not positive-definite: eig={eigvals}"
    # Point-mass pendulum: M ≈ m*L^2 (plus the tiny stabilizing link inertia).
    assert m_mat[0, 0] == pytest.approx(DRAKE_INERTIA_ABOUT_PIVOT, abs=2e-3)


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake not installed")
@pytest.mark.parametrize("theta", TEST_ANGLES_RAD)
def test_drake_gravity_torque(theta: float) -> None:
    """Drake gravity generalized force magnitude ≈ m*g*L*sin(theta)."""
    engine = _make_drake_engine()
    engine.set_state(np.array([theta]), np.array([0.0]))
    engine.forward()
    g_force = np.atleast_1d(engine.compute_gravity_forces())
    expected = _gravity_torque_magnitude(DRAKE_MASS_KG, DRAKE_L_COM_M, theta)
    assert abs(float(g_force[0])) == pytest.approx(expected, abs=1e-2)


@pytest.mark.requires_drake
@pytest.mark.skipif(not is_engine_available("drake"), reason="pydrake not installed")
@pytest.mark.parametrize("theta", TEST_ANGLES_RAD)
@pytest.mark.parametrize("accel", [0.0, 1.5, -2.0])
def test_drake_newton_euler_roundtrip(theta: float, accel: float) -> None:
    """Drake reproduces tau = M @ a + bias to within 1e-10."""
    engine = _make_drake_engine()
    engine.set_state(np.array([theta]), np.array([0.4]))
    engine.forward()

    m_mat = np.atleast_2d(engine.compute_mass_matrix())
    bias = np.atleast_1d(engine.compute_bias_forces())
    a = np.array([accel], dtype=np.float64)

    tau_id = np.atleast_1d(engine.compute_inverse_dynamics(a))
    tau_expected = m_mat @ a + bias
    np.testing.assert_allclose(tau_id, tau_expected, atol=1e-10, rtol=0.0)


# --- OpenSim -------------------------------------------------------------


def _make_opensim_engine() -> object:
    from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
        OpenSimPhysicsEngine,
    )

    engine = OpenSimPhysicsEngine()
    engine.load_from_path(str(OPENSIM_OSIM))
    return engine


@pytest.mark.requires_opensim
@pytest.mark.skipif(not is_engine_available("opensim"), reason="opensim not installed")
def test_opensim_mass_matrix_positive_definite() -> None:
    """OpenSim pendulum mass matrix is SPD and matches Izz + m*L_com^2."""
    engine = _make_opensim_engine()
    engine.set_state(np.array([0.3]), np.array([0.0]))
    engine.forward()
    m_mat = np.atleast_2d(engine.compute_mass_matrix())

    assert m_mat.shape == (1, 1)
    eigvals = np.linalg.eigvalsh(0.5 * (m_mat + m_mat.T))
    assert np.all(eigvals > 0.0), f"M not positive-definite: eig={eigvals}"
    assert m_mat[0, 0] == pytest.approx(OPENSIM_INERTIA_ABOUT_PIVOT, abs=1e-3)


@pytest.mark.requires_opensim
@pytest.mark.skipif(not is_engine_available("opensim"), reason="opensim not installed")
@pytest.mark.parametrize("theta", TEST_ANGLES_RAD)
def test_opensim_gravity_torque(theta: float) -> None:
    """OpenSim gravity generalized force magnitude ≈ m*g*L_com*sin(theta)."""
    engine = _make_opensim_engine()
    engine.set_state(np.array([theta]), np.array([0.0]))
    engine.forward()
    g_force = np.atleast_1d(engine.compute_gravity_forces())
    expected = _gravity_torque_magnitude(OPENSIM_MASS_KG, OPENSIM_L_COM_M, theta)
    assert abs(float(g_force[0])) == pytest.approx(expected, abs=1e-2)


@pytest.mark.requires_opensim
@pytest.mark.skipif(not is_engine_available("opensim"), reason="opensim not installed")
@pytest.mark.parametrize("theta", TEST_ANGLES_RAD)
@pytest.mark.parametrize("accel", [0.0, 1.5, -2.0])
def test_opensim_newton_euler_roundtrip(theta: float, accel: float) -> None:
    """OpenSim reproduces tau = M @ a + bias to within 1e-10."""
    engine = _make_opensim_engine()
    engine.set_state(np.array([theta]), np.array([0.4]))
    engine.forward()

    m_mat = np.atleast_2d(engine.compute_mass_matrix())
    bias = np.atleast_1d(engine.compute_bias_forces())
    a = np.array([accel], dtype=np.float64)

    tau_id = np.atleast_1d(engine.compute_inverse_dynamics(a))
    tau_expected = m_mat @ a + bias
    np.testing.assert_allclose(tau_id, tau_expected, atol=1e-10, rtol=0.0)
