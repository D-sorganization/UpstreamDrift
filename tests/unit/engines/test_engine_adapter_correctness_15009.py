"""Regression tests for engine adapter correctness (issue #15009)."""

from __future__ import annotations

import importlib.util
from unittest.mock import MagicMock

import numpy as np
import pytest

_MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None

from src.engines.physics_engines.opensim.python.muscle_analysis import (
    accumulate_muscle_generalized_forces,
)

pytestmark = pytest.mark.unit


def test_accumulate_muscle_generalized_forces_uses_coordinate_names() -> None:
    """Moment-arm torque must follow coordinate name → speed index, not dict order."""
    moment_arms = {
        "m1": {"hip_flexion": 0.1, "knee_angle": 0.5},
    }
    coord_to_speed = {"knee_angle": 0, "hip_flexion": 1}
    active, passive = accumulate_muscle_generalized_forces(
        moment_arms,
        active_forces={"m1": 100.0},
        passive_forces={"m1": 10.0},
        coord_to_speed=coord_to_speed,
        n_u=2,
    )
    np.testing.assert_allclose(active, [50.0, 10.0])
    np.testing.assert_allclose(passive, [5.0, 1.0])


def test_iaa_gravity_and_velocity_sum_to_minus_mass_inverse_bias() -> None:
    """OpenSim IAA gravity uses the same sign convention as drift (-M^-1 bias)."""
    from src.engines.physics_engines.opensim.python import (
        opensim_physics_engine as osim_mod,
    )

    engine = osim_mod.OpenSimPhysicsEngine()
    engine._model = MagicMock()
    engine._model.getNumSpeeds.return_value = 2
    engine._state = MagicMock()

    m = np.array([[2.0, 0.0], [0.0, 3.0]])
    gravity = np.array([6.0, -3.0])
    bias = np.array([10.0, 4.0])
    engine.compute_mass_matrix = MagicMock(return_value=m)
    engine.compute_gravity_forces = MagicMock(return_value=gravity)
    engine.compute_bias_forces = MagicMock(return_value=bias)

    analyzer = MagicMock()
    analyzer.get_muscle_forces.return_value = {}
    analyzer.get_passive_muscle_forces.return_value = {}
    analyzer.get_moment_arms.return_value = {}
    analyzer.coordinate_speed_indices.return_value = {}
    engine.get_muscle_analyzer = MagicMock(return_value=analyzer)

    result = engine.compute_iaa_decomposition()
    expected_drift = -np.linalg.solve(m, bias)
    reconstructed = result["gravity"] + result["velocity"]
    np.testing.assert_allclose(reconstructed, expected_drift, atol=1e-12)
    np.testing.assert_allclose(result["total"][:2], expected_drift, atol=1e-12)


def test_iaa_total_matches_forward_dynamics_with_muscle_torque() -> None:
    """IAA components must sum to M^-1 (tau_muscle - bias)."""
    from src.engines.physics_engines.opensim.python import (
        opensim_physics_engine as osim_mod,
    )

    engine = osim_mod.OpenSimPhysicsEngine()
    engine._model = MagicMock()
    engine._model.getNumSpeeds.return_value = 2
    engine._state = MagicMock()

    m = np.eye(2)
    gravity = np.array([2.0, 0.0])
    bias = np.array([5.0, 1.0])
    engine.compute_mass_matrix = MagicMock(return_value=m)
    engine.compute_gravity_forces = MagicMock(return_value=gravity)
    engine.compute_bias_forces = MagicMock(return_value=bias)

    analyzer = MagicMock()
    analyzer.get_muscle_forces.return_value = {"m1": 3.0}
    analyzer.get_passive_muscle_forces.return_value = {"m1": 1.0}
    analyzer.get_moment_arms.return_value = {"m1": {"coord_a": 1.0}}
    analyzer.coordinate_speed_indices.return_value = {"coord_a": 0}
    engine.get_muscle_analyzer = MagicMock(return_value=analyzer)

    result = engine.compute_iaa_decomposition()
    tau_muscle = np.array([3.0, 0.0]) + np.array([1.0, 0.0])
    forward = np.linalg.solve(m, tau_muscle - bias)
    np.testing.assert_allclose(result["total"], forward, atol=1e-12)


WELD_TWO_SLIDE_XML = """
<mujoco model="weld_superposition">
  <option gravity="0 0 -9.80665"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <joint name="j1" type="slide" axis="0 0 1" damping="0.05"/>
      <geom type="sphere" size="0.05" mass="0.2"/>
      <site name="s1" pos="0 0 0" size="0.01"/>
    </body>
    <body name="follower" pos="0 0 1">
      <joint name="j2" type="slide" axis="0 0 1" damping="0.05"/>
      <geom type="box" size="0.05 0.05 0.1" mass="0.8"/>
      <site name="s2" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <equality>
    <weld site1="s1" site2="s2"/>
  </equality>
</mujoco>
"""


@pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="mujoco not installed")
def test_mujoco_superposition_holds_under_active_weld() -> None:
    """Under constraints, control must be a_full - a_drift, not unconstrained M^-1 tau."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    engine = MuJoCoPhysicsEngine()
    engine.load_from_string(WELD_TWO_SLIDE_XML)
    assert engine.model is not None
    q = np.zeros(engine.model.nq)
    v = np.array([0.1, -0.05])
    tau = np.array([1.0, 0.5])
    assert len(tau) == engine.model.nv
    engine.set_state(q, v)

    a_drift = engine.compute_drift_acceleration()
    a_control = engine.compute_control_acceleration(tau)
    a_full = engine._constrained_forward_acceleration(np.asarray(tau, dtype=float))
    np.testing.assert_allclose(a_drift + a_control, a_full, atol=1e-6)

    m = engine.compute_mass_matrix()
    naive = np.linalg.solve(m, tau)
    assert np.linalg.norm(naive - a_control) > 1e-4, (
        "fixture must break unconstrained M^-1 tau under weld"
    )


@pytest.mark.skipif(not _MUJOCO_AVAILABLE, reason="mujoco not installed")
def test_mujoco_superposition_unconstrained_matches_mass_inverse_tau() -> None:
    """Without constraints, a_control still matches M^-1 tau."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )

    xml = """
    <mujoco>
      <option gravity="0 0 -9.80665"/>
      <worldbody>
        <body pos="0 0 1">
          <joint name="h" type="hinge" axis="0 1 0"/>
          <geom type="capsule" size="0.02 0.3" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    engine = MuJoCoPhysicsEngine()
    engine.load_from_string(xml)
    q = np.array([0.2])
    v = np.array([0.4])
    tau = np.array([0.7])
    engine.set_state(q, v)

    a_control = engine.compute_control_acceleration(tau)
    m = engine.compute_mass_matrix()
    np.testing.assert_allclose(a_control, np.linalg.solve(m, tau), atol=1e-6)
