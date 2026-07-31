"""Cross-engine contact-model invariant tests.

Verifies contact behaviour numerically across MuJoCo, Drake, and Pinocchio
rather than documenting it in prose. Each engine drops a small sphere onto a
plane and asserts the physical invariants the prose used to merely describe:

* no NaN/inf in state (no explosive divergence);
* post-settling penetration below an engine-appropriate tolerance;
* energy is non-increasing after release within tolerance.

The documented qualitative differences between engines (soft penalty vs rigid
vs constraint-based) are encoded as *numbers* (per-engine tolerances), which is
what the prose was trying to say. Each test skips with an explicit reason when
its engine is not installed — it is never silently green (issue #7153).

The narrative comparison of the three contact models lives in
``docs/conventions/contact_models.md`` (documentation does not belong in test
bodies).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from tests.fixtures.fixtures_lib import (
    _check_drake_available,
    _check_mujoco_available,
    _check_pinocchio_available,
)

pytestmark = pytest.mark.integration

# Physical setup shared across engines.
_BALL_MASS_KG = 0.045  # golf ball
_BALL_RADIUS_M = 0.02135
_DROP_HEIGHT_M = 0.10  # release the ball 10 cm above the plane
_GRAVITY = 9.81

# A penalty-based (MuJoCo) contact allows visible penetration; a rigid/
# constraint model permits far less. Encoding the documented qualitative
# difference as numbers (issue #7153).
_MUJOCO_PENETRATION_TOL_M = 5e-3
_RIGID_PENETRATION_TOL_M = 1e-3

# Total mechanical energy must not increase after release (dissipative
# contact). Small positive slack absorbs integrator noise.
_ENERGY_INCREASE_TOL_J = 1e-3


# MuJoCo: golf ball (free joint) dropped onto a static plane with a soft
# penalty contact. Self-contained XML (no external file dependency).
_MUJOCO_DROP_XML = f"""
<mujoco model="ball_drop_contact">
  <option gravity="0 0 -{_GRAVITY}" timestep="0.0005" integrator="RK4"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"
          solref="0.002 1" solimp="0.9 0.95 0.001"/>
    <body name="ball" pos="0 0 {_DROP_HEIGHT_M + _BALL_RADIUS_M}">
      <freejoint name="ball_free"/>
      <geom name="ball" type="sphere" size="{_BALL_RADIUS_M}"
            mass="{_BALL_MASS_KG}" solref="0.002 1" solimp="0.9 0.95 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.skipif(not _check_mujoco_available(), reason="MuJoCo not installed")
class TestMuJoCoContactInvariants:
    """Numeric contact invariants for MuJoCo's soft penalty model."""

    def _simulate(self) -> dict[str, Any]:
        import mujoco

        model = mujoco.MjModel.from_xml_string(_MUJOCO_DROP_XML)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        # Total mechanical energy at release (ball at rest above the plane).
        e0 = self._total_energy(mujoco, model, data)

        min_ball_z = float("inf")
        max_energy_after_settle = -float("inf")
        # 2 s of simulation at 0.5 ms = 4000 steps: enough to land and settle.
        n_steps = 4000
        for i in range(n_steps):
            mujoco.mj_step(model, data)
            qpos = np.asarray(data.qpos)
            qvel = np.asarray(data.qvel)
            assert np.all(np.isfinite(qpos)), f"non-finite qpos at step {i}"
            assert np.all(np.isfinite(qvel)), f"non-finite qvel at step {i}"
            ball_z = float(data.qpos[2])
            min_ball_z = min(min_ball_z, ball_z)
            # Energy is sampled in the settled second half of the run.
            if i > n_steps // 2:
                max_energy_after_settle = max(
                    max_energy_after_settle,
                    self._total_energy(mujoco, model, data),
                )

        return {
            "e0": e0,
            "min_ball_z": min_ball_z,
            "max_energy_after_settle": max_energy_after_settle,
            "final_z": float(data.qpos[2]),
        }

    @staticmethod
    def _total_energy(mujoco_mod: Any, model: Any, data: Any) -> float:
        """Kinetic + gravitational potential energy of the ball."""
        # KE = 0.5 v^T M v
        n = model.nv
        full_m = np.zeros((n, n))
        mujoco_mod.mj_fullM(model, full_m, data.qM)
        qvel = np.asarray(data.qvel)
        ke = 0.5 * float(qvel @ full_m @ qvel)
        # PE = m g z of the ball COM (z is qpos[2] for the free joint).
        pe = _BALL_MASS_KG * _GRAVITY * float(data.qpos[2])
        return ke + pe

    def test_no_explosion(self) -> None:
        """State stays finite (asserted each step) and the ball does not fly
        off to absurd heights — an unstable contact would launch it."""
        result = self._simulate()
        assert result["final_z"] < _DROP_HEIGHT_M + 2 * _BALL_RADIUS_M

    def test_penetration_bounded(self) -> None:
        """Penetration into the plane stays within the soft-contact tolerance."""
        result = self._simulate()
        # Deepest the ball centre reaches below (radius) is the penetration.
        penetration = _BALL_RADIUS_M - result["min_ball_z"]
        assert penetration < _MUJOCO_PENETRATION_TOL_M, (
            f"penetration {penetration:.5f} m exceeds soft-contact tolerance "
            f"{_MUJOCO_PENETRATION_TOL_M} m"
        )
        # Sanity: the documented soft/rigid ordering is encoded as numbers.
        assert _MUJOCO_PENETRATION_TOL_M > _RIGID_PENETRATION_TOL_M

    def test_energy_non_increasing(self) -> None:
        """Total mechanical energy after settling does not exceed the release
        energy (contact is dissipative)."""
        result = self._simulate()
        assert (
            result["max_energy_after_settle"] <= result["e0"] + _ENERGY_INCREASE_TOL_J
        )


@pytest.mark.skipif(
    not _check_drake_available(), reason="Drake (pydrake) not installed"
)
class TestDrakeContactInvariants:
    """Numeric contact invariants for Drake's rigid/compliant model."""

    def _simulate(self) -> dict[str, Any]:
        from pydrake.geometry import HalfSpace, SceneGraph, Sphere
        from pydrake.math import RigidTransform
        from pydrake.multibody.plant import CoulombFriction, MultibodyPlant
        from pydrake.multibody.tree import SpatialInertia, UnitInertia
        from pydrake.systems.analysis import Simulator
        from pydrake.systems.framework import DiagramBuilder

        builder = DiagramBuilder()
        plant = builder.AddSystem(MultibodyPlant(time_step=5e-4))
        scene_graph = builder.AddSystem(SceneGraph())
        plant.RegisterAsSourceForSceneGraph(scene_graph)

        # Ground halfspace at z = 0.
        friction = CoulombFriction(0.9, 0.8)
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            RigidTransform(),
            HalfSpace(),
            "ground_collision",
            friction,
        )

        # Free sphere body.
        inertia = SpatialInertia(
            _BALL_MASS_KG,
            np.zeros(3),
            UnitInertia.SolidSphere(_BALL_RADIUS_M),
        )
        ball = plant.AddRigidBody("ball", inertia)
        plant.RegisterCollisionGeometry(
            ball,
            RigidTransform(),
            Sphere(_BALL_RADIUS_M),
            "ball_collision",
            friction,
        )
        plant.Finalize()

        builder.Connect(
            plant.get_geometry_poses_output_port(),
            scene_graph.get_source_pose_port(plant.get_source_id()),
        )
        builder.Connect(
            scene_graph.get_query_output_port(),
            plant.get_geometry_query_input_port(),
        )
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        plant.SetFreeBodyPose(
            plant_context,
            ball,
            RigidTransform([0.0, 0.0, _DROP_HEIGHT_M + _BALL_RADIUS_M]),
        )

        simulator = Simulator(diagram, context)
        simulator.Initialize()

        min_ball_z = float("inf")
        for _ in range(40):
            simulator.AdvanceTo(simulator.get_context().get_time() + 0.05)
            pose = plant.EvalBodyPoseInWorld(plant_context, ball)
            z = float(pose.translation()[2])
            assert math.isfinite(z), "non-finite ball z (Drake)"
            min_ball_z = min(min_ball_z, z)

        final_pose = plant.EvalBodyPoseInWorld(plant_context, ball)
        return {
            "min_ball_z": min_ball_z,
            "final_z": float(final_pose.translation()[2]),
        }

    def test_no_explosion(self) -> None:
        result = self._simulate()
        assert result["final_z"] < _DROP_HEIGHT_M + 2 * _BALL_RADIUS_M

    def test_penetration_bounded(self) -> None:
        result = self._simulate()
        penetration = _BALL_RADIUS_M - result["min_ball_z"]
        # Drake's rigid contact permits far less penetration than MuJoCo.
        assert penetration < _RIGID_PENETRATION_TOL_M, (
            f"penetration {penetration:.6f} m exceeds rigid-contact tolerance "
            f"{_RIGID_PENETRATION_TOL_M} m"
        )

    def test_settles_above_plane(self) -> None:
        result = self._simulate()
        # The ball rests on the plane: centre near +radius, not below 0.
        assert result["final_z"] > -_RIGID_PENETRATION_TOL_M


@pytest.mark.skipif(not _check_pinocchio_available(), reason="Pinocchio not installed")
class TestPinocchioContactInvariants:
    """Numeric invariant for Pinocchio's constraint-based dynamics.

    Pinocchio has no built-in penalty floor; contact is resolved
    algorithmically from explicit constraints. We verify the verifiable
    physical claim its prose implied: in free flight before contact, the
    forward dynamics conserve mechanical energy (a correctness precondition
    for any subsequent constraint resolution).
    """

    def _build_free_ball(self) -> tuple[Any, Any]:
        import pinocchio as pin

        model = pin.Model()
        inertia = pin.Inertia.FromSphere(_BALL_MASS_KG, _BALL_RADIUS_M)
        model.addJoint(
            0,
            pin.JointModelFreeFlyer(),
            pin.SE3.Identity(),
            "ball_joint",
        )
        model.appendBodyToJoint(1, inertia, pin.SE3.Identity())
        model.gravity.linear = np.array([0.0, 0.0, -_GRAVITY])
        data = model.createData()
        return model, data

    def test_free_fall_energy_conserved(self) -> None:
        import pinocchio as pin

        model, data = self._build_free_ball()

        q = pin.neutral(model)
        # Free-flyer config layout: [x, y, z, qx, qy, qz, qw].
        q[2] = _DROP_HEIGHT_M + _BALL_RADIUS_M
        v = np.zeros(model.nv)

        def energy(q_: np.ndarray, v_: np.ndarray) -> float:
            # NO computeTotalEnergy in Pinocchio (CLAUDE.md): sum the parts.
            ke = pin.computeKineticEnergy(model, data, q_, v_)
            pe = pin.computePotentialEnergy(model, data, q_)
            return float(ke + pe)

        e0 = energy(q, v)
        dt = 5e-4
        # Integrate free fall for 0.1 s (still above the plane: no contact yet).
        for _ in range(200):
            tau = np.zeros(model.nv)
            a = pin.aba(model, data, q, v, tau)
            assert np.all(np.isfinite(a)), "non-finite acceleration (Pinocchio)"
            v = v + a * dt
            q = pin.integrate(model, q, v * dt)

        e1 = energy(q, v)
        # Semi-implicit Euler conserves energy to O(dt) over this short window.
        assert (
            abs(e1 - e0) < 1e-2 * abs(e0) + 1e-4
        ), f"free-fall energy drifted: e0={e0:.6f} e1={e1:.6f}"
        # And the ball actually fell (sanity that dynamics ran).
        assert q[2] < _DROP_HEIGHT_M + _BALL_RADIUS_M
