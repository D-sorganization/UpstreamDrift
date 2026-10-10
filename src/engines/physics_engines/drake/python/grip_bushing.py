"""Drake bushing grip on the same input as OpenSim (issue #11739, OSV-7 phase 2).

Two Drake ``MultibodyPlant`` objects are used:

* the full-body weld URDF (``export_full_body_urdf``) as Drake's own forward
  kinematics: for the prescribed coordinates and speeds it gives the club
  pose and spatial velocity of the weld model, which carries both hand
  bushing frames (as in the OpenSim model, both on the left hand body);
* a dynamics plant with a free hand body (its pose and spatial velocity are
  overwritten from the kinematics at every derivative evaluation) and a free
  club body (spec mass, centre of mass and inertia, spec gravity) joined by
  two native ``LinearBushingRollPitchYaw`` force elements with the same
  stiffness and damping as the OpenSim ``BushingForce``.

The club state (quaternion, position, world spatial velocity) is the
continuous state of a small ``LeafSystem`` whose derivatives are the club
rows of the dynamics plant's ``EvalTimeDerivatives``; Drake's
``Simulator`` integrates it with error-controlled Runge-Kutta 3.

Convention difference (documented, not corrected).  OpenSim uses body-fixed
X-Y-Z angles of the club frame in the hand frame, translations in hand
axes, the force applied at the club frame origin and the moment mapped by
``N^T``.  Drake uses roll-pitch-yaw (space-fixed X-Y-Z, i.e. body-fixed
Z-Y-X) angles, expresses the translation in the "halfway" frame B and
applies the force at the midpoint of the two frame origins.  The two laws
agree to first order in the deflection; the differences are second order
(about ``theta / 2`` relative, under 1 % for the 0.84 degree peak).
"""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any

import numpy as np

from src.engines.physics_engines.drake.python.full_body_urdf import (
    export_full_body_urdf,
)
from src.shared.python.grip_contact import (
    ClubDynamics,
    CoordinateSpline,
    CoordinateSwing,
    GripInterface,
    RigidBodyState,
)
from src.shared.python.grip_contact.parity import BushingProbe, GripKineticsSeries

ENGINE = "drake"
DEFAULT_ACCURACY = 1.0e-7
DEFAULT_MAX_STEP_S = 2.5e-4
SIDES = ("L", "R")


def _module(name: str) -> Any:
    return import_module(name)


class WeldClubKinematics:
    """Drake forward kinematics of the weld model's club (spec club frame)."""

    def __init__(self, spec_bytes: bytes, names: list[str]) -> None:
        plant_mod = _module("pydrake.multibody.plant")
        parsing = _module("pydrake.multibody.parsing")
        spec = json.loads(spec_bytes)
        xml, meta = export_full_body_urdf(spec_bytes)
        plant = plant_mod.MultibodyPlant(0.0)
        inst = parsing.Parser(plant).AddModelsFromString(xml, "urdf")[0]
        links = meta["body_links"]
        world_link = plant.GetBodyByName(links["world"], inst)
        plant.WeldFrames(plant.world_frame(), world_link.body_frame())
        plant.Finalize()
        if plant.num_positions() != len(names) or plant.num_velocities() != len(names):
            raise ValueError("every URDF coordinate must be prescribed")
        self.plant = plant
        self.context = plant.CreateDefaultContext()
        self._q = np.array(
            [plant.GetJointByName(n, inst).position_start() for n in names]
        )
        self._v = np.array(
            [plant.GetJointByName(n, inst).velocity_start() for n in names]
        )
        self.body = plant.GetBodyByName(links[spec["closure"]["body_b"]], inst)

    def state(self, q: np.ndarray, qdot: np.ndarray) -> RigidBodyState:
        """Pose and spatial velocity of the spec club frame for ``(q, qdot)``."""
        full_q = np.zeros(self.plant.num_positions())
        full_v = np.zeros(self.plant.num_velocities())
        full_q[self._q], full_v[self._v] = q, qdot
        self.plant.SetPositions(self.context, full_q)
        self.plant.SetVelocities(self.context, full_v)
        pose = self.plant.EvalBodyPoseInWorld(self.context, self.body)
        vel = self.plant.EvalBodySpatialVelocityInWorld(self.context, self.body)
        return RigidBodyState(
            np.array(pose.rotation().matrix()),
            np.array(pose.translation()),
            np.array(vel.translational()),
            np.array(vel.rotational()),
        )


_FrameLog = dict[str, tuple[list, list]]


def continuous_state(simulator: Any) -> np.ndarray:
    """The simulator's current continuous state vector as a numpy array."""
    context = simulator.get_mutable_context()
    return np.asarray(context.get_continuous_state_vector().CopyToVector())


def start_simulation(
    simulator: Any, t0: float, x0: np.ndarray
) -> tuple[_FrameLog, _FrameLog, _FrameLog]:
    """Initialise ``simulator`` at ``t0`` with state ``x0``; return empty frame logs."""
    context = simulator.get_mutable_context()
    context.SetTime(t0)
    context.SetContinuousState(x0)
    simulator.Initialize()
    wrench: _FrameLog = {s: ([], []) for s in SIDES}
    hand_pose: _FrameLog = {s: ([], []) for s in SIDES}
    club_pose: _FrameLog = {s: ([], []) for s in SIDES}
    return wrench, hand_pose, club_pose


class FreeClubPlantMixin:
    """Shared hand and club state access for the free-club Drake plants.

    Hosts provide ``plant``, ``context``, ``hand``, ``club``, ``_math``,
    ``_mpl``, ``_q0`` and ``_v0``.
    """

    plant: Any
    context: Any
    hand: Any
    club: Any
    _math: Any
    _mpl: Any
    _q0: int
    _v0: int

    def set_hand(self, weld: RigidBodyState) -> None:
        """Place the hand body at the weld club pose with its spatial velocity."""
        pose = self._math.RigidTransform(
            self._math.RotationMatrix(weld.rotation), weld.position_m
        )
        self.plant.SetFreeBodyPose(self.context, self.hand, pose)
        self.plant.SetFreeBodySpatialVelocity(
            self.context,
            self.hand,
            self._mpl.SpatialVelocity(weld.omega_rad_s, weld.velocity_m_s),
        )

    def set_club(self, x: np.ndarray) -> None:
        """Set the club's 13 floating states (quaternion normalised)."""
        q = np.array(x[:7], float)
        q[:4] /= np.linalg.norm(q[:4])
        positions = self.plant.GetPositions(self.context)
        velocities = self.plant.GetVelocities(self.context)
        positions[self._q0 : self._q0 + 7] = q
        velocities[self._v0 : self._v0 + 6] = x[7:13]
        self.plant.SetPositions(self.context, positions)
        self.plant.SetVelocities(self.context, velocities)

    def club_state(self) -> np.ndarray:
        """The club's 13 floating states."""
        q = self.plant.GetPositions(self.context)[self._q0 : self._q0 + 7]
        v = self.plant.GetVelocities(self.context)[self._v0 : self._v0 + 6]
        return np.concatenate([q, v])

    def club_derivatives(self) -> np.ndarray:
        """Time derivatives of the club's 13 floating states."""
        xdot = self.plant.EvalTimeDerivatives(self.context).CopyToVector()
        nq = self.plant.num_positions()
        return np.concatenate(
            [
                xdot[self._q0 : self._q0 + 7],
                xdot[nq + self._v0 : nq + self._v0 + 6],
            ]
        )

    def place_club_at(self, weld: RigidBodyState) -> None:
        """Place the free club at the weld pose (velocities are left unchanged)."""
        pose = self._math.RigidTransform(
            self._math.RotationMatrix(weld.rotation), weld.position_m
        )
        self.plant.SetFreeBodyPose(self.context, self.club, pose)


class ClubOnBushings(FreeClubPlantMixin):
    """Dynamics plant: free hand body, free club, two native bushings."""

    def __init__(self, spec: dict[str, Any], interface: GripInterface) -> None:
        plant_mod = _module("pydrake.multibody.plant")
        tree = _module("pydrake.multibody.tree")
        math_mod = _module("pydrake.math")
        club = ClubDynamics.from_spec(spec)
        plant = plant_mod.MultibodyPlant(0.0)
        hand = plant.AddRigidBody(
            "hand", tree.SpatialInertia.SolidSphereWithMass(1.0, 0.05)
        )
        i = club.inertia_com_kg_m2
        inertia = tree.RotationalInertia(
            i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]
        )
        body = plant.AddRigidBody(
            "club",
            tree.SpatialInertia.MakeFromCentralInertia(
                club.mass_kg, np.asarray(club.com_m), inertia
            ),
        )
        b = interface.bushing
        self.bushings = {}
        self.frames: dict[str, tuple[Any, Any]] = {}
        for side in SIDES:
            offset = math_mod.RigidTransform(interface.frame(side).matrix())
            frame_a = plant.AddFrame(
                tree.FixedOffsetFrame(f"hand_{side}", hand.body_frame(), offset)
            )
            frame_c = plant.AddFrame(
                tree.FixedOffsetFrame(f"club_{side}", body.body_frame(), offset)
            )
            self.frames[side] = (frame_a, frame_c)
            self.bushings[side] = plant.AddForceElement(
                tree.LinearBushingRollPitchYaw(
                    frame_a,
                    frame_c,
                    np.asarray(b.rotational_stiffness_nm_rad),
                    np.asarray(b.rotational_damping_nms_rad),
                    np.asarray(b.translational_stiffness_n_m),
                    np.asarray(b.translational_damping_ns_m),
                )
            )
        plant.mutable_gravity_field().set_gravity_vector(
            np.asarray(spec["gravity_m_s2"], float)
        )
        plant.Finalize()
        self.plant, self.hand, self.club = plant, hand, body
        self.context = plant.CreateDefaultContext()
        self._math = math_mod
        self._mpl = _module("pydrake.multibody.math")
        self._q0 = body.floating_positions_start()
        self._v0 = body.floating_velocities_start_in_v()

    def wrench(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Native bushing force on the club and moment about the club frame origin."""
        frame_c = self.frames[side][1]
        rot = np.array(
            frame_c.CalcPoseInWorld(self.context).rotation().matrix(), dtype=float
        )
        spatial = self.bushings[side].CalcBushingSpatialForceOnFrameC(self.context)
        return rot @ np.array(spatial.translational()), rot @ np.array(
            spatial.rotational()
        )

    def frame_pose(self, side: str, which: int) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the hand (``which=0``) or club (``1``) bushing frame."""
        pose = self.frames[side][which].CalcPoseInWorld(self.context)
        return np.array(pose.rotation().matrix()), np.array(pose.translation())


def _system(sim: ClubOnBushings, kin: WeldClubKinematics, spline: Any) -> Any:
    leaf_system: Any = _module("pydrake.systems.framework").LeafSystem

    class _Club(leaf_system):
        def __init__(self) -> None:
            super().__init__()
            self.DeclareContinuousState(13)

        def DoCalcTimeDerivatives(self, context: Any, derivatives: Any) -> None:  # noqa: N802
            x = context.get_continuous_state_vector().CopyToVector()
            sim.set_hand(kin.state(*spline.evaluate(context.get_time())))
            sim.set_club(x)
            derivatives.get_mutable_vector().SetFromVector(sim.club_derivatives())

    return _Club()


def simulate_grip_bushing(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    interface: GripInterface | None = None,
    accuracy: float = DEFAULT_ACCURACY,
    max_step_s: float = DEFAULT_MAX_STEP_S,
    t_end_s: float | None = None,
) -> GripKineticsSeries:
    """Integrate the free club on two native bushings over the prescribed swing.

    The club starts at the weld pose of the first sample with zero velocity,
    as in the OpenSim reference; samples are taken at ``swing.time_s``.
    """
    analysis = _module("pydrake.systems.analysis")
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    kin = WeldClubKinematics(spec_bytes, list(swing.names))
    sim = ClubOnBushings(spec, interface)
    spline = CoordinateSpline(swing.time_s, swing.q)
    times = swing.time_s
    if t_end_s is not None:
        times = times[times <= t_end_s + 1e-12]
    weld0 = kin.state(*spline.evaluate(float(times[0])))
    sim.set_hand(weld0)
    sim.place_club_at(weld0)
    x0 = sim.club_state()
    x0[7:] = 0.0
    system = _system(sim, kin, spline)
    simulator = analysis.Simulator(system)
    config = analysis.SimulatorConfig(
        integration_scheme="runge_kutta3",
        max_step_size=max_step_s,
        accuracy=accuracy,
        use_error_control=True,
        start_time=float(times[0]),
    )
    analysis.ApplySimulatorConfig(config, simulator)
    wrench, hand_pose, club_pose = start_simulation(simulator, float(times[0]), x0)
    club_rot = []
    for t in times:
        if t > times[0]:
            simulator.AdvanceTo(float(t))
        sim.set_hand(kin.state(*spline.evaluate(float(t))))
        sim.set_club(continuous_state(simulator))
        for s in SIDES:
            force, moment = sim.wrench(s)
            wrench[s][0].append(force)
            wrench[s][1].append(moment)
            for store, which in ((hand_pose, 0), (club_pose, 1)):
                r, p = sim.frame_pose(s, which)
                store[s][0].append(r)
                store[s][1].append(p)
        club_rot.append(
            np.array(
                sim.plant.EvalBodyPoseInWorld(sim.context, sim.club).rotation().matrix()
            )
        )
    stats = simulator.get_integrator()
    as_arrays = {
        name: {s: (np.array(d[s][0]), np.array(d[s][1])) for s in SIDES}
        for name, d in (("w", wrench), ("h", hand_pose), ("c", club_pose))
    }
    return GripKineticsSeries.from_frames(
        ENGINE,
        times,
        as_arrays["w"],
        as_arrays["h"],
        as_arrays["c"],
        np.array(club_rot),
        metadata={
            "integrator": "Drake Simulator runge_kutta3, error controlled",
            "accuracy": accuracy,
            "max_step_s": max_step_s,
            "steps_taken": int(stats.get_num_steps_taken()),
            "force_law": "native LinearBushingRollPitchYaw",
            "kinematics": "Drake full-body weld URDF forward kinematics",
        },
    )


def probe_bushing_forces(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    translation_hand_m: np.ndarray,
    interface: GripInterface | None = None,
) -> BushingProbe:
    """Native bushing forces for the club displaced by ``translation_hand_m``.

    The engine total is the net force on the club from the plant's
    acceleration, ``m (a_com - g)``, so it passes through Drake's dynamics.
    """
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    kin = WeldClubKinematics(spec_bytes, list(swing.names))
    sim = ClubOnBushings(spec, interface)
    q0 = np.asarray(swing.q[0], float)
    weld0 = kin.state(q0, np.zeros_like(q0))
    rest = RigidBodyState(weld0.rotation, weld0.position_m, np.zeros(3), np.zeros(3))
    sim.set_hand(rest)
    hand_rot = weld0.rotation @ np.asarray(interface.left.rotation, float)
    shifted = weld0.position_m + hand_rot @ np.asarray(translation_hand_m, float)
    pose = sim._math.RigidTransform(  # noqa: SLF001
        sim._math.RotationMatrix(weld0.rotation),  # noqa: SLF001
        shifted,
    )
    sim.plant.SetFreeBodyPose(sim.context, sim.club, pose)
    forces = {s: sim.wrench(s)[0] for s in SIDES}
    club = ClubDynamics.from_spec(spec)
    xdot = sim.club_derivatives()
    com_world = weld0.rotation @ np.asarray(club.com_m) + shifted
    omega_dot, a_origin = xdot[7:10], xdot[10:13]
    a_com = a_origin + np.cross(omega_dot, com_world - shifted)
    gravity = np.asarray(spec["gravity_m_s2"], float)
    return BushingProbe(
        force_n=forces,
        hand_rotation=hand_rot,
        engine_total_force_n=club.mass_kg * (a_com - gravity),
    )


__all__ = [
    "ClubOnBushings",
    "WeldClubKinematics",
    "probe_bushing_forces",
    "simulate_grip_bushing",
]
