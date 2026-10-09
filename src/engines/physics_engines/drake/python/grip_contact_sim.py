"""Drake point-contact grip on the same input as OpenSim (issue #11739, OSV-7).

A continuous-time ``MultibodyPlant`` with a ``SceneGraph``: a free hand body
carrying the spherical pads of both hands (collision spheres, pose and spatial
velocity overwritten from the prescribed swing at every derivative evaluation)
and a free club body (spec inertia, spec gravity) carrying the grip cylinder
as a collision ``Cylinder``.  Drake's point contact is the Hunt-Crossley law
``f_n = k x (1 + d x_dot)`` of the shared contact law; the pad stiffness and
dissipation are the matched values, the cylinder is made rigid by a very large
stiffness (Drake combines the two in series), and friction is Drake's
regularised Coulomb law with a stiction tolerance equal to the shared
transition speed.  Drake has no torsional friction, which the shared law and
MuJoCo condim 6 include; the difference is documented, not corrected.

The club state is the continuous state of a ``LeafSystem`` whose derivatives
are the club rows of ``EvalTimeDerivatives``, integrated by Drake's
``Simulator`` (error-controlled Runge-Kutta 3).  Per-pad forces come from the
plant's contact results, attributed to hands by collision geometry.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np

from src.engines.physics_engines.drake.python.grip_bushing import (
    SIDES,
    FreeClubPlantMixin,
    WeldClubKinematics,
    _module,
)
from src.shared.python.grip_contact import (
    ClubDynamics,
    CoordinateSpline,
    CoordinateSwing,
    GripInterface,
)
from src.shared.python.grip_contact.bushing_law import cross3
from src.shared.python.grip_contact.contact_run import ContactRun, slip_from_frames
from src.shared.python.grip_contact.pad_contact import PadContactModel, grip_axis
from src.shared.python.grip_contact.parity import GripKineticsSeries

ENGINE = "drake_contact"
DEFAULT_ACCURACY = 1.0e-5
DEFAULT_MAX_STEP_S = 2.0e-4
RIGID_STIFFNESS_N_M = 1.0e11
SCHEME = "implicit_euler"
HAND_MASS_KG = 100.0


class ClubInHands(FreeClubPlantMixin):
    """Drake plant with scene graph: free hand body (pads) and free club."""

    def __init__(
        self,
        spec: dict[str, Any],
        interface: GripInterface,
        pads: PadContactModel,
    ) -> None:
        plant_mod = _module("pydrake.multibody.plant")
        tree = _module("pydrake.multibody.tree")
        math_mod = _module("pydrake.math")
        geometry = _module("pydrake.geometry")
        framework = _module("pydrake.systems.framework")
        self._math = math_mod
        self._mpl = _module("pydrake.multibody.math")
        self.interface, self.pads = interface, pads
        self.pad_count = pads.layout.pad_count
        club = ClubDynamics.from_spec(spec)
        builder = framework.DiagramBuilder()
        plant, scene_graph = plant_mod.AddMultibodyPlantSceneGraph(builder, 0.0)
        hand = plant.AddRigidBody(
            "hand", tree.SpatialInertia.SolidSphereWithMass(HAND_MASS_KG, 0.05)
        )
        i = club.inertia_com_kg_m2
        body = plant.AddRigidBody(
            "club",
            tree.SpatialInertia.MakeFromCentralInertia(
                club.mass_kg,
                np.asarray(club.com_m),
                tree.RotationalInertia(
                    i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2]
                ),
            ),
        )
        law = pads.law
        friction = _module("pydrake.multibody.plant").CoulombFriction(
            law.static_friction, law.dynamic_friction
        )

        def material(stiffness: float, dissipation: float) -> Any:
            props = geometry.ProximityProperties()
            geometry.AddContactMaterial(
                dissipation=dissipation,
                point_stiffness=stiffness,
                friction=friction,
                properties=props,
            )
            return props

        axis, axis_point = grip_axis(interface, pads)
        lo, hi = pads.cylinder.axial_range_m
        centre = axis_point + 0.5 * (lo + hi) * axis
        helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.eye(3)[1]
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        frame = np.column_stack([u, np.cross(axis, u), axis])  # z = grip axis
        plant.RegisterCollisionGeometry(
            body,
            math_mod.RigidTransform(math_mod.RotationMatrix(frame), centre),
            geometry.Cylinder(pads.layout.grip_radius_m, hi - lo),
            "grip",
            material(RIGID_STIFFNESS_N_M, 0.0),
        )
        self.pad_ids: dict[Any, tuple[str, int]] = {}
        for side in SIDES:
            offset = interface.frame(side).matrix()
            for k, pos in enumerate(pads.layout.positions_grip_frame(side)):
                local = offset[:3, :3] @ pos + offset[:3, 3]
                gid = plant.RegisterCollisionGeometry(
                    hand,
                    math_mod.RigidTransform(local),
                    geometry.Sphere(pads.layout.pad_radius_m),
                    f"pad_{side}{k}",
                    material(law.stiffness_n_m, law.dissipation_s_m),
                )
                self.pad_ids[gid] = (side, k)
        plant.mutable_gravity_field().set_gravity_vector(
            np.asarray(spec["gravity_m_s2"], float)
        )
        plant.set_contact_model(plant_mod.ContactModel.kPoint)
        plant.set_stiction_tolerance(law.transition_velocity_m_s)
        plant.Finalize()
        diagram = builder.Build()
        self.diagram = diagram
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant, self.hand, self.club = plant, hand, body
        self.context = plant.GetMyMutableContextFromRoot(self.diagram_context)
        self._offsets = {s: interface.frame(s).matrix() for s in SIDES}
        self._q0 = body.floating_positions_start()
        self._v0 = body.floating_velocities_start_in_v()

    def club_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World rotation and origin of the club body."""
        pose = self.plant.EvalBodyPoseInWorld(self.context, self.club)
        return np.array(pose.rotation().matrix()), np.array(pose.translation())

    def wrenches(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per hand: force on the club, moment about the grip origin, pad normals."""
        rot, origin = self.club_pose()
        grip_origin = {s: origin + rot @ self._offsets[s][:3, 3] for s in SIDES}
        out = {s: [np.zeros(3), np.zeros(3), np.zeros(self.pad_count)] for s in SIDES}
        results = self.plant.get_contact_results_output_port().Eval(self.context)
        club_index = self.club.index()
        for n in range(results.num_point_pair_contacts()):
            info = results.point_pair_contact_info(n)
            pair = info.point_pair()
            hit = self.pad_ids.get(pair.id_A) or self.pad_ids.get(pair.id_B)
            if hit is None:
                continue
            side, k = hit
            force = np.array(info.contact_force())  # on body B
            if info.bodyB_index() != club_index:
                force = -force
            out[side][0] += force
            out[side][1] += cross3(
                np.array(info.contact_point()) - grip_origin[side], force
            )
            out[side][2][k] = float(np.linalg.norm(force))  # total, normal + friction
        return {s: (v[0], v[1], v[2]) for s, v in out.items()}

    def frame_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World pose of each hand grip frame and each club grip frame."""
        hand_pose = self.plant.EvalBodyPoseInWorld(self.context, self.hand)
        rh, ph = (
            np.array(hand_pose.rotation().matrix()),
            np.array(hand_pose.translation()),
        )
        rc, pc = self.club_pose()
        out = {}
        for s in SIDES:
            off = self._offsets[s]
            out[f"hand_{s}"] = (rh @ off[:3, :3], ph + rh @ off[:3, 3])
            out[f"club_{s}"] = (rc @ off[:3, :3], pc + rc @ off[:3, 3])
        return out


def _system(sim: ClubInHands, kin: WeldClubKinematics, source: Any) -> Any:
    leaf_system: Any = _module("pydrake.systems.framework").LeafSystem

    class _Club(leaf_system):
        def __init__(self) -> None:
            super().__init__()
            self.DeclareContinuousState(13)

        def DoCalcTimeDerivatives(self, context: Any, derivatives: Any) -> None:  # noqa: N802
            x = context.get_continuous_state_vector().CopyToVector()
            sim.set_hand(kin.state(*source(context.get_time())))
            sim.set_club(x)
            derivatives.get_mutable_vector().SetFromVector(sim.club_derivatives())

    return _Club()


def simulate_grip_contact(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    pads: PadContactModel,
    interface: GripInterface | None = None,
    accuracy: float = DEFAULT_ACCURACY,
    max_step_s: float = DEFAULT_MAX_STEP_S,
    t_end_s: float | None = None,
    hold: bool = False,
) -> ContactRun:
    """Integrate the free club held by pads over the prescribed swing.

    With ``hold`` the hands stay at the first sample's pose.  The club starts
    with the weld velocity of the first sample (see the MuJoCo contact run).
    """
    analysis = _module("pydrake.systems.analysis")
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    kin = WeldClubKinematics(spec_bytes, list(swing.names))
    sim = ClubInHands(spec, interface, pads)
    times = swing.time_s
    if t_end_s is not None:
        times = times[times <= t_end_s + 1e-12]
    q0 = np.asarray(swing.q[0], float)
    source: Callable[[float], tuple[np.ndarray, np.ndarray]]
    if hold:
        still = np.zeros_like(q0)

        def held(_t: float) -> tuple[np.ndarray, np.ndarray]:
            return q0, still

        source = held
    else:
        source = CoordinateSpline(swing.time_s, swing.q).evaluate
    weld0 = kin.state(*source(float(times[0])))
    sim.set_hand(weld0)
    sim.place_club_at(weld0)
    x0 = sim.club_state()
    x0[7:10] = weld0.velocity_m_s  # world linear velocity of the origin
    x0[10:13] = weld0.omega_rad_s
    simulator = analysis.Simulator(_system(sim, kin, source))
    analysis.ApplySimulatorConfig(
        analysis.SimulatorConfig(
            integration_scheme=SCHEME,
            max_step_size=max_step_s,
            accuracy=accuracy,
            use_error_control=True,
            start_time=float(times[0]),
        ),
        simulator,
    )
    context = simulator.get_mutable_context()
    context.SetTime(float(times[0]))
    context.SetContinuousState(x0)
    simulator.Initialize()
    wrench: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    hand_pose: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    club_pose: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    normal: dict[str, list] = {s: [] for s in SIDES}
    club_rot = []
    for t in times:
        if t > times[0]:
            simulator.AdvanceTo(float(t))
        sim.set_hand(kin.state(*source(float(t))))
        sim.set_club(context.get_continuous_state_vector().CopyToVector())
        poses = sim.frame_poses()
        for s, (force, moment, pad_force) in sim.wrenches().items():
            wrench[s][0].append(force)
            wrench[s][1].append(moment)
            normal[s].append(pad_force)
            hand_pose[s][0].append(poses[f"hand_{s}"][0])
            hand_pose[s][1].append(poses[f"hand_{s}"][1])
            club_pose[s][0].append(poses[f"club_{s}"][0])
            club_pose[s][1].append(poses[f"club_{s}"][1])
        club_rot.append(sim.club_pose()[0])
    arrays = {
        name: {s: (np.array(d[s][0]), np.array(d[s][1])) for s in SIDES}
        for name, d in (("w", wrench), ("h", hand_pose), ("c", club_pose))
    }
    series = GripKineticsSeries.from_frames(
        ENGINE,
        times,
        arrays["w"],
        arrays["h"],
        arrays["c"],
        np.array(club_rot),
        metadata={
            "integrator": "Drake Simulator implicit_euler, error controlled",
            "accuracy": accuracy,
            "max_step_s": max_step_s,
            "steps_taken": int(simulator.get_integrator().get_num_steps_taken()),
            "force_law": "Drake point contact (Hunt-Crossley), no torsional friction",
        },
    )
    axial, roll = slip_from_frames(arrays["h"], arrays["c"])
    return ContactRun(series, {s: np.array(normal[s]) for s in SIDES}, roll, axial)


__all__ = ["ClubInHands", "simulate_grip_contact"]
