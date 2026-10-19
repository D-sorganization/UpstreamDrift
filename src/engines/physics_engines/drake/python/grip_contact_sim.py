"""Drake two-hand grip contact and topology contact adapter (#11739, #13307).

A continuous-time ``MultibodyPlant`` with a ``SceneGraph``: two free prescribed
hand bodies (``lead_hand`` for ``'L'``, ``trail_hand`` for ``'R'``), each driven
by its own :class:`GripInterface` hand frame (:func:`hand_frame_states` per
side) at every derivative evaluation, and a free club body (spec inertia, spec
gravity) carrying the grip cylinder.

Two contact modes share the per-side evaluation and recording path:

1. **Pad mode** (``pads`` provided, ``topology_model=None``): each hand body
   carries only that side's spherical collision pads in its local hand grip
   frame.  Drake's native point contact evaluates the Hunt-Crossley normal law
   ``f_n = k x (1 + d x_dot)`` and regularised Coulomb friction without
   torsional friction.
2. **Topology mode** (``topology_model`` provided, HRG-GS-G9 #13307): each hand
   body carries that side's anatomical :class:`ContactTopology` site frames.
   Forces and moments are evaluated with the shared G2/G6 law
   (:func:`topology_side_wrench`, including sphere-capsule, capsule-capsule and
   torsional patch friction) at the current hand and club frames and applied as
   an :class:`ExternallyAppliedSpatialForce` per side on the club, plus the
   equal and opposite reaction spatial force on the corresponding hand body.
   Using the shared law instead of native hydroelastic/point capsule contact
   gives exact force-law parity with the Pinocchio reference while keeping
   native capsule contact out of scope.  Per-pair traces are recorded with
   :class:`PairTraceRecorder`, and allocated hand-hand pairs are reported as
   ``NaN`` with a documented reason.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.engines.physics_engines.drake.python.grip_bushing import (
    SIDES,
    FreeClubPlantMixin,
    WeldClubKinematics,
    _module,
    continuous_state,
    start_simulation,
)
from src.shared.python.grip_contact import (
    BushingState,
    ClubDynamics,
    CoordinateSpline,
    CoordinateSwing,
    GripInterface,
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.grip_contact.bushing_law import cross3
from src.shared.python.grip_contact.contact_run import ContactRun, slip_from_frames
from src.shared.python.grip_contact.pad_contact import (
    GripCylinder,
    PadContactModel,
)
from src.shared.python.grip_contact.parity import GripKineticsSeries
from src.shared.python.grip_contact.topology_contact import (
    PairTraceRecorder,
    TopologyContactModel,
    TopologySideWrench,
    map_site_to_hand_frame,
    topology_side_wrench,
)

ENGINE = "drake_contact"
DEFAULT_ACCURACY = 1.0e-5
DEFAULT_MAX_STEP_S = 2.0e-4
RIGID_STIFFNESS_N_M = 1.0e11
SCHEME = "implicit_euler"
HAND_MASS_KG = 100.0

HAND_BODY_NAMES: dict[str, str] = {"L": "lead_hand", "R": "trail_hand"}
ROLE_TO_SIDE: dict[str, str] = {"lead": "L", "trail": "R"}


@dataclass(frozen=True)
class IntegratorTolerances:
    """Simulator accuracy and step cap for the Drake contact integration."""

    accuracy: float = DEFAULT_ACCURACY
    max_step_s: float = DEFAULT_MAX_STEP_S


@dataclass(frozen=True)
class HandSiteAttachment:
    """Pure description of a pad or topology site attached to one hand body."""

    side: str
    hand_body_name: str
    name: str
    primitive: str
    centre_m: tuple[float, float, float]
    radius_m: float
    endpoints_m: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None
    pad_index: int | None = None


def hand_body_name(side_or_role: str) -> str:
    """Return the Drake rigid-body name (``'lead_hand'`` or ``'trail_hand'``).

    Preconditions:
        ``side_or_role`` in ``('L', 'R', 'lead', 'trail')``.
    Postconditions:
        Returns ``'lead_hand'`` for lead/L and ``'trail_hand'`` for trail/R.
    """
    if not isinstance(side_or_role, str):
        raise TypeError(
            f"side_or_role must be a str, got {type(side_or_role).__name__}"
        )
    if side_or_role in ("L", "lead"):
        return HAND_BODY_NAMES["L"]
    if side_or_role in ("R", "trail"):
        return HAND_BODY_NAMES["R"]
    raise ValueError(
        f"side_or_role must be one of ('L', 'R', 'lead', 'trail'), got {side_or_role!r}"
    )


def hand_site_attachments(
    pads: PadContactModel | None = None,
    topology_model: TopologyContactModel | None = None,
) -> tuple[HandSiteAttachment, ...]:
    """Compute per-hand-body site or pad attachments without building a plant.

    Preconditions:
        At least one of ``pads`` or ``topology_model`` must be provided.
    Postconditions:
        Every returned attachment maps ``'L'``/``'lead'`` to ``'lead_hand'`` and
        ``'R'``/``'trail'`` to ``'trail_hand'``, with coordinates in that hand's
        grip frame.
    """
    if pads is None and topology_model is None:
        raise ValueError("either pads or topology_model must be provided")
    if topology_model is not None:
        if not isinstance(topology_model, TopologyContactModel):
            raise TypeError(
                f"topology_model must be TopologyContactModel, "
                f"got {type(topology_model).__name__}"
            )
        topo = topology_model.topology
        out: list[HandSiteAttachment] = []
        for site in topo.sites:
            side = ROLE_TO_SIDE[site.hand]
            mapped = map_site_to_hand_frame(topo, site, side)
            endpoints = mapped.endpoints_m if mapped.primitive == "capsule" else None
            out.append(
                HandSiteAttachment(
                    side=side,
                    hand_body_name=HAND_BODY_NAMES[side],
                    name=mapped.name,
                    primitive=mapped.primitive,
                    centre_m=mapped.centre_m,
                    radius_m=mapped.radius_m,
                    endpoints_m=endpoints,
                )
            )
        return tuple(out)

    if not isinstance(pads, PadContactModel):
        raise TypeError(f"pads must be PadContactModel, got {type(pads).__name__}")
    layout = pads.layout
    pad_out: list[HandSiteAttachment] = []
    for side in SIDES:
        body_name = HAND_BODY_NAMES[side]
        for k, pos in enumerate(layout.positions_grip_frame(side)):
            pad_out.append(
                HandSiteAttachment(
                    side=side,
                    hand_body_name=body_name,
                    name=f"pad_{side}{k}",
                    primitive="sphere",
                    centre_m=(float(pos[0]), float(pos[1]), float(pos[2])),
                    radius_m=layout.pad_radius_m,
                    pad_index=k,
                )
            )
    return tuple(pad_out)


def _grip_cylinder_pose(
    interface: GripInterface, cylinder: GripCylinder
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return ``(rotation_matrix, centre_m, radius_m, length_m)`` of the grip cylinder."""
    rot_r = np.asarray(interface.right.rotation, dtype=float)
    axis = rot_r[:, 0]
    axis_point = np.asarray(interface.right.position_m) + rot_r @ np.array(
        [0.0, 0.0, cylinder.radius_m]
    )
    lo, hi = cylinder.axial_range_m
    centre = axis_point + 0.5 * (lo + hi) * axis
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.eye(3)[1]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    frame = np.column_stack([u, np.cross(axis, u), axis])
    return frame, centre, cylinder.radius_m, hi - lo


class ClubInHands(FreeClubPlantMixin):
    """Drake plant with scene graph: two free hand bodies and a free club."""

    def __init__(
        self,
        spec: dict[str, Any],
        interface: GripInterface,
        pads: PadContactModel | None = None,
        topology_model: TopologyContactModel | None = None,
    ) -> None:
        if pads is None and topology_model is None:
            raise ValueError("either pads or topology_model must be provided")
        plant_mod = _module("pydrake.multibody.plant")
        tree = _module("pydrake.multibody.tree")
        math_mod = _module("pydrake.math")
        geometry = _module("pydrake.geometry")
        framework = _module("pydrake.systems.framework")
        self._plant_mod = plant_mod
        self._math = math_mod
        self._mpl = _module("pydrake.multibody.math")
        self._abstract_value = _module("pydrake.common.value").AbstractValue
        self.interface = interface
        self.pads = pads
        self.topology_model = topology_model
        self.pad_count = (
            pads.layout.pad_count
            if pads is not None
            else len(topology_model.physical_grip_pairs("L"))  # type: ignore[union-attr]
        )
        law = topology_model.law if topology_model is not None else pads.law  # type: ignore[union-attr]
        cylinder = (
            topology_model.cylinder if topology_model is not None else pads.cylinder  # type: ignore[union-attr]
        )

        club = ClubDynamics.from_spec(spec)
        builder = framework.DiagramBuilder()
        plant, scene_graph = plant_mod.AddMultibodyPlantSceneGraph(builder, 0.0)
        self.hands: dict[str, Any] = {
            s: plant.AddRigidBody(
                HAND_BODY_NAMES[s],
                tree.SpatialInertia.SolidSphereWithMass(HAND_MASS_KG, 0.05),
            )
            for s in SIDES
        }
        self.lead_hand = self.hands["L"]
        self.trail_hand = self.hands["R"]
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
        friction = plant_mod.CoulombFriction(law.static_friction, law.dynamic_friction)

        def material(stiffness: float, dissipation: float) -> Any:
            props = geometry.ProximityProperties()
            geometry.AddContactMaterial(
                dissipation=dissipation,
                point_stiffness=stiffness,
                friction=friction,
                properties=props,
            )
            return props

        frame, centre, radius_m, length_m = _grip_cylinder_pose(interface, cylinder)
        plant.RegisterCollisionGeometry(
            body,
            math_mod.RigidTransform(math_mod.RotationMatrix(frame), centre),
            geometry.Cylinder(radius_m, length_m),
            "grip",
            material(RIGID_STIFFNESS_N_M, 0.0),
        )
        self.pad_ids: dict[Any, tuple[str, int]] = {}
        self.site_frames: dict[str, Any] = {}
        self._register_attachments(plant, scene_graph, geometry, tree, material, law)
        plant.mutable_gravity_field().set_gravity_vector(
            np.asarray(spec["gravity_m_s2"], float)
        )
        plant.set_contact_model(plant_mod.ContactModel.kPoint)
        plant.set_stiction_tolerance(law.transition_velocity_m_s)
        plant.Finalize()
        diagram = builder.Build()
        self.diagram = diagram
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant = plant
        self.scene_graph = scene_graph
        self.club = body
        self.context = plant.GetMyMutableContextFromRoot(self.diagram_context)
        self._offsets = {s: interface.frame(s).matrix() for s in SIDES}
        self._q0 = body.floating_positions_start()
        self._v0 = body.floating_velocities_start_in_v()
        self.last_side_wrenches: dict[str, TopologySideWrench] = {}
        self.last_applied_spatial_forces: tuple[Any, ...] = ()

    def _register_attachments(
        self,
        plant: Any,
        scene_graph: Any,
        geometry: Any,
        tree: Any,
        material: Any,
        law: Any,
    ) -> None:
        """Attach site frames and pad collision spheres to their hand bodies."""
        attachments = hand_site_attachments(
            pads=self.pads, topology_model=self.topology_model
        )
        for att in attachments:
            hand_body = self.hands[att.side]
            tf = self._math.RigidTransform(np.asarray(att.centre_m, dtype=float))
            site_frame = plant.AddFrame(
                tree.FixedOffsetFrame(att.name, hand_body.body_frame(), tf)
            )
            self.site_frames[att.name] = site_frame
            if self.topology_model is None and att.pad_index is not None:
                gid = plant.RegisterCollisionGeometry(
                    hand_body,
                    tf,
                    geometry.Sphere(att.radius_m),
                    att.name,
                    material(law.stiffness_n_m, law.dissipation_s_m),
                )
                self.pad_ids[gid] = (att.side, att.pad_index)

        if self.pad_ids:
            pad_set = geometry.GeometrySet(list(self.pad_ids.keys()))
            decl = geometry.CollisionFilterDeclaration()
            decl.ExcludeWithin(pad_set)
            filter_mgr = scene_graph.collision_filter_manager()
            filter_mgr.Apply(decl)

    def set_hand(self, weld: RigidBodyState) -> None:
        """Place each hand body (``lead_hand``, ``trail_hand``) at its grip frame."""
        self.set_hand_frames(hand_frame_states(weld, self.interface))

    def set_hand_frames(self, hands: Mapping[str, BushingState]) -> None:
        """Place each hand body at the supplied per-side :class:`BushingState`."""
        for side in SIDES:
            h = hands[side]
            pose = self._math.RigidTransform(
                self._math.RotationMatrix(h.rotation), h.position_m
            )
            body = self.hands[side]
            self.plant.SetFreeBodyPose(self.context, body, pose)
            self.plant.SetFreeBodySpatialVelocity(
                self.context,
                body,
                self._mpl.SpatialVelocity(h.omega_rad_s, h.velocity_m_s),
            )

    def validate_hand_attachments(self) -> None:
        """Verify every pad geometry and site frame is attached to the right hand body.

        Raises:
            ValueError: if any pad or site is attached to the wrong hand body.
        """
        for side in SIDES:
            expected = HAND_BODY_NAMES[side]
            actual = self.hands[side].name()
            if actual != expected:
                raise ValueError(
                    f"hand slot {side!r} has wrong hand body {actual!r}, expected {expected!r}"
                )
        if self.hands["L"].index() == self.hands["R"].index():
            raise ValueError("lead_hand and trail_hand must be distinct bodies")

        for name, frame in self.site_frames.items():
            if name.startswith(("pad_L", "lead_")):
                expected_body = HAND_BODY_NAMES["L"]
            elif name.startswith(("pad_R", "trail_")):
                expected_body = HAND_BODY_NAMES["R"]
            else:
                raise ValueError(f"cannot determine hand side for frame {name!r}")
            actual_body = frame.body().name()
            if actual_body != expected_body:
                raise ValueError(
                    f"frame {name!r} attached to wrong hand body {actual_body!r}, "
                    f"expected {expected_body!r}"
                )

        inspector = self.scene_graph.model_inspector()
        for gid, (side, _) in self.pad_ids.items():
            frame_id = inspector.GetFrameId(gid)
            body = self.plant.GetBodyFromFrameId(frame_id)
            expected_body = HAND_BODY_NAMES[side]
            if body.name() != expected_body:
                raise ValueError(
                    f"pad geometry on side {side!r} attached to wrong hand body "
                    f"{body.name()!r}, expected {expected_body!r}"
                )

    def club_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World rotation and origin of the club body."""
        pose = self.plant.EvalBodyPoseInWorld(self.context, self.club)
        rot = pose.rotation()
        return np.array(rot.matrix()), np.array(pose.translation())

    def bushing_states(
        self,
    ) -> tuple[dict[str, BushingState], dict[str, BushingState]]:
        """World :class:`BushingState` of each hand body and each club grip frame."""
        c_rot, c_pos = self.club_pose()
        c_vel = self.plant.EvalBodySpatialVelocityInWorld(self.context, self.club)
        club_body = RigidBodyState(
            c_rot,
            c_pos,
            np.array(c_vel.translational()),
            np.array(c_vel.rotational()),
        )
        hands: dict[str, BushingState] = {}
        frames: dict[str, BushingState] = {}
        for s in SIDES:
            h_pose = self.plant.EvalBodyPoseInWorld(self.context, self.hands[s])
            h_rot = h_pose.rotation()
            h_vel = self.plant.EvalBodySpatialVelocityInWorld(
                self.context, self.hands[s]
            )
            hands[s] = BushingState(
                np.array(h_rot.matrix()),
                np.array(h_pose.translation()),
                np.array(h_vel.translational()),
                np.array(h_vel.rotational()),
            )
            frames[s] = club_body.frame(self._offsets[s])
        return hands, frames

    def apply_topology_forces(self) -> dict[str, TopologySideWrench]:
        """Evaluate :func:`topology_side_wrench` per side and fix spatial forces."""
        if self.topology_model is None:
            raise RuntimeError("topology_model is not configured")
        hands, frames = self.bushing_states()
        side_wrenches = {
            s: topology_side_wrench(self.topology_model, s, hands[s], frames[s])
            for s in SIDES
        }
        self.last_side_wrenches = side_wrenches
        spatial_forces = []
        for s in SIDES:
            sw = side_wrenches[s]
            f_club = self._plant_mod.ExternallyAppliedSpatialForce()
            f_club.body_index = self.club.index()
            f_club.p_BoBq_B = self._offsets[s][:3, 3]
            f_club.F_Bq_W = self._mpl.SpatialForce(tau=sw.moment_nm, f=sw.force_n)
            spatial_forces.append(f_club)

            f_hand = self._plant_mod.ExternallyAppliedSpatialForce()
            f_hand.body_index = self.hands[s].index()
            f_hand.p_BoBq_B = np.zeros(3)
            arm = frames[s].position_m - hands[s].position_m
            tau_hand = -sw.moment_nm - cross3(arm, sw.force_n)
            f_hand.F_Bq_W = self._mpl.SpatialForce(tau=tau_hand, f=-sw.force_n)
            spatial_forces.append(f_hand)

        self.last_applied_spatial_forces = tuple(spatial_forces)
        port = self.plant.get_applied_spatial_force_input_port()
        port.FixValue(self.context, self._abstract_value.Make(spatial_forces))
        return side_wrenches

    def club_derivatives(self) -> np.ndarray:
        """Time derivatives of the club's 13 floating states."""
        if self.topology_model is not None:
            self.apply_topology_forces()
        return super().club_derivatives()

    def _native_pad_wrenches(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate per-hand wrenches from Drake's native point-contact results."""
        rot, origin = self.club_pose()
        grip_origin = {s: origin + rot @ self._offsets[s][:3, 3] for s in SIDES}
        out = {
            s: [
                np.zeros(3),
                np.zeros(3),
                np.zeros(self.pad_count),
                np.zeros(self.pad_count),
            ]
            for s in SIDES
        }
        port = self.plant.get_contact_results_output_port()
        results = port.Eval(self.context)
        club_index = self.club.index()
        for n in range(results.num_point_pair_contacts()):
            info = results.point_pair_contact_info(n)
            pair = info.point_pair()
            hit = self.pad_ids.get(pair.id_A) or self.pad_ids.get(pair.id_B)
            if hit is None:
                continue
            side, k = hit
            force = np.array(info.contact_force())
            if info.bodyB_index() != club_index:
                force = -force
            out[side][0] += force
            out[side][1] += cross3(
                np.array(info.contact_point()) - grip_origin[side], force
            )
            n_raw = np.array(pair.nhat_BA_W)
            n_into_club = -n_raw if info.bodyB_index() == club_index else n_raw
            n_len = float(np.linalg.norm(n_into_club))
            if n_len > 1e-12:
                n_into_club = n_into_club / n_len
            f_norm = max(0.0, float(np.dot(force, n_into_club)))
            f_tang = float(np.linalg.norm(force - f_norm * n_into_club))
            out[side][2][k] += f_norm
            out[side][3][k] += f_tang
        return {s: (v[0], v[1], v[2], v[3]) for s, v in out.items()}

    def wrenches(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Per hand: force on club, moment about grip origin, normals, tangentials."""
        if self.topology_model is not None:
            side_wrenches = self.apply_topology_forces()
            return {
                s: (
                    sw.force_n,
                    sw.moment_nm,
                    sw.normal_force_n,
                    sw.tangential_force_n,
                )
                for s, sw in side_wrenches.items()
            }
        return self._native_pad_wrenches()

    def frame_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World pose of each hand grip frame and each club grip frame."""
        hands, frames = self.bushing_states()
        out = {}
        for s in SIDES:
            out[f"hand_{s}"] = (hands[s].rotation, hands[s].position_m)
            out[f"club_{s}"] = (frames[s].rotation, frames[s].position_m)
        return out


def _system(sim: ClubInHands, kin: WeldClubKinematics, source: Any) -> Any:
    leaf_system: Any = _module("pydrake.systems.framework").LeafSystem

    class _Club(leaf_system):
        def __init__(self) -> None:
            super().__init__()
            self.DeclareContinuousState(13)

        def DoCalcTimeDerivatives(self, context: Any, derivatives: Any) -> None:  # noqa: N802
            state_vec = context.get_continuous_state_vector()
            x = state_vec.CopyToVector()
            sim.set_hand(kin.state(*source(context.get_time())))
            sim.set_club(x)
            mut_vec = derivatives.get_mutable_vector()
            mut_vec.SetFromVector(sim.club_derivatives())

    return _Club()


def _resolve_tolerances(
    tolerances: IntegratorTolerances | float | None,
) -> IntegratorTolerances:
    if tolerances is None:
        return IntegratorTolerances()
    if isinstance(tolerances, IntegratorTolerances):
        return tolerances
    if isinstance(tolerances, (int, float)):
        return IntegratorTolerances(accuracy=float(tolerances))
    raise TypeError(
        f"tolerances must be IntegratorTolerances or float, "
        f"got {type(tolerances).__name__}"
    )


def _hand_source(
    swing: CoordinateSwing, hold: bool
) -> Callable[[float], tuple[np.ndarray, np.ndarray]]:
    """Return coordinate evaluation callable for static hold or spline replay."""
    if not hold:
        return CoordinateSpline(swing.time_s, swing.q).evaluate
    q0 = np.asarray(swing.q[0], float)
    still = np.zeros_like(q0)

    def held(_t: float) -> tuple[np.ndarray, np.ndarray]:
        return q0, still

    return held


def simulate_grip_contact(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    pads: PadContactModel | None = None,
    interface: GripInterface | None = None,
    tolerances: IntegratorTolerances | float | None = None,
    t_end_s: float | None = None,
    hold: bool = False,
    topology_model: TopologyContactModel | None = None,
) -> ContactRun:
    """Integrate the free club held by pads or topology over the prescribed swing.

    With ``hold`` the hands stay at the first sample's pose.  The club starts
    with the weld velocity of the first sample.
    """
    tol = _resolve_tolerances(tolerances)
    analysis = _module("pydrake.systems.analysis")
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    kin = WeldClubKinematics(spec_bytes, list(swing.names))
    sim = ClubInHands(spec, interface, pads=pads, topology_model=topology_model)
    times = swing.time_s
    if t_end_s is not None:
        times = times[times <= t_end_s + 1e-12]
    source = _hand_source(swing, hold)
    weld0 = kin.state(*source(float(times[0])))
    sim.set_hand(weld0)
    sim.place_club_at(weld0)
    x0 = sim.club_state()
    x0[7:10] = weld0.omega_rad_s
    x0[10:13] = weld0.velocity_m_s
    simulator = analysis.Simulator(_system(sim, kin, source))
    analysis.ApplySimulatorConfig(
        analysis.SimulatorConfig(
            integration_scheme=SCHEME,
            max_step_size=tol.max_step_s,
            accuracy=tol.accuracy,
            use_error_control=True,
            start_time=float(times[0]),
        ),
        simulator,
    )
    wrench, hand_pose, club_pose = start_simulation(simulator, float(times[0]), x0)
    normal: dict[str, list] = {s: [] for s in SIDES}
    tangential: dict[str, list] = {s: [] for s in SIDES}
    recorder = PairTraceRecorder(topology_model) if topology_model is not None else None
    club_rot = []
    for t in times:
        if t > times[0]:
            simulator.AdvanceTo(float(t))
        sim.set_hand(kin.state(*source(float(t))))
        sim.set_club(continuous_state(simulator))
        hands, frames = sim.bushing_states()
        for s, (force, moment, pad_normal, pad_tang) in sim.wrenches().items():
            wrench[s][0].append(force)
            wrench[s][1].append(moment)
            normal[s].append(pad_normal)
            tangential[s].append(pad_tang)
            hand_pose[s][0].append(hands[s].rotation)
            hand_pose[s][1].append(hands[s].position_m)
            club_pose[s][0].append(frames[s].rotation)
            club_pose[s][1].append(frames[s].position_m)
        if recorder is not None:
            recorder.record(sim.last_side_wrenches, hands, frames)
        club_rot.append(sim.club_pose()[0])
    arrays = {
        name: {s: (np.array(d[s][0]), np.array(d[s][1])) for s in SIDES}
        for name, d in (("w", wrench), ("h", hand_pose), ("c", club_pose))
    }
    force_law_meta = (
        "ContactTopology shared contact law, Drake ExternallyAppliedSpatialForce"
        if topology_model is not None
        else "Drake point contact (Hunt-Crossley), no torsional friction"
    )
    integrator = simulator.get_integrator()
    series = GripKineticsSeries.from_frames(
        ENGINE,
        times,
        arrays["w"],
        arrays["h"],
        arrays["c"],
        np.array(club_rot),
        metadata={
            "integrator": "Drake Simulator implicit_euler, error controlled",
            "accuracy": tol.accuracy,
            "max_step_s": tol.max_step_s,
            "steps_taken": int(integrator.get_num_steps_taken()),
            "force_law": force_law_meta,
        },
    )
    axial, roll = slip_from_frames(arrays["h"], arrays["c"])
    extra = recorder.result() if recorder is not None else {}
    return ContactRun(
        series,
        {s: np.array(normal[s]) for s in SIDES},
        roll,
        axial,
        tangential_force_n={s: np.array(tangential[s]) for s in SIDES},
        **extra,
    )


__all__ = [
    "HAND_BODY_NAMES",
    "ClubInHands",
    "HandSiteAttachment",
    "IntegratorTolerances",
    "hand_body_name",
    "hand_site_attachments",
    "simulate_grip_contact",
]
