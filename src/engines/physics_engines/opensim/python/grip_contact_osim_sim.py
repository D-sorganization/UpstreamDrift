"""Driven-swing simulation of the OpenSim contact grip (issue #11739, OSV-7).

The same input and topology as the bushing runner
(:class:`~...grip_bushing_sim.BushingGripSimulator`): the 44 body coordinates
are prescribed from a matched trajectory and the club is a free body.  Here
the hand-club interface is the distributed contact of the exporter's
``contact`` grip model: spherical pads on the hands pressed against a closed
grip mesh on the club through one ``ElasticFoundationForce`` per pad.  The
per-pad record values give the per-pad force on the club; summed per hand they
give the per-hand wrench.  Sign convention (binding, as for the bushing): every
wrench is the loading exerted BY THE HAND ON THE CLUB, world axes; the torque
is the free torque at the hand grip point on the club.

The right hand is tied to the left hand body exactly as in the bushing runner
(the matched IK does not close the two-hand loop), so the right pads are
re-attached to the left hand body at the tied-frame offset.

OpenSim is imported lazily; callers must skip when it is unavailable.
"""

from __future__ import annotations

import functools
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.full_body_grip_contact import (
    CONTACT_FORCE_PREFIX,
    ContactGripConfig,
    capsule_sphere_count,
    pad_name,
    shift_topology_site_inward,
)
from src.engines.physics_engines.opensim.python.full_body_grip_topology import (
    CLUB_FREE_COORDINATES,
)
from src.engines.physics_engines.opensim.python.full_body_osim import (
    export_full_body_osim,
)
from src.engines.physics_engines.opensim.python.grip_bushing_sim import (
    CLUB_BODY,
    BushingGripSimulator,
    BushingRun,
    _inv,
    _osim,
    _transform_to_rt,
    _vec,
)
from src.shared.python.grip_contact import CoordinateSwing, GripInterface
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_run import ContactRun
from src.shared.python.grip_contact.contact_topology import (
    DEFAULT_TOPOLOGY_PRELOAD_M,
)
from src.shared.python.grip_contact.elastic_foundation import (
    ElasticFoundationParameters,
    elastic_foundation_parameters,
    winkler_patch_factor_m,
)
from src.shared.python.grip_contact.grip_mesh import capped_cylinder_mesh, write_obj
from src.shared.python.grip_contact.pad_contact import PadContactModel
from src.shared.python.grip_contact.pad_layout import PadLayout
from src.shared.python.grip_contact.pair_contact import PairWrench
from src.shared.python.grip_contact.parity import GripKineticsSeries
from src.shared.python.grip_contact.topology_contact import (
    PairTraceRecorder,
    TopologyContactModel,
    TopologySideWrench,
    map_site_to_hand_frame,
)
from src.shared.python.motion_matching.contact_law import ContactParameters

ENGINE = "opensim_contact"
CONTACT_INTEGRATOR = "CPodes"
CONTACT_ACCURACY = 1e-4
TOPOLOGY_CONTACT_ACCURACY = 1e-6
_SIDES = (("L", "left"), ("R", "right"))
CALIBRATION_HALF_LENGTH_M = 0.02
_GRAVITY_M_S2 = 9.80665


def single_pad_model(
    ef: ElasticFoundationParameters,
    mesh_path: Path,
    pad_mass_kg: float = 0.5,
    gravity_m_s2: float = _GRAVITY_M_S2,
) -> tuple[Any, Any, Any]:
    """One pad sphere on a slider above a fixed grip mesh: ``(model, force, coord)``.

    The mesh axis is the ground z axis through the origin; the pad slides
    along the ground y axis (the radial direction of the mesh vertex ring at
    azimuth 0), so the coordinate is the pad-centre distance from the axis and
    the penetration is ``grip_radius + pad_radius - coordinate``.  The force is
    the same ``ElasticFoundationForce`` the grip model uses.
    """
    osim = _osim()
    osim.Logger.setLevelString("Error")
    model = osim.Model()
    model.setGravity(osim.Vec3(0.0, -gravity_m_s2, 0.0))
    body = osim.Body("pad", pad_mass_kg, osim.Vec3(0), osim.Inertia(1e-3))
    model.addBody(body)
    quarter = osim.Vec3(0.0, 0.0, np.pi / 2)
    joint = osim.SliderJoint(
        "slide", model.getGround(), osim.Vec3(0), quarter, body, osim.Vec3(0), quarter
    )
    model.addJoint(joint)
    layout = ef.layout
    joint.updCoordinate().setDefaultValue(
        layout.grip_radius_m + layout.pad_radius_m - ef.preload_penetration_m
    )
    model.addContactGeometry(
        osim.ContactSphere(layout.pad_radius_m, osim.Vec3(0), body, "pad")
    )
    model.addContactGeometry(
        osim.ContactMesh(
            str(mesh_path), osim.Vec3(0), osim.Vec3(0), model.getGround(), "grip"
        )
    )
    force = osim.ElasticFoundationForce()
    force.setName("pad_force")
    force.addGeometry("pad")
    force.addGeometry("grip")
    force.setStiffness(ef.stiffness_n_m3)
    force.setDissipation(ef.dissipation_s_m)
    force.setStaticFriction(ef.static_friction)
    force.setDynamicFriction(ef.dynamic_friction)
    force.setViscousFriction(ef.viscous_friction)
    force.setTransitionVelocity(ef.transition_velocity_m_s)
    model.addForce(force)
    return model, force, joint.updCoordinate()


def write_calibration_mesh(ef: ElasticFoundationParameters, directory: Path) -> Path:
    """Short closed grip mesh (axis z, ring vertex 0 on +y: the slider ray)."""
    layout = ef.layout
    verts, faces = capped_cylinder_mesh(
        layout.grip_radius_m,
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        (-CALIBRATION_HALF_LENGTH_M, CALIBRATION_HALF_LENGTH_M),
    )
    path = directory / "calibration_mesh.obj"
    write_obj(path, verts, faces)
    return path


def pad_normal_force_n(
    ef: ElasticFoundationParameters, mesh_path: Path, penetration_m: float
) -> float:
    """Static normal force of one pad pressed ``penetration_m`` into the mesh."""
    model, force, coord = single_pad_model(ef, mesh_path)
    state = model.initSystem()
    layout = ef.layout
    coord.setValue(state, layout.grip_radius_m + layout.pad_radius_m - penetration_m)
    model.realizeDynamics(state)
    return float(force.getRecordValues(state).get(1))


def calibrated_foundation(
    ef: ElasticFoundationParameters, directory: Path
) -> ElasticFoundationParameters:
    """Scale the foundation stiffness so one pad carries ``force_per_pad_n`` at the preload.

    The Winkler estimate is within about 15 % of the real mesh contact; the
    force is exactly proportional to the stiffness, so one static evaluation of
    the real mesh contact at the preload fixes it (no integration, no fitting
    to a swing).  Raises ``RuntimeError`` if the pad does not touch the mesh.
    """
    mesh = write_calibration_mesh(ef, directory)
    measured = pad_normal_force_n(ef, mesh, ef.preload_penetration_m)
    if measured <= 0.0:
        raise RuntimeError("calibration pad does not touch the grip mesh")
    return ef.with_stiffness(ef.stiffness_n_m3 * ef.force_per_pad_n / measured)


@functools.lru_cache(maxsize=64)
def _calibrated_site_foundation_cached(
    radius_m: float,
    grip_radius_m: float,
    count: int,
    law: ContactParameters,
) -> ElasticFoundationParameters:
    """Calibrate foundation stiffness for a site radius so summed stiffness matches shared law.

    Because the Winkler foundation law is quadratic in penetration
    (``F(delta) = k_W * Phi * delta^2``), matching both the preload force
    ``f_target = k_law * delta_nom / count`` and the local tangent stiffness
    ``dF/ddelta(delta_0) = 2 * F(delta_0) / delta_0 = k_law / count`` requires
    ``delta_0 = 2 * delta_nom``.  Furthermore, the shared Stribeck-regularized
    friction curve ``[mu_d + (mu_s - mu_d)*exp(-(v/v_tr)^2) + mu_v*v]*tanh(v/v_tr)``
    with ``mu_s = 0.9, mu_d = 0.7`` is monotonically increasing without a static
    peak above ``mu_d`` (at ``v = v_tr``, ``mu_law(v_tr) = 0.589`` whereas Simbody's
    Hollars law peaks at ``mu_s``); mapping ``static_friction = law.dynamic_friction``
    aligns Simbody's Hollars curve with the shared tangential law across all slip
    regimes within 5 %.
    """
    delta_nom = DEFAULT_TOPOLOGY_PRELOAD_M
    f_target = (law.stiffness_n_m * delta_nom) / count
    delta = 2.0 * delta_nom
    factor = winkler_patch_factor_m(radius_m, grip_radius_m)
    k_winkler = f_target / (factor * delta * delta)
    layout = PadLayout(
        grip_radius_m=grip_radius_m,
        pad_radius_m=radius_m,
        preload_penetration_m=delta,
    )
    ef = ElasticFoundationParameters(
        stiffness_n_m3=k_winkler,
        dissipation_s_m=law.dissipation_s_m,
        static_friction=law.dynamic_friction,
        dynamic_friction=law.dynamic_friction,
        viscous_friction=law.viscous_friction,
        transition_velocity_m_s=law.transition_velocity_m_s,
        preload_penetration_m=delta,
        force_per_pad_n=f_target,
        pad_stiffness_n_m=law.stiffness_n_m / count,
        patch_factor_m=factor,
        layout=layout,
    )
    with tempfile.TemporaryDirectory(prefix="calib_site_") as tmp:
        calibrated = calibrated_foundation(ef, Path(tmp))
    return calibrated


def calibrated_site_foundation(
    topology_model: TopologyContactModel,
    radius_m: float,
    grip_radius_m: float,
    mesh_dir: Path | None = None,
    count: int = 1,
) -> ElasticFoundationParameters:
    """Calibrate foundation stiffness for a site radius so summed stiffness matches shared law.

    Capsule phalanx sites approximated by a chain of ``count`` spheres along the
    capsule medial axis divide the target stiffness by ``count`` so their sum matches.
    Delegates to the pure cached function :func:`_calibrated_site_foundation_cached`.
    """
    return _calibrated_site_foundation_cached(
        radius_m, grip_radius_m, count, topology_model.law
    )


@dataclass(frozen=True)
class OpenSimContactRun(BushingRun):
    """A :class:`BushingRun` plus the outputs only a distributed contact has.

    ``normal_force_n[side]`` is ``(n, pads)`` (pressure along the grip),
    ``roll_slip_rad`` / ``axial_slip_m`` the hand-to-club slip.
    """

    normal_force_n: dict[str, np.ndarray]
    roll_slip_rad: dict[str, np.ndarray]
    axial_slip_m: dict[str, np.ndarray]
    pair_normal_force_n: Mapping[str, np.ndarray] | None = None
    pair_tangential_force_n: Mapping[str, np.ndarray] | None = None
    channel_reasons: Mapping[str, str] | None = None
    hand_states: Mapping[str, list[Any]] | None = None
    club_states: Mapping[str, list[Any]] | None = None
    pair_contact_point_drift_m: Mapping[str, np.ndarray] | None = None

    @property
    def squeeze_n(self) -> dict[str, np.ndarray]:
        """Total pad normal force of each hand."""
        return {s: np.asarray(self.normal_force_n[s]).sum(axis=1) for s in ("L", "R")}

    def to_contact_run(self, metadata: dict[str, Any] | None = None) -> ContactRun:
        """The engine-agnostic :class:`ContactRun` (series plus contact outputs)."""
        series = GripKineticsSeries(
            engine=ENGINE,
            time_s=self.time_s,
            force_on_club_n=self.force_on_club_n,
            torque_on_club_nm=self.torque_on_club_nm,
            grip_point_m=self.grip_point_m,
            deflection_m=self.deflection_m,
            rotation_deflection_rad=self.rotation_deflection_rad,
            club_rotation=self.club_rotation,
            metadata=dict(metadata or {}),
        )
        return ContactRun(
            series,
            self.normal_force_n,
            self.roll_slip_rad,
            self.axial_slip_m,
            pair_normal_force_n=self.pair_normal_force_n,
            pair_tangential_force_n=self.pair_tangential_force_n,
            channel_reasons=self.channel_reasons,
            hand_states=self.hand_states,
            club_states=self.club_states,
            pair_contact_point_drift_m=self.pair_contact_point_drift_m,
        )


class ContactGripSimulator(BushingGripSimulator):
    """Prescribed-arm, free-club simulation of the contact grip model.

    Args:
        spec_bytes, names, time_s, q: as :class:`BushingGripSimulator`.
        pads: shared pad model (layout and law) the foundation is matched to.
        interface: grip interface (default: built from the spec).
        foundation: foundation parameters (default: matched to ``pads`` and
            calibrated by one static evaluation, :func:`calibrated_foundation`).
        mesh_dir: directory for the grip OBJ files (default: a temporary
            directory that lives as long as the simulator).
    """

    def __init__(
        self,
        spec_bytes: bytes,
        names: Sequence[str],
        time_s: np.ndarray,
        q: np.ndarray,
        pads: PadContactModel | TopologyContactModel | None = None,
        interface: GripInterface | None = None,
        mesh_dir: Path | None = None,
        topology_model: TopologyContactModel | None = None,
    ) -> None:
        if isinstance(pads, TopologyContactModel):
            topology_model = pads
            pads = None
        self.topology_model = topology_model
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        if mesh_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="grip_meshes_")
            mesh_dir = Path(self._tmp.name)
        interface = interface or GripInterface.from_spec(json.loads(spec_bytes))
        if topology_model is not None:
            site_foundations = self._calibrate_topology_foundations(
                topology_model, mesh_dir
            )
            self.contact_config = ContactGripConfig(
                pads=None,
                mesh_dir=mesh_dir,
                topology_model=topology_model,
                site_foundations=site_foundations,
            )
        else:
            if pads is None:
                raise ValueError("either pads or topology_model must be provided")
            foundation = calibrated_foundation(
                elastic_foundation_parameters(pads), mesh_dir
            )
            self.contact_config = ContactGripConfig(pads, mesh_dir, foundation)
        self.export_receipt: dict[str, Any] = {}
        self._frames: dict[tuple[str, str], Any] = {}
        self._pad_forces: dict[str, list[tuple[Any, int, int]]] = {}
        self._topo_forces: dict[
            str,
            list[tuple[str, list[tuple[Any, int, int, np.ndarray, float, float]]]],
        ] = {}
        super().__init__(spec_bytes, names, time_s, q, interface)
        self._index_records()

    def _calibrate_topology_foundations(
        self, topology_model: TopologyContactModel, mesh_dir: Path
    ) -> dict[tuple[float, int], ElasticFoundationParameters]:
        """Calibrate foundation parameters for unique (radius_m, count) pairs."""
        foundations: dict[tuple[float, int], ElasticFoundationParameters] = {}
        topo = topology_model.topology
        grip_r = topology_model.cylinder.radius_m
        for side, _ in _SIDES:
            for pair in topology_model.physical_grip_pairs(side):
                site = topo.site(pair.site_a)
                mapped = map_site_to_hand_frame(topo, site, side)
                if mapped.primitive == "sphere":
                    key = (mapped.radius_m, 1)
                elif mapped.primitive == "capsule":
                    count = capsule_sphere_count(mapped.half_length_m, mapped.radius_m)
                    key = (mapped.radius_m, count)
                else:
                    continue
                if key not in foundations:
                    foundations[key] = calibrated_site_foundation(
                        topology_model, key[0], grip_r, count=key[1]
                    )
        return foundations

    # ------------------------------------------------------------ build hooks
    def _export_xml(self, spec_bytes: bytes, interface: GripInterface | None) -> str:
        xml, receipt = export_full_body_osim(
            spec_bytes,
            grip_model="contact",
            grip_interface=interface,
            grip_contact=self.contact_config,
        )
        self.export_receipt = receipt
        return xml

    def _strip_forces(self) -> None:
        forces = self._model.updForceSet()
        for i in range(forces.getSize() - 1, -1, -1):
            if not forces.get(i).getName().startswith(CONTACT_FORCE_PREFIX):
                forces.remove(i)

    def _tie_topology_right_hand(self, tie: np.ndarray) -> None:
        assert self.topology_model is not None
        osim = self._osim
        top = self.topology_model
        topo = top.topology
        grip_r = top.cylinder.radius_m
        hand = self._model.getBodySet().get("LGrip")
        for pair in self.topology_model.physical_grip_pairs("R"):
            site = topo.site(pair.site_a)
            mapped = map_site_to_hand_frame(topo, site, "R")
            if mapped.primitive == "sphere":
                ef_site = self.contact_config.site_foundation(mapped.radius_m, 1)
                extra_pen = ef_site.preload_penetration_m - DEFAULT_TOPOLOGY_PRELOAD_M
                shifted = shift_topology_site_inward(
                    mapped.centre_m, "R", grip_r, extra_pen
                )
                geom = osim.ContactSphere.safeDownCast(
                    self._model.getContactGeometrySet().get(f"grip_pad_{pair.name}")
                )
                geom.connectSocket_frame(hand)
                at = tie[:3, :3] @ shifted + tie[:3, 3]
                geom.set_location(osim.Vec3(*(float(x) for x in at)))
            elif mapped.primitive == "capsule":
                p0, p1 = mapped.endpoints_m
                count = capsule_sphere_count(mapped.half_length_m, mapped.radius_m)
                ef_site = self.contact_config.site_foundation(mapped.radius_m, count)
                extra_pen = ef_site.preload_penetration_m - DEFAULT_TOPOLOGY_PRELOAD_M
                for k in range(count):
                    geom = osim.ContactSphere.safeDownCast(
                        self._model.getContactGeometrySet().get(
                            f"grip_pad_{pair.name}_{k}"
                        )
                    )
                    geom.connectSocket_frame(hand)
                    t = k / (count - 1)
                    c_k = np.asarray(p0) + t * (np.asarray(p1) - np.asarray(p0))
                    shifted = shift_topology_site_inward(c_k, "R", grip_r, extra_pen)
                    at = tie[:3, :3] @ shifted + tie[:3, 3]
                    geom.set_location(osim.Vec3(*(float(x) for x in at)))

    def _tie_right_hand_to_left(self, spec: dict[str, Any]) -> None:
        """Hand and club grip frames, and the right pads moved to the left hand."""
        osim = self._osim
        gi = self.interface or GripInterface.from_spec(spec)
        tie = self._tied_right_matrix(spec)
        left_hand = self._left_hand_matrix(spec, gi)
        self._frames[("L", "hand")] = self._add_offset_frame(
            "grip_hand_frame_left", "LGrip", left_hand
        )
        self._frames[("R", "hand")] = self._add_offset_frame(
            "grip_hand_frame_right", "LGrip", tie
        )
        for side, _ in _SIDES:
            self._frames[(side, "club")] = self._add_offset_frame(
                f"grip_club_frame_{side}", CLUB_BODY, gi.frame(side).matrix()
            )
        if self.topology_model is not None:
            self._tie_topology_right_hand(tie)
        else:
            layout = self.contact_config.layout
            hand = self._model.getBodySet().get("LGrip")
            for k, local in enumerate(layout.positions_grip_frame("R")):
                geom = osim.ContactSphere.safeDownCast(
                    self._model.getContactGeometrySet().get(
                        f"grip_pad_{pad_name('R', k)}"
                    )
                )
                geom.connectSocket_frame(hand)
                at = tie[:3, :3] @ local + tie[:3, 3]
                geom.set_location(osim.Vec3(*(float(x) for x in at)))
        self._model.finalizeConnections()

    def _left_hand_matrix(self, spec: dict[str, Any], gi: GripInterface) -> np.ndarray:
        """Left grip frame on the left hand body (which sits at the wrist frame)."""
        club = spec["closure"]["body_b"]
        wrist = next(j for j in spec["joints"] if j["child"] == club)
        return _inv(np.asarray(wrist["child_to_follower"], float)) @ (gi.left.matrix())

    def _frame(self, side_name: str, kind: str) -> Any:
        return self._frames[("L" if side_name == "left" else "R", kind)]

    def _index_topology_records(self) -> None:
        assert self.topology_model is not None
        osim = self._osim
        topo = self.topology_model.topology
        for side, _ in _SIDES:
            rows = []
            for pair in self.topology_model.physical_grip_pairs(side):
                site = topo.site(pair.site_a)
                mapped = map_site_to_hand_frame(topo, site, side)
                entries: list[tuple[str, np.ndarray, float, float]] = []
                if mapped.primitive == "sphere":
                    ef_site = self.contact_config.site_foundation(mapped.radius_m, 1)
                    entries.append(
                        (
                            f"{CONTACT_FORCE_PREFIX}_{pair.name}",
                            np.asarray(mapped.centre_m),
                            mapped.radius_m,
                            ef_site.preload_penetration_m,
                        )
                    )
                elif mapped.primitive == "capsule":
                    p0, p1 = mapped.endpoints_m
                    count = capsule_sphere_count(mapped.half_length_m, mapped.radius_m)
                    ef_site = self.contact_config.site_foundation(
                        mapped.radius_m, count
                    )
                    for k in range(count):
                        t = k / (count - 1)
                        c_k = np.asarray(p0) + t * (np.asarray(p1) - np.asarray(p0))
                        entries.append(
                            (
                                f"{CONTACT_FORCE_PREFIX}_{pair.name}_{k}",
                                c_k,
                                mapped.radius_m,
                                ef_site.preload_penetration_m,
                            )
                        )
                indexed_entries = []
                for name, pos, sphere_r, d0 in entries:
                    force = osim.ElasticFoundationForce.safeDownCast(
                        self._model.getForceSet().get(name)
                    )
                    labels = force.getRecordLabels()
                    names = [labels.get(i) for i in range(labels.size())]
                    i_f = names.index(f"{name}.{CLUB_BODY}.force.X")
                    i_t = names.index(f"{name}.{CLUB_BODY}.torque.X")
                    indexed_entries.append((force, i_f, i_t, pos, sphere_r, d0))
                rows.append((pair.name, indexed_entries))
            self._topo_forces[side] = rows

    def _index_records(self) -> None:
        """Locate the club-body force and torque entries of every pad record."""
        if self.topology_model is not None:
            self._index_topology_records()
            return
        osim = self._osim
        layout = self.contact_config.layout
        for side, _ in _SIDES:
            rows = []
            for k in range(layout.pad_count):
                name = f"{CONTACT_FORCE_PREFIX}_{pad_name(side, k)}"
                force = osim.ElasticFoundationForce.safeDownCast(
                    self._model.getForceSet().get(name)
                )
                labels = force.getRecordLabels()
                names = [labels.get(i) for i in range(labels.size())]
                rows.append(
                    (
                        force,
                        names.index(f"{name}.{CLUB_BODY}.force.X"),
                        names.index(f"{name}.{CLUB_BODY}.torque.X"),
                    )
                )
            self._pad_forces[side] = rows

    # ----------------------------------------------------------------- sample
    def _sample_topology_side(
        self,
        state: Any,
        side: str,
        p1: np.ndarray,
        r1: np.ndarray,
        p2: np.ndarray,
        r2: np.ndarray,
    ) -> tuple[np.ndarray, TopologySideWrench]:
        assert self.topology_model is not None
        top = self.topology_model
        axis = r2[:, 0]
        grip_r = top.cylinder.radius_m
        axis_offset = np.array([0.0, 0.0, -grip_r if side == "L" else grip_r])
        axis_point = p2 + r2 @ axis_offset
        total_f = np.zeros(3)
        total_t = np.zeros(3)
        pair_wrenches: dict[str, PairWrench] = {}
        normal_forces: list[float] = []
        tangential_forces: list[float] = []

        for pair_name, entries in self._topo_forces[side]:
            f_pair = np.zeros(3)
            t_pair = np.zeros(3)
            fn_pair = 0.0
            ft_vec = np.zeros(3)
            pen_max = 0.0
            last_u_rad = np.array([0.0, 0.0, 1.0])
            last_rel = np.zeros(3)
            for force, i_f, i_t, local_pos, sphere_r, d0 in entries:
                rec = force.getRecordValues(state)
                f = np.array([rec.get(i_f + j) for j in range(3)])
                t = np.array([rec.get(i_t + j) for j in range(3)])
                f_pair += f
                t_pair += t
                c_world = p1 + r1 @ local_pos
                rel = c_world - axis_point
                radial = rel - (axis @ rel) * axis
                norm_rad = float(np.linalg.norm(radial))
                u_rad = (
                    radial / norm_rad if norm_rad > 1e-12 else np.array([0.0, 0.0, 1.0])
                )
                last_u_rad = u_rad
                last_rel = rel
                fn_raw = max(0.0, -float(f @ u_rad))
                f_tang_k = f - (-fn_raw) * u_rad
                pen_phys = max(0.0, sphere_r + grip_r - norm_rad)
                pen_k = max(
                    0.0,
                    sphere_r + grip_r - norm_rad + (d0 - DEFAULT_TOPOLOGY_PRELOAD_M),
                )
                if pen_k > 1e-12 and d0 > 0.0:
                    ratio_hc_w = max(0.0, 1.0 - ((pen_k - d0) / pen_k) ** 2)
                    fn_k = fn_raw * ratio_hc_w
                else:
                    fn_k = fn_raw
                fn_pair += fn_k
                ft_vec += f_tang_k
                pen_max = max(pen_max, pen_phys)

            ft_mag = float(np.linalg.norm(ft_vec))
            total_f += f_pair
            total_t += t_pair
            normal_forces.append(fn_pair)
            tangential_forces.append(ft_mag)
            contact_pt = axis_point + (axis @ last_rel) * axis + last_u_rad * grip_r
            pw = PairWrench(
                force_a_n=-f_pair,
                moment_a_nm=np.zeros(3),
                force_b_n=f_pair,
                moment_b_nm=t_pair,
                contact_point_m=contact_pt,
                normal=-last_u_rad,
                penetration_m=pen_max,
                point_a_m=p1,
                point_b_m=p2,
                normal_force_mag_n=fn_pair,
                tangential_force_mag_n=ft_mag,
            )
            pair_wrenches[pair_name] = pw

        total = np.concatenate([total_f, total_t])
        side_wrench = TopologySideWrench(
            force_n=total_f,
            moment_nm=total_t,
            pair_wrenches=pair_wrenches,
            normal_force_n=np.array(normal_forces),
            tangential_force_n=np.array(tangential_forces),
            total_normal_force_n=float(sum(normal_forces)),
        )
        return total, side_wrench

    def _side_sample(self, state: Any, name: str) -> dict[str, Any]:
        side = "L" if name == "left" else "R"
        r1, p1 = _transform_to_rt(self._frame(name, "hand").getTransformInGround(state))
        r2, p2 = _transform_to_rt(self._frame(name, "club").getTransformInGround(state))
        rel_rot = r1.T @ r2
        roll = float(
            np.arctan2(rel_rot[2, 1] - rel_rot[1, 2], rel_rot[1, 1] + rel_rot[2, 2])
        )
        angle = float(np.linalg.norm(Rotation.from_matrix(rel_rot).as_rotvec()))
        delta = r1.T @ (p2 - p1)

        osim = self._osim
        hand_body = self._model.getBodySet().get("LGrip")
        club_body = self._model.getBodySet().get(CLUB_BODY)
        omega_h = _vec(hand_body.getAngularVelocityInGround(state))
        r_hb, p_hb = _transform_to_rt(hand_body.getTransformInGround(state))
        st_h = osim.Vec3(*(float(x) for x in r_hb.T @ (p1 - p_hb)))
        v_h = _vec(hand_body.findStationVelocityInGround(state, st_h))
        hand_bstate = BushingState(r1, p1, v_h, omega_h)

        omega_c = _vec(club_body.getAngularVelocityInGround(state))
        r_cb, p_cb = _transform_to_rt(club_body.getTransformInGround(state))
        st_c = osim.Vec3(*(float(x) for x in r_cb.T @ (p2 - p_cb)))
        v_c = _vec(club_body.findStationVelocityInGround(state, st_c))
        club_bstate = BushingState(r2, p2, v_c, omega_c)

        if self.topology_model is not None:
            total, side_wrench = self._sample_topology_side(state, side, p1, r1, p2, r2)
            return {
                "record": total,
                "rot1": r1,
                "point": p2,
                "delta": delta,
                "angle": angle,
                "normal": side_wrench.normal_force_n,
                "roll": roll,
                "side_wrench": side_wrench,
                "hand_bstate": hand_bstate,
                "club_bstate": club_bstate,
            }

        layout = self.contact_config.layout
        axis = r2[:, 0]
        axis_point = p2 + r2 @ layout.axis_offset_grip_frame(side)
        centres = p1 + layout.positions_grip_frame(side) @ r1.T
        total = np.zeros(6)
        normal = np.zeros(len(centres))
        for k, (force, i_f, i_t) in enumerate(self._pad_forces[side]):
            rec = force.getRecordValues(state)
            f = np.array([rec.get(i_f + j) for j in range(3)])
            total[:3] += f
            total[3:] += np.array([rec.get(i_t + j) for j in range(3)])
            rel = centres[k] - axis_point
            radial = rel - (axis @ rel) * axis
            normal[k] = -float(f @ radial) / float(np.linalg.norm(radial))
        return {
            "record": total,
            "rot1": r1,
            "point": p2,
            "delta": delta,
            "angle": angle,
            "normal": normal,
            "roll": roll,
            "hand_bstate": hand_bstate,
            "club_bstate": club_bstate,
        }

    def _init_club_pose(self) -> None:
        """Pose as the bushing runner, and the club moving with the hand."""
        super()._init_club_pose()
        osim, model, state = self._osim, self._model, self._state
        state.setTime(float(self.time_s[0]))
        model.realizePosition(state)
        model.realizeVelocity(state)
        hand = model.getBodySet().get("LGrip")
        _, pos = self._weld_pose0
        r_h, p_h = _transform_to_rt(hand.getTransformInGround(state))
        station = osim.Vec3(*(float(x) for x in r_h.T @ (pos - p_h)))
        omega = _vec(hand.getAngularVelocityInGround(state))
        v_origin = _vec(hand.findStationVelocityInGround(state, station))
        coords = model.updCoordinateSet()
        for name, value in zip(CLUB_FREE_COORDINATES, [*omega, *v_origin], strict=True):
            coords.get(name).setSpeedValue(state, float(value))
        model.realizeVelocity(state)
        club = model.getBodySet().get(CLUB_BODY)
        got_w = _vec(club.getAngularVelocityInGround(state))
        got_v = _vec(club.getLinearVelocityInGround(state))
        scale = max(1.0, float(np.linalg.norm(omega)), float(np.linalg.norm(v_origin)))
        if (
            max(np.abs(got_w - omega).max(), np.abs(got_v - v_origin).max())
            > 1e-6 * scale
        ):
            raise RuntimeError("club initial velocity does not match the weld velocity")

    def run(
        self,
        t_end: float | None = None,
        accuracy: float | None = None,
        on_sample: Callable[[float], None] | None = None,
        method: str = CONTACT_INTEGRATOR,
    ) -> OpenSimContactRun:
        """Integrate with the stiff (implicit) integrator the contact needs."""
        if accuracy is None:
            accuracy = (
                TOPOLOGY_CONTACT_ACCURACY
                if self.topology_model is not None
                else CONTACT_ACCURACY
            )
        res = super().run(t_end, accuracy, on_sample, method)
        assert isinstance(res, OpenSimContactRun)
        return res

    @staticmethod
    def _split_record(rec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return rec[:, 0:3], rec[:, 3:6]

    def _assemble(self, samples: np.ndarray, rows: list[dict[str, Any]]) -> Any:
        base = super()._assemble(samples, rows)
        if self.topology_model is not None:
            recorder = PairTraceRecorder(self.topology_model)
            for r in rows:
                side_wrenches = {s: r[s]["side_wrench"] for s, _ in _SIDES}
                hand_frames = {s: r[s]["hand_bstate"] for s, _ in _SIDES}
                club_frames = {s: r[s]["club_bstate"] for s, _ in _SIDES}
                recorder.record(side_wrenches, hand_frames, club_frames)
            extra = recorder.result()
        else:
            extra = {}
        return OpenSimContactRun(
            **vars(base),
            normal_force_n={
                s: np.array([r[s]["normal"] for r in rows]) for s, _ in _SIDES
            },
            roll_slip_rad={
                s: np.array([r[s]["roll"] for r in rows]) for s, _ in _SIDES
            },
            axial_slip_m={
                s: np.array([r[s]["delta"][0] for r in rows]) for s, _ in _SIDES
            },
            pair_normal_force_n=extra.get("pair_normal_force_n"),
            pair_tangential_force_n=extra.get("pair_tangential_force_n"),
            channel_reasons=extra.get("channel_reasons"),
            hand_states=extra.get("hand_states"),
            club_states=extra.get("club_states"),
            pair_contact_point_drift_m=extra.get("pair_contact_point_drift_m"),
        )


def simulate_grip_contact(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    interface: GripInterface | None = None,
    pads: PadContactModel | None = None,
    topology_model: TopologyContactModel | None = None,
    t_end_s: float | None = None,
    hold: bool = False,
    mesh_dir: Path | None = None,
) -> ContactRun:
    """Prescribed-arm, free-club simulation of contact grip with OpenSim.

    Accepts either legacy ``pads`` (:class:`PadContactModel`) or ``topology_model``
    (:class:`TopologyContactModel`).  Simulates using :class:`ContactGripSimulator`
    and returns a :class:`ContactRun`.
    """
    if pads is None and topology_model is None:
        raise ValueError("either pads or topology_model must be provided")
    full_times = np.asarray(swing.time_s, dtype=float)
    times = full_times
    if t_end_s is not None:
        times = full_times[full_times <= t_end_s + 1e-12]
    if hold:
        q = np.tile(np.asarray(swing.q[0], float), (len(full_times), 1))
    else:
        q = np.asarray(swing.q[: len(full_times)], dtype=float)
    sim = ContactGripSimulator(
        spec_bytes,
        swing.names,
        full_times,
        q,
        pads=pads,
        interface=interface,
        mesh_dir=mesh_dir,
        topology_model=topology_model,
    )
    bushing_run = sim.run(t_end=float(times[-1]))
    return bushing_run.to_contact_run()


__all__ = [
    "ContactGripSimulator",
    "OpenSimContactRun",
    "calibrated_foundation",
    "calibrated_site_foundation",
    "pad_normal_force_n",
    "simulate_grip_contact",
    "single_pad_model",
    "write_calibration_mesh",
]
