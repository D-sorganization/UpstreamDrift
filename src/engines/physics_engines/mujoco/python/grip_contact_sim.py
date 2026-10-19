"""MuJoCo native-contact grip on the same input as OpenSim (issue #11739, OSV-7).

The ``contact`` grip model with MuJoCo's own collision and constraint solver:
the club is a free body carrying a rigid grip cylinder; each hand is a heavy
free body whose pose and velocity are overwritten from the prescribed swing
before every step, carrying the spherical pads of the shared
:class:`~src.shared.python.grip_contact.pad_layout.PadLayout` or the anatomical
sphere/capsule sites of :class:`~src.shared.python.grip_contact.contact_topology.ContactTopology`
(issue #13306, HRG-GS-G8).  Site-cylinder pairs are explicit ``<pair>``
elements with ``condim=6`` (normal, two tangential, torsional and two rolling
rows), an elliptic cone and a soft ``solref``/``solimp``.

Matching the stiffness.  A MuJoCo soft constraint produces a force
``d^2 k / A`` per unit penetration (``k`` the direct ``solref`` stiffness, ``d``
the impedance, ``A`` the inverse inertia in the contact row), so the force
stiffness depends on the contact's lever arm on the club.  The per-site
``solref`` is therefore calibrated numerically at the held configuration so
each site's force equals the shared law's normal force
(:meth:`ClubInHands.calibrate`), and the damping coefficient follows from the
Hunt-Crossley form linearised at the nominal penetration.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.engines.physics_engines.mujoco.python.grip_bushing import (
    SIDES,
    WeldClubKinematics,
    _mujoco,
)
from src.shared.python.grip_contact import (
    ClubDynamics,
    CoordinateSpline,
    CoordinateSwing,
    GripInterface,
    RigidBodyState,
)
from src.shared.python.grip_contact.contact_run import ContactRun, slip_from_frames
from src.shared.python.grip_contact.contact_topology import (
    DEFAULT_TOPOLOGY_PRELOAD_M,
    pair_surface_gap_m,
)
from src.shared.python.grip_contact.pad_contact import PadContactModel, grip_axis
from src.shared.python.grip_contact.pair_contact import PairWrench
from src.shared.python.grip_contact.parity import GripKineticsSeries
from src.shared.python.grip_contact.topology_contact import (
    PairTraceRecorder,
    TopologyContactModel,
    TopologySideWrench,
    map_site_to_hand_frame,
    topology_side_wrench,
)

ENGINE = "mujoco_contact"
DEFAULT_TIMESTEP_S = 1.0e-5
#: ``hand_mode`` values: both hands prescribed (default), the trail hand
#: following the club (issue #11986), or only the lead hand gripping.
HAND_MODES = ("prescribed", "trail_follows_club", "lead_only")
HAND_MASS_KG = 100.0
IMPEDANCE = 0.95


def _nums(values: Any) -> str:
    return " ".join(format(float(v), ".17g") for v in values)


def _topology_bodies_and_pairs(
    topo_model: TopologyContactModel,
    solref: dict[Any, tuple[float, float]],
    friction_time_s: float,
    sides: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Build hand body XML and explicit <exclude>/<pair> XML from TopologyContactModel."""
    topo = topo_model.topology
    law = topo_model.law
    mu = law.static_friction
    torsional = mu * topo_model.patch_radius_m
    bodies: list[str] = []
    pairs: list[str] = []
    if "L" in sides and "R" in sides:
        pairs.append('<exclude body1="hand_L" body2="hand_R"/>')
    for side in sides:
        geoms: list[str] = []
        for k, pair in enumerate(topo_model.physical_grip_pairs(side)):
            mapped = map_site_to_hand_frame(topo, pair.site_a, side)
            if mapped.primitive == "sphere":
                geoms.append(
                    f'<geom name="{mapped.name}" type="sphere" '
                    f'size="{mapped.radius_m:.17g}" pos="{_nums(mapped.centre_m)}" '
                    f'contype="0" conaffinity="0" mass="0"/>'
                )
            elif mapped.primitive == "capsule":
                p0, p1 = mapped.endpoints_m
                geoms.append(
                    f'<geom name="{mapped.name}" type="capsule" '
                    f'size="{mapped.radius_m:.17g}" fromto="{_nums(p0)} {_nums(p1)}" '
                    f'contype="0" conaffinity="0" mass="0"/>'
                )
            else:
                raise ValueError(f"unsupported primitive {mapped.primitive!r}")
            ref = solref.get((side, k)) or solref[pair.name]
            pairs.append(
                f'<pair geom1="{mapped.name}" geom2="grip" condim="6" '
                f'friction="{mu:.17g} {mu:.17g} {torsional:.17g} 1e-5 1e-5" '
                f'solref="{ref[0]:.17g} {ref[1]:.17g}" '
                f'solreffriction="{friction_time_s:.17g} 1" '
                f'solimp="{IMPEDANCE} {IMPEDANCE} 0.001 0.5 2"/>'
            )
        bodies.append(
            f'<body name="hand_{side}"><freejoint name="hand_{side}_free"/>'
            f'<inertial pos="0 0 0" mass="{HAND_MASS_KG}" diaginertia="10 10 10"/>'
            + "".join(geoms)
            + "</body>"
        )
    return bodies, pairs


def contact_mjcf(
    club: ClubDynamics,
    gravity: np.ndarray,
    timestep_s: float,
    interface: GripInterface,
    pads: PadContactModel | TopologyContactModel | None,
    solref: dict[Any, tuple[float, float]],
    friction_time_s: float | None = None,
    topology_model: TopologyContactModel | tuple[str, ...] | None = None,
) -> str:
    """Free club (grip cylinder), heavy hand bodies with pads or topology sites, explicit pairs.

    ``friction_time_s`` is the ``solreffriction`` time constant (default two
    timesteps). ``topology_model`` optionally supplies a :class:`TopologyContactModel`
    (or legacy ``sides`` tuple when called positionally on the pad path).
    """
    if friction_time_s is None:
        friction_time_s = 2.0 * timestep_s
    if not friction_time_s > 0.0:
        raise ValueError("friction_time_s must be positive")
    topo: TopologyContactModel | None = None
    sides: tuple[str, ...] = SIDES
    if isinstance(topology_model, tuple):
        sides = topology_model
    elif isinstance(topology_model, TopologyContactModel):
        topo = topology_model
        sides = (
            tuple(
                s
                for s in SIDES
                if any(isinstance(k, tuple) and k[0] == s for k in solref)
            )
            or SIDES
        )
    if isinstance(pads, TopologyContactModel):
        topo = pads
    i = club.inertia_com_kg_m2
    full = (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])
    if topo is not None:
        cap_r = topo.grip_capsule("R")
        right_frame = interface.right
        rot_r = np.asarray(right_frame.rotation, dtype=float)
        pos_r = np.asarray(right_frame.position_m, dtype=float)
        a0, a1 = pos_r + rot_r @ cap_r.p0_m, pos_r + rot_r @ cap_r.p1_m
        grip_radius_m = topo.cylinder.radius_m
        bodies, pairs = _topology_bodies_and_pairs(topo, solref, friction_time_s, sides)
    else:
        if not isinstance(pads, PadContactModel):
            raise ValueError("either pads or topology_model must be provided")
        axis, axis_point = grip_axis(interface, pads)
        lo, hi = pads.cylinder.axial_range_m
        a0, a1 = axis_point + lo * axis, axis_point + hi * axis
        grip_radius_m = pads.layout.grip_radius_m
        mu = pads.law.static_friction
        torsional = mu * pads.patch_radius_m
        bodies, pairs = [], []
        for side in sides:
            geoms = []
            for k, pos in enumerate(pads.layout.positions_grip_frame(side)):
                geoms.append(
                    f'<geom name="pad_{side}{k}" type="sphere" size="{pads.layout.pad_radius_m:.17g}" '
                    f'pos="{_nums(pos)}" contype="0" conaffinity="0" mass="0"/>'
                )
                ref = solref[(side, k)]
                pairs.append(
                    f'<pair geom1="pad_{side}{k}" geom2="grip" condim="6" '
                    f'friction="{mu:.17g} {mu:.17g} {torsional:.17g} 1e-5 1e-5" '
                    f'solref="{ref[0]:.17g} {ref[1]:.17g}" '
                    f'solreffriction="{friction_time_s:.17g} 1" '
                    f'solimp="{IMPEDANCE} {IMPEDANCE} 0.001 0.5 2"/>'
                )
            bodies.append(
                f'<body name="hand_{side}"><freejoint name="hand_{side}_free"/>'
                f'<inertial pos="0 0 0" mass="{HAND_MASS_KG}" diaginertia="10 10 10"/>'
                + "".join(geoms)
                + "</body>"
            )
    return (
        '<mujoco model="grip_contact"><compiler inertiafromgeom="false"/>'
        f'<option timestep="{timestep_s:.17g}" integrator="Euler" '
        f'cone="elliptic" impratio="10" gravity="{_nums(gravity)}"/>'
        '<worldbody><body name="club"><freejoint name="club_free"/>'
        f'<inertial pos="{_nums(club.com_m)}" mass="{club.mass_kg:.17g}" '
        f'fullinertia="{_nums(full)}"/>'
        f'<geom name="grip" type="cylinder" fromto="{_nums(a0)} {_nums(a1)}" '
        f'size="{grip_radius_m:.17g}" contype="0" conaffinity="0" mass="0"/>'
        "</body>" + "".join(bodies) + "</worldbody>"
        "<contact>" + "".join(pairs) + "</contact></mujoco>"
    )


@dataclass
class ContactSample:
    """Recorded state at one sample time."""

    hands: dict[str, Any]
    club_frames: dict[str, Any]
    force_n: dict[str, np.ndarray]
    moment_nm: dict[str, np.ndarray]
    normal_n: dict[str, np.ndarray]
    club_rotation: np.ndarray
    side_wrenches: dict[str, TopologySideWrench] | None = None


class ClubInHands:
    """The free club held by pads or topology sites on two prescribed hand bodies."""

    def __init__(
        self,
        spec_bytes: bytes,
        names: list[str],
        interface: GripInterface,
        pads: PadContactModel | TopologyContactModel | None = None,
        timestep_s: float = DEFAULT_TIMESTEP_S,
        friction_time_s: float | None = None,
        **options: Any,
    ) -> None:
        hand_mode: str = str(options.pop("hand_mode", "prescribed"))
        topology_model: TopologyContactModel | None = options.pop(
            "topology_model", None
        )
        if options:
            unknown = next(iter(options))
            raise TypeError(
                f"ClubInHands.__init__() got an unexpected keyword argument {unknown!r}"
            )
        if isinstance(pads, TopologyContactModel) and topology_model is None:
            topology_model, pads = pads, None
        if hand_mode not in HAND_MODES:
            raise ValueError(
                f"hand_mode must be one of {HAND_MODES}, got {hand_mode!r}"
            )
        if pads is None and topology_model is None:
            raise ValueError("either pads or topology_model must be provided")
        mj = _mujoco()
        self.friction_time_s, self.hand_mode = friction_time_s, hand_mode
        self.pad_sides = ("L",) if hand_mode == "lead_only" else SIDES
        spec = json.loads(spec_bytes)
        self._mj = mj
        self.interface = interface
        self.pads: PadContactModel | None = (
            pads if isinstance(pads, PadContactModel) else None
        )
        self.topology_model = topology_model
        self.timestep_s = timestep_s
        if topology_model is not None:
            topo = topology_model.topology
            law = topology_model.law
            self._side_counts = {
                s: len(topology_model.physical_grip_pairs(s)) for s in SIDES
            }
            self.pad_count = max(self._side_counts.values())
            first_pair = topo.hand_grip_pairs()[0]
            self._preload_m = max(
                -pair_surface_gap_m(topo, first_pair), DEFAULT_TOPOLOGY_PRELOAD_M
            )
            self._dissipation_s_m = law.dissipation_s_m
            k0 = law.stiffness_n_m * 4.0 / IMPEDANCE**2
            self.solref = {
                (s, k): self._solref(k0)
                for s in self.pad_sides
                for k in range(self._side_counts[s])
            }
        else:
            assert isinstance(pads, PadContactModel)
            layout = pads.layout
            law = pads.law
            self.pad_count = layout.pad_count
            self._side_counts = {s: self.pad_count for s in SIDES}
            self._preload_m = layout.preload_penetration_m
            self._dissipation_s_m = law.dissipation_s_m
            k0 = law.stiffness_n_m * 4.0 / IMPEDANCE**2
            self.solref = {
                (s, k): self._solref(k0) for s in SIDES for k in range(self.pad_count)
            }
        self.kinematics = WeldClubKinematics(spec_bytes, names)
        self.club = ClubDynamics.from_spec(spec)
        self.gravity = np.asarray(spec["gravity_m_s2"], float)
        self._rebuild()
        self.delta0 = self._preload_m
        self.hand_source: Any = None
        #: diagnostic drift of the trail hand, in its grip frame (issue #11986)
        self.trail_shift_m: np.ndarray = np.zeros(3)

    def _solref(self, k_solref: float) -> tuple[float, float]:
        damping = k_solref * IMPEDANCE * self._preload_m * self._dissipation_s_m
        return (-k_solref, -damping)

    def _rebuild(self) -> None:
        mj = self._mj
        third_arg: Any = (
            self.topology_model if self.topology_model is not None else self.pad_sides
        )
        xml = contact_mjcf(
            self.club,
            self.gravity,
            self.timestep_s,
            self.interface,
            self.pads,
            self.solref,
            self.friction_time_s,
            third_arg,
        )
        self.model = mj.MjModel.from_xml_string(xml)
        self.data = mj.MjData(self.model)
        self._club = int(self.model.body("club").id)
        self._hand = {s: int(self.model.body(f"hand_{s}").id) for s in self.pad_sides}
        self._offsets = {s: self.interface.frame(s).matrix() for s in SIDES}
        self._pad_geom: dict[int, tuple[str, int]] = {}
        if self.topology_model is not None:
            for s in self.pad_sides:
                for k, pair in enumerate(self.topology_model.physical_grip_pairs(s)):
                    self._pad_geom[int(self.model.geom(pair.site_a).id)] = (s, k)
        else:
            for s in self.pad_sides:
                for k in range(self.pad_count):
                    self._pad_geom[int(self.model.geom(f"pad_{s}{k}").id)] = (s, k)

    # ------------------------------------------------------------ placement
    def _set_free(self, body: int, rot: np.ndarray, pos: np.ndarray) -> None:
        quat = np.zeros(4)
        self._mj.mju_mat2Quat(quat, np.asarray(rot, float).ravel())
        adr = int(self.model.body_jntadr[body])
        qadr = int(self.model.jnt_qposadr[adr])
        self.data.qpos[qadr : qadr + 3] = pos
        self.data.qpos[qadr + 3 : qadr + 7] = quat

    def _qadr(self, body: int) -> int:
        adr = int(self.model.body_jntadr[body])
        return int(self.model.jnt_qposadr[adr])

    def _dof(self, body: int) -> int:
        return int(self.model.body_dofadr[body])

    def place_hands(self, t: float, velocity_time: float | None = None) -> dict:
        """Overwrite both hand bodies from the prescribed swing at ``t``.

        The velocity is taken at ``velocity_time`` (default ``t``).
        """
        if (
            self.hand_source is getattr(self, "_held_source", None)
            and self.hand_mode == "prescribed"
            and not self.trail_shift_m.any()
        ):
            for s in self.pad_sides:
                body = self._hand[s]
                qadr = self._qadr(body)
                dof = self._dof(body)
                self.data.qpos[qadr : qadr + 7] = self._held_qpos[s]
                self.data.qvel[dof : dof + 6] = 0.0
            return self._held_hands

        q, qdot = self.hand_source(t)
        weld = self.kinematics.state(q, qdot)
        hands = {s: weld.frame(self._offsets[s]) for s in SIDES}
        if velocity_time is not None and velocity_time != t:
            qv, qvd = self.hand_source(velocity_time)
            weld_v = self.kinematics.state(qv, qvd)
            vel = {s: weld_v.frame(self._offsets[s]) for s in SIDES}
        else:
            vel = hands
        for s in self.pad_sides:
            h = hands[s]
            if s == "R" and self.hand_mode == "trail_follows_club":
                h = hands[s] = vel[s] = self.club_state_now().frame(self._offsets[s])
            if s == "R" and self.trail_shift_m.any():
                h = hands[s] = replace(
                    h, position_m=h.position_m + h.rotation @ self.trail_shift_m
                )
            self._set_free(self._hand[s], h.rotation, h.position_m)
            dof = self._dof(self._hand[s])
            self.data.qvel[dof : dof + 3] = vel[s].velocity_m_s
            self.data.qvel[dof + 3 : dof + 6] = vel[s].rotation.T @ vel[s].omega_rad_s
        return hands

    def place_club(self, state: RigidBodyState) -> None:
        """Set the club pose and velocity (world) from ``state``."""
        self._set_free(self._club, state.rotation, state.position_m)
        dof = self._dof(self._club)
        self.data.qvel[dof : dof + 3] = state.velocity_m_s
        self.data.qvel[dof + 3 : dof + 6] = state.rotation.T @ state.omega_rad_s

    def club_state(self) -> RigidBodyState:
        from src.engines.physics_engines.mujoco.python.grip_bushing import body_state

        return body_state(self._mj, self.model, self.data, self._club)

    def club_state_now(self) -> RigidBodyState:
        """Club state from the current ``qpos``/``qvel`` (kinematics refreshed)."""
        mj, m, d = self._mj, self.model, self.data
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        mj.mj_comVel(m, d)
        return self.club_state()

    # ------------------------------------------------------------ recording
    def _build_side_wrenches(
        self,
        force: dict[str, np.ndarray],
        moment: dict[str, np.ndarray],
        normal: dict[str, np.ndarray],
        tangential: dict[str, np.ndarray],
    ) -> dict[str, TopologySideWrench] | None:
        if self.topology_model is None:
            return None
        side_wrenches: dict[str, TopologySideWrench] = {}
        zeros3 = np.zeros(3)
        unit_n = np.array([1.0, 0.0, 0.0])
        for s in SIDES:
            pairs = self.topology_model.physical_grip_pairs(s)
            pw_map: dict[str, PairWrench] = {}
            for k, p in enumerate(pairs):
                fn = float(normal[s][k])
                ft = float(tangential[s][k])
                pw_map[p.name] = PairWrench(
                    force_a_n=zeros3,
                    moment_a_nm=zeros3,
                    force_b_n=zeros3,
                    moment_b_nm=zeros3,
                    contact_point_m=zeros3,
                    normal=unit_n,
                    penetration_m=0.0,
                    point_a_m=zeros3,
                    point_b_m=zeros3,
                    normal_force_mag_n=fn,
                    tangential_force_mag_n=ft,
                )
            norm_arr = normal[s][: len(pairs)].copy()
            tang_arr = tangential[s][: len(pairs)].copy()
            side_wrenches[s] = TopologySideWrench(
                force_n=force[s].copy(),
                moment_nm=moment[s].copy(),
                pair_wrenches=pw_map,
                normal_force_n=norm_arr,
                tangential_force_n=tang_arr,
                total_normal_force_n=float(np.sum(norm_arr)),
            )
        return side_wrenches

    def read(self, hands: dict) -> ContactSample:
        """Per-hand wrench on the club from the current contacts (after forward)."""
        mj, m, d = self._mj, self.model, self.data
        club = self.club_state()
        frames = {s: club.frame(self._offsets[s]) for s in SIDES}
        force = {s: np.zeros(3) for s in SIDES}
        moment = {s: np.zeros(3) for s in SIDES}
        normal = {s: np.zeros(self._side_counts[s]) for s in SIDES}
        tangential = {s: np.zeros(self._side_counts[s]) for s in SIDES}
        res = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            hit = self._pad_geom.get(int(c.geom1))
            if hit is None:
                continue
            side, k = hit
            mj.mj_contactForce(m, d, i, res)
            frame = np.asarray(c.frame).reshape(3, 3)
            f_world = frame.T @ res[:3]  # on geom2 (the grip), i.e. on the club
            torque_world = frame.T @ res[3:6]
            force[side] += f_world
            moment[side] += np.cross(c.pos - frames[side].position_m, f_world) + (
                torque_world
            )
            normal[side][k] += float(res[0])
            tangential[side][k] += float(np.hypot(res[1], res[2]))
        side_wrenches = self._build_side_wrenches(force, moment, normal, tangential)
        return ContactSample(
            hands, frames, force, moment, normal, club.rotation.copy(), side_wrenches
        )

    # ---------------------------------------------------------- calibration
    def hold_pose(self, q: np.ndarray) -> dict:
        """Hands and club at the weld pose for ``q``, at rest."""
        still = np.zeros_like(q)

        def _hold_src(_t: float) -> tuple[np.ndarray, np.ndarray]:
            return q, still

        self._held_source: Any = None
        self.hand_source = _hold_src
        weld = self.kinematics.state(q, still)
        self.place_club(weld)
        hands = self.place_hands(0.0)
        if self.hand_mode == "prescribed" and not self.trail_shift_m.any():
            self._held_hands = hands
            self._held_qpos = {}
            for s in self.pad_sides:
                qadr = self._qadr(self._hand[s])
                self._held_qpos[s] = self.data.qpos[qadr : qadr + 7].copy()
            self._held_source = _hold_src
        return hands

    def _forward_clamped_club(self) -> None:
        """Run ``mj_forward`` with the free club held in static equilibrium."""
        mj, m, d = self._mj, self.model, self.data
        d.qfrc_applied[:] = 0.0
        mj.mj_forward(m, d)
        if d.nefc > 0:
            j_mat = np.asarray(d.efc_J).reshape(d.nefc, m.nv)
            f_eq = np.asarray(d.efc_aref) / np.asarray(d.efc_R)
            d.qfrc_applied[:] = d.qfrc_bias - j_mat.T @ f_eq
            mj.mj_forward(m, d)

    def pad_forces_and_penetrations(self) -> tuple[dict, dict]:
        """Normal force and penetration of every pad/site contact after ``mj_forward``."""
        mj, m, d = self._mj, self.model, self.data
        if self.topology_model is None:
            mj.mj_forward(m, d)
            res = np.zeros(6)
            force, depth = {}, {}
            for i in range(d.ncon):
                c = d.contact[i]
                hit = self._pad_geom.get(int(c.geom1))
                if hit is None:
                    continue
                mj.mj_contactForce(m, d, i, res)
                force[hit], depth[hit] = float(res[0]), float(-c.dist)
            return force, depth

        self._forward_clamped_club()
        res = np.zeros(6)
        force, depth = {}, {}
        for i in range(d.ncon):
            c = d.contact[i]
            hit = self._pad_geom.get(int(c.geom1))
            if hit is None:
                continue
            mj.mj_contactForce(m, d, i, res)
            force[hit] = force.get(hit, 0.0) + float(res[0])
            depth[hit] = max(depth.get(hit, 0.0), float(-c.dist))
        d.qfrc_applied[:] = 0.0
        return force, depth

    def _default_target_force(
        self, q: np.ndarray
    ) -> Callable[[tuple[str, int], float], float]:
        if self.topology_model is None:
            assert self.pads is not None
            k_pad = self.pads.law.stiffness_n_m
            return lambda _key, depth: k_pad * depth
        hands = self.hold_pose(q)
        club = self.club_state_now()
        targets: dict[tuple[str, int], float] = {}
        for s in self.pad_sides:
            sw = topology_side_wrench(
                self.topology_model, s, hands[s], club.frame(self._offsets[s])
            )
            for k, pair in enumerate(self.topology_model.physical_grip_pairs(s)):
                targets[(s, k)] = sw.pair_wrenches[pair.name].normal_force_mag_n
        return lambda key, _depth: targets[key]

    def calibrate(
        self,
        q: np.ndarray,
        iterations: int = 8,
        tol: float = 1e-3,
        target_force: Callable[[tuple[str, int], float], float] | None = None,
    ) -> float:
        """Scale each site's ``solref`` until its force equals ``target_force(key, depth)``.

        Returns the largest relative force error left.  Raises ``RuntimeError``
        if a site is not in contact or the iteration does not converge.
        """
        target_fn = target_force or self._default_target_force(q)
        expected_count = sum(self._side_counts[s] for s in self.pad_sides)
        worst = float("inf")
        for _ in range(iterations):
            self.hold_pose(q)
            force, depth = self.pad_forces_and_penetrations()
            if len(force) != expected_count:
                raise RuntimeError("every pad must touch the grip at the held pose")
            worst = 0.0
            for key, f in force.items():
                target = float(target_fn(key, depth[key]))
                worst = max(worst, abs(f - target) / target)
                scale = target / max(f, 1e-9)
                self.solref[key] = self._solref(-self.solref[key][0] * scale)
            self._rebuild_keep_state()
            if worst < tol:
                return worst
        raise RuntimeError(f"solref calibration did not converge (error {worst:.3g})")

    def _rebuild_keep_state(self) -> None:
        self._rebuild()

    # ------------------------------------------------------------ stepping
    def advance(self, steps: int) -> None:
        """``steps`` Euler steps with the hands re-prescribed before each."""
        mj, m, d = self._mj, self.model, self.data
        dt = self.timestep_s
        for _ in range(steps):
            t = float(d.time)
            self.place_hands(t, velocity_time=t + 0.5 * dt)
            mj.mj_step(m, d)

    def sample(self) -> ContactSample:
        """Exact hand pose at the current time, forward, and the contact wrench."""
        hands = self.place_hands(float(self.data.time))
        self._mj.mj_forward(self.model, self.data)
        return self.read(hands)


def _stack(samples: list[ContactSample]) -> dict[str, Any]:
    def per_side(get: Any) -> dict[str, np.ndarray]:
        return {s: np.array([get(x)[s] for x in samples]) for s in SIDES}

    def pose(attr: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            s: (
                np.array([getattr(x, attr)[s].rotation for x in samples]),
                np.array([getattr(x, attr)[s].position_m for x in samples]),
            )
            for s in SIDES
        }

    return {
        "force": per_side(lambda x: x.force_n),
        "moment": per_side(lambda x: x.moment_nm),
        "normal": per_side(lambda x: x.normal_n),
        "hand_pose": pose("hands"),
        "club_pose": pose("club_frames"),
    }


def build_run(
    times: np.ndarray,
    samples: list[ContactSample],
    metadata: dict[str, Any],
    topology_model: TopologyContactModel | None = None,
) -> ContactRun:
    """Assemble the series and the contact-only outputs from recorded samples."""
    rec = _stack(samples)
    series = GripKineticsSeries.from_frames(
        ENGINE,
        times,
        {s: (rec["force"][s], rec["moment"][s]) for s in SIDES},
        rec["hand_pose"],
        rec["club_pose"],
        np.array([x.club_rotation for x in samples]),
        metadata,
    )
    axial, roll = slip_from_frames(rec["hand_pose"], rec["club_pose"])
    extra: dict[str, Any] = {}
    if topology_model is not None:
        recorder = PairTraceRecorder(topology_model)
        for smp in samples:
            assert smp.side_wrenches is not None
            recorder.record(smp.side_wrenches, smp.hands, smp.club_frames)
        extra = recorder.result()
    return ContactRun(series, rec["normal"], roll, axial, **extra)


def hold_run(
    sim: ClubInHands, q: np.ndarray, duration_s: float, sample_dt_s: float = 0.002
) -> ContactRun:
    """Hold the hands still and let the club settle; sample every ``sample_dt_s``."""
    sim.hold_pose(q)
    n = int(round(duration_s / sample_dt_s))
    per = int(round(sample_dt_s / sim.timestep_s))
    samples = [sim.sample()]
    for _ in range(n):
        sim.advance(per)
        samples.append(sim.sample())
    times = np.arange(n + 1) * sample_dt_s
    return build_run(
        times, samples, {"scenario": "static hold"}, topology_model=sim.topology_model
    )


def simulate_grip_contact(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    pads: PadContactModel | None = None,
    interface: GripInterface | None = None,
    timestep_s: float = DEFAULT_TIMESTEP_S,
    t_end_s: float | None = None,
    topology_model: TopologyContactModel | None = None,
    **options: Any,
) -> ContactRun:
    """Integrate the free club held by pads or topology over the prescribed swing.

    Optional keyword options in ``options``:
    - ``t_start_s``: starts the run at the first sample at or after that time.
    - ``friction_time_s``: ``solreffriction`` time constant.
    - ``hand_mode``: ``'prescribed'``, ``'trail_follows_club'``, or ``'lead_only'``.
    - ``trail_shift_m``: 3-vector shift of the trail hand in its grip frame.
    - ``hold``: if True, keep the hands at rest at ``swing.q[0]``.

    Raises:
        ValueError: if a sample interval is not a whole number of steps.
    """
    allowed = {"t_start_s", "friction_time_s", "hand_mode", "trail_shift_m", "hold"}
    unknown = set(options) - allowed
    if unknown:
        raise TypeError(f"unexpected keyword argument(s): {sorted(unknown)}")
    t_start_s: float | None = options.get("t_start_s")
    friction_time_s: float | None = options.get("friction_time_s")
    hand_mode: str = options.get("hand_mode", "prescribed")
    trail_shift_m = options.get("trail_shift_m", (0.0, 0.0, 0.0))
    hold: bool = bool(options.get("hold", False))

    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    sim = ClubInHands(
        spec_bytes,
        list(swing.names),
        interface,
        pads,
        timestep_s,
        friction_time_s,
        hand_mode=hand_mode,
        topology_model=topology_model,
    )
    q0 = np.asarray(swing.q[0], float)
    sim.calibrate(q0)
    shift = np.asarray(trail_shift_m, dtype=float)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("trail_shift_m must be a finite 3-vector")
    sim.trail_shift_m = shift
    times: np.ndarray = swing.time_s
    if t_start_s is not None:
        times = times[times >= t_start_s - 1e-12]
    if t_end_s is not None:
        times = times[times <= t_end_s + 1e-12]
    steps = np.diff(times) / timestep_s
    if not np.allclose(steps, np.round(steps), atol=1e-6):
        raise ValueError("sample interval must be a whole number of timesteps")
    if hold:
        sim.hold_pose(q0)
    else:
        spline = CoordinateSpline(swing.time_s, swing.q)
        sim.hand_source = spline.evaluate
        weld0 = sim.kinematics.state(*spline.evaluate(float(times[0])))
        sim.place_club(weld0)
    sim.data.time = float(times[0])
    samples = [sim.sample()]
    for k in range(1, times.size):
        sim.advance(int(round(steps[k - 1])))
        samples.append(sim.sample())
    if topology_model is not None:
        topo = topology_model.topology
        law = topology_model.law
        force_law = (
            "MuJoCo native ContactTopology sphere/capsule-cylinder contact, condim 6"
        )
        squeeze_n = (
            0.5 * len(topo.hand_grip_pairs()) * law.stiffness_n_m * sim._preload_m
        )
    else:
        assert pads is not None
        layout = pads.layout
        law = pads.law
        force_law = "MuJoCo native sphere-cylinder contact, condim 6"
        squeeze_n = layout.pad_count * law.stiffness_n_m * layout.preload_penetration_m
    return build_run(
        times,
        samples,
        {
            "integrator": "MuJoCo Euler, fixed step, soft constraints",
            "timestep_s": timestep_s,
            "force_law": force_law,
            "solreffriction_time_s": friction_time_s or 2.0 * timestep_s,
            "hand_mode": hand_mode,
            "trail_shift_m": [float(v) for v in shift],
            "squeeze_per_hand_n": squeeze_n,
        },
        topology_model=topology_model,
    )
