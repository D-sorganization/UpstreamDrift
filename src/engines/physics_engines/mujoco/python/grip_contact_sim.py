"""MuJoCo native-contact grip on the same input as OpenSim (issue #11739, OSV-7).

The ``contact`` grip model with MuJoCo's own collision and constraint solver:
the club is a free body carrying a rigid grip cylinder; each hand is a heavy
free body whose pose and velocity are overwritten from the prescribed swing
before every step, carrying the spherical pads of the shared
:class:`~src.shared.python.grip_contact.pad_layout.PadLayout`.  Pad-cylinder
pairs are explicit ``<pair>`` elements with ``condim=6`` (normal, two
tangential, torsional and two rolling rows), an elliptic cone and a soft
``solref``/``solimp``.

Matching the stiffness.  A MuJoCo soft constraint produces a force
``d^2 k / A`` per unit penetration (``k`` the direct ``solref`` stiffness, ``d``
the impedance, ``A`` the inverse inertia in the contact row), so the force
stiffness depends on the contact's lever arm on the club.  The per-pad
``solref`` is therefore calibrated numerically at the held configuration so
each pad's force equals the shared law's ``k_pad * penetration``
(:meth:`ClubInHands.calibrate`), and the damping coefficient follows from the
Hunt-Crossley form linearised at the nominal penetration.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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
    hand_frame_states,
)
from src.shared.python.grip_contact.contact_run import ContactRun, slip_from_frames
from src.shared.python.grip_contact.pad_contact import PadContactModel
from src.shared.python.grip_contact.parity import GripKineticsSeries

ENGINE = "mujoco_contact"
DEFAULT_TIMESTEP_S = 1.0e-5
#: ``hand_mode`` values: both hands prescribed (default), the trail hand
#: following the club (issue #11986), or only the lead hand gripping.
HAND_MODES = ("prescribed", "trail_follows_club", "lead_only")
HAND_MASS_KG = 100.0
IMPEDANCE = 0.95


def _nums(values: Any) -> str:
    return " ".join(format(float(v), ".17g") for v in values)


def contact_mjcf(
    club: ClubDynamics,
    gravity: np.ndarray,
    timestep_s: float,
    interface: GripInterface,
    pads: PadContactModel,
    solref: dict[tuple[str, int], tuple[float, float]],
    friction_time_s: float | None = None,
    sides: tuple[str, ...] = SIDES,
) -> str:
    """Free club (grip cylinder), heavy hand bodies with pads, explicit pairs.

    ``friction_time_s`` is the ``solreffriction`` time constant (default two
    timesteps); ``sides`` are the hands that carry pads.
    """
    if friction_time_s is None:
        friction_time_s = 2.0 * timestep_s
    if not friction_time_s > 0.0:
        raise ValueError("friction_time_s must be positive")
    i = club.inertia_com_kg_m2
    full = (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])
    rot_r = np.asarray(interface.right.rotation, dtype=float)
    axis = rot_r[:, 0]
    axis_point = np.asarray(interface.right.position_m) + rot_r @ (
        pads.layout.axis_offset_grip_frame("R")
    )
    lo, hi = pads.cylinder.axial_range_m
    a0, a1 = axis_point + lo * axis, axis_point + hi * axis
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
        f'size="{pads.layout.grip_radius_m:.17g}" contype="0" conaffinity="0" mass="0"/>'
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


class ClubInHands:
    """The free club held by pads on two prescribed hand bodies."""

    def __init__(
        self,
        spec_bytes: bytes,
        names: list[str],
        interface: GripInterface,
        pads: PadContactModel,
        timestep_s: float = DEFAULT_TIMESTEP_S,
        friction_time_s: float | None = None,
        hand_mode: str = "prescribed",
    ) -> None:
        if hand_mode not in HAND_MODES:
            raise ValueError(
                f"hand_mode must be one of {HAND_MODES}, got {hand_mode!r}"
            )
        mj = _mujoco()
        self.friction_time_s, self.hand_mode = friction_time_s, hand_mode
        self.pad_sides = ("L",) if hand_mode == "lead_only" else SIDES
        spec = json.loads(spec_bytes)
        self._mj = mj
        self.interface, self.pads, self.timestep_s = interface, pads, timestep_s
        self.pad_count = pads.layout.pad_count
        self._preload_m = pads.layout.preload_penetration_m
        self._dissipation_s_m = pads.law.dissipation_s_m
        self.kinematics = WeldClubKinematics(spec_bytes, names)
        self.club = ClubDynamics.from_spec(spec)
        self.gravity = np.asarray(spec["gravity_m_s2"], float)
        n_pad = self.pad_count
        delta0 = self._preload_m
        # first guess: unit contact-row inverse inertia of about 4 / kg
        k0 = pads.law.stiffness_n_m * 4.0 / IMPEDANCE**2
        self.solref = {(s, k): self._solref(k0) for s in SIDES for k in range(n_pad)}
        self._rebuild()
        self.delta0 = delta0
        self.hand_source: Any = None
        #: diagnostic drift of the trail hand, in its grip frame (issue #11986)
        self.trail_shift_m = np.zeros(3)

    def _solref(self, k_solref: float) -> tuple[float, float]:
        damping = k_solref * IMPEDANCE * self._preload_m * self._dissipation_s_m
        return (-k_solref, -damping)

    def _rebuild(self) -> None:
        mj = self._mj
        xml = contact_mjcf(
            self.club,
            self.gravity,
            self.timestep_s,
            self.interface,
            self.pads,
            self.solref,
            self.friction_time_s,
            self.pad_sides,
        )
        self.model = mj.MjModel.from_xml_string(xml)
        self.data = mj.MjData(self.model)
        self._club = int(self.model.body("club").id)
        self._hand = {s: int(self.model.body(f"hand_{s}").id) for s in self.pad_sides}
        self._offsets = {s: self.interface.frame(s).matrix() for s in SIDES}
        self._pad_geom = {}
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

    def _dof(self, body: int) -> int:
        return int(self.model.body_dofadr[body])

    def place_hands(self, t: float, velocity_time: float | None = None) -> dict:
        """Overwrite both hand bodies from the prescribed swing at ``t``.

        The velocity is taken at ``velocity_time`` (default ``t``).
        """
        q, qdot = self.hand_source(t)
        hands = hand_frame_states(self.kinematics.state(q, qdot), self.interface)
        if velocity_time is not None and velocity_time != t:
            qv, qvd = self.hand_source(velocity_time)
            vel = hand_frame_states(self.kinematics.state(qv, qvd), self.interface)
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
    def read(self, hands: dict) -> ContactSample:
        """Per-hand wrench on the club from the current contacts (after forward)."""
        mj, m, d = self._mj, self.model, self.data
        club = self.club_state()
        frames = {s: club.frame(self._offsets[s]) for s in SIDES}
        force = {s: np.zeros(3) for s in SIDES}
        moment = {s: np.zeros(3) for s in SIDES}
        normal = {s: np.zeros(self.pad_count) for s in SIDES}
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
            normal[side][k] = res[0]
        return ContactSample(hands, frames, force, moment, normal, club.rotation.copy())

    # ---------------------------------------------------------- calibration
    def hold_pose(self, q: np.ndarray) -> dict:
        """Hands and club at the weld pose for ``q``, at rest."""
        self.hand_source = lambda _t: (q, np.zeros_like(q))
        weld = self.kinematics.state(q, np.zeros_like(q))
        self.place_club(weld)
        return self.place_hands(0.0)

    def pad_forces_and_penetrations(self) -> tuple[dict, dict]:
        """Normal force and penetration of every pad contact after ``mj_forward``."""
        mj, m, d = self._mj, self.model, self.data
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

    def calibrate(self, q: np.ndarray, iterations: int = 8, tol: float = 1e-3) -> float:
        """Scale each pad's ``solref`` until its force is ``k_pad * penetration``.

        Returns the largest relative force error left.  Raises ``RuntimeError``
        if a pad is not in contact or the iteration does not converge.
        """
        law = self.pads.law
        worst = float("inf")
        for _ in range(iterations):
            self.hold_pose(q)
            force, depth = self.pad_forces_and_penetrations()
            if len(force) != len(self.pad_sides) * self.pad_count:
                raise RuntimeError("every pad must touch the grip at the held pose")
            worst = 0.0
            for key, f in force.items():
                target = law.stiffness_n_m * depth[key]
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
    times: np.ndarray, samples: list[ContactSample], metadata: dict[str, Any]
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
    return ContactRun(series, rec["normal"], roll, axial)


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
    return build_run(times, samples, {"scenario": "static hold"})


def simulate_grip_contact(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    pads: PadContactModel,
    interface: GripInterface | None = None,
    timestep_s: float = DEFAULT_TIMESTEP_S,
    t_end_s: float | None = None,
    t_start_s: float | None = None,
    friction_time_s: float | None = None,
    hand_mode: str = "prescribed",
    trail_shift_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> ContactRun:
    """Integrate the free club held by pads over the prescribed swing.

    ``t_start_s`` starts the run at the first sample at or after that time,
    with the club moving as the weld (a window of the swing for diagnosis and
    cross-engine comparison).

    The club starts at the weld pose and with the weld velocity of the first
    sample: unlike a bushing, a stiff frictional contact would turn a start at
    rest against moving hands into a large artificial impulse.  Samples are taken at ``swing.time_s``.

    Raises:
        ValueError: if a sample interval is not a whole number of steps.
    """
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    sim = ClubInHands(
        spec_bytes,
        list(swing.names),
        interface,
        pads,
        timestep_s,
        friction_time_s,
        hand_mode,
    )
    sim.calibrate(np.asarray(swing.q[0], float))
    shift = np.asarray(trail_shift_m, dtype=float)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("trail_shift_m must be a finite 3-vector")
    sim.trail_shift_m = shift
    spline = CoordinateSpline(swing.time_s, swing.q)
    times = swing.time_s
    if t_start_s is not None:
        times = times[times >= t_start_s - 1e-12]
    if t_end_s is not None:
        times = times[times <= t_end_s + 1e-12]
    steps = np.diff(times) / timestep_s
    if not np.allclose(steps, np.round(steps), atol=1e-6):
        raise ValueError("sample interval must be a whole number of timesteps")
    sim.hand_source = spline.evaluate
    weld0 = sim.kinematics.state(*spline.evaluate(float(times[0])))
    sim.place_club(weld0)
    sim.data.time = float(times[0])
    samples = [sim.sample()]
    for k in range(1, times.size):
        sim.advance(int(round(steps[k - 1])))
        samples.append(sim.sample())
    return build_run(
        times,
        samples,
        {
            "integrator": "MuJoCo Euler, fixed step, soft constraints",
            "timestep_s": timestep_s,
            "force_law": "MuJoCo native sphere-cylinder contact, condim 6",
            "solreffriction_time_s": friction_time_s or 2.0 * timestep_s,
            "hand_mode": hand_mode,
            "trail_shift_m": [float(v) for v in shift],
            "squeeze_per_hand_n": pads.layout.pad_count
            * pads.law.stiffness_n_m
            * pads.layout.preload_penetration_m,
        },
    )
