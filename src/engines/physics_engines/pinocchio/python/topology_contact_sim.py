"""Pinocchio grip-contact adapter from the shared topology (HRG-GS-G6, #13304).

Pinocchio has no native contact dynamics for this problem, so the contact
forces are built from :class:`ContactTopology` and the shared G2/G3 contact law
(:mod:`src.shared.python.grip_contact.pair_contact`).

The club is a free-flyer model evaluated via Pinocchio ``aba``. Forward
kinematics of the weld model provides the hand frames. Integration is performed
via SciPy ``solve_ivp`` (Radau).

Hand-grip pairs transmit physical forces to the club.
Hand-hand pairs are allocated / diagnostic only under the welded-kinematic
contact assumption: they exert action-reaction forces between hands and are
tracked in allocated channels, with zero contribution to the club acceleration.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from src.engines.physics_engines.pinocchio.python import grip_bushing as bushing
from src.shared.python.grip_contact import (
    CoordinateSpline,
    CoordinateSwing,
    GripInterface,
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.grip_contact.bushing_law import BushingState, cross3
from src.shared.python.grip_contact.contact_run import ContactRun, slip_from_frames
from src.shared.python.grip_contact.contact_topology import (
    ContactPair,
    ContactSite,
    ContactTopology,
    contact_topology,
)
from src.shared.python.grip_contact.pair_contact import (
    ContactCapsule,
    ContactSphere,
    capsule_capsule,
    capsule_sphere,
    closest_points_segments,
    sphere_capsule,
)
from src.shared.python.grip_contact.parity import GripKineticsSeries
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
    sphere_ground_contact,
    torsional_friction_moment,
)

ENGINE = "pinocchio_topology_contact"
DEFAULT_RTOL = 1.0e-6
DEFAULT_ATOL = 1.0e-9
DEFAULT_MAX_STEP_S = 2.0e-4
DEFAULT_PRELOAD_M = 2.5e-4
SIDES = ("L", "R")

__all__ = [
    "IntegratorTolerances",
    "TopologyClubInHands",
    "TopologyContactRun",
    "TopologyPairResult",
    "default_contact_parameters",
    "evaluate_topology_wrenches",
    "simulate_topology_contact",
]


def default_contact_parameters() -> ContactParameters:
    """Default contact law parameters for humanoid grip contact."""
    return ContactParameters(
        stiffness_n_m=1.0e5,
        dissipation_s_m=0.1,
        static_friction=0.9,
        dynamic_friction=0.7,
        viscous_friction=0.05,
        transition_velocity_m_s=1.0e-3,
    )


@dataclass(frozen=True)
class IntegratorTolerances:
    """Radau tolerances and maximum step cap for topology contact integration."""

    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    max_step_s: float = DEFAULT_MAX_STEP_S

    def __post_init__(self) -> None:
        if not math.isfinite(self.rtol) or self.rtol <= 0.0:
            raise ValueError(f"rtol must be positive and finite, got {self.rtol}")
        if not math.isfinite(self.atol) or self.atol <= 0.0:
            raise ValueError(f"atol must be positive and finite, got {self.atol}")
        if not math.isfinite(self.max_step_s) or self.max_step_s <= 0.0:
            raise ValueError(
                f"max_step_s must be positive and finite, got {self.max_step_s}"
            )


@dataclass(frozen=True)
class TopologyPairResult:
    """Contact outcome for a single topology pair at a given state.

    Attributes:
        pair_name: Name of the contact pair.
        is_hand_grip: True if pair connects a hand site to the club grip.
        is_hand_hand: True if pair connects two hand sites.
        penetration_m: Penetration depth (m), >= 0.0.
        normal_force_mag_n: Normal force magnitude (N), >= 0.0.
        tangential_force_mag_n: Friction force magnitude (N), >= 0.0.
        force_club_n: Force exerted on club in world coordinates (N).
        moment_club_nm: Moment exerted on club about club body origin (N*m).
        force_a_n: Force on site A in world coordinates (N).
        force_b_n: Force on partner (site B or grip) in world coordinates (N).
        contact_point_m: Contact location in world coordinates (m).
        normal: Unit contact normal pointing from body A toward body B.
        allocated_label: Canonical allocated telemetry channel label.
    """

    pair_name: str
    is_hand_grip: bool
    is_hand_hand: bool
    penetration_m: float
    normal_force_mag_n: float
    tangential_force_mag_n: float
    force_club_n: np.ndarray
    moment_club_nm: np.ndarray
    force_a_n: np.ndarray
    force_b_n: np.ndarray
    contact_point_m: np.ndarray
    normal: np.ndarray
    allocated_label: str


@dataclass(frozen=True)
class TopologyContactRun(ContactRun):
    """Simulation run of the free-flyer club under ContactTopology contact law.

    Extends :class:`ContactRun` with per-pair telemetry traces.
    """

    pair_traces: Mapping[str, np.ndarray] = ()  # type: ignore[assignment]
    pair_normal_force_n: Mapping[str, np.ndarray] = ()  # type: ignore[assignment]


def _site_primitive(site: ContactSite) -> ContactSphere | ContactCapsule:
    """Construct a ContactSphere or ContactCapsule from a ContactSite."""
    if site.primitive == "sphere":
        return ContactSphere(
            center_m=np.asarray(site.centre_m, dtype=float),
            radius_m=site.radius_m,
        )
    hl = site.half_length_m
    ax = np.asarray(site.axis_m, dtype=float)
    c = np.asarray(site.centre_m, dtype=float)
    return ContactCapsule(
        p0_m=c - hl * ax,
        p1_m=c + hl * ax,
        radius_m=site.radius_m,
    )


def _resolve_hand_state(
    hands: Mapping[str, BushingState], hand_name: str
) -> BushingState:
    """Retrieve BushingState for lead or trail hand from hands mapping."""
    key = "lead" if hand_name in ("lead", "L") else "trail"
    state = hands.get(key)
    if state is None:
        alt = "L" if key == "lead" else "R"
        state = hands.get(alt)
    if state is None or not isinstance(state, BushingState):
        raise TypeError(f"Missing valid BushingState for hand {hand_name!r}")
    return state


def _evaluate_hand_grip_sphere(
    site: ContactSite,
    st_hand: BushingState,
    club: BushingState | RigidBodyState,
    r_grip: float,
    law: ContactParameters,
    patch_radius_m: float,
    spin_transition_rad_s: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate sphere-cylinder contact between a hand site and club grip."""
    axis = club.rotation[:, 0]
    club_pos = club.position_m
    c_w = st_hand.position_m + st_hand.rotation @ np.asarray(site.centre_m, float)
    rel = c_w - club_pos
    s = float(axis @ rel)
    rho = rel - s * axis
    dist = float(np.linalg.norm(rho))
    n_rad = rho / dist if dist > 1e-12 else club.rotation[:, 1]
    p_surf = club_pos + s * axis + r_grip * n_rad
    gap = dist - site.radius_m - r_grip
    pen = max(0.0, -gap)
    if pen <= 0.0:
        return 0.0, 0.0, 0.0, np.zeros(3), np.zeros(3), p_surf, n_rad

    v_a = st_hand.velocity_m_s + cross3(st_hand.omega_rad_s, c_w - st_hand.position_m)
    v_c = club.velocity_m_s + cross3(club.omega_rad_s, p_surf - club_pos)
    plane = GroundPlane(
        (float(n_rad[0]), float(n_rad[1]), float(n_rad[2])), float(n_rad @ p_surf)
    )
    sample = sphere_ground_contact(c_w, v_a - v_c, site.radius_m, plane, law)
    spin = torsional_friction_moment(
        sample.normal_force_n,
        st_hand.omega_rad_s - club.omega_rad_s,
        plane,
        patch_radius_m=patch_radius_m,
        friction=law.static_friction,
        transition_rad_s=spin_transition_rad_s,
    )
    f_on_club = -(sample.normal_force_n + sample.friction_force_n)
    m_on_club = cross3(p_surf - club_pos, f_on_club) - spin
    fn_mag = float(np.linalg.norm(sample.normal_force_n))
    ft_mag = float(np.linalg.norm(sample.friction_force_n))
    return pen, fn_mag, ft_mag, f_on_club, m_on_club, p_surf, n_rad


def _evaluate_hand_grip_capsule(
    site: ContactSite,
    st_hand: BushingState,
    club: BushingState | RigidBodyState,
    r_grip: float,
    law: ContactParameters,
    patch_radius_m: float,
    spin_transition_rad_s: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate capsule-cylinder contact between a hand site and club grip."""
    axis = club.rotation[:, 0]
    club_pos = club.position_m
    hl = site.half_length_m
    ax = np.asarray(site.axis_m, float)
    c = np.asarray(site.centre_m, float)
    p0_w = st_hand.position_m + st_hand.rotation @ (c - hl * ax)
    p1_w = st_hand.position_m + st_hand.rotation @ (c + hl * ax)
    c0 = club_pos
    c1 = club_pos + 1.0 * axis
    p_cap, p_cyl = closest_points_segments(p0_w, p1_w, c0, c1)
    rho = p_cap - p_cyl
    dist = float(np.linalg.norm(rho))
    n_rad = rho / dist if dist > 1e-12 else club.rotation[:, 1]
    p_surf = p_cyl + r_grip * n_rad
    gap = dist - site.radius_m - r_grip
    pen = max(0.0, -gap)
    if pen <= 0.0:
        return 0.0, 0.0, 0.0, np.zeros(3), np.zeros(3), p_surf, n_rad

    v_a = st_hand.velocity_m_s + cross3(st_hand.omega_rad_s, p_cap - st_hand.position_m)
    v_c = club.velocity_m_s + cross3(club.omega_rad_s, p_surf - club_pos)
    plane = GroundPlane(
        (float(n_rad[0]), float(n_rad[1]), float(n_rad[2])), float(n_rad @ p_surf)
    )
    sample = sphere_ground_contact(p_cap, v_a - v_c, site.radius_m, plane, law)
    spin = torsional_friction_moment(
        sample.normal_force_n,
        st_hand.omega_rad_s - club.omega_rad_s,
        plane,
        patch_radius_m=patch_radius_m,
        friction=law.static_friction,
        transition_rad_s=spin_transition_rad_s,
    )
    f_on_club = -(sample.normal_force_n + sample.friction_force_n)
    m_on_club = cross3(p_surf - club_pos, f_on_club) - spin
    fn_mag = float(np.linalg.norm(sample.normal_force_n))
    ft_mag = float(np.linalg.norm(sample.friction_force_n))
    return pen, fn_mag, ft_mag, f_on_club, m_on_club, p_surf, n_rad


def _evaluate_hand_hand_pair(
    pair: ContactPair,
    site_a: ContactSite,
    site_b: ContactSite,
    st_a: BushingState,
    st_b: BushingState,
    law: ContactParameters,
    patch_radius_m: float,
    spin_transition_rad_s: float,
) -> TopologyPairResult:
    """Evaluate contact between two hand sites via pair_contact dispatch."""
    prim_a = _site_primitive(site_a)
    prim_b = _site_primitive(site_b)

    if isinstance(prim_a, ContactSphere) and isinstance(prim_b, ContactCapsule):
        wrench = sphere_capsule(
            prim_a, prim_b, st_a, st_b, law, patch_radius_m, spin_transition_rad_s
        )
    elif isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactSphere):
        wrench = capsule_sphere(
            prim_a, prim_b, st_a, st_b, law, patch_radius_m, spin_transition_rad_s
        )
    elif isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactCapsule):
        wrench = capsule_capsule(
            prim_a, prim_b, st_a, st_b, law, patch_radius_m, spin_transition_rad_s
        )
    else:
        raise ValueError(
            f"Unsupported hand-hand primitive pair: {type(prim_a).__name__}, {type(prim_b).__name__}"
        )

    return TopologyPairResult(
        pair_name=pair.name,
        is_hand_grip=False,
        is_hand_hand=True,
        penetration_m=wrench.penetration_m,
        normal_force_mag_n=wrench.normal_force_mag_n,
        tangential_force_mag_n=wrench.tangential_force_mag_n,
        force_club_n=np.zeros(3),
        moment_club_nm=np.zeros(3),
        force_a_n=wrench.force_a_n,
        force_b_n=wrench.force_b_n,
        contact_point_m=wrench.contact_point_m,
        normal=wrench.normal,
        allocated_label=pair.allocated_label,
    )


def evaluate_topology_wrenches(
    topology: ContactTopology,
    hands: Mapping[str, BushingState],
    club: BushingState | RigidBodyState,
    law: ContactParameters,
    patch_radius_m: float | None = None,
    spin_transition_rad_s: float = 1.0,
) -> tuple[
    dict[str, TopologyPairResult],
    tuple[np.ndarray, np.ndarray],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    """Evaluate contact wrenches for all pairs in the contact topology.

    Preconditions:
        topology is a ContactTopology.
        hands contains BushingState for lead and trail hands.
        club is a BushingState.
        law is a ContactParameters instance.

    Postconditions:
        Normal forces are non-negative.
        Hand-hand pairs satisfy action-reaction (F_A + F_B = 0).
        Hand-hand pairs do not contribute to club wrench.
        Hand wrenches sum to net club wrench.
    """
    if not isinstance(topology, ContactTopology):
        raise TypeError(
            f"topology must be ContactTopology, got {type(topology).__name__}"
        )
    if not isinstance(hands, Mapping):
        raise TypeError(f"hands must be a Mapping, got {type(hands).__name__}")
    if not isinstance(club, BushingState | RigidBodyState):
        raise TypeError(
            f"club must be BushingState or RigidBodyState, got {type(club).__name__}"
        )
    if not isinstance(law, ContactParameters):
        raise TypeError(f"law must be ContactParameters, got {type(law).__name__}")

    lead_hand = _resolve_hand_state(hands, "lead")
    trail_hand = _resolve_hand_state(hands, "trail")
    hand_states = {"lead": lead_hand, "trail": trail_hand}

    results: dict[str, TopologyPairResult] = {}
    f_club_total = np.zeros(3)
    m_club_total = np.zeros(3)
    hand_wrenches = {
        "lead": (np.zeros(3), np.zeros(3)),
        "trail": (np.zeros(3), np.zeros(3)),
    }

    r_grip = topology.grip_radius_m

    for pair in topology.pairs:
        site_a = topology.site(pair.site_a)
        st_a = hand_states[site_a.hand]

        # Calculate patch radius
        if patch_radius_m is None:
            r_eff = 1.0 / (1.0 / site_a.radius_m + 1.0 / r_grip)
            patch = math.sqrt(2.0 * r_eff * DEFAULT_PRELOAD_M)
        else:
            patch = patch_radius_m

        if pair.is_hand_grip:
            if site_a.primitive == "sphere":
                pen, fn, ft, fc, mc, cp, n = _evaluate_hand_grip_sphere(
                    site_a, st_a, club, r_grip, law, patch, spin_transition_rad_s
                )
            else:
                pen, fn, ft, fc, mc, cp, n = _evaluate_hand_grip_capsule(
                    site_a, st_a, club, r_grip, law, patch, spin_transition_rad_s
                )

            res = TopologyPairResult(
                pair_name=pair.name,
                is_hand_grip=True,
                is_hand_hand=False,
                penetration_m=pen,
                normal_force_mag_n=fn,
                tangential_force_mag_n=ft,
                force_club_n=fc,
                moment_club_nm=mc,
                force_a_n=-fc,
                force_b_n=fc,
                contact_point_m=cp,
                normal=n,
                allocated_label=pair.allocated_label,
            )
            results[pair.name] = res
            f_club_total += fc
            m_club_total += mc
            hf, hm = hand_wrenches[site_a.hand]
            hand_wrenches[site_a.hand] = (hf + fc, hm + mc)
        else:
            site_b = topology.site(pair.site_b)
            st_b = hand_states[site_b.hand]
            res = _evaluate_hand_hand_pair(
                pair, site_a, site_b, st_a, st_b, law, patch, spin_transition_rad_s
            )
            results[pair.name] = res

    # Map L and R aliases for hand wrenches
    hand_wrenches["L"] = hand_wrenches["lead"]
    hand_wrenches["R"] = hand_wrenches["trail"]

    return results, (f_club_total, m_club_total), hand_wrenches


class TopologyClubInHands(bushing.ClubOnBushings):
    """Free-flyer club model driven by ContactTopology pairwise forces."""

    def __init__(
        self,
        spec: dict[str, Any],
        names: list[str],
        interface: GripInterface,
        topology: ContactTopology,
        law: ContactParameters | None = None,
    ) -> None:
        super().__init__(spec, names, interface)
        if not isinstance(topology, ContactTopology):
            raise TypeError(
                f"topology must be ContactTopology, got {type(topology).__name__}"
            )
        self.topology = topology
        self.law = law or default_contact_parameters()
        self.last_pair_results: dict[str, TopologyPairResult] = {}

    def wrenches(  # type: ignore[override]
        self, t: float, y: np.ndarray, hand_source: Any
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, TopologyPairResult],
        tuple[np.ndarray, np.ndarray],
    ]:
        """Evaluate hand frames, club frames, pair results, and net wrench at (t, y)."""
        q, qdot = hand_source(t)
        hands = hand_frame_states(self.kinematics.state(q, qdot), self.interface)
        club = self.club_state(y)
        frames = {s: club.frame(self._offsets[s]) for s in SIDES}
        pair_res, club_wrench, hand_w = evaluate_topology_wrenches(
            self.topology, hands, club, self.law
        )
        self.last_pair_results = pair_res
        return hands, frames, pair_res, club_wrench

    def joint_torque_from_wrench(
        self, club: RigidBodyState, club_wrench: tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """Convert net club wrench in world axes to club body-frame joint torque."""
        force, moment = club_wrench
        rot_t = club.rotation.T
        return np.concatenate([rot_t @ force, rot_t @ moment])

    def derivatives(self, t: float, y: np.ndarray, hand_source: Any) -> np.ndarray:
        """Compute state derivative d/dt [q, v] using Pinocchio aba for club."""
        _, _, _, club_wrench = self.wrenches(t, y, hand_source)
        club = self.club_state(y)
        tau = self.joint_torque_from_wrench(club, club_wrench)
        quat = y[3:7] / np.linalg.norm(y[3:7])
        q = np.concatenate([y[:3], quat])
        v = y[7:13]
        accel = np.array(self._pin.aba(self.model, self.data, q, v, tau))
        omega = v[3:]
        qv, qw = quat[:3], quat[3]
        quat_dot = 0.5 * np.concatenate(
            [qw * omega + cross3(qv, omega), [-float(qv @ omega)]]
        )
        return np.concatenate([club.rotation @ v[:3], quat_dot, accel])


def hold_state(sim: TopologyClubInHands, q: np.ndarray) -> tuple[Any, np.ndarray]:
    """Return hand source at rest and club initial state for static hold."""
    still = np.zeros_like(q)
    weld = sim.kinematics.state(q, still)
    y = bushing._initial_state(weld, sim._pin)  # noqa: SLF001
    return (lambda _t: (q, still)), y


def simulate_topology_contact(
    spec_bytes: bytes,
    swing: CoordinateSwing,
    topology: ContactTopology | None = None,
    law: ContactParameters | None = None,
    interface: GripInterface | None = None,
    tolerances: IntegratorTolerances | None = None,
    t_end_s: float | None = None,
    hold: bool = False,
) -> TopologyContactRun:
    """Integrate free-flyer club under ContactTopology contact law.

    Raises:
        RuntimeError: If numerical integration fails.
    """
    tol = tolerances or IntegratorTolerances()
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    topo = topology or contact_topology()
    contact_law = law or default_contact_parameters()
    sim = TopologyClubInHands(spec, list(swing.names), interface, topo, contact_law)

    times = swing.time_s
    if t_end_s is not None:
        times = times[times <= t_end_s + 1e-12]

    if hold:
        source, y0 = hold_state(sim, np.asarray(swing.q[0], float))
    else:
        spline = CoordinateSpline(swing.time_s, swing.q)
        source = spline.evaluate
        weld0 = sim.kinematics.state(*source(float(times[0])))
        y0 = bushing._initial_state(weld0, sim._pin)  # noqa: SLF001
        y0[7:10] = weld0.rotation.T @ weld0.velocity_m_s
        y0[10:13] = weld0.rotation.T @ weld0.omega_rad_s

    sol = solve_ivp(
        lambda t, y: sim.derivatives(t, y, source),
        (float(times[0]), float(times[-1])),
        y0,
        method="Radau",
        t_eval=times,
        rtol=tol.rtol,
        atol=tol.atol,
        max_step=tol.max_step_s,
    )
    if not sol.success:
        raise RuntimeError(
            f"pinocchio topology contact integration failed: {sol.message}"
        )

    wrench: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    hand_pose: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    club_pose: dict[str, tuple[list, list]] = {s: ([], []) for s in SIDES}
    club_rot = []

    pair_traces_list: dict[str, list] = {p.name: [] for p in topo.pairs}
    pair_normal_list: dict[str, list] = {p.name: [] for p in topo.pairs}
    for p in topo.pairs:
        if p.is_hand_hand:
            pair_traces_list[p.allocated_label] = []

    for k, t in enumerate(times):
        y = sol.y[:, k]
        hands, frames, pair_res, _ = sim.wrenches(float(t), y, source)
        for s in SIDES:
            hand_role = "lead" if s == "L" else "trail"
            hw_f = np.zeros(3)
            hw_m = np.zeros(3)
            for p in topo.hand_grip_pairs():
                if topo.site(p.site_a).hand == hand_role:
                    hw_f += pair_res[p.name].force_club_n
                    hw_m += pair_res[p.name].moment_club_nm
            wrench[s][0].append(hw_f)
            wrench[s][1].append(hw_m)
            hand_pose[s][0].append(hands[s].rotation)
            hand_pose[s][1].append(hands[s].position_m)
            club_pose[s][0].append(frames[s].rotation)
            club_pose[s][1].append(frames[s].position_m)

        for p in topo.pairs:
            pres = pair_res[p.name]
            pair_normal_list[p.name].append(pres.normal_force_mag_n)
            if p.is_hand_grip:
                pair_traces_list[p.name].append(pres.force_club_n)
            else:
                pair_traces_list[p.name].append(pres.force_a_n)
                pair_traces_list[p.allocated_label].append(pres.force_a_n)

        club_rot.append(sim.club_state(y).rotation)

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
            "integrator": "SciPy solve_ivp Radau, error controlled",
            "rtol": tol.rtol,
            "atol": tol.atol,
            "rhs_evaluations": int(sol.nfev),
            "force_law": "ContactTopology pairwise contact, Pinocchio aba",
            "style": topo.style.value,
        },
    )
    axial, roll = slip_from_frames(arrays["h"], arrays["c"])

    normal_force_per_side: dict[str, np.ndarray] = {}
    for s in SIDES:
        hand_role = "lead" if s == "L" else "trail"
        side_pairs = [
            p.name
            for p in topo.hand_grip_pairs()
            if topo.site(p.site_a).hand == hand_role
        ]
        if side_pairs:
            normal_force_per_side[s] = np.column_stack(
                [np.array(pair_normal_list[p]) for p in side_pairs]
            )
        else:
            normal_force_per_side[s] = np.zeros((len(times), 0))

    pair_traces_arr = {k: np.array(v) for k, v in pair_traces_list.items()}
    pair_normal_arr = {k: np.array(v) for k, v in pair_normal_list.items()}

    return TopologyContactRun(
        series=series,
        normal_force_n=normal_force_per_side,
        roll_slip_rad=roll,
        axial_slip_m=axial,
        pair_traces=pair_traces_arr,
        pair_normal_force_n=pair_normal_arr,
    )
