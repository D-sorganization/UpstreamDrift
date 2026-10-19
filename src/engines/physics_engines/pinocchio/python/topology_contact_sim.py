"""Pinocchio grip-contact adapter from the shared topology (HRG-GS-G6, #13304).

Pinocchio has no native contact dynamics for this problem, so the contact
forces are built from :class:`ContactTopology` and the shared G2/G6 contact law
(:mod:`src.shared.python.grip_contact.pair_contact` and
:mod:`src.shared.python.grip_contact.topology_contact`).

The club is a free-flyer model evaluated via Pinocchio ``aba`` in
:mod:`src.engines.physics_engines.pinocchio.python.grip_contact_sim`. Forward
kinematics of the weld model provides the hand frames, and sites are mapped into
each hand's local grip frame via :func:`build_topology_model`. Integration is
performed via SciPy ``solve_ivp`` (Radau).

Hand-grip pairs transmit physical forces to the club.
Hand-hand pairs are allocated / diagnostic only under the welded-kinematic
contact assumption: they exert action-reaction forces between hands and are
tracked in allocated channels, with zero contribution to the club acceleration.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.engines.physics_engines.pinocchio.python import grip_contact_sim
from src.engines.physics_engines.pinocchio.python.grip_contact_sim import (
    DEFAULT_ATOL,
    DEFAULT_MAX_STEP_S,
    DEFAULT_RTOL,
    IntegratorTolerances,
)
from src.shared.python.grip_contact import (
    CoordinateSwing,
    GripInterface,
    RigidBodyState,
)
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_run import ContactRun
from src.shared.python.grip_contact.contact_topology import (
    ContactPair,
    ContactSite,
    ContactTopology,
    contact_topology,
)
from src.shared.python.grip_contact.pair_contact import (
    ContactCapsule,
    ContactSphere,
    PairWrench,
    capsule_capsule,
    capsule_sphere,
    sphere_capsule,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    topology_hand_hand_wrenches,
    topology_side_wrench,
)
from src.shared.python.motion_matching.contact_law import ContactParameters

ENGINE = "pinocchio_topology_contact"
DEFAULT_PRELOAD_M = 2.5e-4
DEFAULT_SQUEEZE_PER_HAND_N = 1100.0

__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_MAX_STEP_S",
    "DEFAULT_PRELOAD_M",
    "DEFAULT_RTOL",
    "DEFAULT_SQUEEZE_PER_HAND_N",
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
        moment_club_nm: Moment exerted on club about club origin (N*m).
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
    p0, p1 = site.endpoints_m
    return ContactCapsule(
        p0_m=np.asarray(p0, dtype=float),
        p1_m=np.asarray(p1, dtype=float),
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


def _as_bushing_state(state: BushingState | RigidBodyState) -> BushingState:
    """Convert a BushingState or RigidBodyState into a BushingState."""
    if isinstance(state, BushingState):
        return state
    return BushingState(
        rotation=state.rotation,
        position_m=state.position_m,
        velocity_m_s=state.velocity_m_s,
        omega_rad_s=state.omega_rad_s,
    )


def _evaluate_pair_primitives(
    prim_a: ContactSphere | ContactCapsule,
    prim_b: ContactSphere | ContactCapsule,
    st_a: BushingState,
    st_b: BushingState,
    law: ContactParameters,
    patch_radius_m: float,
    spin_transition_rad_s: float,
) -> PairWrench:
    """Dispatch primitive pair contact to shared G2 pair_contact kernels."""
    if isinstance(prim_a, ContactSphere) and isinstance(prim_b, ContactCapsule):
        return sphere_capsule(
            prim_a, prim_b, st_a, st_b, law, patch_radius_m, spin_transition_rad_s
        )
    if isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactSphere):
        return capsule_sphere(
            prim_a, prim_b, st_a, st_b, law, patch_radius_m, spin_transition_rad_s
        )
    if isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactCapsule):
        return capsule_capsule(
            prim_a, prim_b, st_a, st_b, law, patch_radius_m, spin_transition_rad_s
        )
    raise ValueError(
        f"Unsupported primitive pair: {type(prim_a).__name__}, {type(prim_b).__name__}"
    )


def _pair_result_from_wrench(
    pair: ContactPair, wrench: PairWrench
) -> TopologyPairResult:
    """Build TopologyPairResult from a ContactPair and shared PairWrench."""
    is_hg = pair.is_hand_grip
    return TopologyPairResult(
        pair_name=pair.name,
        is_hand_grip=is_hg,
        is_hand_hand=pair.is_hand_hand,
        penetration_m=wrench.penetration_m,
        normal_force_mag_n=wrench.normal_force_mag_n,
        tangential_force_mag_n=wrench.tangential_force_mag_n,
        force_club_n=wrench.force_b_n if is_hg else np.zeros(3),
        moment_club_nm=wrench.moment_b_nm if is_hg else np.zeros(3),
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
        club is a BushingState or RigidBodyState.
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
    club_state = _as_bushing_state(club)

    results: dict[str, TopologyPairResult] = {}
    f_club_total = np.zeros(3)
    m_club_total = np.zeros(3)
    hand_wrenches = {
        "lead": (np.zeros(3), np.zeros(3)),
        "trail": (np.zeros(3), np.zeros(3)),
    }

    r_grip = topology.grip_radius_m
    grip_cap = ContactCapsule(
        p0_m=np.array([-0.1, 0.0, 0.0]),
        p1_m=np.array([1.0, 0.0, 0.0]),
        radius_m=r_grip,
    )

    for pair in topology.pairs:
        site_a = topology.site(pair.site_a)
        if patch_radius_m is None:
            r_eff = 1.0 / (1.0 / site_a.radius_m + 1.0 / r_grip)
            patch = math.sqrt(2.0 * r_eff * DEFAULT_PRELOAD_M)
        else:
            patch = patch_radius_m

        prim_a = _site_primitive(site_a)
        st_a = hand_states[site_a.hand]
        if pair.is_hand_grip:
            wrench = _evaluate_pair_primitives(
                prim_a,
                grip_cap,
                st_a,
                club_state,
                law,
                patch,
                spin_transition_rad_s,
            )
        else:
            site_b = topology.site(pair.site_b)
            prim_b = _site_primitive(site_b)
            st_b = hand_states[site_b.hand]
            wrench = _evaluate_pair_primitives(
                prim_a,
                prim_b,
                st_a,
                st_b,
                law,
                patch,
                spin_transition_rad_s,
            )

        res = _pair_result_from_wrench(pair, wrench)
        results[pair.name] = res

        if pair.is_hand_grip:
            f_club_total += res.force_club_n
            m_club_total += res.moment_club_nm
            hf, hm = hand_wrenches[site_a.hand]
            hand_wrenches[site_a.hand] = (
                hf + res.force_club_n,
                hm + res.moment_club_nm,
            )

    hand_wrenches["L"] = hand_wrenches["lead"]
    hand_wrenches["R"] = hand_wrenches["trail"]

    return results, (f_club_total, m_club_total), hand_wrenches


class TopologyClubInHands(grip_contact_sim.ClubInHands):
    """Free-flyer club model driven by ContactTopology pairwise forces.

    Consolidates onto :class:`grip_contact_sim.ClubInHands` and
    :class:`TopologyContactModel` so hand sites are mapped into each hand's
    grip frame and evaluated against the club grip frames with memoised hand FK
    and the batched G6 contact law.
    """

    def __init__(
        self,
        spec: dict[str, Any],
        names: list[str],
        interface: GripInterface,
        topology: ContactTopology,
        law: ContactParameters | None = None,
    ) -> None:
        if not isinstance(topology, ContactTopology):
            raise TypeError(
                f"topology must be ContactTopology, got {type(topology).__name__}"
            )
        topo_model = build_topology_model(
            interface, topology, squeeze_per_hand_n=DEFAULT_SQUEEZE_PER_HAND_N
        )
        if law is not None:
            if not isinstance(law, ContactParameters):
                raise TypeError(
                    f"law must be ContactParameters, got {type(law).__name__}"
                )
            topo_model = replace(topo_model, law=law)
        super().__init__(spec, names, interface, topology_model=topo_model)
        self.topology = topology
        self.law = topo_model.law
        self.last_pair_results: dict[str, TopologyPairResult] = {}


def _extract_pair_traces(
    base_run: ContactRun,
    topo_model: TopologyContactModel,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build per-pair 3D force traces and normal-force traces from a ContactRun."""
    topo = topo_model.topology
    n_steps = len(base_run.series.time_s)
    pair_traces_list: dict[str, list[np.ndarray]] = {p.name: [] for p in topo.pairs}
    for p in topo.hand_hand_pairs():
        pair_traces_list[p.allocated_label] = []

    pair_normal: dict[str, np.ndarray] = {}
    if base_run.pair_normal_force_n is not None:
        for p in topo.hand_grip_pairs():
            pair_normal[p.name] = np.asarray(
                base_run.pair_normal_force_n[p.name], dtype=float
            )

    hh_normal_list: dict[str, list[float]] = {
        p.name: [] for p in topo.hand_hand_pairs()
    }
    hand_states = base_run.hand_states or {"L": [], "R": []}
    club_states = base_run.club_states or {"L": [], "R": []}

    for k in range(n_steps):
        h_l = hand_states["L"][k]
        h_r = hand_states["R"][k]
        c_l = club_states["L"][k]
        c_r = club_states["R"][k]
        for side, h_s, c_s in (("L", h_l, c_l), ("R", h_r, c_r)):
            sw = topology_side_wrench(topo_model, side, h_s, c_s)
            for pname, pw in sw.pair_wrenches.items():
                pair_traces_list[pname].append(pw.force_b_n)
        hh = topology_hand_hand_wrenches(topo_model, h_l, h_r, mode="separated_v2")
        for p in topo.hand_hand_pairs():
            pw = hh.pairs[p.name]
            hh_normal_list[p.name].append(float(pw.normal_force_mag_n))
            pair_traces_list[p.name].append(pw.force_a_n)
            pair_traces_list[p.allocated_label].append(pw.force_a_n)

    for p in topo.hand_hand_pairs():
        pair_normal[p.name] = np.asarray(hh_normal_list[p.name], dtype=float)

    pair_traces = {k: np.asarray(v, dtype=float) for k, v in pair_traces_list.items()}
    return pair_traces, pair_normal


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

    Delegates to :func:`grip_contact_sim.simulate_grip_contact` with a
    :class:`TopologyContactModel` matched to ``interface`` and ``topology``.

    Raises:
        RuntimeError: If numerical integration fails.
    """
    spec = json.loads(spec_bytes)
    interface = interface or GripInterface.from_spec(spec)
    topo = topology or contact_topology()
    topo_model = build_topology_model(
        interface, topo, squeeze_per_hand_n=DEFAULT_SQUEEZE_PER_HAND_N
    )
    if law is not None:
        if not isinstance(law, ContactParameters):
            raise TypeError(f"law must be ContactParameters, got {type(law).__name__}")
        topo_model = replace(topo_model, law=law)

    base_run = grip_contact_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        tolerances=tolerances,
        t_end_s=t_end_s,
        hold=hold,
        topology_model=topo_model,
    )
    traces_arr, normals_arr = _extract_pair_traces(base_run, topo_model)

    return TopologyContactRun(
        series=base_run.series,
        normal_force_n=base_run.normal_force_n,
        roll_slip_rad=base_run.roll_slip_rad,
        axial_slip_m=base_run.axial_slip_m,
        tangential_force_n=base_run.tangential_force_n,
        pair_normal_force_n=normals_arr,
        pair_tangential_force_n=base_run.pair_tangential_force_n,
        channel_reasons=base_run.channel_reasons,
        hand_states=base_run.hand_states,
        club_states=base_run.club_states,
        pair_contact_point_drift_m=base_run.pair_contact_point_drift_m,
        pair_traces=traces_arr,
    )
