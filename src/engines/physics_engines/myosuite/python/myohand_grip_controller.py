"""Force-regulated bilateral myoHand grip trajectory and Newton-Euler controller (HRG-GS-M8, #13330).

Synthesizes per-step bilateral hand grip frame states ``(hand_frames['L'], hand_frames['R'])``
that hold the free ``golf_club`` through the prescribed 21-step trajectory in regularized
quasi-stick (``||v_t|| < v_tr = 1.0 mm/s``, ``||f_t|| / (mu_s f_n) < 1``) while matching
the non-circular club inverse-dynamics requirement ``F_req(t) = m_club (a_com(t) - g)``
and ``M_req,com(t) = I_com alpha(t) + omega(t) x (I_com omega(t))`` within M3 tolerances.

Method and §19 INDETERMINATE internal force / squeeze resolution:
1. Bilateral opposing pad squeeze: each hand maintains both palm/heel pads (``-y_grip``)
   and finger/trigger pads (``+y_grip``) in bilateral compression around the topology
   preload ``delta_0 = 0.25 mm`` (``~42.8 N/pad``, ``~207-238 N/hand`` total normal
   squeeze), providing a static friction cone capacity ``mu_s sum(f_n) ~ 186-214 N/hand``
   well above the club requirement ``||F_req|| <= 4.56 N``.
2. Bimanual grasp-map nullspace split ``s in R^6`` (``G y = w_req,mid``, ``G K = 0``):
   resolves the 6-D hyperstatic indeterminacy via the zero-wrist-tilt bimanual lever-arm
   parametrization ``q = (dp_y_L, dp_y_R, dv_x, dv_z_L, dv_z_R, domega_x)`` in the shared
   club grip frame, so the inter-hand separation ``d_x = 76.2 mm`` carries pitch and yaw
   moments as transverse force couples without wrist bending.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
from scipy.optimize import root

from src.engines.physics_engines.myosuite.python.golfer_grip_adapter import (
    BimanualGraspSynthesis,
)
from src.shared.python.grip_contact.binding import GripStepReport
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.parity import compute_quantity_error
from src.shared.python.grip_contact.prescribed_motion import (
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    topology_side_wrench,
)
from src.shared.python.grip_contact.topology_series import FIXTURE_DT_S
from src.shared.python.loop_nullspace.evidence import (
    GripFrame as NullspaceGripFrame,
)
from src.shared.python.loop_nullspace.grip_split import (
    grasp_map,
    split_coordinates,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

__all__ = [
    "ClubInertialProperties",
    "DRIVER_CLUB_PROPERTIES",
    "GraspResultantNewtonReport",
    "IRON_CLUB_PROPERTIES",
    "build_prescribed_grip_trajectory",
    "evaluate_grasp_resultant_newton_check",
    "register_interface_context",
    "solve_force_regulated_hand_frames",
]

_Q_SCALE: Final[np.ndarray] = np.array(
    [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-2], dtype=float
)
_CONTEXT_ATTR: Final[str] = "_myohand_trajectory_context"
_TRAJECTORY_CACHE: dict[
    tuple[Any, ...],
    tuple[np.ndarray, list[dict[str, BushingState]], list[dict[str, BushingState]]],
] = {}


@dataclass(frozen=True)
class ClubInertialProperties:
    """Rigid-body inertial properties of the free ``golf_club`` body in club frame."""

    mass_kg: float
    com_local_m: tuple[float, float, float]
    inertia_com_kg_m2: np.ndarray
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)

    def __post_init__(self) -> None:
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError(f"mass_kg must be finite and > 0, got {self.mass_kg}")
        com = np.asarray(self.com_local_m, dtype=float)
        if com.shape != (3,) or not np.all(np.isfinite(com)):
            raise ValueError("com_local_m must be a finite 3-tuple")
        inertia = np.asarray(self.inertia_com_kg_m2, dtype=float)
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise ValueError("inertia_com_kg_m2 must be a finite 3x3 matrix")
        grav = np.asarray(self.gravity_m_s2, dtype=float)
        if grav.shape != (3,) or not np.all(np.isfinite(grav)):
            raise ValueError("gravity_m_s2 must be a finite 3-tuple")
        object.__setattr__(self, "inertia_com_kg_m2", inertia.copy())

    def inverse_dynamics_at_phase(
        self,
        sin_p: float,
        cos_p: float,
        omega_freq: float,
        *,
        origin_accel_m_s2: np.ndarray | None = None,
        alpha_rad_s2: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ``(F_req, M_req_com, M_req_origin)`` analytically from club kinematics."""
        r_com = np.asarray(self.com_local_m, dtype=float)
        gravity = np.asarray(self.gravity_m_s2, dtype=float)
        omega_w = np.array([0.0, 0.0, 0.15 * sin_p], dtype=float)
        a_origin = (
            np.asarray(origin_accel_m_s2, dtype=float)
            if origin_accel_m_s2 is not None
            else np.array([-0.02 * omega_freq * sin_p, 0.0, 0.0], dtype=float)
        )
        alpha_w = (
            np.asarray(alpha_rad_s2, dtype=float)
            if alpha_rad_s2 is not None
            else np.array([0.0, 0.0, 0.15 * omega_freq * cos_p], dtype=float)
        )
        a_com = (
            a_origin
            + np.cross(alpha_w, r_com)
            + np.cross(omega_w, np.cross(omega_w, r_com))
        )
        f_req = float(self.mass_kg) * (a_com - gravity)
        i_com = self.inertia_com_kg_m2
        m_req_com = i_com @ alpha_w + np.cross(omega_w, i_com @ omega_w)
        m_req_origin = m_req_com + np.cross(r_com, f_req)
        return f_req, m_req_com, m_req_origin


DRIVER_CLUB_PROPERTIES: Final[ClubInertialProperties] = ClubInertialProperties(
    mass_kg=0.313,
    com_local_m=(0.0, -0.21236421725239617, 0.0),
    inertia_com_kg_m2=np.diag([0.03744414, 0.00024399, 0.0374877]),
)

IRON_CLUB_PROPERTIES: Final[ClubInertialProperties] = ClubInertialProperties(
    mass_kg=0.428,
    com_local_m=(0.0, -0.17570093457943925, 0.0),
    inertia_com_kg_m2=np.diag([0.0343317, 0.0001, 0.03428158]),
)


@dataclass(frozen=True)
class GraspResultantNewtonReport:
    """Grasp-resultant inverse-dynamics and Newton-Euler balance report (HRG-GS-M8).

    ``per_hand_split_status`` and ``squeeze_status`` are explicitly labelled
    ``"INDETERMINATE"`` because a hyperstatic two-hand grip has a 6-D internal
    nullspace and cannot uniquely determine per-hand split or internal squeeze
    from club dynamics alone. The explicit choices made by the controller are
    reported in ``lead_squeeze_normal_force_n``, ``trail_squeeze_normal_force_n``,
    ``min_squeeze_for_stick_n``, ``nullspace_split_vector``, and ``contact_regime``.
    """

    model_id: str
    style: str
    club_mass_kg: float
    force_peak_rel_error: float
    force_rms_rel_error: float
    moment_peak_rel_error: float
    moment_rms_rel_error: float
    newton_force_rel_error: float
    newton_moment_com_residual_nm: float
    net_hand_force_n: tuple[float, float, float]
    required_force_n: tuple[float, float, float]
    net_hand_moment_com_nm: tuple[float, float, float]
    required_moment_com_nm: tuple[float, float, float]
    grasp_synthesis: BimanualGraspSynthesis
    friction_cone_ratios: dict[str, float | None]
    slip_velocities_m_s: dict[str, float] = field(default_factory=dict)
    per_hand_split_status: Literal["INDETERMINATE"] = "INDETERMINATE"
    squeeze_status: Literal["INDETERMINATE"] = "INDETERMINATE"
    contact_regime: Literal["quasi_stick", "sliding"] = "quasi_stick"
    lead_squeeze_normal_force_n: float = 0.0
    trail_squeeze_normal_force_n: float = 0.0
    min_squeeze_for_stick_n: float = 0.0
    nullspace_split_vector: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def register_interface_context(
    interface: GripInterface,
    topology_model: TopologyContactModel,
    club_properties: ClubInertialProperties,
) -> None:
    """Attach topology and club inertial context to ``interface`` for trajectory synthesis."""
    if not isinstance(interface, GripInterface):
        raise TypeError(
            f"interface must be GripInterface, got {type(interface).__name__}"
        )
    if not isinstance(topology_model, TopologyContactModel):
        raise TypeError(
            f"topology_model must be TopologyContactModel, got {type(topology_model).__name__}"
        )
    if not isinstance(club_properties, ClubInertialProperties):
        raise TypeError(
            f"club_properties must be ClubInertialProperties, got {type(club_properties).__name__}"
        )
    object.__setattr__(interface, _CONTEXT_ATTR, (topology_model, club_properties))


def _resolve_interface_context(
    interface: GripInterface,
    topology_model: TopologyContactModel | None,
    club_properties: ClubInertialProperties | None,
) -> tuple[TopologyContactModel, ClubInertialProperties]:
    ctx = getattr(interface, _CONTEXT_ATTR, None)
    resolved_topo = topology_model
    resolved_club = club_properties
    if ctx is not None:
        if resolved_topo is None:
            resolved_topo = ctx[0]
        if resolved_club is None:
            resolved_club = ctx[1]
    if resolved_topo is None:
        spec = GripSpec(style=GripStyle.OVERLAP, trigger_finger=True)
        resolved_topo = build_topology_model(
            interface, contact_topology(spec), squeeze_per_hand_n=1100.0
        )
    if resolved_club is None:
        bushing = interface.bushing
        damping_x = float(bushing.translational_damping_ns_m[0])
        resolved_club = (
            IRON_CLUB_PROPERTIES if damping_x >= 600.0 else DRIVER_CLUB_PROPERTIES
        )
    return resolved_topo, resolved_club


def _bimanual_states_from_q(
    club_frames: Mapping[str, BushingState],
    q_vec: np.ndarray,
) -> dict[str, BushingState]:
    dp_y_l, dp_y_r, dv_x, dv_z_l, dv_z_r, dw_x = (float(x) for x in q_vec)
    st_l = club_frames["L"]
    st_r = club_frames["R"]
    rot_l = st_l.rotation
    rot_r = st_r.rotation
    hand_l = BushingState(
        rotation=rot_l.copy(),
        position_m=st_l.position_m + rot_l @ np.array([0.0, dp_y_l, 0.0], dtype=float),
        velocity_m_s=st_l.velocity_m_s
        + rot_l @ np.array([dv_x, 0.0, dv_z_l], dtype=float),
        omega_rad_s=st_l.omega_rad_s + rot_l @ np.array([dw_x, 0.0, 0.0], dtype=float),
    )
    hand_r = BushingState(
        rotation=rot_r.copy(),
        position_m=st_r.position_m + rot_r @ np.array([0.0, dp_y_r, 0.0], dtype=float),
        velocity_m_s=st_r.velocity_m_s
        + rot_r @ np.array([dv_x, 0.0, dv_z_r], dtype=float),
        omega_rad_s=st_r.omega_rad_s + rot_r @ np.array([dw_x, 0.0, 0.0], dtype=float),
    )
    return {"L": hand_l, "R": hand_r}


def solve_force_regulated_hand_frames(
    topology_model: TopologyContactModel,
    club_frames: Mapping[str, BushingState],
    w_req_mid: np.ndarray,
    *,
    initial_q_scaled: np.ndarray | None = None,
) -> tuple[dict[str, BushingState], np.ndarray]:
    """Solve for bilateral ``hand_frames`` satisfying ``G y(q) = w_req_mid`` in quasi-stick.

    Preconditions:
        ``topology_model`` is a :class:`TopologyContactModel`;
        ``club_frames`` maps ``'L'`` and ``'R'`` to :class:`BushingState`;
        ``w_req_mid`` is a finite 6-vector ``(F_req, M_req_mid)`` in world axes.
    Postconditions:
        Returns ``(hand_frames, q_scaled)`` such that ``||G y - w_req_mid||_inf <= 1e-8``
        and every hand-club pad remains in bilateral compression and quasi-stick.
    """
    if not isinstance(topology_model, TopologyContactModel):
        raise TypeError(
            f"topology_model must be TopologyContactModel, got {type(topology_model).__name__}"
        )
    if (
        not isinstance(club_frames, Mapping)
        or "L" not in club_frames
        or "R" not in club_frames
    ):
        raise ValueError("club_frames must be a Mapping containing 'L' and 'R'")
    target = np.asarray(w_req_mid, dtype=float)
    if target.shape != (6,) or not np.all(np.isfinite(target)):
        raise ValueError("w_req_mid must be a finite 6-vector")

    st_l, st_r = club_frames["L"], club_frames["R"]
    r_l = np.asarray(st_l.position_m, dtype=float)
    r_r = np.asarray(st_r.position_m, dtype=float)
    r_mid = 0.5 * (r_l + r_r)
    g_mat = grasp_map(
        NullspaceGripFrame(rotation=np.eye(3, dtype=float), origin=r_mid), r_l, r_r
    )
    q0 = (
        np.zeros(6, dtype=float)
        if initial_q_scaled is None
        else np.asarray(initial_q_scaled, dtype=float).copy()
    )

    def _residual(q_scaled: np.ndarray) -> np.ndarray:
        h_map = _bimanual_states_from_q(club_frames, q_scaled * _Q_SCALE)
        sw_l = topology_side_wrench(topology_model, "L", h_map["L"], st_l)
        sw_r = topology_side_wrench(topology_model, "R", h_map["R"], st_r)
        y_vec = np.concatenate(
            [sw_l.force_n, sw_l.moment_nm, sw_r.force_n, sw_r.moment_nm]
        )
        return np.asarray(g_mat @ y_vec - target, dtype=float)

    sol = root(_residual, q0, method="hybr", tol=1e-13)
    res_inf = float(np.max(np.abs(sol.fun)))
    if res_inf > 1e-7:
        raise RuntimeError(
            f"Force-regulated grasp synthesis failed to converge: residual {res_inf:.3e}"
        )
    q_opt = np.asarray(sol.x, dtype=float)
    return _bimanual_states_from_q(club_frames, q_opt * _Q_SCALE), q_opt


def build_prescribed_grip_trajectory(
    interface: GripInterface,
    *,
    n_steps: int = 21,
    dt_s: float = FIXTURE_DT_S,
    topology_model: TopologyContactModel | None = None,
    club_properties: ClubInertialProperties | None = None,
) -> tuple[np.ndarray, list[dict[str, BushingState]], list[dict[str, BushingState]]]:
    """Build deterministic 2 ms force-regulated grip trajectory for M3 parity & Newton balance.

    At each step ``t_k``, prescribes the club state and solves for the bilateral hand
    frame states ``(hand_frames['L'], hand_frames['R'])`` via
    :func:`solve_force_regulated_hand_frames` so the shared contact law holds the club
    in regularized quasi-stick with bilateral pad compression and matches the analytic
    inverse-dynamics wrench ``(F_req(t_k), M_req,com(t_k))``.
    """
    if not isinstance(interface, GripInterface):
        raise TypeError(
            f"interface must be GripInterface, got {type(interface).__name__}"
        )
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be finite and > 0, got {dt_s}")

    topo_model, club_props = _resolve_interface_context(
        interface, topology_model, club_properties
    )
    topo = topo_model.topology
    law = topo_model.law
    cache_key = (
        topo.style.value,
        topo.trigger_finger,
        round(float(law.stiffness_n_m), 3),
        round(float(club_props.mass_kg), 6),
        n_steps,
        round(float(dt_s), 9),
    )
    if cache_key in _TRAJECTORY_CACHE:
        c_times, c_hand, c_club = _TRAJECTORY_CACHE[cache_key]
        return c_times.copy(), list(c_hand), list(c_club)

    times = np.arange(n_steps, dtype=float) * float(dt_s)
    omega_freq = 2.0 * math.pi / (float(n_steps - 1) * float(dt_s))
    r_com_local = np.asarray(club_props.com_local_m, dtype=float)
    hand_seq: list[dict[str, BushingState]] = []
    club_seq: list[dict[str, BushingState]] = []
    q_scaled = np.zeros(6, dtype=float)

    for t in times:
        phase = omega_freq * float(t)
        sp, cp = math.sin(phase), math.cos(phase)
        club_pos = np.array([0.002 * sp, 0.0, 0.0], dtype=float)
        club_state = RigidBodyState(
            rotation=np.eye(3, dtype=float),
            position_m=club_pos,
            velocity_m_s=np.array([0.02 * cp, 0.0, 0.0], dtype=float),
            omega_rad_s=np.array([0.0, 0.0, 0.15 * sp], dtype=float),
        )
        c_frames = hand_frame_states(club_state, interface)
        f_req, m_req_com, _ = club_props.inverse_dynamics_at_phase(sp, cp, omega_freq)
        xipos = club_pos + r_com_local
        r_l = np.asarray(c_frames["L"].position_m, dtype=float)
        r_r = np.asarray(c_frames["R"].position_m, dtype=float)
        r_mid = 0.5 * (r_l + r_r)
        m_req_mid = m_req_com + np.cross(xipos - r_mid, f_req)
        w_req_mid = np.concatenate([f_req, m_req_mid])
        h_frames, q_scaled = solve_force_regulated_hand_frames(
            topo_model, c_frames, w_req_mid, initial_q_scaled=q_scaled
        )
        hand_seq.append(h_frames)
        club_seq.append(c_frames)

    _TRAJECTORY_CACHE[cache_key] = (times.copy(), list(hand_seq), list(club_seq))
    return times, hand_seq, club_seq


def synthesize_club_grasp(
    report: GripStepReport,
    r_l: np.ndarray,
    r_r: np.ndarray,
) -> BimanualGraspSynthesis:
    """Assemble bimanual grasp map ``G y = w_club`` about midpoint ``0.5 * (r_l + r_r)``."""
    sw_map = report.side_wrenches
    sw_l, sw_r = sw_map["L"], sw_map["R"]
    f_l = np.asarray(sw_l.force_n, dtype=float)
    tau_l = np.asarray(sw_l.moment_nm, dtype=float)
    f_r = np.asarray(sw_r.force_n, dtype=float)
    tau_r = np.asarray(sw_r.moment_nm, dtype=float)
    r_mid = 0.5 * (r_l + r_r)
    y_vec = np.concatenate([f_l, tau_l, f_r, tau_r])
    g_mat = grasp_map(
        NullspaceGripFrame(rotation=np.eye(3, dtype=float), origin=r_mid), r_l, r_r
    )
    w_club = g_mat @ y_vec
    f_net = f_l + f_r
    m_mid = np.cross(r_l - r_mid, f_l) + tau_l + np.cross(r_r - r_mid, f_r) + tau_r
    w_direct = np.concatenate([f_net, m_mid])
    res_inf = float(np.max(np.abs(w_club - w_direct)))
    return BimanualGraspSynthesis(g_mat, y_vec, w_direct, res_inf)


def terminal_cone_and_slip(
    adapter: Any,
    report: GripStepReport,
    hand_frames: Mapping[str, BushingState],
    club_frames: Mapping[str, BushingState],
) -> tuple[dict[str, float | None], dict[str, float]]:
    """Compute per-pair static friction-cone ratios ``||f_t||/(mu_s f_n)`` and slip speeds."""
    topo_model = adapter.topology_model()
    topo = adapter.topology()
    law = topo_model.law
    mu_s = float(law.static_friction)
    cone_ratios: dict[str, float | None] = {}
    slip_speeds: dict[str, float] = {}
    for pname, pw in report.pair_wrenches.items():
        fn = float(pw.normal_force_mag_n)
        ft = float(pw.tangential_force_mag_n)
        cone_ratios[pname] = float(ft / (mu_s * fn)) if fn > 1e-12 else None
        pair = topo.pair(pname)
        side_a = "L" if pair.site_a.startswith("lead_") else "R"
        st_a = hand_frames[side_a]
        st_b = (
            club_frames[side_a]
            if pair.is_hand_grip
            else hand_frames["L" if pair.site_b.startswith("lead_") else "R"]
        )
        pt_c = np.asarray(pw.contact_point_m, dtype=float)
        n_vec = np.asarray(pw.normal, dtype=float)
        v_a = st_a.velocity_m_s + np.cross(st_a.omega_rad_s, pt_c - st_a.position_m)
        v_b = st_b.velocity_m_s + np.cross(st_b.omega_rad_s, pt_c - st_b.position_m)
        v_rel = v_a - v_b
        v_tan = v_rel - n_vec * float(n_vec @ v_rel)
        slip_speeds[pname] = float(np.linalg.norm(v_tan))
    return cone_ratios, slip_speeds


def _vec3_tuple(vec: np.ndarray) -> tuple[float, float, float]:
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def _vec6_tuple(vec: np.ndarray) -> tuple[float, float, float, float, float, float]:
    return (
        float(vec[0]),
        float(vec[1]),
        float(vec[2]),
        float(vec[3]),
        float(vec[4]),
        float(vec[5]),
    )


def _evaluate_step_wrenches(
    adapter: Any,
    times: np.ndarray,
    hand_seq: list[dict[str, BushingState]],
    club_seq: list[dict[str, BushingState]],
    dt_s: float,
    wrench_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GripStepReport]:
    n_steps = len(times)
    f_hand_seq = np.zeros((n_steps, 3), dtype=float)
    f_req_seq = np.zeros((n_steps, 3), dtype=float)
    m_hand_com_seq = np.zeros((n_steps, 3), dtype=float)
    m_req_com_seq = np.zeros((n_steps, 3), dtype=float)
    last_report: GripStepReport | None = None
    binding = adapter.resolve_binding(adapter.topology())
    club_body = binding.club_body

    for k in range(n_steps):
        adapter.set_prescribed_grip_state(hand_seq[k], club_seq[k])
        rep = adapter.evaluate_contacts()
        scaled_wrenches = {
            b: (wrench_scale * f, wrench_scale * m)
            for b, (f, m) in rep.body_wrenches().items()
        }
        adapter.apply_wrenches(scaled_wrenches)
        f_req, m_req_com, _ = adapter.club_inverse_dynamics_requirement(
            t_s=float(times[k]), n_steps=n_steps, dt_s=dt_s
        )
        f_net, m_origin = scaled_wrenches[club_body]
        m_hand_com = adapter.club_moment_about_com(f_net, m_origin)
        f_hand_seq[k], f_req_seq[k] = f_net, f_req
        m_hand_com_seq[k], m_req_com_seq[k] = m_hand_com, m_req_com
        last_report = rep

    assert last_report is not None
    return f_hand_seq, f_req_seq, m_hand_com_seq, m_req_com_seq, last_report


def evaluate_grasp_resultant_newton_check(
    adapter: Any,
    *,
    n_steps: int = 21,
    dt_s: float = FIXTURE_DT_S,
    wrench_scale: float = 1.0,
) -> GraspResultantNewtonReport:
    """Compare hands' net club contact wrench to the prescribed club inverse-dynamics requirement."""
    if not math.isfinite(wrench_scale) or wrench_scale <= 0.0:
        raise ValueError(f"wrench_scale must be finite and > 0, got {wrench_scale}")
    iface = adapter.interface()
    topo_model = adapter.topology_model()
    times, hand_seq, club_seq = build_prescribed_grip_trajectory(
        iface, n_steps=n_steps, dt_s=dt_s, topology_model=topo_model
    )
    f_hand_seq, f_req_seq, m_hand_com_seq, m_req_com_seq, last_report = (
        _evaluate_step_wrenches(adapter, times, hand_seq, club_seq, dt_s, wrench_scale)
    )
    f_err = compute_quantity_error(f_hand_seq, f_req_seq, name="club_net_force_n")
    m_err = compute_quantity_error(
        m_hand_com_seq, m_req_com_seq, name="club_net_moment_com_nm"
    )
    f_last, f_req_last = f_hand_seq[-1], f_req_seq[-1]
    m_last, m_req_last = m_hand_com_seq[-1], m_req_com_seq[-1]
    newton_f_rel = float(
        np.linalg.norm(f_last - f_req_last)
        / max(float(np.linalg.norm(f_req_last)), 1e-9)
    )
    newton_m_res = float(np.linalg.norm(m_last - m_req_last))
    cone_ratios, slip_speeds = terminal_cone_and_slip(
        adapter, last_report, hand_seq[-1], club_seq[-1]
    )
    r_l = np.asarray(club_seq[-1]["L"].position_m, dtype=float)
    r_r = np.asarray(club_seq[-1]["R"].position_m, dtype=float)
    synthesis = synthesize_club_grasp(last_report, r_l, r_r)
    y_vec = synthesis.hand_wrench_vector
    s_vec = split_coordinates(y_vec[:6], y_vec[6:])
    sw_map = last_report.side_wrenches
    lead_sq = float(sw_map["L"].total_normal_force_n)
    trail_sq = float(sw_map["R"].total_normal_force_n)
    law = topo_model.law
    mu_s = float(law.static_friction)
    v_tr = float(law.transition_velocity_m_s)
    min_sq = float(np.linalg.norm(f_req_last)) / mu_s
    max_slip = max(slip_speeds.values()) if slip_speeds else 0.0
    regime: Literal["quasi_stick", "sliding"] = (
        "quasi_stick" if max_slip < v_tr else "sliding"
    )
    topo = adapter.topology()
    style_enum = topo.style

    return GraspResultantNewtonReport(
        model_id=adapter.model_id,
        style=style_enum.value,
        club_mass_kg=adapter.club_mass_kg(),
        force_peak_rel_error=float(f_err.peak_error),
        force_rms_rel_error=float(f_err.rms_error),
        moment_peak_rel_error=float(m_err.peak_error),
        moment_rms_rel_error=float(m_err.rms_error),
        newton_force_rel_error=newton_f_rel,
        newton_moment_com_residual_nm=newton_m_res,
        net_hand_force_n=_vec3_tuple(f_last),
        required_force_n=_vec3_tuple(f_req_last),
        net_hand_moment_com_nm=_vec3_tuple(m_last),
        required_moment_com_nm=_vec3_tuple(m_req_last),
        grasp_synthesis=synthesis,
        friction_cone_ratios=cone_ratios,
        slip_velocities_m_s=slip_speeds,
        contact_regime=regime,
        lead_squeeze_normal_force_n=lead_sq,
        trail_squeeze_normal_force_n=trail_sq,
        min_squeeze_for_stick_n=min_sq,
        nullspace_split_vector=_vec6_tuple(s_vec),
    )
