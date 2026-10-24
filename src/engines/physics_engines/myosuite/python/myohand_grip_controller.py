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
from scipy.spatial.transform import Rotation

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
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    compute_quantity_error,
)
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
from src.shared.python.motion_matching.club_models import CLUBS, ClubSpec
from src.shared.python.motion_matching.grip_contract import GripSpec

__all__ = [
    "ClubInertialProperties",
    "DRIVER_CLUB_PROPERTIES",
    "ForwardReplayClubReport",
    "GraspResultantNewtonReport",
    "IRON_CLUB_PROPERTIES",
    "build_prescribed_grip_trajectory",
    "evaluate_grasp_resultant_newton_check",
    "register_interface_context",
    "replay_club_forward_trajectory",
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
    """Analytical rigid-body inertial properties of ``golf_club`` derived from ``ClubSpec``."""

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

    @classmethod
    def from_club_spec(
        cls,
        club: ClubSpec,
        gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> ClubInertialProperties:
        """Compute analytical mass, COM, and COM inertia tensor from ``ClubSpec`` primitive geoms."""
        if not isinstance(club, ClubSpec):
            raise TypeError(f"club must be a ClubSpec, got {type(club).__name__}")
        m_s = float(club.shaft_mass_kg + club.grip_mass_kg)
        m_h = float(club.head_mass_kg)
        m_tot = m_s + m_h
        length = float(club.length_m)
        radius = float(club.shaft_radius_m)
        y_com = (m_s * (-0.5 * length)) / m_tot

        vol_cyl = math.pi * radius**2 * length
        vol_sph = (4.0 / 3.0) * math.pi * radius**3
        m_cyl = m_s * vol_cyl / (vol_cyl + vol_sph)
        m_sph = m_s * vol_sph / (vol_cyl + vol_sph)
        i_cap_ax = 0.5 * m_cyl * radius**2 + 0.4 * m_sph * radius**2
        i_cap_tr = m_cyl * (length**2 / 12.0 + 0.25 * radius**2) + m_sph * (
            0.4 * radius**2 + 0.25 * length**2 + 0.375 * length * radius
        )
        hx, hy, hz = (float(x) for x in club.head_half_size_m)
        coeff = 0.2 if club.head_shape == "ellipsoid" else (1.0 / 3.0)
        i_head = np.array(
            [
                coeff * m_h * (hy**2 + hz**2),
                coeff * m_h * (hx**2 + hz**2),
                coeff * m_h * (hx**2 + hy**2),
            ],
            dtype=float,
        )
        i_cap = np.array([i_cap_tr, i_cap_ax, i_cap_tr], dtype=float)
        dy_s = -0.5 * length - y_com
        dy_h = -y_com
        i_diag = (
            i_cap
            + m_s * np.array([dy_s**2, 0.0, dy_s**2], dtype=float)
            + i_head
            + m_h * np.array([dy_h**2, 0.0, dy_h**2], dtype=float)
        )
        return cls(
            mass_kg=m_tot,
            com_local_m=(0.0, y_com, 0.0),
            inertia_com_kg_m2=np.diag(i_diag),
            gravity_m_s2=gravity_m_s2,
        )

    def inverse_dynamics_at_phase(
        self,
        sin_p: float,
        cos_p: float,
        omega_freq: float,
        *,
        origin_accel_m_s2: np.ndarray | None = None,
        alpha_rad_s2: np.ndarray | None = None,
        omega_rad_s: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ``(F_req, M_req_com, M_req_origin)`` analytically from club kinematics."""
        rot = (
            np.asarray(rotation, dtype=float)
            if rotation is not None
            else np.eye(3, dtype=float)
        )
        r_com = rot @ np.asarray(self.com_local_m, dtype=float)
        gravity = np.asarray(self.gravity_m_s2, dtype=float)
        omega_w = (
            np.asarray(omega_rad_s, dtype=float)
            if omega_rad_s is not None
            else np.array([0.0, 0.0, 0.15 * sin_p], dtype=float)
        )
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
        i_com = rot @ self.inertia_com_kg_m2 @ rot.T
        m_req_com = i_com @ alpha_w + np.cross(omega_w, i_com @ omega_w)
        m_req_origin = m_req_com + np.cross(r_com, f_req)
        return f_req, m_req_com, m_req_origin


DRIVER_CLUB_PROPERTIES: Final[ClubInertialProperties] = (
    ClubInertialProperties.from_club_spec(CLUBS["driver"])
)

IRON_CLUB_PROPERTIES: Final[ClubInertialProperties] = (
    ClubInertialProperties.from_club_spec(CLUBS["iron7"])
)


@dataclass(frozen=True)
class ForwardReplayClubReport:
    """Open-loop forward-dynamics club replay report under solved hand contact wrenches."""

    model_id: str
    style: str
    integrator: str
    integrator_order: int
    dt_s: float
    substeps: int
    substep_dt_s: float
    n_steps: int
    com_motion_amplitude_m: float
    orientation_amplitude_rad: float
    com_pos_peak_error_m: float
    com_pos_rms_error_m: float
    com_pos_peak_rel_error: float
    com_pos_rms_rel_error: float
    orientation_peak_error_rad: float
    orientation_rms_error_rad: float
    orientation_peak_rel_error: float
    orientation_rms_rel_error: float
    max_active_club_weld_count: int
    max_club_constraint_force_n: float
    passed: bool


@dataclass(frozen=True)
class GraspResultantNewtonReport:
    """Grasp-resultant inverse-dynamics and Newton-Euler balance report (HRG-GS-M8).

    Primary ``force_*`` and ``moment_*`` error fields compare the hand contact resultant
    against the independent MuJoCo ``mj_inverse`` requirement on ``club_free`` (analytic
    ``qacc``), while ``mj_central_diff_*`` compares against ``mj_inverse`` with central
    differences of ``qvel``, ``analytic_vs_mj_*`` reports the analytical-vs-MuJoCo club
    mass/inertia discrepancy, and ``solver_closure_*`` records the root-solver closure
    residual against the analytic ``ClubInertialProperties`` target (not acceptance evidence).
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
    analytic_required_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0)
    analytic_required_moment_com_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    solver_closure_force_peak_rel_error: float = 0.0
    solver_closure_force_rms_rel_error: float = 0.0
    solver_closure_moment_peak_rel_error: float = 0.0
    solver_closure_moment_rms_rel_error: float = 0.0
    mj_central_diff_force_peak_rel_error: float = 0.0
    mj_central_diff_force_rms_rel_error: float = 0.0
    mj_central_diff_moment_peak_rel_error: float = 0.0
    mj_central_diff_moment_rms_rel_error: float = 0.0
    analytic_vs_mj_force_peak_rel_error: float = 0.0
    analytic_vs_mj_force_rms_rel_error: float = 0.0
    analytic_vs_mj_moment_peak_rel_error: float = 0.0
    analytic_vs_mj_moment_rms_rel_error: float = 0.0
    analytic_vs_mj_cd_force_peak_rel_error: float = 0.0
    analytic_vs_mj_cd_force_rms_rel_error: float = 0.0
    analytic_vs_mj_cd_moment_peak_rel_error: float = 0.0
    analytic_vs_mj_cd_moment_rms_rel_error: float = 0.0
    max_club_constraint_force_n: float = 0.0


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


def _rot_z(theta_rad: float) -> np.ndarray:
    ct, st = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def build_prescribed_grip_trajectory(
    interface: GripInterface,
    *,
    n_steps: int = 21,
    dt_s: float = FIXTURE_DT_S,
    topology_model: TopologyContactModel | None = None,
    club_properties: ClubInertialProperties | None = None,
) -> tuple[np.ndarray, list[dict[str, BushingState]], list[dict[str, BushingState]]]:
    """Build deterministic 2 ms force-regulated grip trajectory for M3 parity & Newton balance.

    At each step ``t_k``, prescribes kinematically consistent club pose and velocity
    (``p_origin(t) = int_0^t v_origin(tau) dtau``, ``R_club(t) = R_z(int_0^t omega_z(tau) dtau)``)
    and solves for the bilateral hand frame states ``(hand_frames['L'], hand_frames['R'])``
    via :func:`solve_force_regulated_hand_frames` so the shared contact law holds the club
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
        len(topo.hand_grip_pairs()),
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
        club_pos = np.array([(0.02 / omega_freq) * sp, 0.0, 0.0], dtype=float)
        theta_z = (0.15 / omega_freq) * (1.0 - cp)
        rot_club = _rot_z(theta_z)
        club_state = RigidBodyState(
            rotation=rot_club,
            position_m=club_pos,
            velocity_m_s=np.array([0.02 * cp, 0.0, 0.0], dtype=float),
            omega_rad_s=np.array([0.0, 0.0, 0.15 * sp], dtype=float),
        )
        c_frames = hand_frame_states(club_state, interface)
        f_req, m_req_com, _ = club_props.inverse_dynamics_at_phase(
            sp, cp, omega_freq, rotation=rot_club
        )
        xipos = club_pos + rot_club @ r_com_local
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


@dataclass(frozen=True)
class _StepWrenchSeries:
    f_hand: np.ndarray
    m_hand_com: np.ndarray
    f_req_ana: np.ndarray
    m_req_ana_com: np.ndarray
    f_req_mj: np.ndarray
    m_req_mj_com: np.ndarray
    f_req_mj_cd: np.ndarray
    m_req_mj_cd_com: np.ndarray
    max_club_constraint_force_n: float
    last_report: GripStepReport


def _evaluate_step_wrenches(
    adapter: Any,
    times: np.ndarray,
    hand_seq: list[dict[str, BushingState]],
    club_seq: list[dict[str, BushingState]],
    dt_s: float,
    wrench_scale: float,
) -> _StepWrenchSeries:
    n_steps = len(times)
    f_hand_seq = np.zeros((n_steps, 3), dtype=float)
    m_hand_com_seq = np.zeros((n_steps, 3), dtype=float)
    f_req_ana_seq = np.zeros((n_steps, 3), dtype=float)
    m_req_ana_seq = np.zeros((n_steps, 3), dtype=float)
    f_req_mj_seq = np.zeros((n_steps, 3), dtype=float)
    m_req_mj_seq = np.zeros((n_steps, 3), dtype=float)
    f_req_mj_cd_seq = np.zeros((n_steps, 3), dtype=float)
    m_req_mj_cd_seq = np.zeros((n_steps, 3), dtype=float)
    qpos_seq = np.zeros((n_steps, 7), dtype=float)
    qvel_seq = np.zeros((n_steps, 6), dtype=float)
    last_report: GripStepReport | None = None
    binding = adapter.resolve_binding(adapter.topology())
    club_body = binding.club_body
    max_c_norm = 0.0

    for k in range(n_steps):
        adapter.set_prescribed_grip_state(hand_seq[k], club_seq[k])
        qpos_k, qvel_k = adapter.club_free_qpos_qvel()
        qpos_seq[k], qvel_seq[k] = qpos_k, qvel_k
        rep = adapter.evaluate_contacts()
        scaled_wrenches = {
            b: (wrench_scale * f, wrench_scale * m)
            for b, (f, m) in rep.body_wrenches().items()
        }
        adapter.apply_wrenches(scaled_wrenches)
        t_k = float(times[k])
        f_ana, m_ana_com, _ = adapter.club_inverse_dynamics_requirement(
            t_s=t_k, n_steps=n_steps, dt_s=dt_s
        )
        f_mj, m_mj_com, _, c_norm = adapter.club_mj_inverse_requirement(
            t_s=t_k,
            n_steps=n_steps,
            dt_s=dt_s,
            qpos_club=qpos_k,
            qvel_club=qvel_k,
        )
        max_c_norm = max(max_c_norm, float(c_norm))
        f_net, m_origin = scaled_wrenches[club_body]
        m_hand_com = adapter.club_moment_about_com(f_net, m_origin)
        f_hand_seq[k], m_hand_com_seq[k] = f_net, m_hand_com
        f_req_ana_seq[k], m_req_ana_seq[k] = f_ana, m_ana_com
        f_req_mj_seq[k], m_req_mj_seq[k] = f_mj, m_mj_com
        last_report = rep

    qacc_cd_seq = np.gradient(qvel_seq, float(dt_s), axis=0, edge_order=2)
    for k in range(n_steps):
        f_cd, m_cd_com, _, c_norm_cd = adapter.club_mj_inverse_requirement(
            qpos_club=qpos_seq[k],
            qvel_club=qvel_seq[k],
            qacc_club=qacc_cd_seq[k],
        )
        max_c_norm = max(max_c_norm, float(c_norm_cd))
        f_req_mj_cd_seq[k], m_req_mj_cd_seq[k] = f_cd, m_cd_com

    assert last_report is not None
    return _StepWrenchSeries(
        f_hand=f_hand_seq,
        m_hand_com=m_hand_com_seq,
        f_req_ana=f_req_ana_seq,
        m_req_ana_com=m_req_ana_seq,
        f_req_mj=f_req_mj_seq,
        m_req_mj_com=m_req_mj_seq,
        f_req_mj_cd=f_req_mj_cd_seq,
        m_req_mj_cd_com=m_req_mj_cd_seq,
        max_club_constraint_force_n=max_c_norm,
        last_report=last_report,
    )


def _peak_rms(cand: np.ndarray, ref: np.ndarray, name: str) -> tuple[float, float]:
    err = compute_quantity_error(cand, ref, name=name)
    return float(err.peak_error), float(err.rms_error)


def _pair_errors(
    f_cand: np.ndarray,
    m_cand: np.ndarray,
    f_ref: np.ndarray,
    m_ref: np.ndarray,
    tag: str,
) -> tuple[float, float, float, float]:
    f_pk, f_rms = _peak_rms(f_cand, f_ref, f"{tag}_force_n")
    m_pk, m_rms = _peak_rms(m_cand, m_ref, f"{tag}_moment_com_nm")
    return f_pk, f_rms, m_pk, m_rms


def _build_newton_report(
    adapter: Any,
    topo_model: TopologyContactModel,
    ws: _StepWrenchSeries,
    hand_last: Mapping[str, BushingState],
    club_last: Mapping[str, BushingState],
) -> GraspResultantNewtonReport:
    f_mj_pk, f_mj_rms, m_mj_pk, m_mj_rms = _pair_errors(
        ws.f_hand, ws.m_hand_com, ws.f_req_mj, ws.m_req_mj_com, "club_net"
    )
    f_cd_pk, f_cd_rms, m_cd_pk, m_cd_rms = _pair_errors(
        ws.f_hand, ws.m_hand_com, ws.f_req_mj_cd, ws.m_req_mj_cd_com, "club_net_cd"
    )
    f_cl_pk, f_cl_rms, m_cl_pk, m_cl_rms = _pair_errors(
        ws.f_hand, ws.m_hand_com, ws.f_req_ana, ws.m_req_ana_com, "solver_closure"
    )
    f_am_pk, f_am_rms, m_am_pk, m_am_rms = _pair_errors(
        ws.f_req_ana, ws.m_req_ana_com, ws.f_req_mj, ws.m_req_mj_com, "analytic_vs_mj"
    )
    f_ac_pk, f_ac_rms, m_ac_pk, m_ac_rms = _pair_errors(
        ws.f_req_ana,
        ws.m_req_ana_com,
        ws.f_req_mj_cd,
        ws.m_req_mj_cd_com,
        "analytic_vs_mj_cd",
    )
    f_last, f_req_last = ws.f_hand[-1], ws.f_req_mj[-1]
    m_last, m_req_last = ws.m_hand_com[-1], ws.m_req_mj_com[-1]
    newton_f_rel = float(
        np.linalg.norm(f_last - f_req_last)
        / max(float(np.linalg.norm(f_req_last)), 1e-9)
    )
    newton_m_res = float(np.linalg.norm(m_last - m_req_last))
    last_rep = ws.last_report
    cone_ratios, slip_speeds = terminal_cone_and_slip(
        adapter, last_rep, hand_last, club_last
    )
    r_l = np.asarray(club_last["L"].position_m, dtype=float)
    r_r = np.asarray(club_last["R"].position_m, dtype=float)
    synthesis = synthesize_club_grasp(last_rep, r_l, r_r)
    y_vec = synthesis.hand_wrench_vector
    s_vec = split_coordinates(y_vec[:6], y_vec[6:])
    sw_map = last_rep.side_wrenches
    lead_sq = float(sw_map["L"].total_normal_force_n)
    trail_sq = float(sw_map["R"].total_normal_force_n)
    law = topo_model.law
    mu_s, v_tr = float(law.static_friction), float(law.transition_velocity_m_s)
    min_sq = float(np.linalg.norm(f_req_last)) / mu_s
    max_slip = max(slip_speeds.values()) if slip_speeds else 0.0
    regime: Literal["quasi_stick", "sliding"] = (
        "quasi_stick" if max_slip < v_tr else "sliding"
    )
    topo = adapter.topology()
    return GraspResultantNewtonReport(
        model_id=adapter.model_id,
        style=topo.style.value,
        club_mass_kg=adapter.club_mass_kg(),
        force_peak_rel_error=f_mj_pk,
        force_rms_rel_error=f_mj_rms,
        moment_peak_rel_error=m_mj_pk,
        moment_rms_rel_error=m_mj_rms,
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
        analytic_required_force_n=_vec3_tuple(ws.f_req_ana[-1]),
        analytic_required_moment_com_nm=_vec3_tuple(ws.m_req_ana_com[-1]),
        solver_closure_force_peak_rel_error=f_cl_pk,
        solver_closure_force_rms_rel_error=f_cl_rms,
        solver_closure_moment_peak_rel_error=m_cl_pk,
        solver_closure_moment_rms_rel_error=m_cl_rms,
        mj_central_diff_force_peak_rel_error=f_cd_pk,
        mj_central_diff_force_rms_rel_error=f_cd_rms,
        mj_central_diff_moment_peak_rel_error=m_cd_pk,
        mj_central_diff_moment_rms_rel_error=m_cd_rms,
        analytic_vs_mj_force_peak_rel_error=f_am_pk,
        analytic_vs_mj_force_rms_rel_error=f_am_rms,
        analytic_vs_mj_moment_peak_rel_error=m_am_pk,
        analytic_vs_mj_moment_rms_rel_error=m_am_rms,
        analytic_vs_mj_cd_force_peak_rel_error=f_ac_pk,
        analytic_vs_mj_cd_force_rms_rel_error=f_ac_rms,
        analytic_vs_mj_cd_moment_peak_rel_error=m_ac_pk,
        analytic_vs_mj_cd_moment_rms_rel_error=m_ac_rms,
        max_club_constraint_force_n=float(ws.max_club_constraint_force_n),
    )


def evaluate_grasp_resultant_newton_check(
    adapter: Any,
    *,
    n_steps: int = 21,
    dt_s: float = FIXTURE_DT_S,
    wrench_scale: float = 1.0,
) -> GraspResultantNewtonReport:
    """Compare hands' net club contact wrench to MuJoCo ``mj_inverse`` and analytic requirements."""
    if not math.isfinite(wrench_scale) or wrench_scale <= 0.0:
        raise ValueError(f"wrench_scale must be finite and > 0, got {wrench_scale}")
    iface = adapter.interface()
    topo_model = adapter.topology_model()
    times, hand_seq, club_seq = build_prescribed_grip_trajectory(
        iface, n_steps=n_steps, dt_s=dt_s, topology_model=topo_model
    )
    ws = _evaluate_step_wrenches(adapter, times, hand_seq, club_seq, dt_s, wrench_scale)
    return _build_newton_report(adapter, topo_model, ws, hand_seq[-1], club_seq[-1])


def _geodesic_angle_rad(r_a: np.ndarray, r_b: np.ndarray) -> float:
    trace_val = float(np.trace(r_a.T @ r_b))
    cos_theta = float(np.clip(0.5 * (trace_val - 1.0), -1.0, 1.0))
    return float(math.acos(cos_theta))


def _collect_prescribed_club_wrenches(
    adapter: Any,
    hand_seq: list[dict[str, BushingState]],
    club_seq: list[dict[str, BushingState]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_steps = len(hand_seq)
    binding = adapter.resolve_binding(adapter.topology())
    club_body = binding.club_body
    prescribed_com = np.zeros((n_steps, 3), dtype=float)
    prescribed_rot = np.zeros((n_steps, 3, 3), dtype=float)
    f_club_seq = np.zeros((n_steps, 3), dtype=float)
    m_club_com_seq = np.zeros((n_steps, 3), dtype=float)
    for k in range(n_steps):
        adapter.set_prescribed_grip_state(hand_seq[k], club_seq[k])
        com_k, rot_k = adapter.club_com_and_rotation()
        prescribed_com[k], prescribed_rot[k] = com_k, rot_k
        rep = adapter.evaluate_contacts()
        f_net, m_origin = rep.body_wrenches()[club_body]
        f_club_seq[k] = f_net
        m_club_com_seq[k] = adapter.club_moment_about_com(f_net, m_origin)
    return prescribed_com, prescribed_rot, f_club_seq, m_club_com_seq


def _compute_replay_errors(
    replayed_com: np.ndarray,
    prescribed_com: np.ndarray,
    replayed_rot: np.ndarray,
    prescribed_rot: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    n_steps = len(prescribed_com)
    pos_err_vec = np.linalg.norm(replayed_com - prescribed_com, axis=1)
    rot_err_vec = np.array(
        [
            _geodesic_angle_rad(replayed_rot[k], prescribed_rot[k])
            for k in range(n_steps)
        ],
        dtype=float,
    )
    com_disp = np.linalg.norm(prescribed_com - prescribed_com[0], axis=1)
    rot_disp = np.array(
        [
            _geodesic_angle_rad(prescribed_rot[k], prescribed_rot[0])
            for k in range(n_steps)
        ],
        dtype=float,
    )
    com_amp, rot_amp = float(np.max(com_disp)), float(np.max(rot_disp))
    pos_peak = float(np.max(pos_err_vec))
    pos_rms = float(np.sqrt(np.mean(pos_err_vec**2)))
    rot_peak = float(np.max(rot_err_vec))
    rot_rms = float(np.sqrt(np.mean(rot_err_vec**2)))
    pos_peak_rel = pos_peak / max(com_amp, 1e-12)
    pos_rms_rel = pos_rms / max(com_amp, 1e-12)
    rot_peak_rel = rot_peak / max(rot_amp, 1e-12)
    rot_rms_rel = rot_rms / max(rot_amp, 1e-12)
    return (
        com_amp,
        rot_amp,
        pos_peak,
        pos_rms,
        pos_peak_rel,
        pos_rms_rel,
        rot_peak,
        rot_rms,
        rot_peak_rel,
        rot_rms_rel,
    )


def replay_club_forward_trajectory(
    adapter: Any,
    *,
    n_steps: int = 21,
    dt_s: float = FIXTURE_DT_S,
    substeps: int = 10,
) -> ForwardReplayClubReport:
    """Replay free ``golf_club`` forward under ``mj_step`` with solved hand wrenches via ``xfrc_applied``."""
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"dt_s must be finite and > 0, got {dt_s}")
    if substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")

    _, hand_seq, club_seq = build_prescribed_grip_trajectory(
        adapter.interface(),
        n_steps=n_steps,
        dt_s=dt_s,
        topology_model=adapter.topology_model(),
    )
    prescribed_com, prescribed_rot, f_club_seq, m_club_com_seq = (
        _collect_prescribed_club_wrenches(adapter, hand_seq, club_seq)
    )
    adapter.set_prescribed_grip_state(hand_seq[0], club_seq[0])
    replayed_com = np.zeros((n_steps, 3), dtype=float)
    replayed_rot = np.zeros((n_steps, 3, 3), dtype=float)
    replayed_com[0], replayed_rot[0] = adapter.club_com_and_rotation()

    dt_sub = float(dt_s) / float(substeps)
    max_weld = int(adapter.active_club_weld_count())
    max_c_norm = 0.0
    for k in range(n_steps - 1):
        for s in range(substeps):
            alpha = (float(s) + 0.5) / float(substeps)
            f_sub = (1.0 - alpha) * f_club_seq[k] + alpha * f_club_seq[k + 1]
            m_sub = (1.0 - alpha) * m_club_com_seq[k] + alpha * m_club_com_seq[k + 1]
            c_norm = adapter.step_club_with_wrench(f_sub, m_sub, dt_s=dt_sub)
            max_c_norm = max(max_c_norm, float(c_norm))
        max_weld = max(max_weld, int(adapter.active_club_weld_count()))
        replayed_com[k + 1], replayed_rot[k + 1] = adapter.club_com_and_rotation()

    (
        com_amp,
        rot_amp,
        pos_peak,
        pos_rms,
        pos_peak_rel,
        pos_rms_rel,
        rot_peak,
        rot_rms,
        rot_peak_rel,
        rot_rms_rel,
    ) = _compute_replay_errors(
        replayed_com, prescribed_com, replayed_rot, prescribed_rot
    )
    int_name, int_order = adapter.integrator_info()
    topo = adapter.topology()
    passed = bool(
        max_weld == 0
        and max_c_norm == 0.0
        and max(pos_peak_rel, rot_peak_rel) <= 0.05
        and max(pos_rms_rel, rot_rms_rel) <= 0.02
    )
    return ForwardReplayClubReport(
        model_id=adapter.model_id,
        style=topo.style.value,
        integrator=int_name,
        integrator_order=int_order,
        dt_s=float(dt_s),
        substeps=substeps,
        substep_dt_s=dt_sub,
        n_steps=n_steps,
        com_motion_amplitude_m=com_amp,
        orientation_amplitude_rad=rot_amp,
        com_pos_peak_error_m=pos_peak,
        com_pos_rms_error_m=pos_rms,
        com_pos_peak_rel_error=pos_peak_rel,
        com_pos_rms_rel_error=pos_rms_rel,
        orientation_peak_error_rad=rot_peak,
        orientation_rms_error_rad=rot_rms,
        orientation_peak_rel_error=rot_peak_rel,
        orientation_rms_rel_error=rot_rms_rel,
        max_active_club_weld_count=max_weld,
        max_club_constraint_force_n=max_c_norm,
        passed=passed,
    )
