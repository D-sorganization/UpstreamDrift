"""Contact mode inference with hysteresis and support geometry resolution (PF-04, #10434).

Provides:
1. Contact state inference for foot spheres with hysteresis on height and velocity.
2. Classification of foot-level support modes (FLIGHT, HEEL_ONLY, TOE_ONLY, FULL_FOOT, SLIPPING).
3. Classification of global support modes (DOUBLE_SUPPORT, LEFT_SUPPORT, RIGHT_SUPPORT, etc.).
4. Detection and recording of ambiguous support states with plausible alternative schedules.
5. Resolution of contact geometry: Center of Pressure (COP), support polygon containment,
   and Coulomb friction cone compliance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.ground_support import convex_hull_contains

Array: TypeAlias = NDArray[np.float64]


class ContactState(str, Enum):
    """Discrete state of a single contact sphere."""

    OPEN = "open"
    STICKING = "sticking"
    SLIPPING = "slipping"


class FootSupportMode(str, Enum):
    """Support mode for an individual foot."""

    FLIGHT = "flight"
    HEEL_ONLY = "heel_only"
    TOE_ONLY = "toe_only"  # Pivot
    FULL_FOOT = "full_foot"
    SLIPPING = "slipping"


class GlobalSupportMode(str, Enum):
    """Bipedal support mode."""

    DOUBLE_SUPPORT = "double_support"
    LEFT_SUPPORT = "left_support"
    RIGHT_SUPPORT = "right_support"
    LEFT_HEEL_PIVOT = "left_heel_pivot"
    RIGHT_HEEL_PIVOT = "right_heel_pivot"
    FLIGHT = "flight"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ContactHysteresisSettings:
    """Thresholds for contact engagement, release, and slip detection with hysteresis."""

    engage_height_m: float = 0.005
    release_height_m: float = 0.015
    engage_velocity_m_s: float = -0.05
    release_velocity_m_s: float = 0.05
    slip_velocity_m_s: float = 0.02
    slip_hysteresis_ratio: float = 1.5

    def __post_init__(self) -> None:
        require(
            self.engage_height_m <= self.release_height_m,
            "engage_height_m must be <= release_height_m",
        )
        require(
            self.engage_velocity_m_s <= self.release_velocity_m_s,
            "engage_velocity_m_s must be <= release_velocity_m_s",
        )
        require(self.slip_velocity_m_s > 0.0, "slip_velocity_m_s must be positive")
        require(
            self.slip_hysteresis_ratio >= 1.0, "slip_hysteresis_ratio must be >= 1.0"
        )


@dataclass(frozen=True)
class ContactPointState:
    """State and kinematic metrics for a single contact sphere at one timestep."""

    name: str
    height_m: float
    normal_velocity_m_s: float
    tangential_velocity_m_s: float
    state: ContactState
    is_ambiguous: bool


@dataclass(frozen=True)
class FootModeState:
    """Resolved support mode for a single foot."""

    heel_state: ContactState
    toe_state: ContactState
    support_mode: FootSupportMode
    is_ambiguous: bool


@dataclass(frozen=True)
class FrameSupportMode:
    """Resolved bipedal contact mode at a single frame."""

    time_s: float
    global_mode: GlobalSupportMode
    left_foot: FootModeState
    right_foot: FootModeState
    is_ambiguous: bool
    sphere_states: dict[str, ContactPointState]


@dataclass(frozen=True)
class ContactModeSequence:
    """Complete sequence of contact modes across a trajectory."""

    frames: tuple[FrameSupportMode, ...]
    ambiguous_frame_indices: tuple[int, ...]
    alternative_schedules: tuple[tuple[GlobalSupportMode, ...], ...]

    @property
    def primary_schedule(self) -> tuple[GlobalSupportMode, ...]:
        return tuple(f.global_mode for f in self.frames)

    @property
    def ambiguity_fraction(self) -> float:
        if not self.frames:
            return 0.0
        n_frames = len(self.frames)
        n_ambig = len(self.ambiguous_frame_indices)
        return float(n_ambig / n_frames)


@dataclass(frozen=True)
class SupportGeometryReport:
    """Instantaneous center of pressure, support polygon containment, and friction checks."""

    centre_of_pressure_m: tuple[float, float, float] | None
    inside_support_polygon: bool
    cop_margin_m: float
    active_points: tuple[str, ...]
    total_normal_force_n: float
    friction_cone_violations: dict[str, float]


def _classify_foot_mode(
    heel: ContactState, toe: ContactState, is_ambiguous: bool
) -> FootModeState:
    """Combine heel and toe contact states into foot-level support mode."""
    if heel == ContactState.OPEN and toe == ContactState.OPEN:
        mode = FootSupportMode.FLIGHT
    elif heel in (ContactState.STICKING, ContactState.SLIPPING) and toe in (
        ContactState.STICKING,
        ContactState.SLIPPING,
    ):
        mode = (
            FootSupportMode.SLIPPING
            if (heel == ContactState.SLIPPING or toe == ContactState.SLIPPING)
            else FootSupportMode.FULL_FOOT
        )
    elif heel != ContactState.OPEN and toe == ContactState.OPEN:
        mode = (
            FootSupportMode.SLIPPING
            if heel == ContactState.SLIPPING
            else FootSupportMode.HEEL_ONLY
        )
    elif heel == ContactState.OPEN and toe != ContactState.OPEN:
        mode = (
            FootSupportMode.SLIPPING
            if toe == ContactState.SLIPPING
            else FootSupportMode.TOE_ONLY
        )
    else:
        mode = FootSupportMode.FLIGHT

    return FootModeState(
        heel_state=heel,
        toe_state=toe,
        support_mode=mode,
        is_ambiguous=is_ambiguous,
    )


def _classify_global_mode(
    left: FootModeState, right: FootModeState
) -> GlobalSupportMode:
    """Determine global bipedal support mode from left and right foot modes."""
    l_active = left.support_mode != FootSupportMode.FLIGHT
    r_active = right.support_mode != FootSupportMode.FLIGHT

    if l_active and r_active:
        if (
            left.support_mode == FootSupportMode.TOE_ONLY
            and right.support_mode != FootSupportMode.TOE_ONLY
        ):
            return GlobalSupportMode.LEFT_HEEL_PIVOT
        if (
            right.support_mode == FootSupportMode.TOE_ONLY
            and left.support_mode != FootSupportMode.TOE_ONLY
        ):
            return GlobalSupportMode.RIGHT_HEEL_PIVOT
        return GlobalSupportMode.DOUBLE_SUPPORT
    if l_active and not r_active:
        if left.support_mode == FootSupportMode.TOE_ONLY:
            return GlobalSupportMode.LEFT_HEEL_PIVOT
        return GlobalSupportMode.LEFT_SUPPORT
    if not l_active and r_active:
        if right.support_mode == FootSupportMode.TOE_ONLY:
            return GlobalSupportMode.RIGHT_HEEL_PIVOT
        return GlobalSupportMode.RIGHT_SUPPORT
    return GlobalSupportMode.FLIGHT


def _update_sphere_contact_state(
    prev_state: ContactState,
    bottom_h: float,
    v_normal: float,
    v_tangent: float,
    opts: ContactHysteresisSettings,
) -> tuple[ContactState, bool]:
    """Evaluate contact engagement/release hysteresis and ambiguity for one sphere."""
    is_ambig = False
    if prev_state == ContactState.OPEN:
        if bottom_h <= opts.engage_height_m or (
            bottom_h <= opts.release_height_m and v_normal <= opts.engage_velocity_m_s
        ):
            new_state = (
                ContactState.SLIPPING
                if v_tangent > opts.slip_velocity_m_s
                else ContactState.STICKING
            )
        else:
            new_state = ContactState.OPEN
            if (
                opts.engage_height_m < bottom_h <= opts.release_height_m
                and abs(v_normal) < 0.1
            ):
                is_ambig = True
    else:
        if bottom_h > opts.release_height_m or (
            bottom_h > opts.engage_height_m and v_normal >= opts.release_velocity_m_s
        ):
            new_state = ContactState.OPEN
        else:
            slip_thresh = (
                opts.slip_velocity_m_s * opts.slip_hysteresis_ratio
                if prev_state == ContactState.STICKING
                else opts.slip_velocity_m_s
            )
            new_state = (
                ContactState.SLIPPING
                if v_tangent > slip_thresh
                else ContactState.STICKING
            )
            if (
                opts.engage_height_m < bottom_h <= opts.release_height_m
                and abs(v_normal) < 0.1
            ):
                is_ambig = True
    return new_state, is_ambig


def _build_alternative_schedules(
    frames: list[FrameSupportMode], ambiguous_indices: list[int]
) -> list[tuple[GlobalSupportMode, ...]]:
    """Generate alternative candidate schedules for ambiguous intervals."""
    if not ambiguous_indices:
        return []
    ambig_set = set(ambiguous_indices)
    cand1 = [
        GlobalSupportMode.FLIGHT if idx in ambig_set else f.global_mode
        for idx, f in enumerate(frames)
    ]
    cand2 = [
        GlobalSupportMode.DOUBLE_SUPPORT if idx in ambig_set else f.global_mode
        for idx, f in enumerate(frames)
    ]
    return [tuple(cand1), tuple(cand2)]


def _classify_frame_support_mode(
    time_s: float,
    sphere_states: dict[str, ContactPointState],
    frame_has_ambiguity: bool,
) -> FrameSupportMode:
    l_heel = sphere_states.get(
        "heel_l",
        ContactPointState("heel_l", 1.0, 0.0, 0.0, ContactState.OPEN, False),
    )
    l_toe = sphere_states.get(
        "forefoot_l",
        sphere_states.get(
            "toe_l",
            ContactPointState("forefoot_l", 1.0, 0.0, 0.0, ContactState.OPEN, False),
        ),
    )
    r_heel = sphere_states.get(
        "heel_r",
        ContactPointState("heel_r", 1.0, 0.0, 0.0, ContactState.OPEN, False),
    )
    r_toe = sphere_states.get(
        "forefoot_r",
        sphere_states.get(
            "toe_r",
            ContactPointState("forefoot_r", 1.0, 0.0, 0.0, ContactState.OPEN, False),
        ),
    )

    l_ambig = l_heel.is_ambiguous or l_toe.is_ambiguous
    r_ambig = r_heel.is_ambiguous or r_toe.is_ambiguous
    l_foot = _classify_foot_mode(l_heel.state, l_toe.state, l_ambig)
    r_foot = _classify_foot_mode(r_heel.state, r_toe.state, r_ambig)
    global_mode = _classify_global_mode(l_foot, r_foot)

    return FrameSupportMode(
        time_s=time_s,
        global_mode=global_mode,
        left_foot=l_foot,
        right_foot=r_foot,
        is_ambiguous=frame_has_ambiguity,
        sphere_states=sphere_states,
    )


def infer_contact_modes(
    sphere_positions_m: Mapping[str, Array],
    sphere_velocities_m_s: Mapping[str, Array],
    ground: GroundPlane,
    times_s: Sequence[float] | Array,
    sphere_radii_m: Mapping[str, float],
    settings: ContactHysteresisSettings | None = None,
) -> ContactModeSequence:
    """Infer contact modes across time using height and velocity hysteresis."""
    opts = settings or ContactHysteresisSettings()
    n_frames = len(times_s)
    require(n_frames > 0, "times_s must not be empty")

    for name, pos in sphere_positions_m.items():
        require(pos.shape == (n_frames, 3), f"Position shape mismatch for {name}")
    for name, vel in sphere_velocities_m_s.items():
        require(vel.shape == (n_frames, 3), f"Velocity shape mismatch for {name}")

    normal = np.asarray(ground.normal, dtype=float)
    normal_unit = normal / np.linalg.norm(normal)

    sphere_prev_state: dict[str, ContactState] = dict.fromkeys(
        sphere_positions_m, ContactState.OPEN
    )
    frames: list[FrameSupportMode] = []
    ambiguous_indices: list[int] = []

    for k in range(n_frames):
        t = float(times_s[k])
        sphere_states: dict[str, ContactPointState] = {}
        frame_has_ambiguity = False

        for name, pos_series in sphere_positions_m.items():
            pos = pos_series[k]
            vel = sphere_velocities_m_s[name][k]
            radius = sphere_radii_m.get(name, 0.03)

            center_h = float(normal_unit @ pos) - ground.height_m
            bottom_h = center_h - radius
            v_normal = float(normal_unit @ vel)
            v_tangent_vec = vel - normal_unit * v_normal
            v_tangent = float(np.linalg.norm(v_tangent_vec))

            new_state, is_ambig = _update_sphere_contact_state(
                sphere_prev_state[name], bottom_h, v_normal, v_tangent, opts
            )
            if is_ambig:
                frame_has_ambiguity = True

            sphere_prev_state[name] = new_state
            sphere_states[name] = ContactPointState(
                name=name,
                height_m=bottom_h,
                normal_velocity_m_s=v_normal,
                tangential_velocity_m_s=v_tangent,
                state=new_state,
                is_ambiguous=is_ambig,
            )

        if frame_has_ambiguity:
            ambiguous_indices.append(k)

        frames.append(
            _classify_frame_support_mode(t, sphere_states, frame_has_ambiguity)
        )

    alt_schedules = _build_alternative_schedules(frames, ambiguous_indices)
    return ContactModeSequence(
        frames=tuple(frames),
        ambiguous_frame_indices=tuple(ambiguous_indices),
        alternative_schedules=tuple(alt_schedules),
    )


def _check_point_in_polygon(
    cop_2d: Array, polygon_2d: Array, tolerance_cop_m: float
) -> tuple[bool, float]:
    """Determine whether COP is inside the 2D contact polygon or segment."""
    n_pts = len(polygon_2d)
    if n_pts == 1:
        dist = float(np.linalg.norm(cop_2d - polygon_2d[0]))
        return dist <= tolerance_cop_m, tolerance_cop_m - dist
    if n_pts == 2:
        p1, p2 = polygon_2d[0], polygon_2d[1]
        seg = p2 - p1
        seg_len_sq = float(np.dot(seg, seg))
        t_proj = (
            max(0.0, min(1.0, float(np.dot(cop_2d - p1, seg) / seg_len_sq)))
            if seg_len_sq > 1e-12
            else 0.0
        )
        closest = p1 + t_proj * seg
        dist = float(np.linalg.norm(cop_2d - closest))
        return dist <= tolerance_cop_m, tolerance_cop_m - dist

    inside = convex_hull_contains(cop_2d, polygon_2d, tolerance_m=tolerance_cop_m)
    dists = np.linalg.norm(polygon_2d - cop_2d, axis=1)
    margin = float(tolerance_cop_m if inside else -np.min(dists))
    return inside, margin


def evaluate_support_geometry(
    contact_positions_m: Mapping[str, Array],
    contact_forces_n: Mapping[str, Array],
    ground: GroundPlane,
    mu_friction: float = 0.8,
    tolerance_cop_m: float = 0.01,
) -> SupportGeometryReport:
    """Evaluate Center of Pressure, support polygon containment, and friction cone."""
    normal = np.asarray(ground.normal, dtype=float)
    normal_unit = normal / np.linalg.norm(normal)

    total_f_n = 0.0
    weighted_cop = np.zeros(3, dtype=float)
    active_points: list[str] = []
    active_positions_2d: list[list[float]] = []
    friction_violations: dict[str, float] = {}

    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(normal_unit[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u_axis = np.cross(normal_unit, helper)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal_unit, u_axis)

    for name, pos_3d in contact_positions_m.items():
        pos = np.asarray(pos_3d, dtype=float)
        force = np.asarray(contact_forces_n.get(name, np.zeros(3)), dtype=float)
        fn = float(normal_unit @ force)
        f_tangent_vec = force - normal_unit * fn
        ft = float(np.linalg.norm(f_tangent_vec))

        if fn > 1e-3:
            total_f_n += fn
            weighted_cop += fn * pos
            active_points.append(name)
            active_positions_2d.append([float(u_axis @ pos), float(v_axis @ pos)])

            f_limit = mu_friction * fn
            if ft > f_limit + 1e-4:
                friction_violations[name] = float(ft - f_limit)

    if total_f_n <= 1e-4 or not active_positions_2d:
        return SupportGeometryReport(
            centre_of_pressure_m=None,
            inside_support_polygon=False,
            cop_margin_m=-1.0,
            active_points=(),
            total_normal_force_n=0.0,
            friction_cone_violations={},
        )

    cop_3d = weighted_cop / total_f_n
    cop_2d = np.array([float(u_axis @ cop_3d), float(v_axis @ cop_3d)])
    polygon_2d = np.array(active_positions_2d)
    inside, margin = _check_point_in_polygon(cop_2d, polygon_2d, tolerance_cop_m)

    return SupportGeometryReport(
        centre_of_pressure_m=(float(cop_3d[0]), float(cop_3d[1]), float(cop_3d[2])),
        inside_support_polygon=inside,
        cop_margin_m=margin,
        active_points=tuple(active_points),
        total_normal_force_n=float(total_f_n),
        friction_cone_violations=friction_violations,
    )
