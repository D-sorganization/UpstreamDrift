"""Sequential and isolated joint kinematics on synthetic fixtures (#15666)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
_FD_STEP_RAD = 1e-6


def _finite_1d(values: ArrayLike, label: str) -> Array:
    """Convert ``values`` to a finite non-empty 1D float64 array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{label} must be finite 1D")
    return arr


def _rodrigues(axis: tuple[float, float, float], angle_rad: float) -> Array:
    """Return 3x3 SO(3) rotation matrix for unit ``axis`` and ``angle_rad``."""
    ux, uy, uz = axis
    k = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]], dtype=np.float64)
    s, c = math.sin(angle_rad), 1.0 - math.cos(angle_rad)
    return np.eye(3, dtype=np.float64) + s * k + c * (k @ k)


def _rotation_angle_rad(rot_a: Array, rot_b: Array) -> float:
    """Return geodesic SO(3) angle in radians between ``rot_a`` and ``rot_b``."""
    rel = rot_a.T @ rot_b
    vee = [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]
    return abs(
        math.atan2(0.5 * float(np.linalg.norm(vee)), 0.5 * (float(np.trace(rel)) - 1.0))
    )


@dataclass(frozen=True)
class JointRomSpec:
    """Declared range of motion bounds for a single revolute joint."""

    joint_name: str
    q_min_rad: float
    q_max_rad: float
    proximal_order: int = 0

    def __post_init__(self) -> None:
        """Validate joint name, finite ROM bounds, and proximal order."""
        if not self.joint_name or not isinstance(self.joint_name, str):
            raise ValueError("joint_name must be non-empty")
        if not (math.isfinite(self.q_min_rad) and math.isfinite(self.q_max_rad)):
            raise ValueError("ROM bounds must be finite")
        if self.q_min_rad >= self.q_max_rad or self.proximal_order < 0:
            raise ValueError("ROM requires q_min_rad < q_max_rad and order >= 0")


@dataclass(frozen=True)
class IsolatedJointResult:
    """Immutable result of sweeping one target joint with all others locked."""

    joint_name: str
    joint_index: int
    joint_names: tuple[str, ...]
    reference_q_rad: tuple[float, ...]
    sweep_angles_rad: tuple[float, ...]
    trajectory_q_rad: tuple[tuple[float, ...], ...]
    max_locked_drift_rad: float
    rom_spec: JointRomSpec

    def __post_init__(self) -> None:
        """Enforce ROM bounds and zero locked-DOF contamination invariants."""
        n, idx = len(self.joint_names), self.joint_index
        if not (0 <= idx < n) or self.joint_names[idx] != self.joint_name:
            raise ValueError("joint_index/joint_name mismatch")
        if len(self.reference_q_rad) != n or len(self.sweep_angles_rad) < 2:
            raise ValueError("Invalid reference_q_rad or sweep_angles_rad length")
        if len(self.trajectory_q_rad) != len(self.sweep_angles_rad):
            raise ValueError("trajectory_q_rad length mismatch")
        lo, hi = self.rom_spec.q_min_rad, self.rom_spec.q_max_rad
        if any(not (lo <= a <= hi) for a in self.sweep_angles_rad):
            raise ValueError("Sweep angle violates declared ROM bounds")
        drift = validate_isolated_trajectory(
            self.trajectory_q_rad, self.reference_q_rad, idx
        )
        if self.max_locked_drift_rad != 0.0 or drift != 0.0:
            raise ValueError("Locked-DOF contamination detected")


@dataclass(frozen=True)
class JointLinkSpec:
    """Single revolute joint and subsequent rigid link offset in a serial chain."""

    name: str
    axis: tuple[float, float, float]
    link_offset_m: tuple[float, float, float]
    proximal_order: int
    is_grip_anchor: bool = False


@dataclass(frozen=True)
class SyntheticArmClubFK:
    """Engine-free 3D forward kinematics fixture for a multi-link arm and club."""

    links: tuple[JointLinkSpec, ...]
    club_tip_local_m: tuple[float, float, float]
    default_rom_specs: tuple[JointRomSpec, ...]

    def __post_init__(self) -> None:
        """Validate chain consistency, unit axes, and default ROM alignment."""
        names = [lnk.name for lnk in self.links]
        if not names or len(set(names)) != len(names) or any(not n for n in names):
            raise ValueError("SyntheticArmClubFK requires unique non-empty links")
        if [s.joint_name for s in self.default_rom_specs] != names:
            raise ValueError("default_rom_specs must align 1:1 with links")
        for lnk in self.links:
            ax = np.asarray(lnk.axis, dtype=np.float64)
            off = np.asarray(lnk.link_offset_m, dtype=np.float64)
            if ax.shape != (3,) or abs(float(np.linalg.norm(ax)) - 1.0) > 1e-6:
                raise ValueError("Link axis must be a unit 3-vector")
            if off.shape != (3,) or not bool(np.all(np.isfinite(off))):
                raise ValueError("link_offset_m must be a finite 3-vector")

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return ordered proximal-to-distal joint names."""
        return tuple(link.name for link in self.links)

    def evaluate(self, q_rad: Sequence[float] | Array) -> tuple[Array, Array, Array]:
        """Evaluate ``(clubhead_pos_m, clubhead_rot, grip_pos_m)`` at ``q_rad``."""
        q_arr = _finite_1d(q_rad, "q_rad")
        if q_arr.size != len(self.links):
            raise ValueError("q_rad length must match number of joints")
        pos: Array = np.zeros(3, dtype=np.float64)
        rot: Array = np.eye(3, dtype=np.float64)
        grip_pos: Array | None = None
        for link, angle in zip(self.links, q_arr, strict=True):
            rot = rot @ _rodrigues(link.axis, float(angle))
            pos = pos + rot @ np.asarray(link.link_offset_m, dtype=np.float64)
            if link.is_grip_anchor:
                grip_pos = pos.copy()
        grip_out = pos.copy() if grip_pos is None else grip_pos
        tip = np.asarray(self.club_tip_local_m, dtype=np.float64)
        return pos + rot @ tip, rot, grip_out

    def evaluate_trajectory(self, traj_q: ArrayLike) -> tuple[Array, Array, Array]:
        """Evaluate ``(clubhead_positions, clubhead_rots, grip_positions)``."""
        traj = np.asarray(traj_q, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != len(self.links):
            raise ValueError("trajectory_q_rad must be (steps, n_joints)")
        pos_l, rot_l, grip_l = zip(*(self.evaluate(r) for r in traj), strict=True)
        return np.stack(pos_l), np.stack(rot_l), np.stack(grip_l)


@dataclass(frozen=True)
class JointSweepSensitivity:
    """Clubhead displacement, orientation sensitivity, and grip drift for one joint."""

    joint_name: str
    proximal_order: int
    isolated_result: IsolatedJointResult
    max_clubhead_displacement_m: float
    clubhead_displacement_sensitivity_m_per_rad: float
    max_clubhead_orientation_delta_rad: float
    clubhead_orientation_sensitivity_rad_per_rad: float
    max_grip_drift_m: float


@dataclass(frozen=True)
class SequentialSweepReport:
    """Aggregate report from an ordered proximal-to-distal joint sweep."""

    sweep_order: tuple[str, ...]
    reference_q_rad: tuple[float, ...]
    reference_clubhead_pos_m: tuple[float, float, float]
    reference_grip_pos_m: tuple[float, float, float]
    joint_sweeps: tuple[JointSweepSensitivity, ...]
    max_locked_drift_rad: float


def validate_isolated_trajectory(
    trajectory_q: ArrayLike,
    reference_q: ArrayLike,
    target_index: int,
    *,
    atol: float = 0.0,
) -> float:
    """Verify non-target DOFs match ``reference_q`` within ``atol`` radians."""
    traj = np.asarray(trajectory_q, dtype=np.float64)
    ref = _finite_1d(reference_q, "reference_q")
    if traj.ndim != 2 or traj.shape[1] != ref.size or traj.shape[0] < 1:
        raise ValueError("trajectory_q must have shape (steps >= 1, n_dofs)")
    if not bool(np.all(np.isfinite(traj))) or not (0 <= target_index < ref.size):
        raise ValueError("trajectory_q must be finite and target_index in bounds")
    if not (math.isfinite(atol) and atol >= 0.0):
        raise ValueError("atol must be finite and non-negative")
    if ref.size == 1:
        return 0.0
    mask = np.ones(ref.size, dtype=bool)
    mask[target_index] = False
    max_drift = float(np.max(np.abs(traj[:, mask] - ref[mask])))
    if max_drift > atol:
        raise ValueError(f"Locked-DOF contamination detected: {max_drift:.6g} > {atol}")
    return max_drift


def isolate_joint_trajectory(
    joint_names: Sequence[str],
    reference_q_rad: Sequence[float],
    target_joint: str,
    rom_spec: JointRomSpec,
    *,
    n_steps: int = 11,
    sweep_angles_rad: Sequence[float] | None = None,
) -> IsolatedJointResult:
    """Sweep ``target_joint`` within ``rom_spec`` while locking all other DOFs."""
    names = tuple(joint_names)
    if not names or len(set(names)) != len(names) or target_joint not in names:
        raise ValueError(f"Unknown target joint or invalid names: {target_joint!r}")
    if rom_spec.joint_name != target_joint:
        raise ValueError("rom_spec.joint_name must match target_joint")
    ref = _finite_1d(reference_q_rad, "reference_q_rad")
    if ref.size != len(names):
        raise ValueError("reference_q_rad length must match joint_names")
    if sweep_angles_rad is None:
        if n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        angles = np.linspace(rom_spec.q_min_rad, rom_spec.q_max_rad, n_steps)
    else:
        angles = _finite_1d(sweep_angles_rad, "sweep_angles_rad")
        lo, hi = rom_spec.q_min_rad, rom_spec.q_max_rad
        if angles.size < 2 or bool(np.any(angles < lo) or np.any(angles > hi)):
            raise ValueError("sweep_angles_rad violates declared ROM bounds")
    idx = names.index(target_joint)
    traj = np.tile(ref, (angles.size, 1))
    traj[:, idx] = angles
    drift = validate_isolated_trajectory(traj, ref, idx, atol=0.0)
    ref_t = tuple(float(v) for v in ref)
    ang_t = tuple(float(v) for v in angles)
    traj_t = tuple(tuple(float(x) for x in row) for row in traj)
    return IsolatedJointResult(
        target_joint, idx, names, ref_t, ang_t, traj_t, drift, rom_spec
    )


def build_synthetic_arm_club_fk(
    *,
    upper_arm_m: float = 0.30,
    forearm_m: float = 0.26,
    club_shaft_m: float = 1.10,
    clubhead_lateral_offset_m: float = 0.05,
) -> SyntheticArmClubFK:
    """Build a 5-DOF proximal-to-distal synthetic arm+club kinematic chain."""
    lens = (upper_arm_m, forearm_m, club_shaft_m)
    if not all(math.isfinite(v) and v > 0.0 for v in lens) or not math.isfinite(
        clubhead_lateral_offset_m
    ):
        raise ValueError("Segment and shaft lengths must be positive and finite")
    rows = (
        ("shoulder_yaw", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), False, -0.8, 0.8),
        ("shoulder_pitch", (0.0, 1.0, 0.0), (upper_arm_m, 0.0, 0.0), False, -0.7, 0.7),
        ("elbow_flex", (0.0, 0.0, 1.0), (forearm_m, 0.0, 0.0), True, -1.2, 0.2),
        ("wrist_cock", (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), False, -0.6, 0.6),
        ("wrist_roll", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, -0.9, 0.9),
    )
    links = tuple(
        JointLinkSpec(n, ax, off, i, g) for i, (n, ax, off, g, _, _) in enumerate(rows)
    )
    roms = tuple(
        JointRomSpec(n, lo, hi, i) for i, (n, _, _, _, lo, hi) in enumerate(rows)
    )
    return SyntheticArmClubFK(
        links, (club_shaft_m, clubhead_lateral_offset_m, 0.0), roms
    )


def _resolve_ordered_specs(
    fk_model: SyntheticArmClubFK,
    rom_specs: Sequence[JointRomSpec] | None,
    sweep_order: Sequence[str] | None,
) -> tuple[list[JointRomSpec], dict[str, JointLinkSpec]]:
    """Validate ROM specs and enforce proximal-to-distal sweep ordering."""
    link_map = {link.name: link for link in fk_model.links}
    specs = list(fk_model.default_rom_specs if rom_specs is None else rom_specs)
    spec_map = {spec.joint_name: spec for spec in specs}
    if len(spec_map) != len(specs) or set(spec_map) - set(link_map):
        raise ValueError("rom_specs must contain unique known joints")
    if sweep_order is None:
        return sorted(
            specs, key=lambda s: link_map[s.joint_name].proximal_order
        ), link_map
    order_t = tuple(sweep_order)
    if set(order_t) != set(spec_map) or len(order_t) != len(spec_map):
        raise ValueError("sweep_order must list each ROM joint once")
    ranks = [link_map[name].proximal_order for name in order_t]
    if ranks != sorted(ranks):
        raise ValueError("sweep_order must follow proximal-to-distal ordering")
    return [spec_map[name] for name in order_t], link_map


def _evaluate_joint_sensitivity(
    fk: SyntheticArmClubFK, ref_q: Array, iso: IsolatedJointResult, order: int
) -> JointSweepSensitivity:
    """Compute clubhead sensitivity and grip drift for one isolated joint sweep."""
    ref_club, ref_rot, ref_grip = fk.evaluate(ref_q)
    c_pos, c_rot, g_pos = fk.evaluate_trajectory(iso.trajectory_q_rad)
    qp, qm = ref_q.copy(), ref_q.copy()
    qp[iso.joint_index] += _FD_STEP_RAD
    qm[iso.joint_index] -= _FD_STEP_RAD
    pp, rp, _ = fk.evaluate(qp)
    pm, rm, _ = fk.evaluate(qm)
    max_d = float(np.max(np.linalg.norm(c_pos - ref_club, axis=1)))
    d_sens = float(np.linalg.norm(pp - pm) / (2.0 * _FD_STEP_RAD))
    max_o = max(_rotation_angle_rad(ref_rot, rt) for rt in c_rot)
    o_sens = _rotation_angle_rad(rm, rp) / (2.0 * _FD_STEP_RAD)
    max_g = float(np.max(np.linalg.norm(g_pos - ref_grip, axis=1)))
    return JointSweepSensitivity(
        iso.joint_name, order, iso, max_d, d_sens, max_o, o_sens, max_g
    )


def run_sequential_joint_sweep(
    fk_model: SyntheticArmClubFK,
    rom_specs: Sequence[JointRomSpec] | None = None,
    reference_q_rad: Sequence[float] | None = None,
    *,
    n_steps: int = 11,
    sweep_order: Sequence[str] | None = None,
) -> SequentialSweepReport:
    """Run an ordered proximal-to-distal isolated joint sweep on ``fk_model``."""
    names = fk_model.joint_names
    ref_in = [0.0] * len(names) if reference_q_rad is None else reference_q_rad
    ref = _finite_1d(ref_in, "reference_q_rad")
    if ref.size != len(names):
        raise ValueError("reference_q_rad length must match fk_model joints")
    ordered, link_map = _resolve_ordered_specs(fk_model, rom_specs, sweep_order)
    ref_club, _, ref_grip = fk_model.evaluate(ref)
    ref_l = ref.tolist()
    sweeps = []
    for s in ordered:
        iso = isolate_joint_trajectory(names, ref_l, s.joint_name, s, n_steps=n_steps)
        rank = link_map[s.joint_name].proximal_order
        sweeps.append(_evaluate_joint_sensitivity(fk_model, ref, iso, rank))
    ref_c = (float(ref_club[0]), float(ref_club[1]), float(ref_club[2]))
    ref_g = (float(ref_grip[0]), float(ref_grip[1]), float(ref_grip[2]))
    max_drift = max(e.isolated_result.max_locked_drift_rad for e in sweeps)
    order_names = tuple(s.joint_name for s in ordered)
    ref_t = tuple(float(v) for v in ref)
    return SequentialSweepReport(
        order_names, ref_t, ref_c, ref_g, tuple(sweeps), max_drift
    )
