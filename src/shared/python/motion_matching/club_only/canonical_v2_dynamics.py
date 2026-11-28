"""Canonical-v2 inverse dynamics & GRF/CoP gate + capped repair (WP4, #15787)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType

from src.shared.python.biomechanics.cop_grf_replay_harness import (
    ContactPointRecord,
    compute_wrench_cop,
    point_in_polygon_2d,
)

__all__ = [
    "DynamicsFeasibilityThresholds",
    "FootContactWrench",
    "InverseDynamicsFrame",
    "InverseDynamicsGateResult",
    "RepairConfig",
    "build_irreparable_dynamics_fixture",
    "evaluate_inverse_dynamics_gate",
    "repair_dynamically_infeasible_trajectory",
]

_TRUNK = ("lumbar_", "thoracic_", "trunk_")
_PL = ((0.0, -0.05), (0.24, -0.05), (0.24, 0.05), (0.0, 0.05))
_PR = ((0.0, 0.25), (0.24, 0.25), (0.24, 0.35), (0.0, 0.35))
_MSG_TAU, _MSG_TRUNK = "torque_ceiling_exceeded", "trunk_torque_ceiling_exceeded"
Evaluator = Callable[
    [Sequence["InverseDynamicsFrame"]], Sequence["InverseDynamicsFrame"]
]


def _req(name: str, v: float, *, pos: bool = False, nn: bool = False) -> float:
    """Validate finite float with optional positive or non-negative check."""
    if isinstance(v, bool) or not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if (pos and v <= 0.0) or (nn and v < 0.0):
        raise ValueError(f"{name} out of bounds: {v}")
    return float(v)


def _chk(obj: object, pos: tuple[str, ...], nn: tuple[str, ...]) -> None:
    """Validate positive and non-negative dataclass attributes."""
    for k in pos:
        _req(k, getattr(obj, k), pos=True)
    for k in nn:
        _req(k, getattr(obj, k), nn=True)


@dataclass(frozen=True)
class DynamicsFeasibilityThresholds:
    """Versioned dynamic feasibility thresholds for canonical-v2 inverse dynamics."""

    tau_max_nm: float = 250.0
    trunk_coupled_tau_max_nm: float = 200.0
    friction_mu: float = 0.8
    min_loaded_fz_n: float = 10.0
    root_force_tol_n: float = 15.0
    root_moment_tol_nm: float = 15.0
    max_club_tracking_error_m: float = 0.04
    max_grip_closure_error_m: float = 0.005
    ground_z_m: float = 0.0
    require_coupled_trunk: bool = True
    joint_tau_max_nm: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate threshold bounds and freeze per-joint torque overrides."""
        pos = ("tau_max_nm", "trunk_coupled_tau_max_nm", "min_loaded_fz_n")
        nn = ("friction_mu", "root_force_tol_n", "root_moment_tol_nm")
        _chk(self, pos, (*nn, "max_club_tracking_error_m", "max_grip_closure_error_m"))
        _req("ground_z_m", self.ground_z_m)
        lim = {k: _req(k, v, pos=True) for k, v in self.joint_tau_max_nm.items()}
        object.__setattr__(self, "joint_tau_max_nm", MappingProxyType(lim))


@dataclass(frozen=True)
class FootContactWrench:
    """Single-foot ground contacts and active 2D support patch polygon."""

    foot: str
    contacts: tuple[ContactPointRecord, ...]
    support_polygon_2d: tuple[tuple[float, float], ...]
    is_loaded: bool = True

    def __post_init__(self) -> None:
        """Validate foot label."""
        if not self.foot.strip():
            raise ValueError("foot must be non-empty")


@dataclass(frozen=True)
class InverseDynamicsFrame:
    """Single canonical-v2 inverse dynamics frame with GRF, root, and tracking."""

    frame_index: int
    time_s: float
    q_rad: tuple[float, ...]
    joint_torques_nm: Mapping[str, float]
    root_residual_force_n: tuple[float, float, float]
    root_residual_moment_nm: tuple[float, float, float]
    foot_wrenches: tuple[FootContactWrench, ...]
    club_tracking_error_m: float = 0.0
    grip_closure_error_m: float = 0.0
    trunk_coupled: bool = True

    def __post_init__(self) -> None:
        """Validate frame scalars, torques, root wrenches, and tolerances."""
        _chk(self, (), ("club_tracking_error_m", "grip_closure_error_m"))
        for v in (self.time_s, *self.q_rad):
            _req("q_or_t", v)
        for v in (*self.root_residual_force_n, *self.root_residual_moment_nm):
            _req("root", v)
        tau = {str(k): _req(k, float(v)) for k, v in self.joint_torques_nm.items()}
        object.__setattr__(self, "joint_torques_nm", MappingProxyType(tau))


@dataclass(frozen=True)
class RepairConfig:
    """Capped iterative perturbation budget and tolerance bounds for repair."""

    max_iters: int = 10
    max_delta_q_rad: float = 0.05
    step_size_rad: float = 0.01
    max_club_tracking_error_m: float = 0.04
    max_grip_closure_error_m: float = 0.005

    def __post_init__(self) -> None:
        """Validate positive iteration cap, perturbation limit, and tolerances."""
        if self.max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        pos = ("max_delta_q_rad", "step_size_rad")
        _chk(self, pos, ("max_club_tracking_error_m", "max_grip_closure_error_m"))


@dataclass(frozen=True)
class InverseDynamicsGateResult:
    """Aggregated feasibility gate and optional capped-repair outcome."""

    passed: bool
    failure_reasons: tuple[str, ...]
    failing_frame_indices: tuple[int, ...]
    peak_torque_nm: float
    peak_trunk_torque_nm: float
    min_normal_force_n: float
    peak_friction_ratio: float
    cop_excursion_count: int
    peak_root_force_n: float
    peak_root_moment_nm: float
    peak_club_tracking_error_m: float
    peak_grip_closure_error_m: float
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_iterations: int = 0
    max_applied_delta_q_rad: float = 0.0
    frames: tuple[InverseDynamicsFrame, ...] = ()


def _check_foot(
    w: FootContactWrench, th: DynamicsFeasibilityThresholds, reasons: set[str]
) -> tuple[float | None, float, int]:
    """Check unilateral normal force, Coulomb friction cone, and CoP polygon."""
    gz = th.ground_z_m
    nf, _, cop = compute_wrench_cop(w.contacts, (0.0, 0.0, gz), gz, th.min_loaded_fz_n)
    fz, active = float(nf[2]), w.is_loaded or bool(w.contacts)
    min_fz = min([fz, *(float(c.force_n[2]) for c in w.contacts)])
    if active and min_fz < 0.0:
        reasons.add("unilateral_normal_force_breach")
    sx, sy = nf[0], nf[1]
    shear = math.hypot(sx, sy)
    ratio = shear / fz if fz > 0.0 else (shear / 1e-6 if shear > 1e-9 else 0.0)
    if (shear > th.friction_mu * fz + 1e-12) if fz > 0.0 else (shear > 1e-9):
        reasons.add("friction_cone_breach")
    exc = 0
    if cop is not None and (
        len(w.support_polygon_2d) < 3
        or not point_in_polygon_2d((cop[0], cop[1]), w.support_polygon_2d)
    ):
        reasons.add("cop_outside_support_polygon")
        exc = 1
    return (min_fz if active else None), ratio, exc


def evaluate_inverse_dynamics_gate(
    frames: Sequence[InverseDynamicsFrame], th: DynamicsFeasibilityThresholds
) -> InverseDynamicsGateResult:
    """Evaluate ADR-0059 coupled-trunk torque, GRF/CoP, and root wrench gates."""
    if not frames:
        raise ValueError("frames must be non-empty")
    reasons: set[str] = set()
    bad: list[int] = []
    pt = ptr = pmu = prf = prm = ptrk = pgrp = 0.0
    mfz, exc = float("inf"), 0
    for f in frames:
        fr: set[str] = set()
        if th.require_coupled_trunk and not f.trunk_coupled:
            fr.add("adr0059_uncoupled_trunk")
        for jnt, val in f.joint_torques_nm.items():
            mag, trunk = abs(val), jnt.startswith(_TRUNK)
            pt, ptr = max(pt, mag), max(ptr, mag) if trunk else ptr
            jmax = th.joint_tau_max_nm.get(jnt, th.tau_max_nm)
            lim = min(jmax, th.trunk_coupled_tau_max_nm) if trunk else jmax
            if mag > lim:
                fr.add(_MSG_TRUNK if trunk else _MSG_TAU)
        for w in f.foot_wrenches:
            wfz, wmu, wexc = _check_foot(w, th, fr)
            mfz = min(mfz, wfz) if wfz is not None else mfz
            pmu, exc = max(pmu, wmu), exc + wexc
        rf = math.sqrt(sum(v * v for v in f.root_residual_force_n))
        rm = math.sqrt(sum(v * v for v in f.root_residual_moment_nm))
        prf, prm = max(prf, rf), max(prm, rm)
        ptrk = max(ptrk, f.club_tracking_error_m)
        pgrp = max(pgrp, f.grip_closure_error_m)
        if rf > th.root_force_tol_n:
            fr.add("root_residual_force_exceeded")
        if rm > th.root_moment_tol_nm:
            fr.add("root_residual_moment_exceeded")
        if f.club_tracking_error_m > th.max_club_tracking_error_m:
            fr.add("club_tracking_tolerance_exceeded")
        if f.grip_closure_error_m > th.max_grip_closure_error_m:
            fr.add("grip_closure_tolerance_exceeded")
        if fr:
            reasons.update(fr)
            bad.append(f.frame_index)
    fz_out = 0.0 if mfz == float("inf") else mfz
    pks = (pt, ptr, fz_out, pmu, exc, prf, prm, ptrk, pgrp)
    rs = tuple(sorted(reasons))
    return InverseDynamicsGateResult(not rs, rs, tuple(bad), *pks, frames=tuple(frames))


def _score(
    g: InverseDynamicsGateResult, th: DynamicsFeasibilityThresholds, cfg: RepairConfig
) -> tuple[bool, float]:
    """Return (tolerances_ok, violation_score) for capped perturbation search."""
    t_ok = g.peak_club_tracking_error_m <= cfg.max_club_tracking_error_m
    g_ok = g.peak_grip_closure_error_m <= cfg.max_grip_closure_error_m
    if g.passed and t_ok and g_ok:
        return True, 0.0
    pairs = (
        (g.peak_torque_nm, th.tau_max_nm),
        (g.peak_trunk_torque_nm, th.trunk_coupled_tau_max_nm),
        (g.peak_root_force_n, th.root_force_tol_n),
        (g.peak_root_moment_nm, th.root_moment_tol_nm),
    )
    s = float(g.cop_excursion_count) + max(0.0, -g.min_normal_force_n) / 100.0
    s += max(0.0, g.peak_friction_ratio - th.friction_mu)
    s += sum(max(0.0, v - lim) / max(lim, 1.0) for v, lim in pairs)
    s += 10.0 * max(0.0, g.peak_club_tracking_error_m - cfg.max_club_tracking_error_m)
    s += 100.0 * max(0.0, g.peak_grip_closure_error_m - cfg.max_grip_closure_error_m)
    return (t_ok and g_ok), (1.0 if (not g.passed and s == 0.0) else s)


def _cand(
    cur: tuple[InverseDynamicsFrame, ...],
    orig: tuple[InverseDynamicsFrame, ...],
    bad: set[int],
    dof_delta: tuple[int, float],
    cap: float,
) -> tuple[InverseDynamicsFrame, ...]:
    """Perturb one DOF on failing frames within +/- cap of original posture."""
    dof, delta = dof_delta
    out: list[InverseDynamicsFrame] = []
    for c, o in zip(cur, orig, strict=True):
        if c.frame_index not in bad:
            out.append(c)
        else:
            q = list(c.q_rad)
            q[dof] = min(o.q_rad[dof] + cap, max(o.q_rad[dof] - cap, q[dof] + delta))
            out.append(replace(c, q_rad=tuple(q)))
    return tuple(out)


def repair_dynamically_infeasible_trajectory(
    frames: Sequence[InverseDynamicsFrame],
    th: DynamicsFeasibilityThresholds,
    cfg: RepairConfig,
    evaluator: Evaluator,
) -> InverseDynamicsGateResult:
    """Apply capped perturbation to fix dynamics breaches preserving tolerances."""
    orig = cur = tuple(frames)
    gate = evaluate_inverse_dynamics_gate(cur, th)
    iters, cap = 0, cfg.max_delta_q_rad
    step = min(cfg.step_size_rad, cap)
    tol_ok, cur_s = _score(gate, th, cfg)
    while iters < cfg.max_iters and not (gate.passed and tol_ok):
        iters += 1
        bf, bg, bs, bad = cur, gate, cur_s, set(gate.failing_frame_indices)
        for dof in range(len(cur[0].q_rad)):
            for sgn in (1.0, -1.0):
                co = tuple(evaluator(_cand(cur, orig, bad, (dof, sgn * step), cap)))
                cg = evaluate_inverse_dynamics_gate(co, th)
                ok_t, cs = _score(cg, th, cfg)
                if ok_t and cs < bs:
                    bf, bg, bs = co, cg, cs
        if bs >= cur_s:
            break
        cur, gate, cur_s = bf, bg, bs
        tol_ok, _ = _score(gate, th, cfg)
    dq = 0.0
    for f1, f2 in zip(cur, orig, strict=True):
        dq = max(dq, *(abs(a - b) for a, b in zip(f1.q_rad, f2.q_rad, strict=True)))
    reasons = set(gate.failure_reasons)
    if gate.peak_club_tracking_error_m > cfg.max_club_tracking_error_m:
        reasons.add("club_tracking_tolerance_exceeded")
    if gate.peak_grip_closure_error_m > cfg.max_grip_closure_error_m:
        reasons.add("grip_closure_tolerance_exceeded")
    ok, rs = not reasons, tuple(sorted(reasons))
    kw: dict[str, object] = {
        "passed": ok,
        "failure_reasons": rs,
        "repair_attempted": True,
        "repair_succeeded": ok,
        "repair_iterations": iters,
        "max_applied_delta_q_rad": dq,
        "frames": cur,
    }
    return replace(gate, **kw)  # type: ignore[arg-type]


def build_irreparable_dynamics_fixture(
    num_frames: int = 4, n_dofs: int = 4
) -> tuple[InverseDynamicsFrame, ...]:
    """Build a deterministic irreparable trajectory with impossible loads."""
    if num_frames < 1 or n_dofs < 1:
        raise ValueError("num_frames and n_dofs must be >= 1")
    q0 = tuple(0.0 for _ in range(n_dofs))
    tau = {"lumbar_L5_S1_flexion": 1800.0, "hip_l_extension": 950.0}
    fz, rf, rm = (0.0, 0.0, 350.0), (1200.0, 0.0, 800.0), (450.0, 0.0, 0.0)
    cpr, fcw = ContactPointRecord, FootContactWrench
    out: list[InverseDynamicsFrame] = []
    for i in range(num_frames):
        c1 = cpr(f"l_{i}", "l", "left", (0.12, 0.0, 0.0), fz)
        c2 = cpr(f"r_{i}", "r", "right", (0.12, 0.3, 0.0), fz)
        ws = (fcw("left", (c1,), _PL), fcw("right", (c2,), _PR))
        out.append(InverseDynamicsFrame(i, 0.01 * i, q0, tau, rf, rm, ws))
    return tuple(out)
