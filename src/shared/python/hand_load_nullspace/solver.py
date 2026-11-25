# fmt: off
# ruff: noqa: E301, E302, E305, E306
"""Analytic two-hand null space. J_F is squared force; C is opposing-force cancellation."""
from __future__ import annotations
import math
from dataclasses import dataclass, replace
import numpy as np
from numpy.typing import ArrayLike
from src.shared.python.contracts import ensure, require
_TOL = 1e-8
_AXIAL = "infeasible: u · m' != 0; axial moment needs hand couples"
_NAMES = ("f_l_x", "f_l_y", "f_l_z", "f_r_x", "f_r_y", "f_r_z", "F_x", "F_y", "F_z", "M_x", "M_y", "M_z")
def _pre(ok: bool, msg: str) -> None:
    require(ok, msg)
    if not ok:
        raise ValueError(msg)
def _post(ok: bool, msg: str) -> None:
    ensure(ok, msg)
    if not ok:
        raise ValueError(msg)
def _vec(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    _pre(arr.shape == (3,) and bool(np.isfinite(arr).all()), f"{name} must be a finite 3-vector")
    return arr
def _tup(v: np.ndarray) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))
@dataclass(frozen=True)
class HandLoadProblem:
    p_l: tuple[float, float, float]
    p_r: tuple[float, float, float]
    force_n: tuple[float, float, float]
    moment_nm: tuple[float, float, float]
    m_l_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    m_r_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reference_point_m: tuple[float, float, float] | None = None
    couple_policy: str = "axial_only"
    couple_bound_nm: float | None = None
    objective: str = "min_squared_force"
    load_kind: str = "total"
    max_force_n: float | None = None
    alpha: float | None = None
@dataclass(frozen=True)
class HandLoadSolution:
    availability: str
    feasible: bool
    reason: str
    replay_badge: str
    load_kind: str
    objective: str
    couple_policy: str
    grip_checked: bool
    alpha: float | None
    f_l_n: tuple[float, float, float] | None
    f_r_n: tuple[float, float, float] | None
    m_l_nm: tuple[float, float, float] | None
    m_r_nm: tuple[float, float, float] | None
    net_force_n: tuple[float, float, float] | None
    midhands_moment_nm: tuple[float, float, float] | None
    squared_force: float | None
    opposing_force_cancellation: float | None
@dataclass(frozen=True)
class KktResidual:
    availability: str
    reason: str
    primal_force: float | None
    primal_moment: float | None
    stationarity: float | None
@dataclass(frozen=True)
class ComponentExtrema:
    names: tuple[str, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    width: tuple[float, ...]
    availability: str
    reason: str
def _geometry(p_l, p_r):
    left, right = _vec(p_l, "p_l"), _vec(p_r, "p_r")
    span = right - left
    dist = float(np.linalg.norm(span))
    _pre(dist > 1e-9, "hand positions must be distinct")
    return left, right, span / dist, dist, 0.5 * (left + right)
def _mid_moment(moment, ref, mid, force):
    if ref is None:
        return moment.copy()
    return moment + np.cross(_vec(ref, "reference_point_m") - mid, force)
def _couples(m_net, m_l, m_r, u, policy, bound):
    raw = m_net - m_l - m_r
    axial = float(u @ raw) * u
    perp = raw - axial
    if policy == "fixed":
        return m_l, m_r, perp, _AXIAL if abs(float(u @ raw)) > _TOL else ""
    if policy == "axial_only":
        return m_l + 0.5 * axial, m_r + 0.5 * axial, perp, ""
    _pre(policy == "bounded", f"unknown couple policy {policy}")
    _pre(bound is not None and math.isfinite(bound) and bound > 0, "bounded couples require couple_bound_nm > 0")
    half_ax = 0.5 * abs(float(u @ raw))
    if half_ax > float(bound) + _TOL:
        return m_l, m_r, raw, "infeasible: axial couple exceeds the bound"
    budget = math.sqrt(max(float(bound) ** 2 - half_ax**2, 0.0))
    need = float(np.linalg.norm(perp))
    take = perp if need <= 2 * budget + _TOL else perp * (2 * budget / need)
    share = 0.5 * axial + 0.5 * take
    _post(float(np.linalg.norm(share)) <= float(bound) + 1e-8, "couple share exceeds its bound")
    return m_l + share, m_r + share, perp - take, ""
def _family(force, residual, u, dist, alpha):
    arm = np.cross(u, residual) / dist
    return 0.5 * force + arm + alpha * u, 0.5 * force - arm - alpha * u
def _cancel_alpha(force, residual, u, dist) -> float:
    arm = np.cross(u, residual) / dist
    half = 0.5 * (force - float(u @ force) * u)
    left, right = float(np.linalg.norm(half + arm)), float(np.linalg.norm(half - arm))
    if left + right <= _TOL:
        return 0.0
    return 0.5 * float(u @ force) * (left - right) / (left + right)
def _alpha(problem, force, residual, u, dist) -> tuple[float, str]:
    if problem.alpha is not None:
        _pre(math.isfinite(problem.alpha), "alpha must be finite")
        return float(problem.alpha), "prescribed_alpha"
    _pre(problem.objective in {"min_squared_force", "min_cancellation"}, "objective is not recognised")
    if problem.objective == "min_squared_force":
        return 0.0, problem.objective
    return _cancel_alpha(force, residual, u, dist), problem.objective
def _none(problem, reason, objective, grip=False) -> HandLoadSolution:
    return HandLoadSolution(
        "UNAVAILABLE", False, reason, "allocation-only", problem.load_kind, objective,
        problem.couple_policy, grip, None, None, None, None, None, None, None, None, None,
    )
def _prepare(problem: HandLoadProblem):
    _pre(problem.couple_policy in {"fixed", "axial_only", "bounded"}, "couple policy is not recognised")
    _pre(problem.load_kind in {"total", "ztcf", "zvcf"}, "load_kind must be total, ztcf, or zvcf")
    left, right, u, dist, mid = _geometry(problem.p_l, problem.p_r)
    force = _vec(problem.force_n, "force_n")
    moment = _mid_moment(_vec(problem.moment_nm, "moment_nm"), problem.reference_point_m, mid, force)
    m_l, m_r, residual, reason = _couples(
        moment, _vec(problem.m_l_nm, "m_l_nm"), _vec(problem.m_r_nm, "m_r_nm"),
        u, problem.couple_policy, problem.couple_bound_nm,
    )
    return left, right, u, dist, mid, force, moment, m_l, m_r, residual, reason
def solve(problem: HandLoadProblem) -> HandLoadSolution:
    left, right, u, dist, mid, force, moment, m_l, m_r, residual, reason = _prepare(problem)
    alpha, objective = _alpha(problem, force, residual, u, dist)
    if reason:
        return _none(problem, reason, objective)
    f_l, f_r = _family(force, residual, u, dist, alpha)
    raw_limit = problem.max_force_n
    grip = problem.load_kind == "total" and raw_limit is not None
    if raw_limit is not None and grip:
        limit = float(raw_limit)
        _pre(math.isfinite(limit) and limit > 0, "max_force_n must be positive")
        peak = max(float(np.linalg.norm(f_l)), float(np.linalg.norm(f_r)))
        if peak > limit + _TOL:
            return _none(problem, "infeasible: grip force limit exceeded on the Total load", objective, True)
    closed = m_l + m_r + np.cross(left - mid, f_l) + np.cross(right - mid, f_r)
    cancel = float(np.linalg.norm(f_l) + np.linalg.norm(f_r) - np.linalg.norm(force))
    _post(float(np.linalg.norm(f_l + f_r - force)) <= _TOL, "hand forces must sum to F")
    _post(float(np.linalg.norm(closed - moment)) <= _TOL, "midhands moment must close")
    _post(cancel >= -1e-8, "opposing-force cancellation must be non-negative")
    if objective == "min_squared_force":
        _post(abs(alpha) <= _TOL, "minimum squared force is at alpha = 0")
    return HandLoadSolution(
        "available", True, "", "allocation-only", problem.load_kind, objective, problem.couple_policy, grip,
        alpha, _tup(f_l), _tup(f_r), _tup(m_l), _tup(m_r), _tup(force), _tup(moment),
        float(f_l @ f_l + f_r @ f_r), cancel,
    )
def grasp_null_residual(p_l, p_r) -> float:
    left, right, u, _dist, mid = _geometry(p_l, p_r)
    def skew(v):
        x, y, z = (float(v[0]), float(v[1]), float(v[2]))
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    grasp, eye, direction = np.zeros((6, 12)), np.eye(3), np.zeros(12)
    grasp[0:3, 0:3] = grasp[0:3, 6:9] = eye
    grasp[3:6, 3:6] = grasp[3:6, 9:12] = eye
    grasp[3:6, 0:3], grasp[3:6, 6:9] = skew(left - mid), skew(right - mid)
    direction[0:3], direction[6:9] = u, -u
    residual = float(np.linalg.norm(grasp @ direction))
    _post(residual <= _TOL, "G N must be 0")
    return residual
def kkt_residual(problem: HandLoadProblem, solution: HandLoadSolution) -> KktResidual:
    if not solution.feasible or solution.f_l_n is None or solution.f_r_n is None or solution.m_l_nm is None or solution.m_r_nm is None:
        return KktResidual("UNAVAILABLE", solution.reason, None, None, None)
    left, right, u, dist, mid, force, moment, _ml, _mr, residual, reason = _prepare(problem)
    if reason:
        return KktResidual("UNAVAILABLE", reason, None, None, None)
    f_l, f_r = _vec(solution.f_l_n, "f_l_n"), _vec(solution.f_r_n, "f_r_n")
    m_l, m_r = _vec(solution.m_l_nm, "m_l_nm"), _vec(solution.m_r_nm, "m_r_nm")
    closed = m_l + m_r + np.cross(left - mid, f_l) + np.cross(right - mid, f_r)
    force_res = float(np.linalg.norm(f_l + f_r - force))
    moment_res = float(np.linalg.norm(closed - moment))
    if solution.objective == "min_cancellation":
        n_l, n_r = float(np.linalg.norm(f_l)), float(np.linalg.norm(f_r))
        if min(n_l, n_r) <= _TOL:
            return KktResidual("UNAVAILABLE", "norm stationarity needs a nonzero hand force", force_res, moment_res, None)
        u_l, u_r = f_l / n_l, f_r / n_r
        lam_f, lam_m = 0.5 * (u_l + u_r), np.cross(u, u_r - u_l) / dist
        rows = (u_l - lam_f + np.cross(left - mid, lam_m), u_r - lam_f + np.cross(right - mid, lam_m))
    else:
        lam_m = 4.0 * residual / (dist * dist)
        rows = (2 * f_l - force + np.cross(left - mid, lam_m), 2 * f_r - force + np.cross(right - mid, lam_m))
    station = max(float(np.linalg.norm(row)) for row in rows)
    return KktResidual("available", "", force_res, moment_res, station)
def component_extrema(problem: HandLoadProblem, alpha_lo: float, alpha_hi: float) -> ComponentExtrema:
    _pre(math.isfinite(alpha_lo) and math.isfinite(alpha_hi) and alpha_hi > alpha_lo, "alpha interval must be finite with lo < hi")
    ends = [solve(replace(problem, alpha=value)) for value in (alpha_lo, alpha_hi)]
    if not all(item.feasible for item in ends):
        return ComponentExtrema(_NAMES, (), (), (), "UNAVAILABLE", ends[0].reason or ends[1].reason)
    packed = [np.concatenate([s.f_l_n, s.f_r_n, s.net_force_n, s.midhands_moment_nm]) for s in ends]
    low, high = np.minimum(packed[0], packed[1]), np.maximum(packed[0], packed[1])
    return ComponentExtrema(_NAMES, tuple(map(float, low)), tuple(map(float, high)), tuple(map(float, high - low)), "available", "")
def time_weighted_squared_force(solutions, dt: float) -> float | None:
    _pre(math.isfinite(dt) and dt > 0, "dt must be positive")
    if not solutions or any(item.squared_force is None for item in solutions):
        return None
    return float(sum(item.squared_force for item in solutions) * dt)
def synthetic_forward_step(position, velocity, omega, force, moment, mass, inertia, dt):
    _pre(math.isfinite(mass) and mass > 0 and math.isfinite(dt) and dt > 0, "mass and dt must be positive")
    iner = np.asarray(inertia, dtype=np.float64)
    _pre(iner.shape == (3, 3) and bool(np.isfinite(iner).all()), "inertia must be a finite 3x3")
    try:
        np.linalg.cholesky(iner)
    except np.linalg.LinAlgError as exc:
        raise ValueError("inertia must be symmetric positive definite") from exc
    vel = _vec(velocity, "velocity") + _vec(force, "force") / mass * dt
    omg = _vec(omega, "omega") + np.linalg.solve(iner, _vec(moment, "moment")) * dt
    return _vec(position, "position") + vel * dt, vel, omg
def certify_forward_replay(solution, *, mass, inertia, position, velocity, omega, dt, reference_position, reference_velocity, reference_omega, atol=1e-8):
    if not solution.feasible:
        return solution
    pos, vel, omg = synthetic_forward_step(
        position, velocity, omega, solution.net_force_n, solution.midhands_moment_nm, mass, inertia, dt,
    )
    matched = (
        np.allclose(pos, reference_position, atol=atol)
        and np.allclose(vel, reference_velocity, atol=atol)
        and np.allclose(omg, reference_omega, atol=atol)
    )
    return replace(solution, replay_badge="forward-replayed" if matched else "allocation-only")
# fmt: on
