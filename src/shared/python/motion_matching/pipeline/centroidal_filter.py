"""Centroidal feasibility filter v2 for the tracked reference (#11669).

The tracked reference of a full-body swing demands a ground reaction the feet
cannot deliver in the finish: the full zero-moment point (including the rate of
angular momentum) leaves the support polygon, the vertical force dips below a
fraction of body weight, and the horizontal force leaves the friction cone.
``zmp_filter`` only shifts the centre of mass by the cart-table rule and so
cannot see the angular-momentum term.

This filter perturbs the joint trajectory ``q`` over the finish window,
``q + dq``, with ``dq`` a cubic B-spline in time.  Each outer iteration
linearises, per frame, the centre of mass, centroidal angular momentum, marker
residuals, loop closure and stance-foot contact velocity at the current
trajectory and solves one bounded least-squares problem:

* minimise the marker-metric size of ``dq`` (stay close to the IK reference),
  with a small ridge and acceleration penalty;
* hold loop closures and stance feet fixed (soft equalities);
* keep the linearised ZMP inside each support edge shrunk by a margin, keep
  ``Fz >= min_load * W`` and keep ``|Fh| <= mu_c * Fz`` (octagonal cone),
  through an active-set penalty;
* bound every spline coefficient (trust region).

A step is accepted only if the exact (nonlinear) violation merit falls;
otherwise the trust region is halved.  Nothing is smoothed or loosened to
reach a target: the report carries the before and after numbers whether or
not the acceptance fractions are met.

Frames before ``t_on_s`` are untouched; ``dq`` is zero with zero slope at
``t_on_s``.  Pure numerics live in the module level helpers so they can be
tested on synthetic problems; ``centroidal_filter`` is the pipeline entry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import logging
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear
from scipy.spatial import ConvexHull, QhullError
import scipy.sparse as sp

GRAVITY_M_S2 = 9.81
FRICTION_OCTAGON = tuple(np.arange(8) * np.pi / 4)
OCTAGON_INSCRIBED = float(np.cos(np.pi / 8))
UNLOADED_OUTSIDE_M = 1.0
CONTACT_TOLERANCE_M = 0.005
MARKER_PRIOR_M = 0.05
VIOLATION_TOL = 1e-5


@dataclass(frozen=True)
class CentroidalFilterConfig:
    """Tuning of the centroidal feasibility filter (all strictly validated)."""

    t_on_s: float = 0.9
    t_con_s: float = 1.0
    friction_mu: float = 0.6
    min_load_bw: float = 0.5
    margin_m: float = 0.03
    ridge: float = 1.0
    accel_weight: float = 1e-8
    penalty: float = 1e2
    spacing_s: float = 0.05
    bound_rad: float = 0.03
    iterations: int = 10
    active_passes: int = 12
    closure_weight: float = 1e4
    foot_weight: float = 1e3

    def __post_init__(self) -> None:
        if not 0.0 <= self.t_on_s <= self.t_con_s:
            raise ValueError("need 0 <= t_on_s <= t_con_s")
        if not 0.0 < self.friction_mu <= 2.0:
            raise ValueError("friction_mu must lie in (0, 2]")
        if not 0.0 <= self.min_load_bw < 1.0:
            raise ValueError("min_load_bw must lie in [0, 1)")
        positive = (
            self.margin_m + 1.0,
            self.ridge,
            self.penalty,
            self.spacing_s,
            self.bound_rad,
        )
        if min(positive) <= 0.0 or self.margin_m < 0.0:
            raise ValueError("margin, ridge, penalty, spacing, bound must be positive")
        if self.iterations < 1 or self.active_passes < 1:
            raise ValueError("iterations and active_passes must be at least 1")


# ---------------------------------------------------------------------------
# Pure numerics (synthetic-testable)
# ---------------------------------------------------------------------------


def ccw_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull vertices of (n, 2) points (n >= 3)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("hull needs at least three (x, y) points")
    return pts[ConvexHull(pts).vertices]


def support_edges(hull: np.ndarray, margin_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normals and offsets of a CCW hull shrunk by ``margin_m``.

    A point ``p`` is inside the shrunk polygon when ``normals @ p <= offsets``.
    """
    poly = np.asarray(hull, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("hull must be a (n>=3, 2) polygon")
    if margin_m < 0.0:
        raise ValueError("margin must be nonnegative")
    edge = np.roll(poly, -1, axis=0) - poly
    normals = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals, np.einsum("ij,ij->i", normals, poly) - margin_m


def time_basis(times: np.ndarray, spacing_s: float, drop: int = 2) -> np.ndarray:
    """Cubic B-spline basis on ``times`` (nW, nb); first ``drop`` columns removed.

    Dropping two columns makes the correction and its slope vanish at the
    first sample, so the filtered path joins the reference smoothly.
    """
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 8 or np.any(np.diff(t) <= 0):
        raise ValueError("need at least eight increasing times")
    span = t[-1] - t[0]
    nseg = max(int(round(span / spacing_s)), 4)
    inner = np.linspace(t[0], t[-1], nseg + 1)
    knots = np.r_[[t[0]] * 3, inner, [t[-1]] * 3]
    design = BSpline.design_matrix(np.clip(t, t[0], t[-1] - 1e-12), knots, 3)
    return design.toarray()[:, drop:]


def _stable_cholesky(matrix: np.ndarray) -> np.ndarray:
    """Cholesky factor, adding growing diagonal jitter if round-off breaks it.

    Active rows with a very small vertical force give huge, nearly parallel
    constraint rows; the penalty Hessian then loses definiteness to round-off
    even though it is positive semidefinite by construction.
    """
    scale = float(np.trace(matrix)) / len(matrix)
    jitter = 0.0
    for _ in range(12):
        try:
            return np.linalg.cholesky(matrix + jitter * np.eye(len(matrix)))
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, 1e-14 * scale)
    raise ValueError("penalty Hessian is not positive definite")


def solve_bounded_penalty(
    objective: sp.csr_matrix,
    target: np.ndarray,
    rows: sp.csr_matrix,
    limits: np.ndarray,
    config: CentroidalFilterConfig,
) -> np.ndarray:
    """Minimise ``|objective x - target|^2`` s.t. ``rows x <= limits``, box bound.

    The inequalities enter through an active-set quadratic penalty
    (``config.penalty``); the box ``|x| <= config.bound_rad`` is exact.

    Returns:
        Coefficient vector ``x`` (length ``objective.shape[1]``).
    """
    if objective.shape[0] != len(target) or rows.shape[0] != len(limits):
        raise ValueError("objective/rows and right-hand sides must agree in length")
    if not (np.isfinite(target).all() and np.isfinite(limits).all()):
        raise ValueError("targets and limits must be finite")
    if objective.shape[1] != rows.shape[1]:
        raise ValueError("objective and constraint rows need the same columns")
    gram = np.asarray((objective.T @ objective).todense())
    rhs = np.asarray(objective.T @ target).ravel()
    dense_rows = np.asarray(rows.todense())
    n = gram.shape[0]
    x = np.zeros(n)
    active = np.zeros(len(limits), dtype=bool)
    bound = config.bound_rad
    for sweep in range(config.active_passes):
        violated = (dense_rows @ x - limits) > VIOLATION_TOL
        if sweep > 0 and not (violated & ~active).any():
            break
        active |= violated
        act = dense_rows[active]
        hess = gram + config.penalty * (act.T @ act) + 1e-12 * np.eye(n)
        grad = rhs + config.penalty * (act.T @ limits[active])
        chol = _stable_cholesky(hess)
        reduced = np.linalg.solve(chol, grad)
        x = lsq_linear(
            chol.T, reduced, bounds=(-bound, bound), method="bvls", tol=1e-8
        ).x
    return x


def violation_merit(
    zmp: dict[str, Any], times: np.ndarray, config: CentroidalFilterConfig
) -> float:
    """Exact (nonlinear) sum of squared violations over ``t >= t_con_s``.

    Terms: ZMP distance outside the support polygon, vertical-force shortfall
    below ``min_load_bw`` (in body weights) and horizontal force beyond the
    friction cone (in body weights).  Unloaded frames count as a full metre.
    """
    window = np.asarray(times) >= config.t_con_s
    outside = np.where(zmp["unloaded"], UNLOADED_OUTSIDE_M, zmp["outside_m"])[window]
    grf = np.asarray(zmp["grf_over_weight"])[window]
    shortfall = np.maximum(0.0, config.min_load_bw - grf[:, 2])
    horizontal = np.linalg.norm(grf[:, :2], axis=1)
    excess = np.maximum(0.0, horizontal - config.friction_mu * np.maximum(grf[:, 2], 0))
    return float(np.sum(outside**2) + np.sum(shortfall**2) + np.sum(excess**2))


# ---------------------------------------------------------------------------
# Linearisation of one frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameLinearisation:
    """First-order model of one frame (all Jacobians w.r.t. the nv dofs)."""

    com_jac: np.ndarray  # (3, nv) centre of mass
    momentum_jac: np.ndarray  # (3, nv) centroidal angular momentum
    marker_factor: np.ndarray  # (nv, nv) L with Gram = L L^T
    closure_residual: np.ndarray
    closure_jac: np.ndarray
    foot_jac: np.ndarray  # (3 * contacts, nv)

    def __post_init__(self) -> None:
        nv = self.com_jac.shape[1]
        if self.com_jac.shape != (3, nv) or self.momentum_jac.shape != (3, nv):
            raise ValueError("com and momentum Jacobians must be (3, nv)")
        if self.marker_factor.shape != (nv, nv):
            raise ValueError("marker factor must be (nv, nv)")
        if self.closure_jac.shape[1] != nv or self.foot_jac.shape[1] != nv:
            raise ValueError("closure and foot Jacobians must have nv columns")


def zmp_sensitivity(
    wrench: np.ndarray, zmp_point: Callable[[np.ndarray, np.ndarray, np.ndarray], Any]
) -> np.ndarray:
    """Central-difference ``d zmp_xy / d (com, force, moment)`` (2, 9)."""
    x0 = np.asarray(wrench, dtype=float)
    if x0.shape != (9,):
        raise ValueError("wrench must hold (com, force, moment): 9 numbers")
    sens = np.zeros((2, 9))
    for j in range(9):
        step = np.zeros(9)
        eps = 1e-3 * max(1.0, abs(x0[j]))
        step[j] = eps
        hi = np.asarray(zmp_point(*np.split(x0 + step, 3)))[:2]
        lo = np.asarray(zmp_point(*np.split(x0 - step, 3)))[:2]
        sens[:, j] = (hi - lo) / (2 * eps)
    return sens


def constraint_rows(
    lin: FrameLinearisation,
    wrench: np.ndarray,
    base_zmp: np.ndarray,
    hull: np.ndarray,
    sens: np.ndarray,
    mass_kg: float,
    config: CentroidalFilterConfig,
) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    """Linearised feasibility rows of one frame.

    ``wrench`` is ``(com, force, moment)`` and ``base_zmp`` the ZMP it implies.
    Each row is ``(pos_vec, acc_vec, bound, kind)`` meaning
    ``pos_vec . dq_k + acc_vec . ddq_k <= bound`` where ``dq_k`` and ``ddq_k``
    are the correction and its second time derivative at the frame
    (``dC = Jc dq``, ``dF = m Jc ddq``, ``dM = H ddq``).
    """
    weight = mass_kg * GRAVITY_M_S2
    jc, hm = lin.com_jac, lin.momentum_jac
    force = np.asarray(wrench, dtype=float)[3:6]
    point = np.asarray(base_zmp, dtype=float)
    rows: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    normals, offsets = support_edges(hull, config.margin_m)
    for normal, offset in zip(normals, offsets, strict=True):
        w9 = normal @ sens
        pos = w9[0:3] @ jc
        acc = mass_kg * (w9[3:6] @ jc) + w9[6:9] @ hm
        rows.append((pos, acc, float(offset - normal @ point), "zmp"))
    zero = np.zeros(jc.shape[1])
    rows.append(
        (
            zero,
            -mass_kg * jc[2] / weight,
            float((force[2] - config.min_load_bw * weight) / weight),
            "fz",
        )
    )
    cone = config.friction_mu * OCTAGON_INSCRIBED
    for angle in FRICTION_OCTAGON:
        unit = np.array([np.cos(angle), np.sin(angle)])
        acc = (mass_kg * (unit @ jc[:2]) - cone * mass_kg * jc[2]) / weight
        rows.append(
            (zero, acc, float((cone * force[2] - unit @ force[:2]) / weight), "cone")
        )
    return rows


# ---------------------------------------------------------------------------
# MuJoCo access (kept in one class so the private adapter reach stays local)
# ---------------------------------------------------------------------------


class MujocoFrames:
    """Per-frame linearisation and wrench of a MuJoCo full-body model."""

    def __init__(self, sim: Any, kin: Any, ground: Any, valid: np.ndarray) -> None:
        adapter = sim.adapter
        self.sim, self.kin, self.ground, self.valid = sim, kin, ground, valid
        self.mj, self.model, self.data = adapter._mj, adapter.model, adapter.data
        self.dof = np.asarray(sim._dof)
        self.nv = int(sim.nv)
        self.qpos = np.array([self.model.joint(n).qposadr[0] for n in sim.names])
        spheres = adapter._spheres
        self.site_ids = [spheres[s]["site_id"] for s in spheres]
        self.radii = np.array([spheres[s]["radius"] for s in spheres])
        self.adapter = adapter

    def _forward(self, q_row: np.ndarray) -> None:
        self.data.qpos[self.qpos] = q_row
        self.mj.mj_fwdPosition(self.model, self.data)

    def _touching(self) -> np.ndarray:
        centres = np.array([self.data.site_xpos[i] for i in self.site_ids])
        height = centres[:, 2] - self.radii - self.ground.height_m
        return height <= CONTACT_TOLERANCE_M

    def wrench_series(
        self, times: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centre of mass, net contact force and moment about it, per frame."""
        from src.shared.python.motion_matching import full_body_forward_dynamics as fs

        vel = np.gradient(q, times, axis=0)
        acc = np.gradient(vel, times, axis=0)
        n, dt = len(times), 1e-4
        com, force, moment = np.empty((n, 3)), np.empty((n, 3)), np.empty((n, 3))
        for k in range(n):
            c0, p0, l0 = fs._compute_frame_momentum(
                self.adapter, self.sim, q[k], vel[k]
            )
            c1, p1, l1 = fs._compute_frame_momentum(
                self.adapter, self.sim, q[k] + vel[k] * dt, vel[k] + acc[k] * dt
            )
            com[k] = c0
            force[k] = (p1 - p0) / dt - self.sim.gravity * self.sim.mass_kg
            moment[k] = (l1 - l0) / dt
        return com, force, moment

    def zmp_point(self, com: Any, force: Any, moment: Any) -> np.ndarray:
        """ZMP of a (com, force, moment) triple on this model's ground."""
        from src.shared.python.motion_matching import full_body_forward_dynamics as fs

        return np.asarray(fs._compute_zmp_point(com, force, moment, self.ground))[:2]

    def linearise(self, q_row: np.ndarray, frame: int) -> FrameLinearisation:
        """First-order model of frame ``frame`` at coordinates ``q_row``."""
        self._forward(q_row)
        nv_full = self.model.nv
        jac_com = np.zeros((3, nv_full))
        self.mj.mj_jacSubtreeCom(self.model, self.data, jac_com, 1)
        momentum = np.zeros((3, nv_full))
        self.mj.mj_angmomMat(self.model, self.data, momentum, 1)
        jm = self.kin._marker_jacobian(self.kin._positions())
        ok = self.valid[frame]
        gram = np.einsum("mik,mil->kl", jm[ok], jm[ok]) / max(int(ok.sum()), 1)
        gram += MARKER_PRIOR_M**2 * np.eye(self.nv)
        rows: list[np.ndarray] = []
        jacs: list[np.ndarray] = []
        self.kin._append_closure(rows, jacs, 1.0, 1.0)
        feet = []
        for touching, site in zip(self._touching(), self.site_ids, strict=True):
            if touching:
                jp = np.zeros((3, nv_full))
                self.mj.mj_jacSite(self.model, self.data, jp, None, site)
                feet.append(jp[:, self.dof])
        return FrameLinearisation(
            com_jac=jac_com[:, self.dof],
            momentum_jac=momentum[:, self.dof],
            marker_factor=np.linalg.cholesky(gram),
            closure_residual=np.concatenate(rows),
            closure_jac=np.vstack(jacs),
            foot_jac=np.vstack(feet) if feet else np.zeros((0, self.nv)),
        )


# ---------------------------------------------------------------------------
# One linearised step and the outer loop
# ---------------------------------------------------------------------------


def _dense_block(
    parts: dict[str, list], first_row: int, col0: int, block: np.ndarray, rhs: Any
) -> int:
    """Append a dense row block at (first_row, col0) to COO ``parts``."""
    rows, cols = block.shape
    parts["i"].append(np.repeat(first_row + np.arange(rows), cols))
    parts["j"].append(np.tile(col0 + np.arange(cols), rows))
    parts["v"].append(block.ravel())
    parts["d"].append(np.asarray(rhs, dtype=float))
    return first_row + rows


def objective_system(
    lins: list[FrameLinearisation],
    accel: np.ndarray,
    config: CentroidalFilterConfig,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Least-squares objective over all window frames (marker metric first).

    ``accel`` is the (nW, nW) second-difference operator on the window.
    """
    nv = lins[0].com_jac.shape[1]
    n = len(lins) * nv
    parts: dict[str, list] = {"i": [], "j": [], "v": [], "d": []}
    row = 0
    w_cl, w_ft = np.sqrt(config.closure_weight), np.sqrt(config.foot_weight)
    for r, lin in enumerate(lins):
        col = r * nv
        row = _dense_block(parts, row, col, lin.marker_factor.T, np.zeros(nv))
        row = _dense_block(
            parts, row, col, w_cl * lin.closure_jac, -w_cl * lin.closure_residual
        )
        if len(lin.foot_jac):
            row = _dense_block(
                parts, row, col, w_ft * lin.foot_jac, np.zeros(len(lin.foot_jac))
            )
    base = sp.csr_matrix(
        (
            np.concatenate(parts["v"]),
            (np.concatenate(parts["i"]), np.concatenate(parts["j"])),
        ),
        shape=(row, n),
    )
    smooth = sp.kron(
        sp.csr_matrix(np.sqrt(config.accel_weight) * accel), sp.identity(nv)
    )
    ridge = np.sqrt(config.ridge) * sp.identity(n)
    stacked = sp.vstack([base, smooth, ridge], format="csr")
    target = np.concatenate([*parts["d"], np.zeros(2 * n)])
    return stacked, target


def constraint_system(
    frame_rows: list[list[tuple[np.ndarray, np.ndarray, float, str]]],
    accel: np.ndarray,
    nv: int,
) -> tuple[sp.csr_matrix, np.ndarray, list[str]]:
    """Stack per-window-frame rows into ``R x <= b`` over all frame corrections.

    ``frame_rows[r]`` is empty for frames before the constraint onset.
    """
    n = len(frame_rows) * nv
    ii: list[np.ndarray] = []
    jj: list[np.ndarray] = []
    vv: list[np.ndarray] = []
    limits: list[float] = []
    kinds: list[str] = []
    for r, rows in enumerate(frame_rows):
        coupled = np.flatnonzero(accel[r])
        for pos, acc, bound, kind in rows:
            row = len(limits)
            entries: dict[int, np.ndarray] = {r: pos.copy()}
            for other in (int(c) for c in coupled):
                entries[other] = entries.get(other, 0.0) + accel[r, other] * acc
            for frame, vec in entries.items():
                ii.append(np.full(nv, row))
                jj.append(frame * nv + np.arange(nv))
                vv.append(np.asarray(vec, dtype=float))
            limits.append(bound)
            kinds.append(kind)
    mat = sp.csr_matrix(
        (np.concatenate(vv), (np.concatenate(ii), np.concatenate(jj))),
        shape=(len(limits), n),
    )
    return mat, np.asarray(limits), kinds


def _finite_row(row: tuple[np.ndarray, np.ndarray, float, str]) -> bool:
    """True when a constraint row has only finite numbers."""
    pos, acc, bound, _ = row
    return bool(
        np.isfinite(pos).all() and np.isfinite(acc).all() and np.isfinite(bound)
    )


def second_difference(times: np.ndarray) -> np.ndarray:
    """Dense (N, N) operator of the second time derivative (np.gradient twice)."""
    first = np.asarray(np.gradient(np.eye(len(times)), times, axis=0))
    return first @ first


def centroidal_step(
    frames: MujocoFrames,
    times: np.ndarray,
    q: np.ndarray,
    zmp: dict[str, Any],
    config: CentroidalFilterConfig,
) -> np.ndarray:
    """One linearised, trust-region-bounded correction ``dq`` (N, nv)."""
    window = np.flatnonzero(times >= config.t_on_s)
    if len(window) < 8:
        raise ValueError("finish window holds fewer than eight frames")
    nv, mass = frames.nv, frames.sim.mass_kg
    accel = second_difference(times)
    local = accel[np.ix_(window, window)]
    com, force, moment = frames.wrench_series(times, q)
    lins = [frames.linearise(q[k], int(k)) for k in window]
    frame_rows: list[list[tuple[np.ndarray, np.ndarray, float, str]]] = []
    for r, k in enumerate(window):
        if times[k] < config.t_con_s:
            frame_rows.append([])
            continue
        wrench = np.r_[com[k], force[k], moment[k]]
        base = frames.zmp_point(com[k], force[k], moment[k])
        sens = zmp_sensitivity(wrench, frames.zmp_point)
        try:
            hull = ccw_hull(np.asarray(zmp["hull_xy"][k]))
        except (ValueError, QhullError):
            frame_rows.append([])  # no support polygon: nothing to linearise against
            continue
        rows_k = constraint_rows(lins[r], wrench, base, hull, sens, mass, config)
        frame_rows.append(
            [row for row in rows_k if _finite_row(row)]  # Fz ~ 0: ZMP undefined
        )
    objective, target = objective_system(lins, local, config)
    rows, limits, _ = constraint_system(frame_rows, local, nv)
    basis = time_basis(times[window], config.spacing_s)
    lift = sp.kron(sp.csr_matrix(basis), sp.identity(nv), format="csr")
    coeffs = solve_bounded_penalty(
        objective @ lift, target, rows @ lift, limits, config
    )
    step = np.zeros_like(q)
    step[window] = (lift @ coeffs).reshape(len(window), nv)
    return step


def centroidal_filter(
    lane: Any,
    kin: Any,
    sim: Any,
    q_track: np.ndarray,
    zmp: dict[str, Any],
    log: logging.Logger,
    config: CentroidalFilterConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Centroidal feasibility filter on the tracked reference (#11669).

    Same contract as ``dynamics.zmp_filter``: returns the filtered trajectory,
    its reference ZMP and a report.  Steps that do not lower the exact
    violation merit are rejected and the trust region halves.
    """
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs
    from src.shared.python.motion_matching.pipeline.dynamics import zmp_summary
    from src.shared.python.motion_matching.pipeline.reference import marker_errors

    cfg = config or CentroidalFilterConfig()
    times = np.asarray(lane.times, dtype=float)
    frames = MujocoFrames(sim, kin, lane.ground, np.asarray(lane.valid))

    def markers(q: np.ndarray) -> float:
        err = marker_errors(kin, q, lane.points)
        return float(np.sqrt(np.mean(err[lane.valid] ** 2)))

    report: dict[str, Any] = {
        "config": asdict(cfg),
        "before": {**zmp_summary(zmp, times), "marker_rms_m": markers(q_track)},
        "passes": [],
    }
    merit = violation_merit(zmp, times, cfg)
    bound = cfg.bound_rad
    for it in range(cfg.iterations):
        step_cfg = replace(cfg, bound_rad=bound)
        trial = q_track + centroidal_step(frames, times, q_track, zmp, step_cfg)
        trial_zmp = fs.reference_zmp(sim, times, trial, lane.ground)
        trial_merit = violation_merit(trial_zmp, times, cfg)
        accepted = trial_merit < merit
        report["passes"].append(
            {
                "pass": it + 1,
                "bound_rad": bound,
                "accepted": accepted,
                "merit_before": merit,
                "merit_after": trial_merit,
                "marker_rms_m": markers(trial),
                **zmp_summary(trial_zmp, times),
            }
        )
        log.info(
            "centroidal filter pass %d: merit %.3f -> %.3f (%s), 1.0-1.5 s outside %.2f",
            it + 1,
            merit,
            trial_merit,
            "accepted" if accepted else "rejected",
            report["passes"][-1]["outside_fraction_1s_to_1_5s"],
        )
        if accepted:
            q_track, zmp, merit = trial, trial_zmp, trial_merit
        else:
            bound *= 0.5
    report["after"] = {**zmp_summary(zmp, times), "marker_rms_m": markers(q_track)}
    return q_track, zmp, report
