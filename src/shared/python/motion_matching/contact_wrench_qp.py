# mypy: disable-error-code="arg-type"
# pydrake stubs demand ndarray[float64, Any]; the arrays here are float64.
"""Contact-wrench tracking QP for the full-body computed-torque controller (#11670).

The baseline controller (:func:`full_body_forward_dynamics.tracking_controller`)
asks every actuated joint for the reference acceleration and lets the unactuated
root absorb whatever ground reaction that implies. Late in the swing the
reaction it implies lies outside what a foot can deliver: the tangential force
leaves the friction cone, the centre of pressure leaves the sole and the foot
slides and pivots. This controller keeps the baseline torque law but first asks
whether the resulting reaction is deliverable, and bends the joint accelerations
only as far as needed to make it so.

Per control step it solves one convex QP over the actuated joint accelerations
``a``, the root acceleration ``a_r`` and one contact wrench per foot
``w_k = (f_k, M_k)`` taken about the foot's support centre:

.. math::

    \\min\\; \\|a - a^*\\|^2_{W_a} + \\|a_r - a_r^0\\|^2_{W_r}
        + \\|w - w^0\\|^2_{W_w} + \\rho \\|s\\|^2

    \\text{s.t. } A_G \\begin{bmatrix}a_r\\\\a\\end{bmatrix} + \\dot A_G v
        = \\sum_k \\begin{bmatrix} f_k \\\\ (p_k - c) \\times f_k + M_k
        \\end{bmatrix} + \\begin{bmatrix} m g \\\\ 0 \\end{bmatrix},
    \\qquad C_k w_k \\le s_k

where ``A_G`` is the centroidal momentum matrix (internal forces such as the
dual-grip weld cancel), ``a*`` is the baseline target acceleration, ``a_r^0``
and ``w^0`` are the root acceleration and the wrench the plant would produce
from the baseline torques, and ``C_k w_k <= s_k`` is the wrench cone of one foot
(Caron et al. 2015): a linearised friction pyramid, centre-of-pressure bounds
over the touching sphere polygon, a torsion bound and ``f_z >= 0``, with a
nonnegative slack ``s_k`` so the QP is always feasible.

Because the nominal point ``(a*, a_r^0, w^0)`` has zero cost, the controller is
exactly the baseline whenever the baseline's own reaction satisfies the cone.
The solver is Drake's ``ClarabelSolver``. The joint torques follow from the
modified target through the same pseudo-inverse of the plant's affine map as
the baseline.

Preconditions are validated and raise ``ValueError``; only the MuJoCo adapter is
supported because the momentum matrix comes from MuJoCo.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import logging
import math
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        FullBodySimulator,
    )

logger = logging.getLogger(__name__)

Array: TypeAlias = NDArray[np.float64]
Controller: TypeAlias = Callable[[float, Array, Array], Array]

CONTACT_CLEARANCE_M = 0.008
CONE_ROWS_PER_FOOT = 11
MIN_EXTENT_M = 1e-3


@dataclass(frozen=True)
class WrenchQPConfig:
    """Weights and limits of the contact-wrench QP; all validated."""

    friction: float | None = None
    """Cone friction coefficient; ``None`` takes the plant's dynamic friction."""
    torsion_length_m: float = 0.03
    """Moment arm of the torsion bound ``|M_n| <= mu f_z * length``."""
    cop_margin_m: float = 0.0
    """Shrinks the centre-of-pressure rectangle on every side."""
    joint_weight: float = 1.0
    root_weight: float = 1.0
    wrench_weight: float = 1e-2
    slack_weight: float = 1e4
    root_gain: float = 0.0
    """Fraction of the root PD toward the reference added to ``a_r^0``."""
    root_omega_rad_s: float = 12.0

    def __post_init__(self) -> None:
        if self.friction is not None and not (
            math.isfinite(self.friction) and self.friction > 0
        ):
            raise ValueError("friction must be positive and finite or None")
        for name in ("torsion_length_m", "cop_margin_m", "root_gain"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "joint_weight",
            "root_weight",
            "wrench_weight",
            "slack_weight",
            "root_omega_rad_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class FootSupport:
    """Support geometry of one foot at one instant, in the world frame."""

    name: str
    centre: Array  # (3,) support centre on the ground plane
    axis_long: Array  # (3,) unit tangent along the foot
    axis_lat: Array  # (3,) unit tangent across the foot, normal x long
    extent_long: tuple[float, float]  # (min, max) of the sole along axis_long
    extent_lat: tuple[float, float]
    touching: bool

    def __post_init__(self) -> None:
        for label, (lo, hi) in (
            ("extent_long", self.extent_long),
            ("extent_lat", self.extent_lat),
        ):
            if not lo <= 0.0 <= hi:
                raise ValueError(f"{label} must bracket the support centre")


def wrench_cone_rows(
    support: FootSupport,
    normal: Array,
    *,
    friction: float,
    torsion_length_m: float,
    cop_margin_m: float = 0.0,
) -> Array:
    """Rows ``C`` of the foot wrench cone, ``C @ [f, M] <= 0`` (world frame).

    Eleven rows: four friction-pyramid faces (inscribed, ``mu / sqrt(2)`` per
    axis), four centre-of-pressure bounds over the sole rectangle, two torsion
    bounds and ``f_z >= 0``. For a foot that is not touching, every row is
    replaced by the bound ``w = 0`` elsewhere; this function still returns the
    rows so callers can test them.

    Preconditions: ``friction > 0``; ``torsion_length_m, cop_margin_m >= 0``;
    ``normal`` a unit vector. Postcondition: shape ``(11, 6)``.
    """
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isclose(np.linalg.norm(n), 1.0):
        raise ValueError("normal must be a unit 3-vector")
    if not friction > 0 or torsion_length_m < 0 or cop_margin_m < 0:
        raise ValueError("friction must be positive, lengths nonnegative")
    mu = friction / math.sqrt(2.0)
    e1, e2 = support.axis_long, support.axis_lat
    x_lo = support.extent_long[0] + cop_margin_m
    x_hi = support.extent_long[1] - cop_margin_m
    y_lo = support.extent_lat[0] + cop_margin_m
    y_hi = support.extent_lat[1] - cop_margin_m
    if x_lo > x_hi or y_lo > y_hi:
        x_lo = x_hi = 0.5 * (x_lo + x_hi)
        y_lo = y_hi = 0.5 * (y_lo + y_hi)
    zero = np.zeros(3)

    def row(force: Array, moment: Array) -> Array:
        return np.concatenate([force, moment])

    return np.array(
        [
            row(e1 - mu * n, zero),
            row(-e1 - mu * n, zero),
            row(e2 - mu * n, zero),
            row(-e2 - mu * n, zero),
            row(-x_hi * n, -e2),  # M.e2 >= -x_hi fz
            row(x_lo * n, e2),  # M.e2 <= -x_lo fz
            row(-y_hi * n, e1),  # M.e1 <= y_hi fz
            row(y_lo * n, -e1),  # M.e1 >= y_lo fz
            row(-friction * torsion_length_m * n, n),  # M.n <= mu L fz
            row(-friction * torsion_length_m * n, -n),
            row(-n, zero),  # fz >= 0
        ]
    )


def foot_supports(
    names: Sequence[str],
    centres: Array,
    radii: Array,
    normal: Array,
    ground_height_m: float,
) -> list[FootSupport]:
    """Group contact spheres into feet (name suffix ``_r`` / ``_l``) and measure them.

    A sphere is touching when its lowest point is within ``CONTACT_CLEARANCE_M``
    of the ground. The support centre is the mean of the touching sphere
    contact points; the sole rectangle is their extent along and across the
    heel-to-toe direction. Postcondition: one entry per foot side with a heel
    and a toe or forefoot sphere, in sorted side order.
    """
    pts = np.asarray(centres, dtype=float)
    rad = np.asarray(radii, dtype=float)
    n = np.asarray(normal, dtype=float)
    if pts.shape != (len(names), 3) or rad.shape != (len(names),):
        raise ValueError("centres and radii must match the sphere names")
    clearance = pts @ n - rad - ground_height_m
    ground_pts = pts - np.outer(pts @ n - ground_height_m, n)
    sides: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        sides.setdefault(name.rsplit("_", 1)[-1], []).append(i)
    out = []
    for side in sorted(sides):
        members = sides[side]
        by_name = {names[i]: i for i in members}
        far = next(
            (by_name[s] for s in (f"toe_{side}", f"forefoot_{side}") if s in by_name),
            None,
        )
        if f"heel_{side}" not in by_name or far is None:
            continue
        axis = ground_pts[far] - ground_pts[by_name[f"heel_{side}"]]
        axis = axis - (axis @ n) * n
        length = np.linalg.norm(axis)
        if length < 1e-6:
            continue
        axis = axis / length
        lateral = np.cross(n, axis)
        touch = [i for i in members if clearance[i] <= CONTACT_CLEARANCE_M]
        use = touch or members
        centre = ground_pts[use].mean(axis=0)
        rel = ground_pts[use] - centre
        long_, lat = rel @ axis, rel @ lateral
        out.append(
            FootSupport(
                name=side,
                centre=centre,
                axis_long=axis,
                axis_lat=lateral,
                extent_long=(
                    min(float(long_.min()), -MIN_EXTENT_M),
                    max(float(long_.max()), MIN_EXTENT_M),
                ),
                extent_lat=(
                    min(float(lat.min()), -MIN_EXTENT_M),
                    max(float(lat.max()), MIN_EXTENT_M),
                ),
                touching=bool(touch),
            )
        )
    return out


def skew(vector: Array) -> Array:
    """Cross-product matrix of a 3-vector."""
    x, y, z = (float(c) for c in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@dataclass(frozen=True)
class WrenchQPProblem:
    """Everything one control step hands to the solver (all in the stated units)."""

    a_target: Array  # (na,) baseline actuated-joint acceleration target
    a_root_nominal: Array  # (6,) root acceleration the baseline produces
    a_root_target: Array  # (6,) root acceleration the cost pulls toward
    wrenches_nominal: Array  # (nf, 6) per-foot wrench the baseline produces, N, N m
    momentum_root: Array  # (6, 6) columns of A_G for the root coordinates
    momentum_joint: Array  # (6, na)
    wrench_map: Array  # (6, 6 nf) sum of wrenches about the CoM
    cone: Array  # (nf, 11, 6) rows per foot
    touching: Sequence[bool]
    body_weight_n: float

    def __post_init__(self) -> None:
        nf = len(self.touching)
        if self.wrenches_nominal.shape != (nf, 6) or self.cone.shape != (
            nf,
            CONE_ROWS_PER_FOOT,
            6,
        ):
            raise ValueError("per-foot arrays must match the foot count")
        if self.wrench_map.shape != (6, 6 * nf):
            raise ValueError("wrench_map must be 6 x 6 nf")
        if not self.body_weight_n > 0:
            raise ValueError("body weight must be positive")


@dataclass(frozen=True)
class WrenchQPSolution:
    """Optimal joint accelerations, root acceleration, wrenches and slack."""

    a_joint: Array
    a_root: Array
    wrenches: Array
    slack: Array
    success: bool

    @property
    def max_slack(self) -> float:
        return float(self.slack.max()) if self.slack.size else 0.0


def solve_wrench_qp(problem: WrenchQPProblem, cfg: WrenchQPConfig) -> WrenchQPSolution:
    """Solve the contact-wrench QP with Drake's Clarabel.

    Variables are ``[a (na), a_r (6), w (6 nf, scaled by body weight), s]``.
    Postcondition: when ``success`` the momentum equality holds to solver
    tolerance; a failed solve returns the nominal point.
    """
    from pydrake.solvers import ClarabelSolver, MathematicalProgram

    na = problem.a_target.size
    nf = len(problem.touching)
    bw = problem.body_weight_n
    nw, ns = 6 * nf, CONE_ROWS_PER_FOOT * nf
    prog = MathematicalProgram()
    a = prog.NewContinuousVariables(na, "a")
    ar = prog.NewContinuousVariables(6, "a_root")
    w = prog.NewContinuousVariables(nw, "w")
    s = prog.NewContinuousVariables(ns, "s")

    w0 = problem.wrenches_nominal.reshape(-1) / bw
    for var, target, weight in (
        (a, problem.a_target, cfg.joint_weight),
        (ar, problem.a_root_target, cfg.root_weight),
        (w, w0, cfg.wrench_weight),
    ):
        prog.AddQuadraticErrorCost(2.0 * weight * np.eye(var.size), target, var)
    prog.AddQuadraticCost(
        2.0 * cfg.slack_weight * np.eye(ns), np.zeros(ns), s, is_convex=True
    )
    prog.AddBoundingBoxConstraint(0.0, np.inf, s)

    equality = np.hstack(
        [problem.momentum_joint, problem.momentum_root, -bw * problem.wrench_map]
    )
    # Momentum balance written about the nominal point, so the plant's own
    # reaction satisfies it exactly and numerical residuals of A_G cannot
    # push a feasible baseline off its torques.
    balance_rhs = (
        problem.momentum_joint @ problem.a_target
        + problem.momentum_root @ problem.a_root_nominal
        - problem.wrench_map @ problem.wrenches_nominal.reshape(-1)
    )
    prog.AddLinearEqualityConstraint(equality, balance_rhs, np.concatenate([a, ar, w]))

    for k in range(nf):
        wk = w[6 * k : 6 * k + 6]
        if not problem.touching[k]:
            prog.AddBoundingBoxConstraint(0.0, 0.0, wk)
            continue
        sk = s[CONE_ROWS_PER_FOOT * k : CONE_ROWS_PER_FOOT * (k + 1)]
        rows = np.hstack([problem.cone[k], -np.eye(CONE_ROWS_PER_FOOT)])
        prog.AddLinearConstraint(
            rows,
            np.full(CONE_ROWS_PER_FOOT, -np.inf),
            np.zeros(CONE_ROWS_PER_FOOT),
            np.concatenate([wk, sk]),
        )
    result = ClarabelSolver().Solve(prog)
    if not result.is_success():
        logger.warning("contact-wrench QP failed; using the baseline target")
        return WrenchQPSolution(
            problem.a_target,
            problem.a_root_nominal,
            problem.wrenches_nominal,
            np.zeros(ns),
            False,
        )
    return WrenchQPSolution(
        a_joint=np.asarray(result.GetSolution(a)),
        a_root=np.asarray(result.GetSolution(ar)),
        wrenches=np.asarray(result.GetSolution(w)).reshape(nf, 6) * bw,
        slack=np.asarray(result.GetSolution(s)),
        success=True,
    )


class ContactWrenchTracker:
    """Computed-torque tracking whose joint targets respect the foot wrench cone."""

    def __init__(
        self,
        sim: FullBodySimulator,
        times: Array,
        q_ref: Array,
        *,
        omega_rad_s: float,
        zeta: float = 1.0,
        balance: tuple[float, float] | None = None,
        config: WrenchQPConfig | None = None,
    ) -> None:
        from src.shared.python.motion_matching import full_body_forward_dynamics as fs

        if not hasattr(sim.adapter, "_mj") or not hasattr(sim.adapter, "_spheres"):
            raise ValueError("The contact-wrench QP needs the MuJoCo full-body adapter")
        self.sim, self.cfg = sim, config or WrenchQPConfig()
        self.times = np.asarray(times, dtype=float)
        self.q_ref = np.asarray(q_ref, dtype=float)
        if (
            self.times.ndim != 1
            or np.any(np.diff(self.times) <= 0)
            or self.q_ref.shape != (self.times.size, sim.nv)
            or not np.isfinite(self.q_ref).all()
        ):
            raise ValueError("Reference times must increase with one finite q row each")
        self.gains = fs._tracking_gains(omega_rad_s, zeta, balance, None)
        self.v_ref = np.gradient(self.q_ref, self.times, axis=0)
        self.a_ref = np.gradient(self.v_ref, self.times, axis=0)
        self.last: WrenchQPSolution | None = None
        self.activations = 0
        self.calls = 0
        self.max_slack = 0.0
        self.history: list[tuple[float, float, float]] = []
        """``(t, max slack in body weights, |a* - a_baseline|)`` of each solve."""

    def _sample(self, table: Array, t: float) -> Array:
        return np.array(
            [np.interp(t, self.times, table[:, k]) for k in range(self.sim.nv)]
        )

    def __call__(self, t: float, q: Array, v: Array) -> Array:
        from src.shared.python.motion_matching import full_body_forward_dynamics as fs

        sim, cfg = self.sim, self.cfg
        act, root = sim.actuated, sim.root
        q_t, v_t, a_t = (
            self._sample(x, t) for x in (self.q_ref, self.v_ref, self.a_ref)
        )
        com_ref = sim.centre_of_mass(q_t)[0] if self.gains.balance is not None else None
        omega = np.broadcast_to(np.asarray(self.gains.omega_rad_s, float), (sim.nv,))[
            act
        ]
        wanted = (
            a_t[act]
            + 2.0 * self.gains.zeta * omega * (v_t[act] - v[act])
            + omega**2 * (q_t[act] - q[act])
        )
        if com_ref is not None:
            wanted = wanted + fs._balance_acceleration(  # type: ignore[operator]  # fs helper returns an untyped ndarray
                sim,
                q,
                v,
                com_ref,
                self.gains.balance,  # type: ignore[arg-type]
            )
        affine, offset = sim.affine_dynamics(q, v)
        tau_base = _pseudo_inverse_torque(sim, affine, offset, wanted)
        accel0 = affine @ tau_base[act] + offset
        problem = self._problem(q, v, wanted, accel0, q_t, v_t, a_t)
        self.calls += 1
        if problem is None:
            return tau_base
        solution = solve_wrench_qp(problem, cfg)
        self.last = solution
        self.max_slack = max(self.max_slack, solution.max_slack)
        if not solution.success:
            return tau_base
        self.history.append(
            (
                float(t),
                solution.max_slack,
                float(np.linalg.norm(solution.a_joint - problem.a_target)),
            )
        )
        if solution.max_slack > 1e-4 or (
            np.linalg.norm(solution.a_joint - problem.a_target) > 1e-4
        ):
            self.activations += 1
        return _pseudo_inverse_torque(sim, affine, offset, solution.a_joint)

    def _problem(
        self,
        q: Array,
        v: Array,
        wanted: Array,
        accel0: Array,
        q_t: Array,
        v_t: Array,
        a_t: Array,
    ) -> WrenchQPProblem | None:
        sim, cfg = self.sim, self.cfg
        adapter = sim.adapter
        ground = adapter.ground_plane
        normal = np.asarray(ground.normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        centres, radii, names = _sphere_state(sim, q)
        feet = foot_supports(names, centres, radii, normal, ground.height_m)
        if not feet:
            return None
        com, a_g = _centroidal_matrix(sim, q)
        samples = adapter.evaluate_contact_samples(sim._map(q), sim._map(v))
        friction = cfg.friction or adapter.contact_parameters.dynamic_friction
        nominal = np.zeros((len(feet), 6))
        cone = np.empty((len(feet), CONE_ROWS_PER_FOOT, 6))
        wrench_map = np.zeros((6, 6 * len(feet)))
        for k, foot in enumerate(feet):
            members = [i for i, n in enumerate(names) if n.endswith("_" + foot.name)]
            for i in members:
                s = samples[names[i]]
                force = s.normal_force_n + s.friction_force_n
                nominal[k, :3] += force
                nominal[k, 3:] += np.cross(centres[i] - foot.centre, force)
            cone[k] = wrench_cone_rows(
                foot,
                normal,
                friction=friction,
                torsion_length_m=cfg.torsion_length_m,
                cop_margin_m=cfg.cop_margin_m,
            )
            wrench_map[:3, 6 * k : 6 * k + 3] = np.eye(3)
            wrench_map[3:, 6 * k : 6 * k + 3] = skew(foot.centre - com)
            wrench_map[3:, 6 * k + 3 : 6 * k + 6] = np.eye(3)
        root = sim.root
        a_root_nominal = accel0[root].copy()
        if cfg.root_gain > 0.0:
            w2 = cfg.root_omega_rad_s**2
            a_root_nominal += cfg.root_gain * (
                a_t[root]
                + 2.0 * cfg.root_omega_rad_s * (v_t[root] - v[root])
                + w2 * (q_t[root] - q[root])
                - a_root_nominal
            )
        return WrenchQPProblem(
            a_target=accel0[sim.actuated].copy(),
            a_root_nominal=accel0[root].copy(),
            a_root_target=a_root_nominal,
            wrenches_nominal=nominal,
            momentum_root=a_g[:, root],
            momentum_joint=a_g[:, sim.actuated],
            wrench_map=wrench_map,
            cone=cone,
            touching=[f.touching for f in feet],
            body_weight_n=sim.mass_kg * float(np.linalg.norm(sim.gravity)),
        )


def _pseudo_inverse_torque(
    sim: FullBodySimulator, affine: Array, offset: Array, target: Array
) -> Array:
    """Torques giving the actuated joints ``target`` (the baseline's inverse)."""
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        MIN_SINGULAR_VALUE,
    )

    rows = sim.actuated
    u, sv, vt = np.linalg.svd(affine[rows], full_matrices=False)
    keep = sv > MIN_SINGULAR_VALUE
    inverse = (vt[keep].T / sv[keep]) @ u[:, keep].T
    tau = np.zeros(sim.nv)
    tau[rows] = inverse @ (target - offset[rows])
    return tau


def _sphere_state(sim: FullBodySimulator, q: Array) -> tuple[Array, Array, list[str]]:
    adapter = sim.adapter
    adapter.frame_poses(sim._map(q))
    names = list(adapter._spheres)
    centres = np.array(
        [adapter.data.site_xpos[adapter._spheres[n]["site_id"]] for n in names]
    )
    radii = np.array([adapter._spheres[n]["radius"] for n in names])
    return centres, radii, names


def _centroidal_matrix(sim: FullBodySimulator, q: Array) -> tuple[Array, Array]:
    """CoM and centroidal momentum matrix ``A_G`` (6 x nv, spec order)."""
    adapter = sim.adapter
    com, jac_com = sim.centre_of_mass(q)
    model = adapter.model
    angular = np.zeros((3, model.nv))
    adapter._mj.mj_angmomMat(model, adapter.data, angular, 1)
    matrix = np.vstack([sim.mass_kg * jac_com, angular[:, sim._dof]])
    return np.asarray(com, dtype=float), matrix


def contact_wrench_controller(
    sim: Any,
    times: Array,
    q_ref: Array,
    *,
    omega_rad_s: float,
    zeta: float = 1.0,
    balance: tuple[float, float] | None = None,
    config: WrenchQPConfig | None = None,
) -> ContactWrenchTracker:
    """Build the contact-wrench tracking controller (see the module docstring)."""
    return ContactWrenchTracker(
        sim,
        times,
        q_ref,
        omega_rad_s=omega_rad_s,
        zeta=zeta,
        balance=balance,
        config=config,
    )
