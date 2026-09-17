"""Discrete Crocoddyl action models for the full-body marker fit (MS-31, #10338).

The shared contact law (Hunt-Crossley normal force, regularised Coulomb
friction with a 0.05 m/s transition velocity) is stiff in velocity: at the
address pose the linearised plant has real eigenvalues near -8000 1/s, so
Crocoddyl's explicit Euler and RK integrators are unstable at the 360 Hz
capture step. These models integrate one capture step with a linearly
implicit (Rosenbrock) Euler step that treats the velocity dependence of
the acceleration implicitly:

    a      = f(q, v, u)                          (plant acceleration)
    S      = (I - dt * df/dv)^-1
    v_next = v + dt * S * a
    q_next = q + dt * v_next

The state map derivatives hold df/dv fixed inside ``S`` (a Gauss-Newton style
approximation of the second-order term), which DDP tolerates. By default the
*rollout* of a node is integrated exactly with adaptive RK45 (the shared
lane's integrator) and only the derivatives use the implicit-Euler
linearisation, so the solver's trajectory is the acceptance replay by
construction; ``node_integrator="implicit_euler"`` keeps the cheaper
first-order step for both. Crocoddyl and
the plant are reached only through the ``ctx`` object handed in by the fit
driver, so this module stays free of engine imports.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.pinocchio.python.crocoddyl_problem import (
    FitWeights,
    MarkerTargets,
    least_squares_controls,
    range_barrier,
)
from src.shared.python.contracts import require

Array = NDArray[np.float64]

INFEASIBLE_COST = 1e12


class PlantContext(Protocol):
    """What the action models need from the fit driver's plant context."""

    n: int
    actuated: NDArray[np.bool_]
    lower: Array
    upper: Array

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array: ...

    def derivatives(self, q: Array, v: Array, tau: Array) -> Any: ...

    def markers(self, q: Array) -> Array: ...

    def markers_and_jacobians(self, q: Array) -> tuple[Array, Array]: ...

    def effort_vector(self, u: Array) -> Array: ...


class _NodeCost:
    """Marker, effort, rate and range-barrier cost of one node (pure numpy)."""

    def __init__(
        self,
        ctx: PlantContext,
        target: Array,
        valid: NDArray[np.bool_],
        marker_weights: Array,
        weights: FitWeights,
        *,
        marker_weight: float,
    ) -> None:
        self.ctx = ctx
        self.target = target
        self.rows = np.flatnonzero(valid)
        self.marker_w = marker_weight * marker_weights[self.rows]
        self.weights = weights

    def value(self, q: Array, v: Array, u: Array | None) -> float:
        positions = self.ctx.markers(q)
        error = positions[self.rows] - self.target[self.rows]
        cost = 0.5 * float(np.sum(self.marker_w[:, None] * error**2))
        barrier, _, _ = range_barrier(
            q, self.ctx.lower, self.ctx.upper, self.weights.range_barrier
        )
        cost += barrier + 0.5 * self.weights.velocity * float(v @ v)
        if u is not None and u.size:
            cost += 0.5 * self.weights.effort * float(u @ u)
        return cost

    def gradient_hessian(self, q: Array) -> tuple[Array, Array]:
        """Gauss-Newton gradient and Hessian of the configuration-dependent terms."""
        positions, jac = self.ctx.markers_and_jacobians(q)
        error = positions[self.rows] - self.target[self.rows]
        jac_rows = jac[self.rows]
        weighted = self.marker_w[:, None] * error
        _, barrier_grad, barrier_hess = range_barrier(
            q, self.ctx.lower, self.ctx.upper, self.weights.range_barrier
        )
        grad = np.einsum("mij,mi->j", jac_rows, weighted) + barrier_grad
        hess = np.einsum(
            "mij,mik->jk", jac_rows * self.marker_w[:, None, None], jac_rows
        )
        return grad, hess + np.diag(barrier_hess)


def rk45_node_step(
    ctx: PlantContext,
    q: Array,
    v: Array,
    tau: Array,
    dt: float,
    *,
    rtol: float,
    atol: float,
) -> tuple[Array, Array]:
    """Exact (adaptive RK45) integration of one node with zero-order-hold effort."""
    from scipy.integrate import solve_ivp

    n = ctx.n

    def rhs(_t: float, y: Array) -> Array:
        return np.concatenate([y[n:], ctx.acceleration(y[:n], y[n:], tau)])

    sol = solve_ivp(
        rhs, (0.0, dt), np.concatenate([q, v]), method="RK45", rtol=rtol, atol=atol
    )
    if not sol.success or not np.isfinite(sol.y[:, -1]).all():
        raise FloatingPointError(f"RK45 node step failed: {sol.message}")
    return np.asarray(sol.y[:n, -1], dtype=float), np.asarray(
        sol.y[n:, -1], dtype=float
    )


def make_action_models(
    crocoddyl: Any,
    ctx: PlantContext,
    targets: MarkerTargets,
    weights: FitWeights,
    effort_bounds: Array,
    dt: float,
    *,
    node_integrator: str = "rk45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    substeps: int = 1,
) -> tuple[list[Any], Any]:
    """Running models for every node but the last, plus the terminal model."""
    require(dt > 0.0, "dt must be positive", dt)
    require(
        node_integrator in ("rk45", "implicit_euler"),
        "unknown node integrator",
        node_integrator,
    )
    require(substeps >= 1, "substeps must be at least 1", substeps)
    n_nodes = targets.targets.shape[0]
    require(n_nodes >= 2, "at least two nodes are required", n_nodes)
    nu = int(ctx.actuated.sum())
    require(effort_bounds.shape == (nu,), "one effort bound per actuated coordinate")
    state = crocoddyl.StateVector(2 * ctx.n)

    class ImplicitEulerAction(crocoddyl.ActionModelAbstract):  # type: ignore[misc]
        def __init__(self, node: int) -> None:
            crocoddyl.ActionModelAbstract.__init__(self, state, nu, 1)
            self.node = node
            self.cost_model = _NodeCost(
                ctx,
                targets.targets[node],
                targets.valid[node],
                targets.weights,
                weights,
                marker_weight=weights.marker,
            )
            self.u_lb = -effort_bounds
            self.u_ub = effort_bounds
            self._cache: dict[str, Any] = {}

        def _dynamics(self, x: Array, u: Array) -> tuple[Array, Array, Any]:
            q, v = x[: ctx.n], x[ctx.n :]
            tau = ctx.effort_vector(u)
            der = ctx.derivatives(q, v, tau)
            solve = np.asarray(
                np.linalg.solve(np.eye(ctx.n) - dt * np.asarray(der.dv), np.eye(ctx.n)),
                dtype=float,
            )
            if node_integrator == "rk45":
                q_next, v_next = rk45_node_step(
                    ctx, q, v, tau, dt, rtol=rtol, atol=atol
                )
            elif substeps == 1:
                accel = ctx.acceleration(q, v, tau)
                v_next = v + dt * solve @ accel
                q_next = q + dt * v_next
            else:
                q_next, v_next = implicit_euler_substeps(ctx, q, v, tau, dt, substeps)
            return np.concatenate([q_next, v_next]), solve, der

        def calc(self, data: Any, x: Array, u: Array | None = None) -> None:
            u_arr = np.zeros(nu) if u is None else np.asarray(u, dtype=float)
            try:
                x_next, solve, der = self._dynamics(np.asarray(x, dtype=float), u_arr)
                data.xnext[:] = x_next
                data.cost = dt * self.cost_model.value(x[: ctx.n], x[ctx.n :], u_arr)
                self._cache = {
                    "x": np.array(x, dtype=float),
                    "u": u_arr.copy(),
                    "solve": solve,
                    "der": der,
                }
            except FloatingPointError:
                data.xnext[:] = x
                data.cost = INFEASIBLE_COST
                self._cache = {}

        def calcDiff(self, data: Any, x: Array, u: Array | None = None) -> None:
            u_arr = np.zeros(nu) if u is None else np.asarray(u, dtype=float)
            x_arr = np.asarray(x, dtype=float)
            cache = self._cache
            if (
                not cache
                or not np.array_equal(cache["x"], x_arr)
                or not np.array_equal(cache["u"], u_arr)
            ):
                try:
                    _, solve, der = self._dynamics(x_arr, u_arr)
                except FloatingPointError:
                    data.Fx[:, :] = 0.0
                    data.Fu[:, :] = 0.0
                    data.Lx[:] = 0.0
                    data.Lu[:] = 0.0
                    data.Lxx[:, :] = 0.0
                    data.Luu[:, :] = 0.0
                    data.Lxu[:, :] = 0.0
                    return
            else:
                solve, der = cache["solve"], cache["der"]
            n = ctx.n
            dv_dq = dt * solve @ np.asarray(der.dq)
            dv_dv = solve
            dv_du = dt * solve @ np.asarray(der.deffort)[:, ctx.actuated]
            data.Fx[n:, :n] = dv_dq
            data.Fx[n:, n:] = dv_dv
            data.Fx[:n, :n] = np.eye(n) + dt * dv_dq
            data.Fx[:n, n:] = dt * dv_dv
            data.Fu[n:, :] = dv_du
            data.Fu[:n, :] = dt * dv_du
            q, v = x_arr[:n], x_arr[n:]
            grad, hess = self.cost_model.gradient_hessian(q)
            data.Lx[:n] = dt * grad
            data.Lx[n:] = dt * weights.velocity * v
            data.Lxx[:, :] = 0.0
            data.Lxx[:n, :n] = dt * hess
            data.Lxx[n:, n:] = dt * weights.velocity * np.eye(n)
            data.Lu[:] = dt * weights.effort * u_arr
            data.Luu[:, :] = dt * weights.effort * np.eye(nu)
            data.Lxu[:, :] = 0.0

        def createData(self) -> Any:
            return crocoddyl.ActionDataAbstract(self)

    class TerminalAction(crocoddyl.ActionModelAbstract):  # type: ignore[misc]
        def __init__(self) -> None:
            crocoddyl.ActionModelAbstract.__init__(self, state, 0, 1)
            self.cost_model = _NodeCost(
                ctx,
                targets.targets[-1],
                targets.valid[-1],
                targets.weights,
                weights,
                marker_weight=weights.terminal_marker,
            )

        def calc(self, data: Any, x: Array, u: Array | None = None) -> None:
            data.xnext[:] = x
            data.cost = self.cost_model.value(x[: ctx.n], x[ctx.n :], None)

        def calcDiff(self, data: Any, x: Array, u: Array | None = None) -> None:
            n = ctx.n
            q, v = np.asarray(x[:n], dtype=float), np.asarray(x[n:], dtype=float)
            grad, hess = self.cost_model.gradient_hessian(q)
            data.Fx[:, :] = np.eye(2 * n)
            data.Lx[:n] = grad
            data.Lx[n:] = weights.velocity * v
            data.Lxx[:, :] = 0.0
            data.Lxx[:n, :n] = hess
            data.Lxx[n:, n:] = weights.velocity * np.eye(n)

        def createData(self) -> Any:
            return crocoddyl.ActionDataAbstract(self)

    running = [ImplicitEulerAction(node) for node in range(n_nodes - 1)]
    return running, TerminalAction()


def implicit_euler_substeps(
    ctx: PlantContext, q: Array, v: Array, tau: Array, dt: float, substeps: int
) -> tuple[Array, Array]:
    """Linearly implicit Euler over ``substeps`` sub-intervals, fresh velocity Jacobian each."""
    h = dt / substeps
    qk, vk = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
    eye = np.eye(ctx.n)
    for _ in range(substeps):
        der = ctx.derivatives(qk, vk, tau)
        accel = ctx.acceleration(qk, vk, tau)
        vk = vk + h * np.linalg.solve(eye - h * np.asarray(der.dv), accel)
        qk = qk + h * vk
    if not np.isfinite(qk).all():
        raise FloatingPointError("implicit substeps diverged")
    return qk, vk


def implicit_euler_rollout(
    ctx: PlantContext, q0: Array, v0: Array, us: Array, dt: float
) -> tuple[Array, Array]:
    """Replay controls with the same linearly implicit Euler step the solver used."""
    n_nodes = us.shape[0] + 1
    q = np.empty((n_nodes, ctx.n))
    v = np.empty((n_nodes, ctx.n))
    q[0], v[0] = q0, v0
    for k in range(us.shape[0]):
        tau = ctx.effort_vector(us[k])
        der = ctx.derivatives(q[k], v[k], tau)
        accel = ctx.acceleration(q[k], v[k], tau)
        v[k + 1] = v[k] + dt * np.linalg.solve(
            np.eye(ctx.n) - dt * np.asarray(der.dv), accel
        )
        q[k + 1] = q[k] + dt * v[k + 1]
        if not np.isfinite(q[k + 1]).all():
            raise FloatingPointError(f"implicit rollout diverged at node {k + 1}")
    return q, v


def tracking_rollout(
    ctx: PlantContext,
    q_ref: Array,
    v_ref: Array,
    a_ref: Array,
    dt: float,
    *,
    kp: float,
    kd: float,
    effort_bounds: Array,
    ridge: float,
    q0: Array | None = None,
    v0: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Dynamically consistent warm start: computed-torque tracking of the IK reference.

    ``q0``/``v0`` override the initial state (used to continue from a previous
    candidate's final node).

    At every node the desired acceleration ``a_ref + kp (q_ref - q) + kd (v_ref - v)``
    is mapped to bounded efforts through the plant's effort sensitivity (ridge
    least squares), then the state is advanced with the same linearly implicit
    Euler step the action models use. The result is a gap-free (xs, us) pair,
    which FDDP handles far better than a kinematic guess with a finite-difference
    velocity. Returns (q, v, us).
    """
    require(kp >= 0.0 and kd >= 0.0, "tracking gains must be nonnegative", (kp, kd))
    n_nodes = q_ref.shape[0]
    q = np.empty_like(q_ref)
    v = np.empty_like(v_ref)
    us = np.empty((n_nodes - 1, int(ctx.actuated.sum())))
    q[0] = q_ref[0] if q0 is None else np.asarray(q0, dtype=float)
    v[0] = v_ref[0] if v0 is None else np.asarray(v0, dtype=float)
    for k in range(n_nodes - 1):
        zero = np.zeros(ctx.n)
        der = ctx.derivatives(q[k], v[k], zero)
        a_zero = ctx.acceleration(q[k], v[k], zero)
        a_des = a_ref[k] + kp * (q_ref[k] - q[k]) + kd * (v_ref[k] - v[k])
        us[k] = np.clip(
            least_squares_controls(
                a_des, a_zero, np.asarray(der.deffort), ctx.actuated, ridge=ridge
            ),
            -effort_bounds,
            effort_bounds,
        )
        tau = ctx.effort_vector(us[k])
        accel = ctx.acceleration(q[k], v[k], tau)
        v[k + 1] = v[k] + dt * np.linalg.solve(
            np.eye(ctx.n) - dt * np.asarray(der.dv), accel
        )
        q[k + 1] = q[k] + dt * v[k + 1]
        if not np.isfinite(q[k + 1]).all():
            raise FloatingPointError(f"tracking rollout diverged at node {k + 1}")
    return q, v, us
