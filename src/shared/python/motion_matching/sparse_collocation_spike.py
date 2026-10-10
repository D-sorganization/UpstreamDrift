"""Bounded sparse implicit inverse-dynamics collocation spike for F03.

This deliberately small second-order synthetic plant is an algorithm check,
not a full-body solver or a native swing qualification. Torque is ZOH over
each interval. Fresh replay uses adaptive forward integration independently
of the midpoint transcription.
"""

from __future__ import annotations

import platform
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.sparse import csr_matrix, lil_matrix, vstack

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class SecondOrderFixture:
    """Known synthetic plant: ``m qdd + c qd + k q = tau`` in SI units."""

    inertia_kg_m2: float
    damping_nm_s_rad: float
    stiffness_nm_rad: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(
                [self.inertia_kg_m2, self.damping_nm_s_rad, self.stiffness_nm_rad]
            ).all()
            or self.inertia_kg_m2 <= 0
            or self.damping_nm_s_rad < 0
            or self.stiffness_nm_rad < 0
        ):
            raise ValueError(
                "fixture mass must be positive; damping/stiffness nonnegative"
            )

    def forward_held(
        self, times_s: Array, initial_state: Array, torque_nm: Array
    ) -> Array:
        """Integrate one full ZOH history from its supplied initial state."""
        times = np.asarray(times_s, dtype=float)
        initial = np.asarray(initial_state, dtype=float)
        torque = np.asarray(torque_nm, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)
            or initial.shape != (2,)
            or not np.isfinite(initial).all()
            or torque.shape != (times.size - 1,)
            or not np.isfinite(torque).all()
        ):
            raise ValueError("forward solve needs finite state, clock and ZOH input")
        state = initial.copy()
        states = [state.copy()]
        for start, end, applied in zip(times[:-1], times[1:], torque, strict=True):

            def rhs(
                time_s: float, current: Array, applied_torque: float = float(applied)
            ) -> Array:
                q, v = current
                acceleration = (
                    applied_torque
                    - self.damping_nm_s_rad * v
                    - self.stiffness_nm_rad * q
                ) / self.inertia_kg_m2
                return np.array([v, acceleration])

            solution = solve_ivp(
                rhs, (float(start), float(end)), state, rtol=1e-10, atol=1e-12
            )
            if not solution.success or not np.isfinite(solution.y).all():
                raise ValueError("fresh forward solve failed")
            state = solution.y[:, -1].copy()
            states.append(state)
        return np.asarray(states)


@dataclass(frozen=True)
class ReplayReceipt:
    """Fresh continuous forward rollout; never inferred from collocation nodes."""

    states: Array
    initial_state_equal: bool
    state_resets: int
    max_state_gap: float
    position_rmse_rad: float


@dataclass(frozen=True)
class CollocationResult:
    """Numerical feasibility, objective and replay evidence stay separate."""

    states: Array
    torque_nm: Array
    objective: float
    max_dynamics_defect: float
    max_torque_violation: float
    max_rate_violation: float
    optimizer_converged: bool
    solver_message: str
    solver_iterations: int
    replay: ReplayReceipt | None
    input_degrees_of_freedom: int
    defect_kind: str
    solve_seconds: float
    replay_seconds: float

    def replay_accepted(
        self,
        *,
        max_state_gap: float,
        max_constraint_defect: float,
        max_bound_violation: float,
    ) -> bool:
        """Apply an externally ratified gap; never infer it from node objective."""
        if (
            not np.isfinite(
                [max_state_gap, max_constraint_defect, max_bound_violation]
            ).all()
            or min(max_state_gap, max_constraint_defect) <= 0
            or max_bound_violation < 0
        ):
            raise ValueError("replay/constraint thresholds must be finite and valid")
        return bool(
            self.optimizer_converged
            and self.max_dynamics_defect <= max_constraint_defect
            and self.max_torque_violation <= max_bound_violation
            and self.max_rate_violation <= max_bound_violation
            and self.replay is not None
            and self.replay.initial_state_equal
            and self.replay.state_resets == 0
            and self.replay.max_state_gap <= max_state_gap
        )


@dataclass(frozen=True)
class SparseCollocationProblem:
    """One-dimensional midpoint collocation with hard dynamics/torque/rate rows."""

    fixture: SecondOrderFixture
    times_s: Array
    initial_state: Array
    target_q_rad: Array
    torque_lower_nm: float
    torque_upper_nm: float
    rate_limit_nm_s: float
    position_weight: float
    effort_weight: float
    rate_weight: float

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float)
        state = np.asarray(self.initial_state, dtype=float)
        target = np.asarray(self.target_q_rad, dtype=float)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)
        ):
            raise ValueError("time grid must be finite and strictly increasing")
        if state.shape != (2,) or not np.isfinite(state).all():
            raise ValueError("initial q/v state must be finite and complete")
        if target.shape != times.shape or not np.isfinite(target).all():
            raise ValueError("target position must cover the finite time grid")
        scalars = np.array(
            [
                self.torque_lower_nm,
                self.torque_upper_nm,
                self.rate_limit_nm_s,
                self.position_weight,
                self.effort_weight,
                self.rate_weight,
            ],
            dtype=float,
        )
        if (
            not np.isfinite(scalars).all()
            or self.torque_lower_nm >= self.torque_upper_nm
            or self.rate_limit_nm_s <= 0
            or self.position_weight <= 0
            or min(self.effort_weight, self.rate_weight) < 0
        ):
            raise ValueError("torque/rate bounds and objective weights are invalid")
        for name, value in (
            ("times_s", times),
            ("initial_state", state),
            ("target_q_rad", target),
        ):
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    @property
    def intervals(self) -> int:
        return self.times_s.size - 1

    def _unpack(self, variables: Array) -> tuple[Array, Array, Array]:
        n = self.intervals
        z = np.asarray(variables, dtype=float)
        if z.shape != (3 * n,) or not np.isfinite(z).all():
            raise ValueError("collocation vector must be finite (q, v, torque)")
        q = np.r_[self.initial_state[0], z[:n]]
        v = np.r_[self.initial_state[1], z[n : 2 * n]]
        return q, v, z[2 * n :]

    def defect_system(self) -> tuple[csr_matrix, Array]:
        """Return affine residual ``A z + offset`` and sparse exact Jacobian."""
        n = self.intervals
        matrix = lil_matrix((2 * n, 3 * n), dtype=float)
        offset = np.zeros(2 * n)
        mass = self.fixture.inertia_kg_m2
        damping = self.fixture.damping_nm_s_rad
        stiffness = self.fixture.stiffness_nm_rad
        for step, dt in enumerate(np.diff(self.times_s)):
            kinematic, dynamic = 2 * step, 2 * step + 1
            matrix[kinematic, step] += 1.0 / dt
            matrix[kinematic, n + step] += -0.5
            matrix[dynamic, step] += 0.5 * stiffness
            matrix[dynamic, n + step] += mass / dt + 0.5 * damping
            matrix[dynamic, 2 * n + step] = -1.0
            if step:
                matrix[kinematic, step - 1] += -1.0 / dt
                matrix[kinematic, n + step - 1] += -0.5
                matrix[dynamic, step - 1] += 0.5 * stiffness
                matrix[dynamic, n + step - 1] += -mass / dt + 0.5 * damping
            else:
                q0, v0 = self.initial_state
                offset[kinematic] = -q0 / dt - 0.5 * v0
                offset[dynamic] = (
                    0.5 * stiffness * q0 + (-mass / dt + 0.5 * damping) * v0
                )
        return matrix.tocsr(), offset

    def defect_jacobian(self) -> csr_matrix:
        return self.defect_system()[0]

    def defects(self, variables: Array) -> Array:
        self._unpack(variables)
        matrix, offset = self.defect_system()
        return np.asarray(matrix @ variables + offset, dtype=float)

    def cost_system(self) -> tuple[csr_matrix, Array]:
        """Return sparse weighted residual ``C z - target``."""
        n = self.intervals
        rows: list[csr_matrix] = []
        targets: list[Array] = []
        position = lil_matrix((n, 3 * n), dtype=float)
        position[np.arange(n), np.arange(n)] = np.sqrt(self.position_weight)
        rows.append(position.tocsr())
        targets.append(np.sqrt(self.position_weight) * self.target_q_rad[1:])
        if self.effort_weight > 0:
            effort = lil_matrix((n, 3 * n), dtype=float)
            effort[np.arange(n), 2 * n + np.arange(n)] = np.sqrt(
                self.effort_weight * np.diff(self.times_s)
            )
            rows.append(effort.tocsr())
            targets.append(np.zeros(n))
        if self.rate_weight > 0 and n > 1:
            rates = self.rate_matrix()
            rows.append(np.sqrt(self.rate_weight) * rates)
            targets.append(np.zeros(n - 1))
        return vstack(rows, format="csr"), np.concatenate(targets)

    def rate_matrix(self) -> csr_matrix:
        n = self.intervals
        matrix = lil_matrix((n - 1, 3 * n), dtype=float)
        for step, dt in enumerate(np.diff(self.times_s)[:-1], start=1):
            matrix[step - 1, 2 * n + step] = 1.0 / dt
            matrix[step - 1, 2 * n + step - 1] = -1.0 / dt
        return matrix.tocsr()

    def replay(self, torque_nm: Array) -> Array:
        """Fresh RK45 ZOH forward solve, independent of transcription nodes."""
        return self.fixture.forward_held(self.times_s, self.initial_state, torque_nm)

    def initial_guess(self, torque_nm: Array) -> Array:
        states = self.replay(torque_nm)
        return np.r_[states[1:, 0], states[1:, 1], torque_nm]


def solve_sparse_collocation(
    problem: SparseCollocationProblem,
    *,
    initial_torque: Array,
    max_iterations: int = 300,
) -> CollocationResult:
    """Run one bounded trust-region solve, then independently reintegrate."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    solve_started = time.perf_counter()
    initial = problem.initial_guess(initial_torque)
    dynamics, offset = problem.defect_system()
    cost, target = problem.cost_system()
    hessian = (cost.T @ cost).tocsr()
    lower = np.full(3 * problem.intervals, -np.inf)
    upper = np.full(3 * problem.intervals, np.inf)
    lower[2 * problem.intervals :] = problem.torque_lower_nm
    upper[2 * problem.intervals :] = problem.torque_upper_nm
    constraints = [LinearConstraint(dynamics, -offset, -offset)]
    if problem.intervals > 1:
        constraints.append(
            LinearConstraint(
                problem.rate_matrix(), -problem.rate_limit_nm_s, problem.rate_limit_nm_s
            )
        )

    def objective(variables: Array) -> float:
        residual = cost @ variables - target
        return 0.5 * float(residual @ residual)

    def gradient(variables: Array) -> Array:
        return np.asarray(cost.T @ (cost @ variables - target), dtype=float)

    # SciPy accepts a sparse Hessian here; its mypy overload omits that type.
    result = minimize(  # type: ignore[call-overload]
        objective,
        initial,
        jac=gradient,
        hess=lambda variables: hessian,
        method="trust-constr",
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={
            "maxiter": max_iterations,
            "gtol": 1e-10,
            "xtol": 1e-10,
            "sparse_jacobian": True,
            "verbose": 0,
        },
    )
    q, v, torque = problem._unpack(result.x)
    states = np.column_stack((q, v))
    solve_seconds = time.perf_counter() - solve_started
    replay_started = time.perf_counter()
    fresh = problem.replay(torque)
    replay_seconds = time.perf_counter() - replay_started
    rate = problem.rate_matrix() @ result.x
    torque_violation = max(
        0.0,
        float(np.max(problem.torque_lower_nm - torque)),
        float(np.max(torque - problem.torque_upper_nm)),
    )
    rate_violation = max(
        0.0, float(np.max(np.abs(rate), initial=0.0) - problem.rate_limit_nm_s)
    )
    replay = ReplayReceipt(
        states=fresh,
        initial_state_equal=bool(np.array_equal(fresh[0], problem.initial_state)),
        state_resets=0,
        max_state_gap=float(np.max(np.abs(fresh - states))),
        position_rmse_rad=float(
            np.sqrt(np.mean((fresh[:, 0] - problem.target_q_rad) ** 2))
        ),
    )
    return CollocationResult(
        states=states,
        torque_nm=torque.copy(),
        objective=objective(result.x),
        max_dynamics_defect=float(np.max(np.abs(problem.defects(result.x)))),
        max_torque_violation=torque_violation,
        max_rate_violation=rate_violation,
        optimizer_converged=bool(result.success),
        solver_message=str(result.message),
        solver_iterations=int(result.nit),
        replay=replay,
        input_degrees_of_freedom=problem.intervals,
        defect_kind="midpoint_inverse_dynamics",
        solve_seconds=solve_seconds,
        replay_seconds=replay_seconds,
    )


def _shooting_segmented_states(
    problem: SparseCollocationProblem,
    torque: Array,
    middle_index: int,
    middle_state: Array,
) -> Array:
    """Reconstruct the existing shooting transcription around its restart."""
    first = problem.fixture.forward_held(
        problem.times_s[: middle_index + 1],
        problem.initial_state,
        torque[:middle_index],
    )
    second = problem.fixture.forward_held(
        problem.times_s[middle_index:],
        middle_state,
        torque[middle_index:],
    )
    return np.vstack((first, second[1:]))


def _shooting_warm_start(
    problem: SparseCollocationProblem, initial_torque: Array
) -> float:
    """Validate the existing solver's one-parameter start."""
    initial = np.asarray(initial_torque, dtype=float)
    if initial.shape != (problem.intervals,) or not np.isfinite(initial).all():
        raise ValueError("shooting start must cover finite input intervals")
    if problem.intervals < 2:
        raise ValueError("shooting fixture needs at least two intervals")
    warm = float(np.mean(initial))
    if not problem.torque_lower_nm <= warm <= problem.torque_upper_nm:
        raise ValueError("shooting start violates torque bounds")
    return warm


def solve_existing_shooting_fixture(
    problem: SparseCollocationProblem, *, initial_torque: Array
) -> CollocationResult:
    """Exercise the existing multiple-shooting solver with one constant torque.

    This adapter is a synthetic benchmark only: shooting has one input DOF,
    while the collocation spike has one per interval. The common post-hoc
    objective is reportable; neither formulation is a fair native winner yet.
    """
    from src.shared.python.motion_matching.multi_shooting_fit import (
        MultipleShootingOptions,
        fit_multiple_shooting,
    )
    from src.shared.python.motion_matching.prefix_fit import MarkerTarget

    warm = _shooting_warm_start(problem, initial_torque)
    middle_index = problem.intervals // 2
    middle_time = float(problem.times_s[middle_index])
    points = np.zeros((problem.intervals + 1, 1, 3))
    points[:, 0, 0] = problem.target_q_rad
    target = MarkerTarget(problem.times_s, points, np.ones(1))

    def segment(
        theta: Array, clock: Array, initial_state: Array | None
    ) -> tuple[Array, Array]:
        state = problem.initial_state if initial_state is None else initial_state
        states = problem.fixture.forward_held(
            clock, state, np.full(len(clock) - 1, float(theta[0]))
        )
        markers = np.zeros((len(clock), 1, 3))
        markers[:, 0, 0] = states[:, 0]
        return markers, states[-1].copy()

    def uninterrupted(theta: Array, clock: Array) -> Array:
        markers, _ = segment(theta, clock, None)
        return markers

    warm_states = problem.replay(np.full(problem.intervals, warm))
    solve_started = time.perf_counter()
    fit = fit_multiple_shooting(
        target,
        segment,
        uninterrupted,
        initial_theta=np.array([warm]),
        lower_theta=np.array([problem.torque_lower_nm]),
        upper_theta=np.array([problem.torque_upper_nm]),
        initial_states={middle_time: warm_states[middle_index]},
        state_bounds={middle_time: (np.full(2, -20.0), np.full(2, 20.0))},
        options=MultipleShootingOptions(
            shooting_nodes=(middle_time, float(problem.times_s[-1])),
            state_dim=2,
            defect_tolerance=1e-6,
            max_nfev=100,
            shared_boundary_policy="once",
            acceptance=lambda observed: bool(np.isfinite(observed).all()),
        ),
    )
    solve_seconds = time.perf_counter() - solve_started
    torque = np.full(problem.intervals, float(fit.theta[0]))
    segmented_states = _shooting_segmented_states(
        problem, torque, middle_index, fit.intermediate_states[middle_time]
    )
    replay_started = time.perf_counter()
    fresh = problem.replay(torque)
    replay_seconds = time.perf_counter() - replay_started
    vector = np.r_[segmented_states[1:, 0], segmented_states[1:, 1], torque]
    cost, target_vector = problem.cost_system()
    residual = cost @ vector - target_vector
    replay = ReplayReceipt(
        states=fresh,
        initial_state_equal=bool(np.array_equal(fresh[0], problem.initial_state)),
        state_resets=0,
        max_state_gap=float(np.max(np.abs(fresh - segmented_states))),
        position_rmse_rad=float(
            np.sqrt(np.mean((fresh[:, 0] - problem.target_q_rad) ** 2))
        ),
    )
    return CollocationResult(
        states=segmented_states,
        torque_nm=torque,
        objective=0.5 * float(residual @ residual),
        max_dynamics_defect=fit.max_defect_norm,
        max_torque_violation=max(
            0.0,
            problem.torque_lower_nm - float(torque[0]),
            float(torque[0]) - problem.torque_upper_nm,
        ),
        max_rate_violation=0.0,
        optimizer_converged=fit.optimizer_converged,
        solver_message=fit.message,
        solver_iterations=fit.function_evaluations,
        replay=replay,
        input_degrees_of_freedom=1,
        defect_kind="shooting_state",
        solve_seconds=solve_seconds,
        replay_seconds=replay_seconds,
    )


@dataclass(frozen=True)
class BenchmarkStart:
    """A predeclared candidate initialization, including its warm/cold status."""

    name: str
    initial_torque_nm: Array
    warm: bool

    def __post_init__(self) -> None:
        torque = np.asarray(self.initial_torque_nm, dtype=float)
        if not self.name or torque.ndim != 1 or not np.isfinite(torque).all():
            raise ValueError("benchmark start must have a name and finite torque")
        frozen = torque.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "initial_torque_nm", frozen)


@dataclass(frozen=True)
class BenchmarkBackend:
    """Adapter for an existing shooting/FDDP or the bounded collocation spike."""

    name: str
    solve: Callable[[SparseCollocationProblem, Array], CollocationResult]

    def __post_init__(self) -> None:
        if not self.name or not callable(self.solve):
            raise ValueError("benchmark backend requires a name and callable solver")


@dataclass(frozen=True)
class BenchmarkGate:
    """Fixture-specific replay and observation thresholds frozen before solves."""

    max_replay_gap: float
    max_observation_rmse_rad: float
    max_constraint_defect_by_kind: Mapping[str, float]
    max_bound_violation: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(
                [
                    self.max_replay_gap,
                    self.max_observation_rmse_rad,
                    self.max_bound_violation,
                ]
            ).all()
            or min(self.max_replay_gap, self.max_observation_rmse_rad) <= 0
            or self.max_bound_violation < 0
            or not self.max_constraint_defect_by_kind
        ):
            raise ValueError("benchmark gate thresholds must be finite and positive")
        for kind, threshold in self.max_constraint_defect_by_kind.items():
            if not kind or not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("each defect kind needs a finite positive threshold")
        object.__setattr__(
            self,
            "max_constraint_defect_by_kind",
            MappingProxyType(dict(self.max_constraint_defect_by_kind)),
        )


@dataclass(frozen=True)
class BenchmarkHardware:
    host: str
    machine: str
    processor: str
    python_version: str
    numpy_version: str
    scipy_version: str


def capture_benchmark_hardware() -> BenchmarkHardware:
    """One canonical environment receipt for synthetic and native candidates."""
    from scipy import __version__ as scipy_version

    return BenchmarkHardware(
        host=platform.node() or "unknown-host",
        machine=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy_version,
    )


@dataclass(frozen=True)
class BenchmarkAttempt:
    backend: str
    start: str
    warm: bool
    optimizer_converged: bool
    accepted: bool
    failure_reason: str | None
    objective: float | None
    max_dynamics_defect: float | None
    max_torque_violation: float | None
    max_rate_violation: float | None
    max_replay_gap: float | None
    observation_rmse_rad: float | None
    input_degrees_of_freedom: int | None
    defect_kind: str | None
    solve_seconds: float | None
    replay_seconds: float | None
    total_seconds: float
    peak_python_bytes: int


@dataclass(frozen=True)
class BenchmarkSummary:
    backend: str
    warm: bool
    attempts: int
    accepted: int
    p50_total_seconds: float
    p95_total_seconds: float
    p50_solve_seconds: float | None
    p95_solve_seconds: float | None
    p50_replay_seconds: float | None
    p95_replay_seconds: float | None
    p50_peak_python_bytes: float
    p95_peak_python_bytes: float
    time_to_first_accepted_s: float | None


@dataclass(frozen=True)
class BenchmarkReport:
    """No global winner is inferred from a synthetic fixture."""

    hardware: BenchmarkHardware
    attempts: tuple[BenchmarkAttempt, ...]

    def summary(self, backend: str, *, warm: bool) -> BenchmarkSummary:
        selected = tuple(
            attempt
            for attempt in self.attempts
            if attempt.backend == backend and attempt.warm == warm
        )
        if not selected:
            raise ValueError("no attempts for requested backend/start class")
        times = np.array([attempt.total_seconds for attempt in selected])
        peak_bytes = np.array([attempt.peak_python_bytes for attempt in selected])
        solve_times = np.array(
            [
                attempt.solve_seconds
                for attempt in selected
                if attempt.solve_seconds is not None
            ]
        )
        replay_times = np.array(
            [
                attempt.replay_seconds
                for attempt in selected
                if attempt.replay_seconds is not None
            ]
        )
        elapsed = 0.0
        first_accepted: float | None = None
        for attempt in selected:
            elapsed += attempt.total_seconds
            if attempt.accepted and first_accepted is None:
                first_accepted = elapsed
        return BenchmarkSummary(
            backend=backend,
            warm=warm,
            attempts=len(selected),
            accepted=sum(attempt.accepted for attempt in selected),
            p50_total_seconds=float(np.percentile(times, 50)),
            p95_total_seconds=float(np.percentile(times, 95)),
            p50_solve_seconds=(
                float(np.percentile(solve_times, 50)) if solve_times.size else None
            ),
            p95_solve_seconds=(
                float(np.percentile(solve_times, 95)) if solve_times.size else None
            ),
            p50_replay_seconds=(
                float(np.percentile(replay_times, 50)) if replay_times.size else None
            ),
            p95_replay_seconds=(
                float(np.percentile(replay_times, 95)) if replay_times.size else None
            ),
            p50_peak_python_bytes=float(np.percentile(peak_bytes, 50)),
            p95_peak_python_bytes=float(np.percentile(peak_bytes, 95)),
            time_to_first_accepted_s=first_accepted,
        )


def _run_benchmark_attempt(
    problem: SparseCollocationProblem,
    backend: BenchmarkBackend,
    start: BenchmarkStart,
    gate: BenchmarkGate,
) -> BenchmarkAttempt:
    """Time one declared start, retaining failed solves and replay omissions."""
    tracemalloc.start()
    began = time.perf_counter()
    try:
        result = backend.solve(problem, start.initial_torque_nm.copy())
        if result.replay is None:
            raise ValueError("backend omitted fresh independent replay")
        if result.defect_kind not in gate.max_constraint_defect_by_kind:
            raise ValueError("backend defect kind has no predeclared gate")
        accepted = bool(
            result.replay_accepted(
                max_state_gap=gate.max_replay_gap,
                max_constraint_defect=gate.max_constraint_defect_by_kind[
                    result.defect_kind
                ],
                max_bound_violation=gate.max_bound_violation,
            )
            and result.replay.position_rmse_rad <= gate.max_observation_rmse_rad
        )
        failure_reason = None
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        result = None
        accepted = False
        failure_reason = str(exc)
    finally:
        elapsed = time.perf_counter() - began
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    replay_result = None if result is None else result.replay
    return BenchmarkAttempt(
        backend=backend.name,
        start=start.name,
        warm=start.warm,
        optimizer_converged=bool(result and result.optimizer_converged),
        accepted=accepted,
        failure_reason=failure_reason,
        objective=None if result is None else result.objective,
        max_dynamics_defect=None if result is None else result.max_dynamics_defect,
        max_torque_violation=None if result is None else result.max_torque_violation,
        max_rate_violation=None if result is None else result.max_rate_violation,
        max_replay_gap=(None if replay_result is None else replay_result.max_state_gap),
        observation_rmse_rad=(
            None if replay_result is None else replay_result.position_rmse_rad
        ),
        input_degrees_of_freedom=(
            None if result is None else result.input_degrees_of_freedom
        ),
        defect_kind=None if result is None else result.defect_kind,
        solve_seconds=None if result is None else result.solve_seconds,
        replay_seconds=None if result is None else result.replay_seconds,
        total_seconds=elapsed,
        peak_python_bytes=peak,
    )


def benchmark_backends(
    problem: SparseCollocationProblem,
    *,
    backends: Sequence[BenchmarkBackend],
    starts: Sequence[BenchmarkStart],
    gate: BenchmarkGate,
) -> BenchmarkReport:
    """Run predeclared starts serially, keeping failures and Python memory.

    The timer covers the complete solve callback, including its preparation,
    derivative construction, fresh replay and receipt assembly. The memory
    measure is Python-tracked peak allocation, not process RSS or GPU memory.
    """
    if (
        not backends
        or not starts
        or len({backend.name for backend in backends}) != len(backends)
        or len({start.name for start in starts}) != len(starts)
    ):
        raise ValueError("benchmark needs unique backend and start names")
    for start in starts:
        if start.initial_torque_nm.shape != (problem.intervals,):
            raise ValueError("benchmark start torque dimension differs from grid")
    hardware = capture_benchmark_hardware()
    attempts: list[BenchmarkAttempt] = []
    for backend in backends:
        for start in starts:
            attempts.append(_run_benchmark_attempt(problem, backend, start, gate))
    return BenchmarkReport(hardware, tuple(attempts))
