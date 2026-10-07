"""Moving-horizon estimator with arrival information and safe window commits.

Maintains a bounded sample window, solves that window with fixed parameters or
dynamic models, updates arrival factors via rank-revealing marginalization, and
enforces safe window commits so failed or non-finite solves never poison future states.

Arrival cost (#11545). The window decision vector holds cubic-Hermite knot
positions and velocities; the arrival state is the first knot's
``x_0 = (q_0, v_0)``, of size ``2 * n_dof``. Each window minimises

    0.5 * || R (x_0 - x_ref) - r ||^2 + 0.5 * sum_k || rho_k(c) ||^2,

where ``rho_k`` collects the user residual rows of window sample ``k`` and of
transitions between consecutive samples. When the window slides from first
sample ``A`` to ``F``, the rows that will not be evaluated again (the arrival
rows, the rows of samples ``A..F-1`` and the transitions ``A..F``) are
linearised at the last accepted solution and the knots ``A..F-1`` are
eliminated by a Schur complement (:func:`marginalize_arrival_factor`), leaving a
factor on knot ``F``. The update is committed only when the window solve is
accepted. For a linear-Gaussian model this is exact: every window equals the
batch MAP (Kalman/RTS) estimate over all samples seen so far. Earlier versions
stored the factor but never applied or propagated it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np

from src.shared.python.contracts import PreconditionError, require
from src.shared.python.estimation.map_estimator import (
    CubicHermiteSplineTrajectory,
    MapDecisionLayout,
    MapEstimatorOptions,
    MapEstimatorProblem,
    MapEstimatorResult,
    SharedParameterBlock,
    SplineTrajectoryEvaluation,
    solve_single_trial_map,
)

FixedResidualFn = Callable[
    [SplineTrajectoryEvaluation, Mapping[str, float]], np.ndarray
]
FixedJacobianFn = Callable[
    [SplineTrajectoryEvaluation, Mapping[str, float], MapDecisionLayout],
    np.ndarray,
]
ResultCallback = Callable[["MovingHorizonResult"], None]

logger = logging.getLogger(__name__)


def _make_readonly(arr: np.ndarray) -> np.ndarray:
    """Return a read-only float64 numpy array copy."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


class LateSamplePolicy(str, Enum):
    """Policy for handling out-of-order or duplicate sample timestamps."""

    REJECT = "reject"
    DROP_LATE = "drop_late"


class WindowCommitStatus(str, Enum):
    """Outcome status of a moving-horizon window solve commit."""

    ACCEPTED = "accepted"
    REJECTED_NONFINITE = "rejected_nonfinite"
    REJECTED_UNSUCCESSFUL = "rejected_unsuccessful"
    REJECTED_CONSTRAINT_VIOLATION = "rejected_constraint_violation"


@dataclass(frozen=True)
class FailureDiagnostics:
    """Diagnostics recorded when a window solve is rejected."""

    window_index: int
    reason: str
    status: WindowCommitStatus
    solver_message: str = ""
    n_iterations: int = 0
    non_finite_indices: tuple[int, ...] = ()
    constraint_residuals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _ArrivalUpdate:
    """Tentative arrival factor for a window, committed only on acceptance.

    ``failure`` is non-empty when the propagation linearisation was
    non-finite; the window is then rejected and committed state is unchanged.
    ``discard_reason`` is non-empty when information is lost on commit.
    """

    factor: ArrivalFactor | None
    anchor: int
    marginalized: tuple[tuple[int, float], ...] = ()
    failure: str = ""
    discard_reason: str = ""


@dataclass(frozen=True)
class ArrivalFactor:
    """Square-root quadratic arrival factor in tangent coordinates about a reference.

    Cost: 0.5 * || R * (x - x_ref) - r ||^2
    """

    reference_state: np.ndarray
    sqrt_information: np.ndarray
    residual_offset: np.ndarray
    rank: int
    linearization_point: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ref = np.asarray(self.reference_state, dtype=np.float64)
        sqrt_info = np.asarray(self.sqrt_information, dtype=np.float64)
        offset = np.asarray(self.residual_offset, dtype=np.float64)
        require(ref.ndim == 1, "reference_state must be a 1D vector")
        require(sqrt_info.ndim == 2, "sqrt_information must be a 2D matrix")
        require(offset.ndim == 1, "residual_offset must be a 1D vector")
        require(
            sqrt_info.shape[0] == offset.size,
            "sqrt_information rows must match residual_offset size",
        )
        require(
            sqrt_info.shape[1] == ref.size,
            "sqrt_information columns must match reference_state size",
        )
        require(0 <= self.rank <= sqrt_info.shape[1], "rank out of valid bounds")
        object.__setattr__(self, "reference_state", _make_readonly(ref))
        object.__setattr__(self, "sqrt_information", _make_readonly(sqrt_info))
        object.__setattr__(self, "residual_offset", _make_readonly(offset))
        if self.linearization_point is not None:
            lin = np.asarray(self.linearization_point, dtype=np.float64)
            require(lin.shape == ref.shape, "linearization_point shape mismatch")
            object.__setattr__(self, "linearization_point", _make_readonly(lin))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_gaussian_prior(
        cls,
        mean: np.ndarray,
        information: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArrivalFactor:
        """Build the factor of a Gaussian prior ``N(mean, information^-1)``.

        Preconditions: ``mean`` is a finite 1D vector of size ``n`` and
        ``information`` is a finite, symmetric, positive-definite ``(n, n)``
        matrix (checked by Cholesky factorisation).

        Postcondition: ``R^T R == information`` with ``R`` upper triangular,
        ``residual_offset == 0`` and ``rank == n``, so
        ``evaluate_cost(x) == 0.5 (x - mean)^T information (x - mean)``.
        """
        mu = np.asarray(mean, dtype=np.float64)
        info = np.asarray(information, dtype=np.float64)
        require(mu.ndim == 1 and mu.size > 0, "mean must be a non-empty 1D vector")
        require(bool(np.all(np.isfinite(mu))), "mean must be finite")
        require(
            info.shape == (mu.size, mu.size),
            f"information must have shape {(mu.size, mu.size)}",
        )
        require(bool(np.all(np.isfinite(info))), "information must be finite")
        require(
            bool(np.allclose(info, info.T, rtol=1e-12, atol=1e-12)),
            "information must be symmetric",
        )
        try:
            lower = np.linalg.cholesky(info)
        except np.linalg.LinAlgError:
            lower = None
        require(lower is not None, "information must be positive definite")
        assert lower is not None  # narrowed by the precondition above
        return cls(
            reference_state=mu,
            sqrt_information=lower.T,
            residual_offset=np.zeros(mu.size),
            rank=mu.size,
            metadata=dict(metadata or {}),
        )

    @property
    def is_rank_deficient(self) -> bool:
        """Whether the information matrix has revealed rank deficiency."""
        return self.rank < self.sqrt_information.shape[1]

    def evaluate_residual(self, state: np.ndarray) -> np.ndarray:
        """Compute arrival residual R * (state - ref) - r."""
        x = np.asarray(state, dtype=np.float64)
        require(x.shape == self.reference_state.shape, "state shape mismatch")
        delta_x = x - self.reference_state
        return self.sqrt_information @ delta_x - self.residual_offset

    def evaluate_cost(self, state: np.ndarray) -> float:
        """Compute quadratic cost 0.5 * || R * (state - ref) - r ||^2."""
        res = self.evaluate_residual(state)
        return 0.5 * float(np.sum(res**2))

    def evaluate_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Evaluate Jacobian with respect to tangent perturbation."""
        x = np.asarray(state, dtype=np.float64)
        require(x.shape == self.reference_state.shape, "state shape mismatch")
        return np.array(self.sqrt_information, copy=True)


def marginalize_arrival_factor(
    A0: np.ndarray,
    A1: np.ndarray,
    b: np.ndarray,
    reference_state_k1: np.ndarray,
    gauge_policy: str = "retain_rank_deficiency",
    metadata: Mapping[str, Any] | None = None,
    rank_tol: float = 1e-10,
) -> ArrivalFactor:
    """Perform rank-revealing marginalization of A0 delta x0 + A1 delta x1 ~ b.

    Eliminates delta x0 and forms an updated square-root arrival factor on delta x1.
    If A0 is singular/unsupported, rank deficiency is retained without diagonal jitter.
    """
    mat_a0 = np.asarray(A0, dtype=np.float64)
    mat_a1 = np.asarray(A1, dtype=np.float64)
    vec_b = np.asarray(b, dtype=np.float64)
    require(mat_a0.ndim == 2 and mat_a1.ndim == 2, "A0 and A1 must be 2D matrices")
    require(mat_a0.shape[0] == mat_a1.shape[0] == vec_b.size, "row dimension mismatch")
    n0 = mat_a0.shape[1]
    n1 = mat_a1.shape[1]

    # Full SVD on A0: A0 = U @ diag(S) @ Vt
    U, S, _ = np.linalg.svd(mat_a0, full_matrices=True)
    tol = max(rank_tol, S[0] * max(mat_a0.shape) * 1e-12) if S.size > 0 else rank_tol
    rank_a0 = int(np.sum(tol < S))

    if rank_a0 < n0 and gauge_policy == "fail_closed":
        raise PreconditionError(
            f"Singular/rank-deficient marginalization (rank {rank_a0} < {n0}) under fail_closed gauge policy."
        )

    # Rotate system by U^T:
    # First rank_a0 rows depend on x0 in observed subspace.
    # Remaining M - rank_a0 rows are completely decoupled from x0!
    U2 = U[:, rank_a0:]
    R_rem = U2.T @ mat_a1
    b_rem = U2.T @ vec_b

    # SVD on R_rem to reveal rank and obtain canonical square-root form without jitter
    if R_rem.shape[0] > 0 and R_rem.shape[1] > 0:
        U_rem, S_rem, Vt_rem = np.linalg.svd(R_rem, full_matrices=False)
        tol_rem = (
            max(rank_tol, S_rem[0] * max(R_rem.shape) * 1e-12)
            if S_rem.size > 0
            else rank_tol
        )
        rank_1 = int(np.sum(S_rem > tol_rem))
        # Zero out below-threshold singular values strictly without jitter
        S_clean = np.where(S_rem > tol_rem, S_rem, 0.0)
        # Pad or form square (n1, n1) matrix
        k_modes = len(S_clean)
        R_sq = np.zeros((n1, n1), dtype=np.float64)
        R_sq[:k_modes, :] = np.diag(S_clean) @ Vt_rem
        r_sq = np.zeros(n1, dtype=np.float64)
        r_sq[:k_modes] = U_rem.T @ b_rem
    else:
        R_sq = np.zeros((n1, n1), dtype=np.float64)
        r_sq = np.zeros(n1, dtype=np.float64)
        rank_1 = 0

    meta = dict(metadata or {})
    meta["gauge_policy"] = gauge_policy
    meta["eliminated_dim"] = n0
    meta["eliminated_rank"] = rank_a0

    return ArrivalFactor(
        reference_state=np.asarray(reference_state_k1, dtype=np.float64),
        sqrt_information=R_sq,
        residual_offset=r_sq,
        rank=rank_1,
        linearization_point=np.asarray(reference_state_k1, dtype=np.float64),
        metadata=meta,
    )


class AccumulationGuard:
    """Guard against double-counted measurements across receding window advances."""

    def __init__(self) -> None:
        self._marginalized_indices: set[int] = set()
        self._marginalized_timestamps: set[float] = set()

    def record_marginalized(self, sample_index: int, timestamp: float) -> None:
        """Record a sample that has been marginalized into the arrival factor."""
        require(
            isinstance(sample_index, int) and sample_index >= 0, "invalid sample_index"
        )
        require(np.isfinite(timestamp), "invalid timestamp")
        self._marginalized_indices.add(sample_index)
        self._marginalized_timestamps.add(float(timestamp))

    def is_marginalized(self, sample_index: int) -> bool:
        """Check if sample_index has been marginalized into arrival factor."""
        return sample_index in self._marginalized_indices

    def validate_sample(self, sample_index: int, timestamp: float) -> None:
        """Ensure sample has not been previously marginalized into the arrival factor."""
        if sample_index in self._marginalized_indices:
            raise PreconditionError(
                f"Sample index {sample_index} at t={timestamp} is already marginalized into the arrival factor; double-counting prevented."
            )


@dataclass(frozen=True)
class MovingHorizonOptions:
    """Configuration for bounded near-real-time window solves."""

    window_size: int
    step_size: int = 1
    latency_budget_ms: float = 50.0
    solver_options: MapEstimatorOptions = MapEstimatorOptions(max_iterations=20)
    enable_safe_commits: bool = True
    late_sample_policy: LateSamplePolicy = LateSamplePolicy.REJECT
    max_history_diagnostics: int = 50

    def __post_init__(self) -> None:
        require(self.window_size >= 2, "window_size must be at least 2")
        require(self.step_size >= 1, "step_size must be positive")
        require(self.step_size <= self.window_size, "step_size must fit the window")
        require(self.latency_budget_ms > 0.0, "latency_budget_ms must be positive")
        require(
            self.max_history_diagnostics > 0, "max_history_diagnostics must be positive"
        )


@dataclass(frozen=True)
class MovingHorizonProblem:
    """Problem definition for a fixed-parameter moving-horizon solve."""

    n_dof: int
    fixed_parameters: Mapping[str, float]
    residual: FixedResidualFn
    jacobian: FixedJacobianFn | None = None
    options: MovingHorizonOptions = MovingHorizonOptions(window_size=2)
    callback: ResultCallback | None = None
    arrival_factor: ArrivalFactor | None = None

    def __post_init__(self) -> None:
        require(self.n_dof > 0, "n_dof must be positive")
        for name, value in self.fixed_parameters.items():
            require(str(name).strip() != "", "parameter names must be non-empty")
            require(np.isfinite(float(value)), f"{name} must be finite")
        if self.arrival_factor is not None:
            arrival = self.arrival_factor
            require(
                arrival.reference_state.size == 2 * self.n_dof,
                "arrival_factor state must be the first knot (q, v): size "
                f"2 * n_dof = {2 * self.n_dof}, got {arrival.reference_state.size}",
            )
            require(
                bool(np.all(np.isfinite(arrival.sqrt_information)))
                and bool(np.all(np.isfinite(arrival.reference_state)))
                and bool(np.all(np.isfinite(arrival.residual_offset))),
                "arrival_factor must be finite",
            )


@dataclass(frozen=True)
class MovingHorizonResult:
    """Result of one deterministic moving-horizon window solve."""

    success: bool
    window_index: int
    sample_start: int
    sample_stop: int
    window_times: np.ndarray
    coefficients: np.ndarray
    initial_coefficients: np.ndarray
    parameters: dict[str, float]
    residual: np.ndarray
    objective: float
    n_iterations: int
    latency_ms: float
    latency_budget_ms: float
    warm_started: bool
    message: str
    commit_status: WindowCommitStatus = WindowCommitStatus.ACCEPTED
    failure_diagnostics: FailureDiagnostics | None = None
    arrival_factor: ArrivalFactor | None = None

    @property
    def over_budget(self) -> bool:
        """Whether the solve exceeded the configured per-window budget."""
        return self.latency_ms > self.latency_budget_ms

    def callback_payload(self) -> dict[str, Any]:
        """Return a JSON-serialisable callback payload for realtime bridges."""
        return {
            "success": self.success,
            "window_index": self.window_index,
            "sample_start": self.sample_start,
            "sample_stop": self.sample_stop,
            "objective": self.objective,
            "n_iterations": self.n_iterations,
            "latency_ms": self.latency_ms,
            "latency_budget_ms": self.latency_budget_ms,
            "over_budget": self.over_budget,
            "warm_started": self.warm_started,
            "parameters": dict(self.parameters),
            "commit_status": self.commit_status.value,
        }


@dataclass
class _SampleBuffer:
    times: list[float] = field(default_factory=list)
    q_rows: list[np.ndarray] = field(default_factory=list)
    first_sample_index: int = 0
    late_policy: LateSamplePolicy = LateSamplePolicy.REJECT

    @property
    def size(self) -> int:
        return len(self.times)

    def append(self, times: Sequence[float], q_samples: np.ndarray, n_dof: int) -> None:
        sample_times = np.asarray(times, dtype=float)
        sample_q = np.asarray(q_samples, dtype=float)
        require(sample_times.ndim == 1, "times must be a 1D array")
        require(sample_q.shape == (sample_times.size, n_dof), "q_samples shape invalid")
        require(bool(np.all(np.isfinite(sample_times))), "times must be finite")
        require(bool(np.all(np.isfinite(sample_q))), "q_samples must be finite")

        if self.late_policy == LateSamplePolicy.DROP_LATE:
            keep_indices = []
            last_t = self.times[-1] if self.times else -np.inf
            for idx, t_val in enumerate(sample_times):
                if t_val > last_t:
                    keep_indices.append(idx)
                    last_t = t_val
            if not keep_indices:
                return
            sample_times = sample_times[keep_indices]
            sample_q = sample_q[keep_indices]
        else:
            if self.times and sample_times.size:
                require(
                    sample_times[0] > self.times[-1],
                    "new samples must advance monotonically",
                )
            if sample_times.size > 1:
                require(
                    bool(np.all(np.diff(sample_times) > 0.0)), "times must increase"
                )

        self.times.extend(float(value) for value in sample_times)
        self.q_rows.extend(np.array(row, dtype=float) for row in sample_q)

    def trim_to(self, max_size: int) -> None:
        overflow = self.size - max_size
        if overflow <= 0:
            return
        del self.times[:overflow]
        del self.q_rows[:overflow]
        self.first_sample_index += overflow

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.times, dtype=float), np.vstack(self.q_rows)


class MovingHorizonEstimator:
    """Bounded moving-horizon estimator with deterministic window advancement."""

    def __init__(self, problem: MovingHorizonProblem) -> None:
        self._problem = problem
        self._buffer = _SampleBuffer(late_policy=problem.options.late_sample_policy)
        self._last_solved_stop = 0
        self._window_index = 0
        self._previous_trajectory: CubicHermiteSplineTrajectory | None = None
        self._previous_coefficients: np.ndarray | None = None
        self._last_accepted_coefficients: np.ndarray | None = None
        self._last_accepted_trajectory: CubicHermiteSplineTrajectory | None = None
        self._last_failure_diagnostics: FailureDiagnostics | None = None
        self._failure_history: list[FailureDiagnostics] = []
        self._arrival_factor: ArrivalFactor | None = problem.arrival_factor
        # Sample index whose first-knot state the arrival factor constrains.
        self._arrival_sample_index = 0
        self._pending_arrival = _ArrivalUpdate(problem.arrival_factor, 0)
        self._last_accepted_first_index = 0
        self._accumulation_guard = AccumulationGuard()

    @property
    def buffered_sample_count(self) -> int:
        """Number of retained samples in the bounded window buffer."""
        return self._buffer.size

    @property
    def last_accepted_coefficients(self) -> np.ndarray | None:
        """Last successfully solved and accepted spline coefficients."""
        return (
            None
            if self._last_accepted_coefficients is None
            else self._last_accepted_coefficients.copy()
        )

    @property
    def last_failure_diagnostics(self) -> FailureDiagnostics | None:
        """Diagnostics from the most recent rejected solve, if any."""
        return self._last_failure_diagnostics

    @property
    def arrival_factor(self) -> ArrivalFactor | None:
        """Current arrival information factor on the window horizon."""
        return self._arrival_factor

    @property
    def arrival_sample_index(self) -> int:
        """Sample index whose knot state the current arrival factor constrains."""
        return self._arrival_sample_index

    @property
    def accumulation_guard(self) -> AccumulationGuard:
        """Accumulation guard protecting against double-counted measurements."""
        return self._accumulation_guard

    def buffer_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, q_samples) currently retained in the window buffer."""
        return self._buffer.arrays()

    def append_samples(self, times: Sequence[float], q_samples: np.ndarray) -> None:
        """Append strictly increasing samples and retain only the active window."""
        opts = self._problem.options
        self._buffer.append(times, q_samples, self._problem.n_dof)
        self._buffer.trim_to(opts.window_size)

    def ready(self) -> bool:
        """Return true when enough new samples exist for the next solve."""
        opts = self._problem.options
        if self._buffer.size < opts.window_size:
            return False
        retained_stop = self._buffer.first_sample_index + self._buffer.size
        if self._last_solved_stop == 0:
            return True
        return retained_stop - self._last_solved_stop >= opts.step_size

    def build_current_problem(self) -> MapEstimatorProblem:
        """Build the fixed-parameter MAP problem for the retained window.

        Computes the arrival factor for the window's first sample (tentative;
        committed state changes only when :meth:`solve_next` accepts the
        window), so the returned problem includes the arrival rows.
        """
        require(self.ready(), "not enough new samples for a moving-horizon solve")
        self._pending_arrival = self._propagate_arrival()
        times, q_samples = self._buffer.arrays()
        trajectory = CubicHermiteSplineTrajectory(times, self._problem.n_dof)
        initial_coefficients = self._initial_coefficients(trajectory, times, q_samples)
        return self._map_problem(trajectory, times, initial_coefficients)

    def solve_next(self) -> MovingHorizonResult | None:
        """Solve the next ready window, or return ``None`` if no window advanced."""
        if not self.ready():
            return None
        map_problem = self.build_current_problem()
        opts = self._problem.options
        warm_started = (
            self._last_accepted_coefficients is not None
            if opts.enable_safe_commits
            else self._previous_coefficients is not None
        )
        started = perf_counter()
        map_result = solve_single_trial_map(map_problem)
        latency_ms = (perf_counter() - started) * 1000.0

        commit_status, failure_diag = self._evaluate_commit(map_result)
        pending = self._pending_arrival
        if pending.failure and commit_status == WindowCommitStatus.ACCEPTED:
            commit_status = WindowCommitStatus.REJECTED_NONFINITE
            failure_diag = FailureDiagnostics(
                window_index=self._window_index,
                reason=f"non_finite arrival propagation: {pending.failure}",
                status=commit_status,
                solver_message=map_result.message,
                n_iterations=map_result.n_iterations,
            )

        if commit_status == WindowCommitStatus.ACCEPTED:
            self._commit_arrival(pending)
            self._last_accepted_first_index = self._buffer.first_sample_index
            self._last_accepted_coefficients = map_result.coefficients
            self._last_accepted_trajectory = map_problem.trajectory
            self._previous_coefficients = map_result.coefficients
            self._previous_trajectory = map_problem.trajectory
        else:
            self._last_failure_diagnostics = failure_diag
            if failure_diag is not None:
                self._record_failure(failure_diag)
            if not opts.enable_safe_commits:
                self._previous_coefficients = map_result.coefficients
                self._previous_trajectory = map_problem.trajectory

        self._last_solved_stop = self._buffer.first_sample_index + self._buffer.size
        result = self._to_result(
            map_problem,
            map_result,
            latency_ms,
            warm_started,
            commit_status,
            failure_diag,
        )
        self._window_index += 1
        if self._problem.callback is not None:
            self._problem.callback(result)
        return result

    def _evaluate_commit(
        self, map_result: MapEstimatorResult
    ) -> tuple[WindowCommitStatus, FailureDiagnostics | None]:
        coeffs = np.asarray(map_result.coefficients, dtype=float)
        res = np.asarray(map_result.residual, dtype=float)
        if (
            not np.all(np.isfinite(coeffs))
            or not np.all(np.isfinite(res))
            or getattr(map_result, "n_non_finite_evaluations", 0) > 0
            or bool(np.any(np.abs(res) >= 1e11))
        ):
            non_finite = tuple(int(i) for i in np.where(~np.isfinite(coeffs))[0])
            diag = FailureDiagnostics(
                window_index=self._window_index,
                reason="non_finite coefficients or residuals detected",
                status=WindowCommitStatus.REJECTED_NONFINITE,
                solver_message=map_result.message,
                n_iterations=map_result.n_iterations,
                non_finite_indices=non_finite,
            )
            return WindowCommitStatus.REJECTED_NONFINITE, diag

        if not map_result.success:
            diag = FailureDiagnostics(
                window_index=self._window_index,
                reason=f"solver reported unsuccessful exit: {map_result.message}",
                status=WindowCommitStatus.REJECTED_UNSUCCESSFUL,
                solver_message=map_result.message,
                n_iterations=map_result.n_iterations,
            )
            return WindowCommitStatus.REJECTED_UNSUCCESSFUL, diag

        return WindowCommitStatus.ACCEPTED, None

    def _record_failure(self, diag: FailureDiagnostics) -> None:
        self._failure_history.append(diag)
        opts = self._problem.options
        overflow = len(self._failure_history) - opts.max_history_diagnostics
        if overflow > 0:
            del self._failure_history[:overflow]

    def _initial_coefficients(
        self,
        trajectory: CubicHermiteSplineTrajectory,
        times: np.ndarray,
        q_samples: np.ndarray,
    ) -> np.ndarray:
        opts = self._problem.options
        source_coeffs = (
            self._last_accepted_coefficients
            if opts.enable_safe_commits
            else self._previous_coefficients
        )
        source_traj = (
            self._last_accepted_trajectory
            if opts.enable_safe_commits
            else self._previous_trajectory
        )
        if source_traj is None or source_coeffs is None:
            return trajectory.initial_coefficients_from_samples(times, q_samples)
        previous = source_traj.evaluate(source_coeffs, times)
        return trajectory.pack(previous.q, previous.v)

    def _map_problem(
        self,
        trajectory: CubicHermiteSplineTrajectory,
        times: np.ndarray,
        initial_coefficients: np.ndarray,
    ) -> MapEstimatorProblem:
        fixed = dict(self._problem.fixed_parameters)
        arrival = self._pending_arrival.factor

        def residual(
            evaluation: SplineTrajectoryEvaluation, _parameters: Mapping[str, float]
        ) -> np.ndarray:
            data = np.asarray(self._problem.residual(evaluation, fixed), dtype=float)
            if arrival is None:
                return data
            return np.concatenate([data, _arrival_rows(arrival, evaluation, 0)[0]])

        jacobian_wrapper = None
        problem_jacobian = self._problem.jacobian
        if problem_jacobian is not None:

            def jacobian_wrapper(
                evaluation: SplineTrajectoryEvaluation,
                _parameters: Mapping[str, float],
                layout: MapDecisionLayout,
            ) -> np.ndarray:
                data = np.asarray(
                    problem_jacobian(evaluation, fixed, layout), dtype=float
                )
                if arrival is None:
                    return data
                arrival_jac = np.zeros((arrival.residual_offset.size, layout.size))
                arrival_jac[:, : layout.trajectory_size] = _arrival_rows(
                    arrival, evaluation, 0
                )[1]
                return np.vstack([data, arrival_jac])

        opts = self._problem.options
        solver_opts = opts.solver_options
        return MapEstimatorProblem(
            trajectory=trajectory,
            evaluation_times=times,
            initial_coefficients=initial_coefficients,
            shared_parameters=SharedParameterBlock.from_specs([]),
            residual=residual,
            jacobian=jacobian_wrapper,
            options=solver_opts,
        )

    def _to_result(
        self,
        problem: MapEstimatorProblem,
        map_result: MapEstimatorResult,
        latency_ms: float,
        warm_started: bool,
        commit_status: WindowCommitStatus = WindowCommitStatus.ACCEPTED,
        failure_diag: FailureDiagnostics | None = None,
    ) -> MovingHorizonResult:
        opts = self._problem.options
        latency_budget = opts.latency_budget_ms
        sample_start = self._buffer.first_sample_index
        sample_stop = sample_start + self._buffer.size
        return MovingHorizonResult(
            success=map_result.success
            and (commit_status == WindowCommitStatus.ACCEPTED),
            window_index=self._window_index,
            sample_start=sample_start,
            sample_stop=sample_stop,
            window_times=np.array(problem.evaluation_times, dtype=float),
            coefficients=map_result.coefficients,
            initial_coefficients=np.array(problem.initial_coefficients, dtype=float),
            parameters=dict(self._problem.fixed_parameters),
            residual=map_result.residual,
            objective=map_result.objective,
            n_iterations=map_result.n_iterations,
            latency_ms=latency_ms,
            latency_budget_ms=latency_budget,
            warm_started=warm_started,
            message=map_result.message,
            commit_status=commit_status,
            failure_diagnostics=failure_diag,
            arrival_factor=self._pending_arrival.factor,
        )

    def _propagate_arrival(self) -> _ArrivalUpdate:
        """Return the arrival factor for the current window without committing.

        Moves the committed factor from sample ``A`` to the window's first
        sample ``F`` using the last accepted window (first sample ``S``) as
        the linearisation point; requires ``S <= A < F < S + N``. Otherwise the
        dropped samples were never solved, the information cannot be carried
        and the factor is discarded on commit with a warning (never silently).
        """
        first = self._buffer.first_sample_index
        anchor = self._arrival_sample_index
        if anchor >= first:
            return _ArrivalUpdate(self._arrival_factor, anchor)
        traj = self._last_accepted_trajectory
        coeffs = self._last_accepted_coefficients
        source = self._last_accepted_first_index
        if traj is None or coeffs is None:
            return _ArrivalUpdate(
                None, first, discard_reason="no accepted window to linearise"
            )
        if not (source <= anchor and first - source < traj.n_knots):
            return _ArrivalUpdate(None, first, discard_reason="samples never solved")
        factor = _marginalize_window_prefix(
            traj,
            coeffs,
            self._arrival_factor,
            (anchor - source, first - source),
            self._dropped_residual_fn(traj),
        )
        if factor is None:
            return _ArrivalUpdate(
                None, anchor, failure="residual or Jacobian non-finite"
            )
        times = traj.knot_times[anchor - source : first - source]
        dropped = tuple(
            (anchor + offset, float(time)) for offset, time in enumerate(times)
        )
        return _ArrivalUpdate(factor, first, dropped)

    def _commit_arrival(self, update: _ArrivalUpdate) -> None:
        """Make an accepted window's arrival update the committed state."""
        if update.discard_reason and self._arrival_factor is not None:
            logger.warning(
                "MHE arrival factor discarded moving from sample %d to %d: %s",
                self._arrival_sample_index,
                update.anchor,
                update.discard_reason,
            )
        for index, time in update.marginalized:
            self._accumulation_guard.validate_sample(index, time)
            self._accumulation_guard.record_marginalized(index, time)
        self._arrival_factor = update.factor
        self._arrival_sample_index = update.anchor

    def _dropped_residual_fn(
        self, traj: CubicHermiteSplineTrajectory
    ) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Return ``(coeffs, times) -> (residual, jacobian)`` on ``traj``."""
        fixed = dict(self._problem.fixed_parameters)
        user_jacobian = self._problem.jacobian
        layout = MapDecisionLayout(traj.coefficient_size, ())

        def residual_at(coeffs: np.ndarray, times: np.ndarray) -> np.ndarray:
            evaluation = traj.evaluate(coeffs, times)
            return np.asarray(self._problem.residual(evaluation, fixed), dtype=float)

        def rows(
            coeffs: np.ndarray, times: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            value = residual_at(coeffs, times)
            if user_jacobian is not None:
                evaluation = traj.evaluate(coeffs, times)
                jac = np.asarray(user_jacobian(evaluation, fixed, layout), dtype=float)
                return value, jac[:, : traj.coefficient_size]
            return value, _forward_difference(
                lambda c: residual_at(c, times), coeffs, value
            )

        return rows


def _knot_state_columns(n_knots: int, n_dof: int, knot: int) -> np.ndarray:
    """Decision columns of knot ``knot``'s state ``(q, v)`` in spline order."""
    q_cols = knot * n_dof + np.arange(n_dof)
    return np.concatenate([q_cols, n_knots * n_dof + q_cols])


def _arrival_rows(
    arrival: ArrivalFactor, evaluation: SplineTrajectoryEvaluation, sample: int
) -> tuple[np.ndarray, np.ndarray]:
    """Arrival residual and coefficient Jacobian at evaluation sample ``sample``.

    The sample must coincide with a knot so ``(q, v)`` there is the knot state.
    """
    state = np.concatenate([evaluation.q[sample], evaluation.v[sample]])
    basis = np.vstack([evaluation.q_basis[sample], evaluation.v_basis[sample]])
    return arrival.evaluate_residual(state), arrival.sqrt_information @ basis


def _forward_difference(
    fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, f0: np.ndarray
) -> np.ndarray:
    jac = np.zeros((f0.size, x.size))
    for col in range(x.size):
        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(x[col])))
        probe = np.array(x, dtype=float, copy=True)
        probe[col] += step
        jac[:, col] = (fn(probe) - f0) / step
    return jac


def _marginalize_window_prefix(
    traj: CubicHermiteSplineTrajectory,
    coeffs: np.ndarray,
    arrival: ArrivalFactor | None,
    knot_span: tuple[int, int],
    rows_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> ArrivalFactor | None:
    """Eliminate knots ``[a, f)`` of a solved window onto knot ``f``.

    The rows that leave the problem are those evaluated on samples ``a..f``
    (including transitions into ``f``) minus those of sample ``f`` alone,
    which the next window evaluates again, plus the arrival rows on knot
    ``a``. Linearised at ``coeffs`` they give the information
    ``H = J_a^T J_a - J_f^T J_f`` and gradient ``g``; a square root
    ``A^T A = H``, ``A^T b = -g`` is marginalised by
    :func:`marginalize_arrival_factor`. Returns ``None`` when the
    linearisation is non-finite.

    Preconditions: residual rows are per-sample or couple consecutive
    samples only, so ``H`` is positive semidefinite and touches no knot
    outside ``a..f``; anything else would drop or double count information
    and is rejected.
    """
    start, stop = knot_span
    n_knots, n_dof = traj.n_knots, traj.n_dof
    value, jac = rows_fn(coeffs, traj.knot_times[start : stop + 1])
    boundary_value, boundary_jac = rows_fn(coeffs, traj.knot_times[stop : stop + 1])
    if arrival is not None:
        evaluation = traj.evaluate(coeffs, traj.knot_times[start : start + 1])
        arrival_value, arrival_jac = _arrival_rows(arrival, evaluation, 0)
        value = np.concatenate([value, arrival_value])
        jac = np.vstack([jac, arrival_jac])
    arrays = (value, jac, boundary_value, boundary_jac)
    if not all(bool(np.all(np.isfinite(arr))) for arr in arrays):
        return None
    info = jac.T @ jac - boundary_jac.T @ boundary_jac
    info = 0.5 * (info + info.T)
    grad = jac.T @ value - boundary_jac.T @ boundary_value
    eliminated = np.concatenate(
        [_knot_state_columns(n_knots, n_dof, k) for k in range(start, stop)]
    )
    kept = _knot_state_columns(n_knots, n_dof, stop)
    cols = np.concatenate([eliminated, kept])
    other = np.setdiff1d(np.arange(traj.coefficient_size), cols)
    scale = max(1.0, float(np.max(np.abs(info))))
    require(
        other.size == 0 or float(np.max(np.abs(info[other]))) <= 1e-9 * scale,
        "marginalised residual rows must touch only the dropped knots and the "
        "new first knot (per-sample or consecutive-sample residuals); otherwise "
        "information is dropped or double counted",
    )
    eigvals, eigvecs = np.linalg.eigh(info[np.ix_(cols, cols)])
    require(
        float(eigvals.min()) >= -1e-9 * scale,
        "boundary-sample rows must be a subset of the span rows (information "
        "of the marginalised rows is not positive semidefinite)",
    )
    keep = eigvals > 1e-12 * scale * cols.size
    root = np.sqrt(eigvals[keep])
    sqrt_rows = root[:, None] * eigvecs[:, keep].T
    target = -(eigvecs[:, keep].T @ grad[cols]) / root
    n_elim = eliminated.size
    return marginalize_arrival_factor(
        A0=sqrt_rows[:, :n_elim],
        A1=sqrt_rows[:, n_elim:],
        b=target,
        reference_state_k1=coeffs[kept],
        metadata={"eliminated_knots": stop - start},
    )
