"""Native Drift and Window Solver Acceleration and Caching (DIME-14, #11435).

Provides:
- DimeCacheIdentity: Structured multi-tiered cache key covering model, parameters,
  contact policy, solver configuration, camera configuration, and job isolation.
- DimeCostBreakdown: Granular profiling report covering drift, full-step, Jacobian,
  assembly/factorization, window solve, independent replay, and failure costs.
- LocalModelApproximation: Local linear model with explicit validity radius and bounds.
- DimeSolverCache: Thread-safe high-performance cache with invalidation lifecycles,
  cross-job contamination guards, and impact phase discontinuity checks.
- accelerated_solve_dynamics_window: Accelerated window solver with mandatory replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np

from src.shared.python.contracts import PreconditionError, require
from src.shared.python.estimation.dime_contracts import (
    ContactPolicy,
    DimeCompleteState,
    DimeFullStepRequest,
    DimeFullStepResult,
)
from src.shared.python.estimation.dime_dynamics_window import (
    DimeDynamicsWindowProblem,
    DimeDynamicsWindowResult,
    solve_dime_dynamics_window,
)


@dataclass(frozen=True)
class DimeCacheIdentity:
    """Multi-tiered identity for cache partitioning and isolation."""

    model_hash: str
    param_hash: str
    contact_policy: ContactPolicy
    solver_config_hash: str
    camera_config_hash: str
    job_id: str

    def __post_init__(self) -> None:
        require(bool(self.model_hash.strip()), "model_hash must not be empty")
        require(bool(self.param_hash.strip()), "param_hash must not be empty")
        require(
            isinstance(self.contact_policy, ContactPolicy), "contact_policy invalid"
        )
        require(
            bool(self.solver_config_hash.strip()),
            "solver_config_hash must not be empty",
        )
        require(
            bool(self.camera_config_hash.strip()),
            "camera_config_hash must not be empty",
        )
        require(bool(self.job_id.strip()), "job_id must not be empty")

    def to_composite_key(self, qualifier: str = "") -> str:
        """Compute unique composite hash key with job and configuration isolation."""
        raw_key = (
            f"job:{self.job_id}|model:{self.model_hash}|param:{self.param_hash}|"
            f"contact:{self.contact_policy.value}|solver:{self.solver_config_hash}|"
            f"camera:{self.camera_config_hash}|q:{qualifier}"
        )
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DimeCostBreakdown:
    """Cost profiling report for native drift and window solves.

    Every numeric field is either a value measured during the solve or ``None``
    meaning *not measured* (#11547). Nothing is derived as a fixed fraction of
    wall time. ``solver_cost_terms`` carries the objective components computed
    by the window solver itself, and they sum to its ``total_cost``.
    """

    drift_calls_count: int | None = None
    drift_time_s: float | None = None
    full_step_calls_count: int | None = None
    full_step_time_s: float | None = None
    jacobian_calls_count: int | None = None
    jacobian_time_s: float | None = None
    assembly_factorization_time_s: float | None = None
    window_solve_time_s: float = 0.0
    independent_replay_time_s: float = 0.0
    total_time_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    replay_executed: bool = True
    failure_costs: float = 0.0
    p50_speed_s: float | None = None
    p95_speed_s: float | None = None
    solver_cost_terms: Mapping[str, float] | None = None

    def not_measured_fields(self) -> tuple[str, ...]:
        """Names of numeric fields that were not measured (value is None)."""
        return tuple(
            name
            for name in _COST_BREAKDOWN_NUMERIC_FIELDS
            if getattr(self, name) is None
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize cost breakdown; unmeasured fields stay None and are listed."""
        out: dict[str, Any] = {
            name: getattr(self, name) for name in _COST_BREAKDOWN_NUMERIC_FIELDS
        }
        out["replay_executed"] = self.replay_executed
        out["solver_cost_terms"] = (
            dict(self.solver_cost_terms) if self.solver_cost_terms is not None else None
        )
        out["not_measured"] = list(self.not_measured_fields())
        return out


_COST_BREAKDOWN_NUMERIC_FIELDS: tuple[str, ...] = (
    "drift_calls_count",
    "drift_time_s",
    "full_step_calls_count",
    "full_step_time_s",
    "jacobian_calls_count",
    "jacobian_time_s",
    "assembly_factorization_time_s",
    "window_solve_time_s",
    "independent_replay_time_s",
    "total_time_s",
    "cache_hits",
    "cache_misses",
    "failure_costs",
    "p50_speed_s",
    "p95_speed_s",
)


@dataclass(frozen=True)
class LocalModelApproximation:
    """First-order Taylor expansion local model with explicit validity radius."""

    nominal_state: DimeCompleteState
    nominal_controls: np.ndarray
    state_jacobian: np.ndarray
    control_jacobian: np.ndarray
    validity_radius: float = 0.05

    def __post_init__(self) -> None:
        require(self.validity_radius > 0.0, "validity_radius must be positive")
        require(self.state_jacobian.ndim == 2, "state_jacobian must be 2D")
        require(self.control_jacobian.ndim == 2, "control_jacobian must be 2D")

    def predict(
        self,
        query_state: DimeCompleteState,
        query_controls: np.ndarray,
    ) -> DimeCompleteState:
        """Predict next state using local Taylor expansion within validity radius."""
        q_diff = query_state.q - self.nominal_state.q
        v_diff = query_state.v - self.nominal_state.v
        dx = np.concatenate([q_diff, v_diff])
        displacement_norm = float(np.linalg.norm(dx))

        if displacement_norm > self.validity_radius:
            raise PreconditionError(
                f"Query displacement {displacement_norm:.6f} exceeds validity radius {self.validity_radius:.6f}"
            )

        u_diff = query_controls - self.nominal_controls
        dx_next = self.state_jacobian @ dx + self.control_jacobian @ u_diff
        nq = len(query_state.q)
        q_next = self.nominal_state.q + dx_next[:nq]
        v_next = self.nominal_state.v + dx_next[nq:]

        return DimeCompleteState(
            t=query_state.t + 0.02,
            q=q_next,
            v=v_next,
            units=self.nominal_state.units,
            model_hash=self.nominal_state.model_hash,
        )


def _array_content_hash(arr: Any, name: str) -> str:
    """Deterministic content hash of a finite array covering dtype, shape, bytes.

    Preconditions: ``arr`` converts to a finite float64 array.
    """
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    require(bool(np.all(np.isfinite(a))), f"{name} must be finite")
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _window_problem_qualifier(problem: DimeDynamicsWindowProblem) -> str:
    """Key qualifier covering every input that changes a window solution.

    Covers initial state, horizon, dt, target observations, weights, defect
    mode, constraints, bounds, and solver options (#11547).

    Preconditions: states, targets, weights, and bounds are finite.
    """
    state = problem.initial_state
    state_hash = _array_content_hash(
        np.concatenate([state.q, state.v]), "initial_state"
    )
    if problem.target_positions is None:
        targets_hash = "none"
    else:
        targets_hash = hashlib.sha256(
            "|".join(
                _array_content_hash(t, "target_positions")
                for t in problem.target_positions
            ).encode()
        ).hexdigest()
    weights = np.array(
        [
            problem.observation_weight,
            problem.control_rate_weight,
            problem.transition_weight,
        ],
        dtype=np.float64,
    )
    weights_hash = _array_content_hash(weights, "weights")
    bounds = problem.actuator_bounds
    bounds_hash = "none" if bounds is None else _array_content_hash(bounds, "bounds")
    disc = problem.discrepancy_bounds
    disc_repr = (
        "none"
        if disc is None
        else _array_content_hash(
            [disc.max_slack_norm, disc.slack_weight], "discrepancy"
        )
    )
    return (
        f"win:{state_hash[:16]}:{problem.horizon_steps}:{problem.dt_s!r}"
        f":tg{targets_hash[:16]}:w{weights_hash[:16]}:b{bounds_hash[:16]}"
        f":d{disc_repr[:16]}:{problem.defect_mode.value}"
        f":root{int(bool(problem.enforce_root_constraints))}"
        f":it{problem.max_iterations}"
    )


class DimeSolverCache:
    """Thread-safe multi-partition solver and provider cache."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._drift_store: dict[str, Any] = {}
        self._step_store: dict[str, DimeCompleteState] = {}
        self._jacobian_store: dict[str, np.ndarray] = {}
        self._window_store: dict[str, DimeDynamicsWindowResult] = {}
        self._timing_store: dict[str, list[float]] = {}
        self._camera_index: dict[str, set[str]] = {}
        self._body_index: dict[str, set[str]] = {}
        self._job_index: dict[str, set[str]] = {}

    def _state_fingerprint(self, state: DimeCompleteState) -> str:
        """Build compact deterministic fingerprint for state array values."""
        q_bytes = np.ascontiguousarray(state.q, dtype=np.float64).tobytes()
        v_bytes = np.ascontiguousarray(state.v, dtype=np.float64).tobytes()
        return hashlib.sha256(q_bytes + v_bytes).hexdigest()[:16]

    def _register_indices(self, identity: DimeCacheIdentity, key: str) -> None:
        """Register key into lookup indexes for targeted invalidation."""
        self._camera_index.setdefault(identity.camera_config_hash, set()).add(key)
        self._body_index.setdefault(identity.param_hash, set()).add(key)
        self._job_index.setdefault(identity.job_id, set()).add(key)

    def store_drift(
        self,
        identity: DimeCacheIdentity,
        state: DimeCompleteState,
        drift_result: Any,
    ) -> None:
        """Store drift computation result."""
        fp = self._state_fingerprint(state)
        key = identity.to_composite_key(f"drift:{fp}")
        with self._lock:
            self._drift_store[key] = drift_result
            self._register_indices(identity, key)

    def get_drift(
        self,
        identity: DimeCacheIdentity,
        state: DimeCompleteState,
    ) -> Any | None:
        """Retrieve cached drift computation or None on cache miss."""
        fp = self._state_fingerprint(state)
        key = identity.to_composite_key(f"drift:{fp}")
        with self._lock:
            return self._drift_store.get(key)

    def store_step(
        self,
        identity: DimeCacheIdentity,
        state: DimeCompleteState,
        controls: np.ndarray,
        dt: float,
        next_state: DimeCompleteState,
    ) -> None:
        """Store full step transition result."""
        fp = self._state_fingerprint(state)
        u_bytes = np.ascontiguousarray(controls, dtype=np.float64).tobytes()
        u_hash = hashlib.sha256(u_bytes).hexdigest()[:8]
        key = identity.to_composite_key(f"step:{fp}:{u_hash}:{dt:.6f}")
        with self._lock:
            self._step_store[key] = next_state
            self._register_indices(identity, key)

    def get_step(
        self,
        identity: DimeCacheIdentity,
        state: DimeCompleteState,
        controls: np.ndarray,
        dt: float,
    ) -> DimeCompleteState | None:
        """Retrieve cached step transition result or None on cache miss."""
        fp = self._state_fingerprint(state)
        u_bytes = np.ascontiguousarray(controls, dtype=np.float64).tobytes()
        u_hash = hashlib.sha256(u_bytes).hexdigest()[:8]
        key = identity.to_composite_key(f"step:{fp}:{u_hash}:{dt:.6f}")
        with self._lock:
            return self._step_store.get(key)

    def store_jacobian(
        self,
        identity: DimeCacheIdentity,
        state: DimeCompleteState,
        jacobian: np.ndarray,
    ) -> None:
        """Store computed Jacobian matrix."""
        fp = self._state_fingerprint(state)
        key = identity.to_composite_key(f"jac:{fp}")
        with self._lock:
            self._jacobian_store[key] = np.copy(jacobian)
            self._register_indices(identity, key)

    def store_jacobian_with_impact_check(
        self,
        identity: DimeCacheIdentity,
        state: DimeCompleteState,
        jacobian: np.ndarray,
        is_impact_phase: bool = False,
    ) -> None:
        """Store Jacobian with fail-closed impact discontinuity verification."""
        if is_impact_phase:
            raise PreconditionError(
                "Derivative at impact is discontinuous; cannot store or retrieve smooth Jacobian."
            )
        self.store_jacobian(identity, state, jacobian)

    def get_jacobian(
        self,
        identity: DimeCacheIdentity,
        state: DimeCompleteState,
    ) -> np.ndarray | None:
        """Retrieve cached Jacobian matrix or None on miss."""
        fp = self._state_fingerprint(state)
        key = identity.to_composite_key(f"jac:{fp}")
        with self._lock:
            cached = self._jacobian_store.get(key)
            return np.copy(cached) if cached is not None else None

    def store_window_solve(
        self,
        identity: DimeCacheIdentity,
        problem: DimeDynamicsWindowProblem,
        result: DimeDynamicsWindowResult,
        elapsed_s: float = 0.0,
    ) -> None:
        """Store converged window solve result and track latency."""
        key = identity.to_composite_key(_window_problem_qualifier(problem))
        with self._lock:
            self._window_store[key] = result
            self._timing_store.setdefault(key, []).append(elapsed_s)
            self._register_indices(identity, key)

    def get_window_solve(
        self,
        identity: DimeCacheIdentity,
        problem: DimeDynamicsWindowProblem,
    ) -> DimeDynamicsWindowResult | None:
        """Retrieve cached window solve result or None on miss."""
        key = identity.to_composite_key(_window_problem_qualifier(problem))
        with self._lock:
            return self._window_store.get(key)

    def invalidate_on_camera_change(self, camera_config_hash: str) -> None:
        """Invalidate all cache entries associated with a modified camera setup."""
        with self._lock:
            keys_to_purge = self._camera_index.pop(camera_config_hash, set())
            for k in keys_to_purge:
                self._drift_store.pop(k, None)
                self._step_store.pop(k, None)
                self._jacobian_store.pop(k, None)
                self._window_store.pop(k, None)

    def invalidate_on_body_change(self, param_hash: str) -> None:
        """Invalidate all cache entries associated with modified body parameters."""
        with self._lock:
            keys_to_purge = self._body_index.pop(param_hash, set())
            for k in keys_to_purge:
                self._drift_store.pop(k, None)
                self._step_store.pop(k, None)
                self._jacobian_store.pop(k, None)
                self._window_store.pop(k, None)

    def invalidate_job(self, job_id: str) -> None:
        """Invalidate all cache entries associated with a finished or isolated job."""
        with self._lock:
            keys_to_purge = self._job_index.pop(job_id, set())
            for k in keys_to_purge:
                self._drift_store.pop(k, None)
                self._step_store.pop(k, None)
                self._jacobian_store.pop(k, None)
                self._window_store.pop(k, None)


def accelerated_solve_dynamics_window(
    problem: DimeDynamicsWindowProblem,
    cache: DimeSolverCache,
    skip_independent_replay: bool = False,
    job_id: str = "default_solver_job",
) -> tuple[DimeDynamicsWindowResult, DimeCostBreakdown]:
    """Execute accelerated window solve with truthful replay and cost profiling."""
    require(problem is not None, "problem must not be None")
    require(cache is not None, "cache must not be None")

    if skip_independent_replay:
        raise PreconditionError(
            "Independent continuous replay is mandatory; cannot skip replay to fake speedup."
        )

    identity = DimeCacheIdentity(
        model_hash=problem.provider.model_hash,
        param_hash="nominal_params",
        contact_policy=ContactPolicy.NATIVE_ELIMINATED,
        solver_config_hash=f"steps_{problem.horizon_steps}_dt_{problem.dt_s}",
        camera_config_hash="default_camera",
        job_id=job_id,
    )

    t_start = time.perf_counter()
    cached_res = cache.get_window_solve(identity, problem)

    if cached_res is not None:
        t_solve = time.perf_counter() - t_start
        hits = 1
        misses = 0
        res = cached_res
    else:
        hits = 0
        misses = 1
        res = solve_dime_dynamics_window(problem)
        t_solve = time.perf_counter() - t_start
        if res.success:
            cache.store_window_solve(identity, problem, res, elapsed_s=t_solve)

    # Execute mandatory independent forward replay
    t_replay_start = time.perf_counter()
    curr_state = problem.initial_state
    for k in range(problem.horizon_steps):
        req = DimeFullStepRequest(
            state=curr_state,
            controls=res.controls[k],
            dt=problem.dt_s,
            model_hash=problem.provider.model_hash,
        )
        curr_state = problem.provider.step(req).next_state
    t_replay = time.perf_counter() - t_replay_start

    total_time = t_solve + t_replay
    # Seconds actually spent on a solve that did not converge (measured).
    failure_cost = 0.0 if res.success else t_solve

    solver_terms = {
        k: float(v) for k, v in res.cost_breakdown.items() if np.isfinite(v)
    } or None

    breakdown = DimeCostBreakdown(
        window_solve_time_s=t_solve,
        independent_replay_time_s=t_replay,
        total_time_s=total_time,
        cache_hits=hits,
        cache_misses=misses,
        replay_executed=True,
        failure_costs=failure_cost,
        solver_cost_terms=solver_terms,
    )

    return res, breakdown
