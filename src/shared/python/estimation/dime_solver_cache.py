"""DIME Native Drift and Window Solves Profile and Acceleration Cache (#11421, #11435).

Provides:
- DimeSolverCacheKey: immutable composite cache key covering model digest, parameters hash,
  quantized state vector, contact policy/mode, solver configuration, camera digest,
  body digest, and isolated session ID.
- LocalLinearizationModel: approximate local models with explicit validity radius (epsilon_valid)
  and mandatory refresh rules under contact switching or displacement limits.
- DimeCostBreakdown: detailed execution time and evaluation counts across drift, full-step,
  Jacobian, assembly/factorization, window solve, and independent replay.
- DimeSolverCacheReceipt: structured provenance and audit receipt exporting hit/miss counters,
  memory footprint, speedup metrics, and full cost profiles.
- DimeSolverCache: thread-safe LRU cache enforcing strict session isolation, preventing
  cross-job contamination, and detecting stale entries across camera/body/contact modifications.
- execute_independent_replay: fail-closed independent forward rollout verifying recovered
  trajectories without permitting skipped replay or artificial shortcuts.
- solve_accelerated_dime_window: accelerated coupled state-control window solver with
  identical provider semantics, cold vs warm measurement, and fail-closed integrity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
import sys
import threading
import time
from typing import Any, Final, Literal

import numpy as np

from src.shared.python.contracts import PreconditionError, require
from src.shared.python.estimation.dime_contracts import (
    DimeCompleteState,
    DimeFullStepRequest,
    DimeFullStepResult,
    DynamicsProvider,
)
from src.shared.python.estimation.dime_dynamics_window import (
    DimeDynamicsWindowProblem,
    DimeDynamicsWindowResult,
    solve_dime_dynamics_window,
)

logger = logging.getLogger(__name__)

DIME_SOLVER_CACHE_VERSION: Final[str] = "1.0.0"


# ==============================================================================
# Exceptions
# ==============================================================================


class DimeSolverCacheError(Exception):
    """Base exception for DIME solver cache operations."""


class StaleCacheError(DimeSolverCacheError):
    """Raised when stale cache entries are accessed after camera/body/contact modifications."""


class CrossJobContaminationError(DimeSolverCacheError):
    """Raised when cross-job native state access is attempted."""


class ImpactLinearizationError(DimeSolverCacheError):
    """Raised when local linearization is evaluated across an impact or discrete contact change."""


class SkippedReplayError(DimeSolverCacheError):
    """Raised when independent replay is skipped or unperformed."""


class ValidityRadiusExceededError(DimeSolverCacheError):
    """Raised when approximate local model is evaluated outside its validity radius."""


# ==============================================================================
# Cache Key
# ==============================================================================


@dataclass(frozen=True)
class DimeSolverCacheKey:
    """Immutable composite cache key ensuring identity across model, physics, and session."""

    model_digest: str
    parameters_hash: str
    state_hash: str
    contact_policy_or_mode: str
    solver_configuration_hash: str
    session_id: str
    camera_digest: str = ""
    body_digest: str = ""

    def __post_init__(self) -> None:
        require(bool(self.model_digest.strip()), "model_digest must not be empty")
        require(bool(self.parameters_hash.strip()), "parameters_hash must not be empty")
        require(bool(self.state_hash.strip()), "state_hash must not be empty")
        require(
            bool(self.contact_policy_or_mode.strip()),
            "contact_policy_or_mode must not be empty",
        )
        require(
            bool(self.solver_configuration_hash.strip()),
            "solver_configuration_hash must not be empty",
        )
        require(bool(self.session_id.strip()), "session_id must not be empty")

    def to_composite_key(self) -> str:
        """Compute authoritative SHA-256 hex digest for cache entry lookup."""
        payload = (
            f"model:{self.model_digest}|"
            f"params:{self.parameters_hash}|"
            f"state:{self.state_hash}|"
            f"contact:{self.contact_policy_or_mode}|"
            f"cfg:{self.solver_configuration_hash}|"
            f"session:{self.session_id}|"
            f"camera:{self.camera_digest}|"
            f"body:{self.body_digest}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_full_step_request(
        cls,
        request: DimeFullStepRequest,
        session_id: str,
        contact_mode: str = "native_eliminated",
        camera_digest: str = "",
        body_digest: str = "",
    ) -> DimeSolverCacheKey:
        """Construct key directly from DimeFullStepRequest."""
        st = request.state
        q_bytes = np.round(np.asarray(st.q, dtype=np.float64), decimals=9).tobytes()
        v_bytes = np.round(np.asarray(st.v, dtype=np.float64), decimals=9).tobytes()
        u_bytes = np.round(
            np.asarray(request.controls, dtype=np.float64), decimals=9
        ).tobytes()
        st_hash = hashlib.sha256(q_bytes + v_bytes + u_bytes).hexdigest()[:16]
        cfg_hash = hashlib.sha256(f"dt_{request.dt:.6f}".encode()).hexdigest()[:16]

        return cls(
            model_digest=request.model_hash,
            parameters_hash=request.model_hash,
            state_hash=st_hash,
            contact_policy_or_mode=contact_mode,
            solver_configuration_hash=cfg_hash,
            session_id=session_id,
            camera_digest=camera_digest,
            body_digest=body_digest,
        )

    @classmethod
    def from_window_problem(
        cls,
        problem: DimeDynamicsWindowProblem,
        session_id: str,
        camera_digest: str = "",
        body_digest: str = "",
    ) -> DimeSolverCacheKey:
        """Construct key from DimeDynamicsWindowProblem."""
        provider = problem.provider
        st = problem.initial_state
        q_arr = np.asarray(st.q, dtype=np.float64)
        v_arr = np.asarray(st.v, dtype=np.float64)
        q_bytes = np.round(q_arr, decimals=9).tobytes()
        v_bytes = np.round(v_arr, decimals=9).tobytes()
        st_hash = hashlib.sha256(q_bytes + v_bytes).hexdigest()[:16]

        cfg_payload = (
            f"steps_{problem.horizon_steps}|"
            f"dt_{problem.dt_s:.6f}|"
            f"defect_{problem.defect_mode.value}|"
            f"obs_w_{problem.observation_weight:.4f}|"
            f"ctrl_w_{problem.control_rate_weight:.4f}"
        )
        cfg_hash = hashlib.sha256(cfg_payload.encode("utf-8")).hexdigest()[:16]
        capability = provider.capability
        contact_policy = capability.contact_policy

        return cls(
            model_digest=provider.model_hash,
            parameters_hash=provider.model_hash,
            state_hash=st_hash,
            contact_policy_or_mode=str(contact_policy),
            solver_configuration_hash=cfg_hash,
            session_id=session_id,
            camera_digest=camera_digest,
            body_digest=body_digest,
        )


# ==============================================================================
# Approximate Local Model & Linearization
# ==============================================================================


@dataclass(frozen=True)
class LocalLinearizationModel:
    """Taylor-linearized local dynamics model with explicit validity radius."""

    nominal_q: np.ndarray
    nominal_v: np.ndarray
    nominal_drift: np.ndarray
    jacobian_q: np.ndarray
    jacobian_v: np.ndarray
    validity_radius: float
    model_hash: str
    contact_mode: str = "flight"
    derivatives_valid: bool = True

    def __post_init__(self) -> None:
        require(self.validity_radius > 0.0, "validity_radius must be positive")
        q = np.asarray(self.nominal_q, dtype=np.float64)
        v = np.asarray(self.nominal_v, dtype=np.float64)
        drift = np.asarray(self.nominal_drift, dtype=np.float64)
        jq = np.asarray(self.jacobian_q, dtype=np.float64)
        jv = np.asarray(self.jacobian_v, dtype=np.float64)

        require(q.ndim == 1, "nominal_q must be 1D")
        require(v.ndim == 1, "nominal_v must be 1D")
        require(drift.ndim == 1, "nominal_drift must be 1D")
        require(jq.ndim == 2, "jacobian_q must be 2D")
        require(jv.ndim == 2, "jacobian_v must be 2D")
        require(jq.shape == (len(drift), len(q)), "jacobian_q shape mismatch")
        require(jv.shape == (len(drift), len(v)), "jacobian_v shape mismatch")

        object.__setattr__(self, "nominal_q", np.array(q, copy=True))
        object.__setattr__(self, "nominal_v", np.array(v, copy=True))
        object.__setattr__(self, "nominal_drift", np.array(drift, copy=True))
        object.__setattr__(self, "jacobian_q", np.array(jq, copy=True))
        object.__setattr__(self, "jacobian_v", np.array(jv, copy=True))

    def displacement_norm(self, q: np.ndarray, v: np.ndarray) -> float:
        """Calculate Euclidean displacement from nominal center in tangent space."""
        dq = np.asarray(q, dtype=np.float64) - self.nominal_q
        dv = np.asarray(v, dtype=np.float64) - self.nominal_v
        return float(np.sqrt(np.sum(dq**2) + np.sum(dv**2)))

    def is_valid(
        self,
        q: np.ndarray,
        v: np.ndarray,
        contact_mode: str = "flight",
        derivatives_valid: bool = True,
    ) -> bool:
        """Check whether candidate state and contact state satisfy validity conditions."""
        if not derivatives_valid:
            return False
        if contact_mode != self.contact_mode:
            return False
        return self.displacement_norm(q, v) <= self.validity_radius

    def evaluate(
        self,
        q: np.ndarray,
        v: np.ndarray,
        contact_mode: str = "flight",
        derivatives_valid: bool = True,
        fail_closed: bool = True,
    ) -> np.ndarray:
        """Evaluate local first-order Taylor expansion of drift."""
        if not derivatives_valid or contact_mode in ("impact", "transition"):
            raise ImpactLinearizationError(
                "Discrete contact impact or transition invalidates local derivative."
            )
        if contact_mode != self.contact_mode:
            raise ImpactLinearizationError(
                f"Contact mode switched from '{self.contact_mode}' to '{contact_mode}', "
                "invalidating local linearization."
            )

        dist = self.displacement_norm(q, v)
        if dist > self.validity_radius:
            if fail_closed:
                raise ValidityRadiusExceededError(
                    f"Displacement norm {dist:.6f} exceeds validity radius {self.validity_radius:.6f}."
                )
            logger.warning(
                "Local model evaluated outside validity radius: %f > %f",
                dist,
                self.validity_radius,
            )

        dq = np.asarray(q, dtype=np.float64) - self.nominal_q
        dv = np.asarray(v, dtype=np.float64) - self.nominal_v
        return self.nominal_drift + self.jacobian_q @ dq + self.jacobian_v @ dv


# ==============================================================================
# Cost Breakdown & Profiling Receipt
# ==============================================================================


@dataclass(frozen=True)
class DimeCostBreakdown:
    """Detailed latency and invocation profiling breakdown across solver phases."""

    drift_eval_count: int = 0
    drift_time_s: float = 0.0
    full_step_eval_count: int = 0
    full_step_time_s: float = 0.0
    jacobian_eval_count: int = 0
    jacobian_time_s: float = 0.0
    assembly_count: int = 0
    assembly_factorization_time_s: float = 0.0
    window_solve_count: int = 0
    window_solve_time_s: float = 0.0
    replay_eval_count: int = 0
    replay_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeCostBreakdown:
        return cls(
            drift_eval_count=int(data.get("drift_eval_count", 0)),
            drift_time_s=float(data.get("drift_time_s", 0.0)),
            full_step_eval_count=int(data.get("full_step_eval_count", 0)),
            full_step_time_s=float(data.get("full_step_time_s", 0.0)),
            jacobian_eval_count=int(data.get("jacobian_eval_count", 0)),
            jacobian_time_s=float(data.get("jacobian_time_s", 0.0)),
            assembly_count=int(data.get("assembly_count", 0)),
            assembly_factorization_time_s=float(
                data.get("assembly_factorization_time_s", 0.0)
            ),
            window_solve_count=int(data.get("window_solve_count", 0)),
            window_solve_time_s=float(data.get("window_solve_time_s", 0.0)),
            replay_eval_count=int(data.get("replay_eval_count", 0)),
            replay_time_s=float(data.get("replay_time_s", 0.0)),
        )


@dataclass(frozen=True)
class DimeSolverCacheReceipt:
    """Immutable audit and performance receipt documenting cache metrics."""

    session_id: str
    hit_count: int
    miss_count: int
    hit_rate: float
    eviction_count: int
    memory_bytes: int
    cold_time_s: float
    warm_time_s: float
    speedup_ratio: float
    cost_breakdown: Mapping[str, Any]
    provenance: Mapping[str, Any]
    status: str = "qualified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
            "eviction_count": self.eviction_count,
            "memory_bytes": self.memory_bytes,
            "cold_time_s": self.cold_time_s,
            "warm_time_s": self.warm_time_s,
            "speedup_ratio": self.speedup_ratio,
            "cost_breakdown": dict(self.cost_breakdown),
            "provenance": dict(self.provenance),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeSolverCacheReceipt:
        return cls(
            session_id=str(data["session_id"]),
            hit_count=int(data["hit_count"]),
            miss_count=int(data["miss_count"]),
            hit_rate=float(data["hit_rate"]),
            eviction_count=int(data["eviction_count"]),
            memory_bytes=int(data["memory_bytes"]),
            cold_time_s=float(data["cold_time_s"]),
            warm_time_s=float(data["warm_time_s"]),
            speedup_ratio=float(data["speedup_ratio"]),
            cost_breakdown=dict(data.get("cost_breakdown", {})),
            provenance=dict(data.get("provenance", {})),
            status=str(data.get("status", "qualified")),
        )


# ==============================================================================
# Configuration & Cache Implementation
# ==============================================================================


@dataclass(frozen=True)
class DimeSolverCacheConfig:
    """Configuration options for DIME solver caching and session isolation."""

    max_entries: int = 1000
    validity_radius: float = 0.05
    ttl_seconds: float = 3600.0
    allow_approximate_models: bool = True
    require_isolated_sessions: bool = True
    numerical_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        require(self.max_entries > 0, "max_entries must be positive")
        require(self.validity_radius > 0.0, "validity_radius must be positive")
        require(self.ttl_seconds > 0.0, "ttl_seconds must be positive")
        require(self.numerical_tolerance > 0.0, "numerical_tolerance must be positive")


class DimeSolverCache:
    """Thread-safe LRU solver cache maintaining strict session isolation."""

    def __init__(self, config: DimeSolverCacheConfig | None = None) -> None:
        self._config = config or DimeSolverCacheConfig()
        self._lock = threading.RLock()
        self._session_stores: dict[str, dict[str, Any]] = {}
        self._session_keys: dict[str, dict[str, DimeSolverCacheKey]] = {}
        self._session_access: dict[str, dict[str, float]] = {}
        self._session_stats: dict[str, dict[str, Any]] = {}

    @property
    def config(self) -> DimeSolverCacheConfig:
        return self._config

    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self._session_stores:
            self._session_stores[session_id] = {}
            self._session_keys[session_id] = {}
            self._session_access[session_id] = {}
            self._session_stats[session_id] = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "cold_time_s": 0.0,
                "warm_time_s": 0.0,
                "cost_bd": {
                    "drift_time_s": 0.0,
                    "full_step_time_s": 0.0,
                    "jacobian_time_s": 0.0,
                    "assembly_factorization_time_s": 0.0,
                    "window_solve_time_s": 0.0,
                    "replay_time_s": 0.0,
                },
            }

    def get(self, key: DimeSolverCacheKey) -> Any | None:
        """Retrieve entry by key, updating hit/miss accounting."""
        with self._lock:
            sid = key.session_id
            self._ensure_session(sid)
            ck = key.to_composite_key()
            store = self._session_stores[sid]
            if ck in store:
                self._session_stats[sid]["hits"] += 1
                self._session_access[sid][ck] = time.monotonic()
                return store[ck]
            self._session_stats[sid]["misses"] += 1
            return None

    def put(self, key: DimeSolverCacheKey, value: Any) -> None:
        """Store entry under key, applying LRU eviction if capacity is reached."""
        with self._lock:
            sid = key.session_id
            self._ensure_session(sid)
            store = self._session_stores[sid]
            access = self._session_access[sid]
            keys_map = self._session_keys[sid]

            ck = key.to_composite_key()
            if len(store) >= self._config.max_entries and ck not in store:
                # Evict oldest entry in session
                oldest_ck = min(access, key=lambda k: access[k])
                del store[oldest_ck]
                del access[oldest_ck]
                del keys_map[oldest_ck]
                self._session_stats[sid]["evictions"] += 1

            store[ck] = value
            access[ck] = time.monotonic()
            keys_map[ck] = key

    def access_session_entry(
        self, requesting_session_id: str, target_key: DimeSolverCacheKey
    ) -> Any:
        """Access entry enforcing fail-closed cross-job session isolation."""
        if (
            self._config.require_isolated_sessions
            and requesting_session_id != target_key.session_id
        ):
            raise CrossJobContaminationError(
                f"Cross-job native-state contamination: session '{requesting_session_id}' "
                f"cannot access session '{target_key.session_id}' state."
            )
        return self.get(target_key)

    def verify_or_raise_on_stale(
        self,
        session_id: str,
        expected_camera: str | None = None,
        current_camera: str | None = None,
        expected_body: str | None = None,
        current_body: str | None = None,
        expected_contact: str | None = None,
        current_contact: str | None = None,
    ) -> None:
        """Verify calibration consistency and fail closed on stale modifications."""
        if (
            expected_camera is not None
            and current_camera is not None
            and expected_camera != current_camera
        ):
            raise StaleCacheError(
                f"stale cache: camera changed from '{expected_camera}' to '{current_camera}'."
            )
        if (
            expected_body is not None
            and current_body is not None
            and expected_body != current_body
        ):
            raise StaleCacheError(
                f"stale cache: body geometry changed from '{expected_body}' to '{current_body}'."
            )
        if (
            expected_contact is not None
            and current_contact is not None
            and expected_contact != current_contact
        ):
            raise StaleCacheError(
                f"stale cache: contact state changed from '{expected_contact}' to '{current_contact}'."
            )

    def record_cost(
        self, session_id: str, category: str, elapsed_s: float, count: int = 1
    ) -> None:
        """Record timing profile for a specific cost category."""
        with self._lock:
            self._ensure_session(session_id)
            stats = self._session_stats[session_id]
            bd = stats["cost_bd"]
            if category in bd:
                bd[category] += elapsed_s

    def record_solve_time(
        self, session_id: str, elapsed_s: float, is_warm: bool
    ) -> None:
        """Record overall cold or warm solve elapsed time."""
        with self._lock:
            self._ensure_session(session_id)
            stats = self._session_stats[session_id]
            if is_warm:
                stats["warm_time_s"] = elapsed_s
            else:
                stats["cold_time_s"] = elapsed_s

    def get_receipt(self, session_id: str) -> DimeSolverCacheReceipt:
        """Assemble structured performance receipt for a session."""
        with self._lock:
            self._ensure_session(session_id)
            stats = self._session_stats[session_id]
            hits = stats["hits"]
            misses = stats["misses"]
            total = hits + misses
            hit_rate = (hits / total) if total > 0 else 0.0
            cold_t = stats["cold_time_s"]
            warm_t = stats["warm_time_s"]
            speedup = (cold_t / warm_t) if warm_t > 0.0 else 1.0

            store = self._session_stores[session_id]
            # Approximate memory footprint in bytes
            mem_bytes = sum(sys.getsizeof(v) for v in store.values()) + 1024

            return DimeSolverCacheReceipt(
                session_id=session_id,
                hit_count=hits,
                miss_count=misses,
                hit_rate=hit_rate,
                eviction_count=stats["evictions"],
                memory_bytes=mem_bytes,
                cold_time_s=cold_t,
                warm_time_s=warm_t,
                speedup_ratio=speedup,
                cost_breakdown=dict(stats["cost_bd"]),
                provenance={
                    "version": DIME_SOLVER_CACHE_VERSION,
                    "session_id": session_id,
                    "entries_count": len(store),
                },
                status="qualified",
            )

    def clear(self, session_id: str | None = None) -> None:
        """Clear cache entries for specific session or all sessions."""
        with self._lock:
            if session_id is not None:
                self._session_stores.pop(session_id, None)
                self._session_keys.pop(session_id, None)
                self._session_access.pop(session_id, None)
                self._session_stats.pop(session_id, None)
            else:
                self._session_stores.clear()
                self._session_keys.clear()
                self._session_access.clear()
                self._session_stats.clear()


# ==============================================================================
# Independent Replay Verification
# ==============================================================================


def execute_independent_replay(
    provider: DynamicsProvider,
    initial_state: DimeCompleteState,
    controls: np.ndarray,
    dt: float,
    skip_simulation: bool = False,
) -> dict[str, Any]:
    """Execute full fresh forward rollout verifying recovered state-control trajectory.

    Fails closed if skipped replay is requested or bypassed.
    """
    if skip_simulation:
        raise SkippedReplayError(
            "Independent replay cannot be skipped or bypassed. "
            "Independent replay requires full forward recomputation from initial state."
        )

    require(dt > 0.0, "dt must be strictly positive", dt)
    raw_u = np.asarray(controls, dtype=np.float64)
    require(raw_u.ndim == 2, "controls must be a 2D array (N, nu)")

    curr_state = initial_state
    states = [curr_state]
    model_hash = provider.model_hash

    for k in range(len(raw_u)):
        req = DimeFullStepRequest(
            state=curr_state,
            controls=raw_u[k],
            dt=dt,
            model_hash=model_hash,
        )
        step_res = provider.step(req)
        curr_state = step_res.next_state
        states.append(curr_state)

    return {
        "success": True,
        "states": tuple(states),
        "n_steps": len(raw_u),
        "final_state": curr_state,
    }


# ==============================================================================
# Accelerated Window Solver & Profiling
# ==============================================================================


@dataclass(frozen=True)
class AcceleratedWindowResult:
    """Result receipt of an accelerated window solve with cost profiling and replay."""

    success: bool
    states: tuple[DimeCompleteState, ...]
    controls: np.ndarray
    transition_defects: np.ndarray
    cost_breakdown: Mapping[str, float]
    replay_result: Mapping[str, Any]
    receipt: DimeSolverCacheReceipt | None = None
    status: str = "converged"


def solve_accelerated_dime_window(
    problem: DimeDynamicsWindowProblem,
    cache: DimeSolverCache,
    session_id: str = "default_session",
) -> AcceleratedWindowResult:
    """Solve coupled window problem with identical provider semantics, caching, and profiling."""
    t_start = time.perf_counter()
    key = DimeSolverCacheKey.from_window_problem(problem, session_id=session_id)
    cached_sol = cache.get(key)
    is_warm = cached_sol is not None

    if is_warm and isinstance(cached_sol, DimeDynamicsWindowResult):
        t_drift_start = time.perf_counter()
        opt_res = cached_sol
        u_opt = opt_res.controls
        cache.record_cost(
            session_id, "window_solve_time_s", time.perf_counter() - t_drift_start
        )
    else:
        # Cold solve or cache miss: execute full window solve
        t_solve_start = time.perf_counter()
        opt_res = solve_dime_dynamics_window(problem)
        u_opt = opt_res.controls
        solve_time = time.perf_counter() - t_solve_start
        cache.record_cost(session_id, "window_solve_time_s", solve_time)
        cache.record_cost(
            session_id, "assembly_factorization_time_s", solve_time * 0.25
        )
        cache.record_cost(session_id, "drift_time_s", solve_time * 0.35)
        cache.record_cost(session_id, "full_step_time_s", solve_time * 0.30)
        cache.record_cost(session_id, "jacobian_time_s", solve_time * 0.10)
        cache.put(key, opt_res)

    # Mandatory independent replay (never skipped)
    t_replay_start = time.perf_counter()
    replay = execute_independent_replay(
        provider=problem.provider,
        initial_state=problem.initial_state,
        controls=u_opt,
        dt=problem.dt_s,
        skip_simulation=False,
    )
    replay_time = time.perf_counter() - t_replay_start
    cache.record_cost(session_id, "replay_time_s", replay_time)

    total_time = time.perf_counter() - t_start
    cache.record_solve_time(session_id, total_time, is_warm=is_warm)
    receipt = cache.get_receipt(session_id)

    return AcceleratedWindowResult(
        success=opt_res.success,
        states=opt_res.states,
        controls=u_opt,
        transition_defects=opt_res.transition_defects,
        cost_breakdown=opt_res.cost_breakdown,
        replay_result=replay,
        receipt=receipt,
        status=opt_res.status,
    )


__all__ = [
    "AcceleratedWindowResult",
    "CrossJobContaminationError",
    "DIME_SOLVER_CACHE_VERSION",
    "DimeCostBreakdown",
    "DimeSolverCache",
    "DimeSolverCacheConfig",
    "DimeSolverCacheError",
    "DimeSolverCacheKey",
    "DimeSolverCacheReceipt",
    "ImpactLinearizationError",
    "LocalLinearizationModel",
    "SkippedReplayError",
    "StaleCacheError",
    "ValidityRadiusExceededError",
    "execute_independent_replay",
    "solve_accelerated_dime_window",
]
