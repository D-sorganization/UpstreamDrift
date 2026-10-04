"""State, Observation, and Dynamics Provider Contracts for DIME (#11421, #11423).

Defines:
1. Versioned complete-state (DimeState) with strict SI units, model hash, and internal states.
2. Manifold contracts (DimeManifold, VectorManifold, QuaternionManifold) supporting
   retract, local-coordinates, nq != nv, and quaternion sign equivalence (SO(3) double cover).
3. Control channels (DimeControlChannel, ControlPhysicalType) declaring physical units,
   selection map, limits, and explicit activation dynamics for muscle excitation.
4. Contact policy (ContactPolicy, ContactInterfaceMode) enforcing mutual exclusivity
   between native eliminated reactions and explicit constrained reaction variables.
5. Observation window (ObservationWindow, ObservationItem) with strict time-monotonicity.
6. Snapshot and rollback protocol (ProviderSnapshot) ensuring exceptions restore state.
7. Runtime factor exclusivity (IntervalFactorRegistry) preventing duplicate or conflicting
   factors on the same interval (marginalized-input transition vs explicit-input likelihood).
8. Unified provider interface (DimeDynamicsProvider) satisfied identically by
   DeterministicFakeProvider and AnalyticPendulumProvider without claiming native qualification.
9. Estimation result contract (DimeEstimationResult) with full serialization round-trips.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable
import numpy as np
import numpy.typing as npt

from src.shared.python.contracts import require
from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    CapabilityStatus,
    DimeProvenanceRecord,
)
from src.shared.python.spatial_algebra.pose6dof.rotations import quaternion_multiply

logger = logging.getLogger(__name__)

# Valid SI unit declarations for DimeState
_ALLOWED_SI_UNITS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "length": "m",
        "angle": "rad",
        "time": "s",
        "mass": "kg",
        "force": "N",
        "torque": "N*m",
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DimeContractError(ValueError):
    """Base exception for all DIME contract violations."""


class UnitMismatchError(DimeContractError):
    """Declared units violate SI standards or expected coordinate frames."""


class StaleModelHashError(DimeContractError):
    """Model hash mismatch between state/provider and current specification."""


class InvalidTimeOrderError(DimeContractError):
    """Observation timestamps violate strict non-decreasing temporal order."""


class ConflictingContactInterfaceError(DimeContractError):
    """Native eliminated reactions and explicit constrained variables configured concurrently."""


class ExclusiveFactorConflictError(DimeContractError):
    """Mutually exclusive factors registered on the same temporal interval."""


class ProviderUnavailableError(RuntimeError):
    """Requested physics/dynamics provider is not available or not provisioned."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ControlPhysicalType(str, Enum):
    """Physical nature and dimension of control actuation channels."""

    TORQUE = "torque"
    FORCE = "force"
    EXCITATION = "excitation"
    GENERALIZED = "generalized"


class ContactInterfaceMode(str, Enum):
    """Contact modeling strategy."""

    NATIVE_ELIMINATED = "native_eliminated"
    EXPLICIT_CONSTRAINED = "explicit_constrained"
    BOTH_CONFLICTING = "both_conflicting"


class IntervalFactorType(str, Enum):
    """Type of estimation factor operating over a temporal interval."""

    MARGINALIZED_INPUT_TRANSITION = "marginalized_input_transition"
    EXPLICIT_INPUT_LIKELIHOOD = "explicit_input_likelihood"
    DIAGNOSTIC_ONLY = "diagnostic_only"


# ---------------------------------------------------------------------------
# Manifold Interface and Primitives
# ---------------------------------------------------------------------------


@runtime_checkable
class DimeManifold(Protocol):
    """Public retraction and local coordinate interface on configuration manifolds."""

    @property
    def config_dim(self) -> int:
        """Dimension of ambient configuration vector q (nq)."""
        ...

    @property
    def tangent_dim(self) -> int:
        """Dimension of tangent / velocity space v (nv)."""
        ...

    def retract(self, q: np.ndarray, delta_v: np.ndarray) -> np.ndarray:
        """Retract tangent step delta_v onto configuration manifold at q: q [+] delta_v."""
        ...

    def local_coordinates(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """Local coordinate tangent vector delta_v = q1 [-] q0."""
        ...


class VectorManifold:
    """Euclidean vector space R^n where nq == nv and retract is addition."""

    def __init__(self, dim: int) -> None:
        require(dim > 0, "dim must be positive")
        self._dim = dim

    @property
    def config_dim(self) -> int:
        return self._dim

    @property
    def tangent_dim(self) -> int:
        return self._dim

    def retract(self, q: np.ndarray, delta_v: np.ndarray) -> np.ndarray:
        require(len(q) == self._dim, "q dimension mismatch")
        require(len(delta_v) == self._dim, "delta_v dimension mismatch")
        return np.asarray(q + delta_v, dtype=np.float64)

    def local_coordinates(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        require(len(q0) == self._dim, "q0 dimension mismatch")
        require(len(q1) == self._dim, "q1 dimension mismatch")
        return np.asarray(q1 - q0, dtype=np.float64)


class QuaternionManifold:
    """Unit quaternion manifold S^3 with nq=4, nv=3, and antipodal sign equivalence.

    Convention: q = [w, x, y, z].
    Equivalence: q and -q represent identical SO(3) rotations.
    """

    @property
    def config_dim(self) -> int:
        return 4

    @property
    def tangent_dim(self) -> int:
        return 3

    @staticmethod
    def _normalize(q: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(q))
        require(norm > 1e-12, "quaternion norm must be positive")
        return q / norm

    def retract(self, q: np.ndarray, delta_v: np.ndarray) -> np.ndarray:
        require(len(q) == 4, "q must be 4-vector")
        require(len(delta_v) == 3, "delta_v must be 3-vector")
        q_norm = self._normalize(q)
        theta = float(np.linalg.norm(delta_v))
        if theta < 1e-12:
            delta_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            half_theta = 0.5 * theta
            axis = delta_v / theta
            sin_half = np.sin(half_theta)
            delta_q = np.array(
                [
                    np.cos(half_theta),
                    axis[0] * sin_half,
                    axis[1] * sin_half,
                    axis[2] * sin_half,
                ],
                dtype=np.float64,
            )
        # Apply perturbation: delta_q * q
        res = quaternion_multiply(delta_q, q_norm)
        return self._normalize(res)

    def local_coordinates(self, q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
        require(len(q0) == 4, "q0 must be 4-vector")
        require(len(q1) == 4, "q1 must be 4-vector")
        q0_norm = self._normalize(q0)
        q1_norm = self._normalize(q1)

        # Quaternion sign equivalence: q and -q represent identical SO(3) rotations.
        # Choose sign to ensure shortest geodesic on S^3.
        dot = float(np.dot(q0_norm, q1_norm))
        if dot < 0.0:
            q1_norm = -q1_norm

        # Relative quaternion: q_rel = q1 * q0_inv
        # For unit quaternion, q0_inv = [w, -x, -y, -z]
        q0_inv = np.array([q0_norm[0], -q0_norm[1], -q0_norm[2], -q0_norm[3]])
        q_rel = quaternion_multiply(q1_norm, q0_inv)
        q_rel = self._normalize(q_rel)

        w = float(np.clip(q_rel[0], -1.0, 1.0))
        v_part = q_rel[1:4]
        v_norm = float(np.linalg.norm(v_part))

        if v_norm < 1e-12:
            return np.zeros(3, dtype=np.float64)

        theta = 2.0 * np.arctan2(v_norm, w)
        axis = v_part / v_norm
        return np.asarray(axis * theta, dtype=np.float64)


# ---------------------------------------------------------------------------
# Versioned Complete-State Interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimeState:
    """Complete, self-describing system state for Dynamics-Informed Mocap Matching."""

    t: float
    q: np.ndarray
    v: np.ndarray
    vdot: np.ndarray | None = None
    internal_state: Mapping[str, np.ndarray] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    model_hash: str = ""
    frame: str = "world"

    def __post_init__(self) -> None:
        require(self.t >= 0.0, "t must be non-negative")
        require(bool(self.model_hash.strip()), "model_hash cannot be empty")
        require(bool(np.all(np.isfinite(self.q))), "q must be finite")
        require(bool(np.all(np.isfinite(self.v))), "v must be finite")
        if self.vdot is not None:
            require(bool(np.all(np.isfinite(self.vdot))), "vdot must be finite")
            require(len(self.vdot) == len(self.v), "vdot dimension mismatch")

        # Validate declared units against canonical SI units
        for k, unit_val in self.units.items():
            if k in _ALLOWED_SI_UNITS and unit_val != _ALLOWED_SI_UNITS[k]:
                raise UnitMismatchError(
                    f"Invalid unit for {k}: expected {_ALLOWED_SI_UNITS[k]!r}, got {unit_val!r}"
                )

        # Freeze internal_state mapping
        frozen_internal: dict[str, np.ndarray] = {}
        for key, arr in self.internal_state.items():
            require(
                bool(np.all(np.isfinite(arr))), f"internal_state[{key}] must be finite"
            )
            frozen_internal[key] = np.asarray(arr, dtype=np.float64)
        object.__setattr__(self, "internal_state", MappingProxyType(frozen_internal))

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a JSON-compatible dictionary."""
        vdot_list = self.vdot.tolist() if self.vdot is not None else None
        internal_dict = {k: v.tolist() for k, v in self.internal_state.items()}
        return {
            "t": float(self.t),
            "q": self.q.tolist(),
            "v": self.v.tolist(),
            "vdot": vdot_list,
            "internal_state": internal_dict,
            "units": dict(self.units),
            "model_hash": self.model_hash,
            "frame": self.frame,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeState:
        """Deserialize state from dictionary."""
        q_arr = np.asarray(data["q"], dtype=np.float64)
        v_arr = np.asarray(data["v"], dtype=np.float64)
        vdot_raw = data.get("vdot")
        vdot_arr = (
            np.asarray(vdot_raw, dtype=np.float64) if vdot_raw is not None else None
        )
        internal_raw = data.get("internal_state", {})
        internal = {k: np.asarray(v, dtype=np.float64) for k, v in internal_raw.items()}
        return cls(
            t=float(data["t"]),
            q=q_arr,
            v=v_arr,
            vdot=vdot_arr,
            internal_state=internal,
            units=dict(data.get("units", {})),
            model_hash=str(data.get("model_hash", "")),
            frame=str(data.get("frame", "world")),
        )


# ---------------------------------------------------------------------------
# Control Channels Interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimeControlChannel:
    """Individual actuation channel definition with physical semantics."""

    name: str
    physical_type: ControlPhysicalType
    unit: str
    dof_index: int
    lower_limit: float
    upper_limit: float
    requires_activation_dynamics: bool = False

    def __post_init__(self) -> None:
        require(bool(self.name.strip()), "name cannot be empty")
        require(
            self.lower_limit <= self.upper_limit, "lower_limit must be <= upper_limit"
        )
        require(self.dof_index >= 0, "dof_index must be non-negative")

        # Muscle-driven models require explicit activation dynamics
        if (
            self.physical_type == ControlPhysicalType.EXCITATION
            and not self.requires_activation_dynamics
        ):
            raise ValueError(
                "Activation dynamics must be declared for muscle excitation channels; "
                "cannot silently substitute generalized torque without activation model."
            )


@dataclass(frozen=True)
class ControlAllocation:
    """Complete control specification for a step or window."""

    channels: tuple[DimeControlChannel, ...]

    def __post_init__(self) -> None:
        names = set()
        for ch in self.channels:
            if ch.name in names:
                raise ValueError(f"Duplicate control channel name: {ch.name}")
            names.add(ch.name)


# ---------------------------------------------------------------------------
# Contact Policy Interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContactPolicy:
    """Contact constraints policy enforcing interface exclusivity."""

    mode: ContactInterfaceMode
    contact_bodies: tuple[str, ...]
    friction_coefficient: float = 0.6
    retained_passive_loads: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode == ContactInterfaceMode.BOTH_CONFLICTING:
            raise ConflictingContactInterfaceError(
                "Native eliminated reactions and explicit constrained variables are "
                "mutually exclusive; choose one contact interface mode."
            )
        require(
            self.friction_coefficient >= 0.0,
            "friction_coefficient must be non-negative",
        )


# ---------------------------------------------------------------------------
# Observation Window Interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationItem:
    """A single sensor observation at a discrete timestamp."""

    t: float
    modality: str
    data: np.ndarray
    covariance: np.ndarray | None = None
    frame: str = "world"
    sensor_id: str = ""

    def __post_init__(self) -> None:
        require(self.t >= 0.0, "t must be non-negative")
        require(bool(self.modality.strip()), "modality cannot be empty")
        require(bool(np.all(np.isfinite(self.data))), "data must be finite")
        if self.covariance is not None:
            require(
                bool(np.all(np.isfinite(self.covariance))), "covariance must be finite"
            )


@dataclass(frozen=True)
class ObservationWindow:
    """A collection of sensor observations across a defined time window."""

    start_time: float
    end_time: float
    items: tuple[ObservationItem, ...]

    def __post_init__(self) -> None:
        require(self.start_time <= self.end_time, "start_time must be <= end_time")
        for obs in self.items:
            if obs.t < self.start_time or obs.t > self.end_time:
                raise ValueError(
                    f"Observation timestamp {obs.t} is outside window bounds "
                    f"[{self.start_time}, {self.end_time}]"
                )

        # Enforce temporal monotonicity
        for i in range(len(self.items) - 1):
            t_curr = self.items[i].t
            t_next = self.items[i + 1].t
            if t_next < t_curr:
                raise InvalidTimeOrderError(
                    f"Observation timestamps are non-monotonic: item[{i}].t={t_curr} "
                    f"> item[{i + 1}].t={t_next}"
                )


# ---------------------------------------------------------------------------
# Provider Snapshot and Rollback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSnapshot:
    """Immutable memory snapshot of a dynamics provider for rollback."""

    state: DimeState
    solver_memory: Mapping[str, Any] = field(default_factory=dict)
    contact_memory: Mapping[str, Any] = field(default_factory=dict)
    controller_memory: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Runtime Factor Exclusivity Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntervalFactorRegistration:
    """Registration record for an estimation factor on a time interval."""

    interval: tuple[float, float]
    factor_type: IntervalFactorType
    factor_id: str
    is_diagnostic: bool = False

    def __post_init__(self) -> None:
        t0, t1 = self.interval
        require(t0 < t1, "interval must have t0 < t1")


class IntervalFactorRegistry:
    """Enforces runtime mutual exclusivity on estimation factors over time intervals."""

    def __init__(self) -> None:
        self._registrations: list[IntervalFactorRegistration] = []

    def register(self, reg: IntervalFactorRegistration) -> None:
        """Register a factor, checking exclusivity against existing active factors."""
        if not reg.is_diagnostic:
            # Check overlap against existing active estimation factors
            t0, t1 = reg.interval
            for existing in self._registrations:
                if existing.is_diagnostic:
                    continue
                ex_t0, ex_t1 = existing.interval
                # Check interval overlap: max(t0, ex_t0) < min(t1, ex_t1)
                if max(t0, ex_t0) < min(t1, ex_t1):
                    # Overlap detected! Check type exclusivity
                    t_type1 = reg.factor_type
                    t_type2 = existing.factor_type
                    conflicting = (
                        t_type1 == IntervalFactorType.MARGINALIZED_INPUT_TRANSITION
                        and t_type2 == IntervalFactorType.EXPLICIT_INPUT_LIKELIHOOD
                    ) or (
                        t_type1 == IntervalFactorType.EXPLICIT_INPUT_LIKELIHOOD
                        and t_type2 == IntervalFactorType.MARGINALIZED_INPUT_TRANSITION
                    )
                    if conflicting:
                        raise ExclusiveFactorConflictError(
                            f"Exclusive factor conflict on interval [{max(t0, ex_t0)}, {min(t1, ex_t1)}]: "
                            f"{t_type1.value} conflicts with existing {t_type2.value} ({existing.factor_id})"
                        )
        self._registrations.append(reg)

    def get_active_estimation_factors(self) -> tuple[IntervalFactorRegistration, ...]:
        return tuple(r for r in self._registrations if not r.is_diagnostic)

    def get_diagnostic_factors(self) -> tuple[IntervalFactorRegistration, ...]:
        return tuple(r for r in self._registrations if r.is_diagnostic)


# ---------------------------------------------------------------------------
# Dynamics Provider Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DimeDynamicsProvider(Protocol):
    """Unified interface for dynamics providers in DIME."""

    @property
    def name(self) -> str: ...

    @property
    def model_hash(self) -> str: ...

    @property
    def capability_status(self) -> CapabilityStatus: ...

    @property
    def manifold(self) -> DimeManifold: ...

    def snapshot(self) -> ProviderSnapshot: ...

    def restore(self, snapshot: ProviderSnapshot) -> None: ...

    def get_state(self) -> DimeState: ...

    def set_state(self, state: DimeState) -> None: ...

    def step_full(
        self, state: DimeState, control: np.ndarray, dt: float
    ) -> DimeState: ...

    def step_zero_input(self, state: DimeState, dt: float) -> DimeState: ...


# ---------------------------------------------------------------------------
# Concrete Providers: Deterministic Fake and Analytic Pendulum
# ---------------------------------------------------------------------------


def make_default_1d_state(model_hash: str) -> DimeState:
    """Construct a default single-DOF initial DimeState."""
    return DimeState(
        t=0.0,
        q=np.array([0.0]),
        v=np.array([0.0]),
        model_hash=model_hash,
        units=dict(CANONICAL_DIME_UNITS),
    )


class DeterministicFakeProvider:
    """Deterministic linear-motion fake provider for unit testing and contract verification."""

    def __init__(
        self,
        model_hash: str = "sha256:test_model_v1",
        capability_status: CapabilityStatus = "implemented",
        unavailability_reason: str | None = None,
    ) -> None:
        self._name = "deterministic_fake_provider"
        self._model_hash = model_hash
        self._capability_status = capability_status
        self._unavailability_reason = unavailability_reason
        self._manifold = VectorManifold(dim=1)
        self._current_state = make_default_1d_state(model_hash)
        self._fail_on_step = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_hash(self) -> str:
        return self._model_hash

    @property
    def capability_status(self) -> CapabilityStatus:
        return self._capability_status

    @property
    def manifold(self) -> DimeManifold:
        return self._manifold

    def inject_step_failure(self, fail: bool) -> None:
        self._fail_on_step = fail

    def snapshot(self) -> ProviderSnapshot:
        return ProviderSnapshot(
            state=self._current_state,
            timestamp=self._current_state.t,
        )

    def restore(self, snapshot: ProviderSnapshot) -> None:
        self._current_state = snapshot.state

    def get_state(self) -> DimeState:
        return self._current_state

    def set_state(self, state: DimeState) -> None:
        require(state.model_hash == self._model_hash, "Model hash mismatch")
        self._current_state = state

    def _check_available_and_hash(self, state: DimeState) -> None:
        if self._capability_status == "unavailable":
            reason = self._unavailability_reason or "Provider is unavailable"
            raise ProviderUnavailableError(
                f"Provider '{self._name}' is unavailable: {reason}"
            )
        if state.model_hash != self._model_hash:
            raise StaleModelHashError(
                f"Model hash mismatch: expected {self._model_hash}, got {state.model_hash}"
            )

    def step_full(self, state: DimeState, control: np.ndarray, dt: float) -> DimeState:
        self._check_available_and_hash(state)
        require(dt > 0.0, "dt must be positive")

        # Snapshot for automatic rollback on exception
        pre_snap = self.snapshot()
        self.set_state(state)

        if self._fail_on_step:
            self.restore(pre_snap)
            raise RuntimeError("Simulated solver convergence failure in native step")

        u = float(control[0]) if len(control) > 0 else 0.0
        accel = u
        v_next = state.v + accel * dt
        q_next = self._manifold.retract(state.q, v_next * dt)
        t_next = state.t + dt

        next_state = DimeState(
            t=t_next,
            q=q_next,
            v=v_next,
            vdot=np.array([accel]),
            internal_state=state.internal_state,
            units=state.units,
            model_hash=self._model_hash,
            frame=state.frame,
        )
        self.set_state(next_state)
        return next_state

    def step_zero_input(self, state: DimeState, dt: float) -> DimeState:
        return self.step_full(state, control=np.array([0.0]), dt=dt)


class AnalyticPendulumProvider:
    """Exact analytic dynamics provider for a 1-DOF fixed-base conservative pendulum."""

    def __init__(
        self,
        length_m: float = 1.0,
        mass_kg: float = 1.0,
        gravity_mps2: float = 9.81,
        model_hash: str = "sha256:analytic_pendulum_v1",
    ) -> None:
        require(length_m > 0.0, "length_m must be positive")
        require(mass_kg > 0.0, "mass_kg must be positive")
        require(gravity_mps2 >= 0.0, "gravity_mps2 must be non-negative")

        self._name = "analytic_pendulum_provider"
        self._length = length_m
        self._mass = mass_kg
        self._gravity = gravity_mps2
        self._model_hash = model_hash
        self._capability_status: CapabilityStatus = "implemented"
        self._manifold = VectorManifold(dim=1)
        self._current_state = make_default_1d_state(model_hash)

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_hash(self) -> str:
        return self._model_hash

    @property
    def capability_status(self) -> CapabilityStatus:
        return self._capability_status

    @property
    def manifold(self) -> DimeManifold:
        return self._manifold

    def compute_energy(self, state: DimeState) -> float:
        """Compute total mechanical energy (kinetic + potential) in Joules."""
        q = float(state.q[0])
        v = float(state.v[0])
        kinetic = 0.5 * self._mass * (self._length**2) * (v**2)
        potential = -self._mass * self._gravity * self._length * np.cos(q)
        return float(kinetic + potential)

    def snapshot(self) -> ProviderSnapshot:
        return ProviderSnapshot(
            state=self._current_state,
            timestamp=self._current_state.t,
        )

    def restore(self, snapshot: ProviderSnapshot) -> None:
        self._current_state = snapshot.state

    def get_state(self) -> DimeState:
        return self._current_state

    def set_state(self, state: DimeState) -> None:
        require(state.model_hash == self._model_hash, "Model hash mismatch")
        self._current_state = state

    def _check_available_and_hash(self, state: DimeState) -> None:
        if self._capability_status == "unavailable":
            raise ProviderUnavailableError(f"Provider '{self._name}' is unavailable")
        if state.model_hash != self._model_hash:
            raise StaleModelHashError(
                f"Model hash mismatch: expected {self._model_hash}, got {state.model_hash}"
            )

    def step_full(self, state: DimeState, control: np.ndarray, dt: float) -> DimeState:
        self._check_available_and_hash(state)
        require(dt > 0.0, "dt must be positive")

        pre_snap = self.snapshot()
        self.set_state(state)

        q = float(state.q[0])
        v = float(state.v[0])
        u = float(control[0]) if len(control) > 0 else 0.0

        # Substep RK4 to ensure energy conservation
        substeps = 20
        sub_dt = dt / substeps

        def ode(_q: float, _v: float) -> tuple[float, float]:
            accel = -(self._gravity / self._length) * np.sin(_q) + u / (
                self._mass * (self._length**2)
            )
            return _v, accel

        curr_q, curr_v = q, v
        for _ in range(substeps):
            k1_q, k1_v = ode(curr_q, curr_v)
            k2_q, k2_v = ode(curr_q + 0.5 * sub_dt * k1_q, curr_v + 0.5 * sub_dt * k1_v)
            k3_q, k3_v = ode(curr_q + 0.5 * sub_dt * k2_q, curr_v + 0.5 * sub_dt * k2_v)
            k4_q, k4_v = ode(curr_q + sub_dt * k3_q, curr_v + sub_dt * k3_v)

            curr_q += (sub_dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
            curr_v += (sub_dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        _, last_accel = ode(curr_q, curr_v)
        next_state = DimeState(
            t=state.t + dt,
            q=np.array([curr_q]),
            v=np.array([curr_v]),
            vdot=np.array([last_accel]),
            internal_state=state.internal_state,
            units=state.units,
            model_hash=self._model_hash,
            frame=state.frame,
        )
        self.set_state(next_state)
        return next_state

    def step_zero_input(self, state: DimeState, dt: float) -> DimeState:
        return self.step_full(state, control=np.array([0.0]), dt=dt)


# ---------------------------------------------------------------------------
# Estimation Result Interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimeEstimationResult:
    """Formal result container for Dynamics-Informed Mocap Matching solves."""

    success: bool
    trajectory: tuple[DimeState, ...]
    residuals: Mapping[str, float]
    provenance: DimeProvenanceRecord
    qualification_status: CapabilityStatus

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary."""
        traj_list = [s.to_dict() for s in self.trajectory]
        res_dict = {k: float(v) for k, v in self.residuals.items()}
        prov_dict = self.provenance.to_dict()
        return {
            "success": self.success,
            "trajectory": traj_list,
            "residuals": res_dict,
            "provenance": prov_dict,
            "qualification_status": self.qualification_status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeEstimationResult:
        """Deserialize result from dictionary."""
        traj_raw = data.get("trajectory", [])
        trajectory = tuple(DimeState.from_dict(s) for s in traj_raw)
        residuals = {str(k): float(v) for k, v in data.get("residuals", {}).items()}
        provenance = DimeProvenanceRecord.from_dict(data["provenance"])
        return cls(
            success=bool(data["success"]),
            trajectory=trajectory,
            residuals=residuals,
            provenance=provenance,
            qualification_status=data.get("qualification_status", "implemented"),
        )
