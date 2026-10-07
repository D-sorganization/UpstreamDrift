"""DIME State, Observation and Dynamics Provider Contracts (#11421, #11423).

Provides versioned complete-state, observation-window, provider capability,
full-step, zero-input proposal, and estimation-result interfaces for dynamics-informed
mocap matching. Integrates the intervention contract (#10286) for acceleration
decomposition, enforces runtime factor exclusivity, and provides public manifold
operations with support for nq != nv and quaternion sign equivalence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import enum
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from src.shared.python.core.deprecation import deprecated_alias_getattr
from src.shared.python.contracts import (
    ContractViolationError,
    PreconditionError,
    require,
)
from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    CapabilityStatus,
)
from src.shared.python.motion_matching.counterfactual import (
    AccelerationDecomposition,
)

DIME_CONTRACTS_VERSION: Final[str] = "1.0.0"


class ContactPolicy(str, enum.Enum):
    """Supported contact reaction interfaces; never both independently."""

    NATIVE_ELIMINATED = "native_eliminated"
    EXPLICIT_CONSTRAINED = "explicit_constrained"


UncertaintyKind = Literal["gaussian", "laplace", "covariance", "unweighted"]
ControlPhysicalType = Literal["torque", "force", "excitation"]

_VALID_CONTACT_POLICIES: Final[tuple[str, ...]] = (
    "native_eliminated",
    "explicit_constrained",
)
_VALID_UNCERTAINTY_KINDS: Final[tuple[str, ...]] = (
    "gaussian",
    "laplace",
    "covariance",
    "unweighted",
)
_VALID_PHYSICAL_TYPES: Final[tuple[str, ...]] = ("torque", "force", "excitation")


def _make_readonly_array(arr: np.ndarray) -> np.ndarray:
    """Return a read-only copy of a float64 numpy array."""
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


def _validate_units_dict(units: Mapping[str, str], name: str = "units") -> None:
    """Ensure units dictionary only contains recognized physical dimension keys and valid units."""
    valid_units_lookup = {
        "length": ("m", "meter", "meters"),
        "angle": ("rad", "radian", "radians"),
        "time": ("s", "sec", "second", "seconds"),
        "mass": ("kg", "kilogram", "kilograms"),
        "force": ("N", "newton", "newtons"),
        "torque": ("N*m", "N·m", "Nm", "newton_meter"),
    }
    for dim, unit_str in units.items():
        require(
            dim in valid_units_lookup,
            f"{name} contains unrecognized dimension key: {dim}",
        )
        require(
            unit_str in valid_units_lookup[dim],
            f"{name} dimension '{dim}' has invalid/non-SI unit: {unit_str}",
        )


# ==============================================================================
# Complete State & Observation Window
# ==============================================================================


@dataclass(frozen=True)
class DimeCompleteState:
    """Versioned complete dynamics state representing generalized coordinates and memory."""

    t: float
    q: np.ndarray
    v: np.ndarray
    v_dot: np.ndarray | None = None
    internal_state: Mapping[str, Any] = field(default_factory=dict)
    model_hash: str = ""
    units: Mapping[str, str] = field(default_factory=dict)
    frame: str = "world"

    def __post_init__(self) -> None:
        require(
            isinstance(self.t, (int, float)) and np.isfinite(self.t) and self.t >= 0.0,
            "State timestamp t must be non-negative and finite",
            self.t,
        )
        require(
            isinstance(self.model_hash, str) and bool(self.model_hash.strip()),
            "model_hash must be non-empty",
        )

        raw_q = np.asarray(self.q)
        require(raw_q.dtype.kind in "iuf", "q must be numeric", raw_q.dtype)
        require(
            not isinstance(self.q, bool)
            and not (isinstance(self.q, np.ndarray) and self.q.dtype == bool)
            and not any(isinstance(x, (bool, np.bool_)) for x in raw_q.flat),
            "q must not contain booleans",
        )
        require(raw_q.ndim == 1, "q must be a 1D vector", raw_q.shape)
        require(
            bool(np.all(np.isfinite(raw_q))),
            "q must contain strictly finite real numbers",
        )

        raw_v = np.asarray(self.v)
        require(raw_v.dtype.kind in "iuf", "v must be numeric", raw_v.dtype)
        require(
            not isinstance(self.v, bool)
            and not (isinstance(self.v, np.ndarray) and self.v.dtype == bool)
            and not any(isinstance(x, (bool, np.bool_)) for x in raw_v.flat),
            "v must not contain booleans",
        )
        require(raw_v.ndim == 1, "v must be a 1D vector", raw_v.shape)
        require(
            bool(np.all(np.isfinite(raw_v))),
            "v must contain strictly finite real numbers",
        )

        if self.v_dot is not None:
            raw_vdot = np.asarray(self.v_dot)
            require(raw_vdot.ndim == 1, "v_dot must be 1D", raw_vdot.shape)
            require(raw_vdot.shape == raw_v.shape, "v_dot shape must match v shape")
            require(bool(np.all(np.isfinite(raw_vdot))), "v_dot must be finite")
            object.__setattr__(self, "v_dot", _make_readonly_array(raw_vdot))

        _validate_units_dict(self.units, "DimeCompleteState.units")

        object.__setattr__(self, "q", _make_readonly_array(raw_q))
        object.__setattr__(self, "v", _make_readonly_array(raw_v))
        object.__setattr__(
            self, "internal_state", MappingProxyType(dict(self.internal_state))
        )
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": float(self.t),
            "q": self.q.tolist(),
            "v": self.v.tolist(),
            "v_dot": self.v_dot.tolist() if self.v_dot is not None else None,
            "internal_state": dict(self.internal_state),
            "model_hash": self.model_hash,
            "units": dict(self.units),
            "frame": self.frame,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeCompleteState:
        v_dot_raw = data.get("v_dot")
        return cls(
            t=float(data["t"]),
            q=np.array(data["q"], dtype=np.float64),
            v=np.array(data["v"], dtype=np.float64),
            v_dot=np.array(v_dot_raw, dtype=np.float64)
            if v_dot_raw is not None
            else None,
            internal_state=dict(data.get("internal_state", {})),
            model_hash=str(data.get("model_hash", "")),
            units=dict(data.get("units", {})),
            frame=str(data.get("frame", "world")),
        )


@dataclass(frozen=True)
class DimeObservationWindow:
    """Windowed observation container enforcing strict temporal monotonicity and uncertainty."""

    t_start: float
    t_end: float
    times: np.ndarray
    observations: Mapping[str, np.ndarray]
    uncertainty_kind: UncertaintyKind = "gaussian"
    uncertainty_values: Mapping[str, np.ndarray] | None = None
    units: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(
            self.t_start <= self.t_end,
            "t_start must be <= t_end",
            (self.t_start, self.t_end),
        )
        raw_times = np.asarray(self.times, dtype=np.float64)
        require(
            raw_times.ndim == 1 and len(raw_times) > 0,
            "times must be a non-empty 1D array",
        )
        require(bool(np.all(np.isfinite(raw_times))), "times must be strictly finite")
        require(
            bool(np.all(np.diff(raw_times) > 0.0)),
            "Observation timestamps must be strictly monotonic",
        )
        require(
            self.t_start <= raw_times[0] and raw_times[-1] <= self.t_end,
            "times must fall within [t_start, t_end]",
        )
        require(
            self.uncertainty_kind in _VALID_UNCERTAINTY_KINDS,
            f"Invalid uncertainty_kind: {self.uncertainty_kind}",
        )

        _validate_units_dict(self.units, "DimeObservationWindow.units")

        obs_dict = {}
        for key, val in self.observations.items():
            arr = np.asarray(val, dtype=np.float64)
            require(
                bool(np.all(np.isfinite(arr))),
                f"Observation '{key}' contains non-finite values",
            )
            require(
                arr.shape[0] == len(raw_times),
                f"Observation '{key}' leading dimension does not match times",
            )
            obs_dict[key] = _make_readonly_array(arr)

        unc_dict = None
        if self.uncertainty_values is not None:
            unc_dict = {}
            for key, val in self.uncertainty_values.items():
                arr = np.asarray(val, dtype=np.float64)
                require(
                    bool(np.all(np.isfinite(arr))),
                    f"Uncertainty '{key}' contains non-finite values",
                )
                unc_dict[key] = _make_readonly_array(arr)

        object.__setattr__(self, "times", _make_readonly_array(raw_times))
        object.__setattr__(self, "observations", MappingProxyType(obs_dict))
        object.__setattr__(
            self,
            "uncertainty_values",
            MappingProxyType(unc_dict) if unc_dict is not None else None,
        )
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_start": float(self.t_start),
            "t_end": float(self.t_end),
            "times": self.times.tolist(),
            "observations": {k: v.tolist() for k, v in self.observations.items()},
            "uncertainty_kind": self.uncertainty_kind,
            "uncertainty_values": (
                {k: v.tolist() for k, v in self.uncertainty_values.items()}
                if self.uncertainty_values is not None
                else None
            ),
            "units": dict(self.units),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeObservationWindow:
        unc_raw = data.get("uncertainty_values")
        return cls(
            t_start=float(data["t_start"]),
            t_end=float(data["t_end"]),
            times=np.array(data["times"], dtype=np.float64),
            observations={
                str(k): np.array(v, dtype=np.float64)
                for k, v in data.get("observations", {}).items()
            },
            uncertainty_kind=data.get("uncertainty_kind", "gaussian"),
            uncertainty_values=(
                {str(k): np.array(v, dtype=np.float64) for k, v in unc_raw.items()}
                if unc_raw is not None
                else None
            ),
            units=dict(data.get("units", {})),
        )


# ==============================================================================
# Control Channels, Passive Loads & Contact Policy
# ==============================================================================


@dataclass(frozen=True)
class ControlChannelSpec:
    """Declared control channel specifying physical type, units, map and limits."""

    name: str
    physical_type: ControlPhysicalType
    units: str
    selection_map: tuple[int, ...]
    limits: tuple[float, float]
    internal_state_semantics: str = "instantaneous"

    def __post_init__(self) -> None:
        require(
            self.physical_type in _VALID_PHYSICAL_TYPES,
            f"Invalid physical_type: {self.physical_type}",
        )
        require(
            self.limits[0] <= self.limits[1],
            "Control channel limit min must be <= max",
            self.limits,
        )
        require(len(self.selection_map) > 0, "selection_map cannot be empty")
        for idx in self.selection_map:
            require(idx >= 0, "selection_map indices must be non-negative", idx)

        # Unit verification against physical type
        if self.physical_type == "torque":
            require(
                self.units in ("N*m", "N·m", "Nm"),
                "Torque units must be N*m",
                self.units,
            )
        elif self.physical_type == "force":
            require(self.units == "N", "Force units must be N", self.units)
        elif self.physical_type == "excitation":
            require(
                self.units in ("dimensionless", "", "1"),
                "Excitation units must be dimensionless",
                self.units,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "physical_type": self.physical_type,
            "units": self.units,
            "selection_map": list(self.selection_map),
            "limits": list(self.limits),
            "internal_state_semantics": self.internal_state_semantics,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ControlChannelSpec:
        return cls(
            name=str(data["name"]),
            physical_type=data["physical_type"],
            units=str(data["units"]),
            selection_map=tuple(int(x) for x in data["selection_map"]),
            limits=(float(data["limits"][0]), float(data["limits"][1])),
            internal_state_semantics=str(
                data.get("internal_state_semantics", "instantaneous")
            ),
        )


@dataclass(frozen=True)
class PassiveLoadSpec:
    """Inventoried passive load evaluated where modeled; ROM priors do not supply unmeasured law."""

    name: str
    load_type: str
    dof_indices: tuple[int, ...]
    stiffness: float | None = None
    damping: float | None = None
    is_measured: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(bool(self.name.strip()), "Passive load name cannot be empty")
        for idx in self.dof_indices:
            require(idx >= 0, "dof_indices must be non-negative", idx)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def evaluate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate generalized passive resistive force vector on declared DOFs."""
        force = np.zeros(len(self.dof_indices), dtype=np.float64)
        for i, dof in enumerate(self.dof_indices):
            k_term = (
                -self.stiffness * q[dof]
                if self.stiffness is not None and dof < len(q)
                else 0.0
            )
            d_term = (
                -self.damping * v[dof]
                if self.damping is not None and dof < len(v)
                else 0.0
            )
            force[i] = k_term + d_term
        return force

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "load_type": self.load_type,
            "dof_indices": list(self.dof_indices),
            "stiffness": self.stiffness,
            "damping": self.damping,
            "is_measured": self.is_measured,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PassiveLoadSpec:
        return cls(
            name=str(data["name"]),
            load_type=str(data["load_type"]),
            dof_indices=tuple(int(x) for x in data["dof_indices"]),
            stiffness=float(data["stiffness"])
            if data.get("stiffness") is not None
            else None,
            damping=float(data["damping"]) if data.get("damping") is not None else None,
            is_measured=bool(data.get("is_measured", True)),
            metadata=dict(data.get("metadata", {})),
        )


from src.shared.python.estimation.dime_manifold import (
    ManifoldContract,
    QuaternionManifold,
    SE3Manifold,
    VectorSpaceManifold,
    manifold_from_dict,
)


# ==============================================================================
# Provider Capability & Snapshots
# ==============================================================================


@dataclass(frozen=True)
class ProviderCapability:
    """Truthful capability report declaring manifold, control channels, contact and memory."""

    provider_id: str
    version: str
    status: CapabilityStatus
    n_q: int
    n_v: int
    manifold: ManifoldContract
    control_channels: tuple[ControlChannelSpec, ...]
    contact_policy: ContactPolicy
    retained_passive_loads: tuple[PassiveLoadSpec, ...]
    supports_snapshot: bool = True
    supports_ztcf: bool = True
    supports_zvcf: bool = True
    has_activation_dynamics: bool = False
    is_qualified: bool = False

    def __post_init__(self) -> None:
        require(
            self.contact_policy in _VALID_CONTACT_POLICIES,
            f"Unsupported contact_policy: {self.contact_policy}",
        )
        require(self.n_q > 0 and self.n_v > 0, "n_q and n_v must be strictly positive")
        require(
            self.manifold.n_q == self.n_q, "Manifold n_q mismatch with capability n_q"
        )
        require(
            self.manifold.n_v == self.n_v, "Manifold n_v mismatch with capability n_v"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "version": self.version,
            "status": self.status,
            "n_q": self.n_q,
            "n_v": self.n_v,
            "manifold": self.manifold.to_dict(),
            "control_channels": [c.to_dict() for c in self.control_channels],
            "contact_policy": self.contact_policy,
            "retained_passive_loads": [
                p.to_dict() for p in self.retained_passive_loads
            ],
            "supports_snapshot": self.supports_snapshot,
            "supports_ztcf": self.supports_ztcf,
            "supports_zvcf": self.supports_zvcf,
            "has_activation_dynamics": self.has_activation_dynamics,
            "is_qualified": self.is_qualified,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProviderCapability:
        return cls(
            provider_id=str(data["provider_id"]),
            version=str(data["version"]),
            status=data["status"],
            n_q=int(data["n_q"]),
            n_v=int(data["n_v"]),
            manifold=manifold_from_dict(data["manifold"]),
            control_channels=tuple(
                ControlChannelSpec.from_dict(c)
                for c in data.get("control_channels", ())
            ),
            contact_policy=data["contact_policy"],
            retained_passive_loads=tuple(
                PassiveLoadSpec.from_dict(p)
                for p in data.get("retained_passive_loads", ())
            ),
            supports_snapshot=bool(data.get("supports_snapshot", True)),
            supports_ztcf=bool(data.get("supports_ztcf", True)),
            supports_zvcf=bool(data.get("supports_zvcf", True)),
            has_activation_dynamics=bool(data.get("has_activation_dynamics", False)),
            is_qualified=bool(data.get("is_qualified", False)),
        )


def check_qualification_rules(cap: ProviderCapability) -> tuple[bool, str]:
    """Evaluate fail-closed qualification rules on dynamics provider capabilities."""
    for ch in cap.control_channels:
        if ch.physical_type == "excitation":
            if not cap.has_activation_dynamics:
                return (
                    False,
                    "Muscle-driven model requires full activation dynamics, never silent substitution.",
                )
            if "activation" not in ch.internal_state_semantics.lower():
                return (
                    False,
                    f"Muscle channel '{ch.name}' lacks explicit activation dynamics semantics.",
                )

    if cap.contact_policy not in _VALID_CONTACT_POLICIES:
        return False, f"Unsupported contact policy '{cap.contact_policy}'"

    for load in cap.retained_passive_loads:
        if not load.is_measured:
            return (
                False,
                f"Unmeasured passive stiffness law in '{load.name}' cannot be qualified; "
                "ROM priors do not supply an unmeasured passive stiffness law.",
            )

    return True, "Qualification checks passed."


@dataclass(frozen=True)
class ProviderSnapshot:
    """Serializable snapshot of complete state and solver/contact/controller memory."""

    provider_id: str
    model_hash: str
    timestamp: float
    state: DimeCompleteState
    solver_memory: Mapping[str, Any] = field(default_factory=dict)
    contact_memory: Mapping[str, Any] = field(default_factory=dict)
    controller_memory: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "solver_memory", MappingProxyType(dict(self.solver_memory))
        )
        object.__setattr__(
            self, "contact_memory", MappingProxyType(dict(self.contact_memory))
        )
        object.__setattr__(
            self, "controller_memory", MappingProxyType(dict(self.controller_memory))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "model_hash": self.model_hash,
            "timestamp": float(self.timestamp),
            "state": self.state.to_dict(),
            "solver_memory": dict(self.solver_memory),
            "contact_memory": dict(self.contact_memory),
            "controller_memory": dict(self.controller_memory),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProviderSnapshot:
        return cls(
            provider_id=str(data["provider_id"]),
            model_hash=str(data["model_hash"]),
            timestamp=float(data["timestamp"]),
            state=DimeCompleteState.from_dict(data["state"]),
            solver_memory=dict(data.get("solver_memory", {})),
            contact_memory=dict(data.get("contact_memory", {})),
            controller_memory=dict(data.get("controller_memory", {})),
        )


# ==============================================================================
# Full Step & Zero-Input Proposals
# ==============================================================================


@dataclass(frozen=True)
class DimeFullStepRequest:
    """Input request for forward simulation step."""

    state: DimeCompleteState
    controls: np.ndarray
    dt: float
    model_hash: str

    def __post_init__(self) -> None:
        require(
            self.dt > 0.0 and np.isfinite(self.dt),
            "dt must be strictly positive and finite",
            self.dt,
        )
        raw_u = np.asarray(self.controls, dtype=np.float64)
        require(bool(np.all(np.isfinite(raw_u))), "Controls must be finite")
        object.__setattr__(self, "controls", _make_readonly_array(raw_u))

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "controls": self.controls.tolist(),
            "dt": float(self.dt),
            "model_hash": self.model_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeFullStepRequest:
        return cls(
            state=DimeCompleteState.from_dict(data["state"]),
            controls=np.array(data["controls"], dtype=np.float64),
            dt=float(data["dt"]),
            model_hash=str(data["model_hash"]),
        )


@dataclass(frozen=True)
class DimeFullStepResult:
    """Output result from forward simulation step."""

    next_state: DimeCompleteState
    accelerations: np.ndarray
    reaction_forces: np.ndarray | None = None
    decomposition: AccelerationDecomposition | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_a = np.asarray(self.accelerations, dtype=np.float64)
        require(bool(np.all(np.isfinite(raw_a))), "Accelerations must be finite")
        object.__setattr__(self, "accelerations", _make_readonly_array(raw_a))
        if self.reaction_forces is not None:
            raw_rf = np.asarray(self.reaction_forces, dtype=np.float64)
            require(bool(np.all(np.isfinite(raw_rf))), "Reaction forces must be finite")
            object.__setattr__(self, "reaction_forces", _make_readonly_array(raw_rf))
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_state": self.next_state.to_dict(),
            "accelerations": self.accelerations.tolist(),
            "reaction_forces": self.reaction_forces.tolist()
            if self.reaction_forces is not None
            else None,
            "decomposition": (
                {
                    "a_grav": self.decomposition.a_grav.tolist(),
                    "a_drift": self.decomposition.a_drift.tolist(),
                    "a_ctrl": self.decomposition.a_ctrl.tolist(),
                }
                if self.decomposition is not None
                else None
            ),
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeFullStepResult:
        rf_raw = data.get("reaction_forces")
        decomp_raw = data.get("decomposition")
        decomp = None
        if decomp_raw is not None:
            decomp = AccelerationDecomposition(
                a_grav=np.array(decomp_raw["a_grav"], dtype=np.float64),
                a_drift=np.array(decomp_raw["a_drift"], dtype=np.float64),
                a_ctrl=np.array(decomp_raw["a_ctrl"], dtype=np.float64),
            )
        return cls(
            next_state=DimeCompleteState.from_dict(data["next_state"]),
            accelerations=np.array(data["accelerations"], dtype=np.float64),
            reaction_forces=np.array(rf_raw, dtype=np.float64)
            if rf_raw is not None
            else None,
            decomposition=decomp,
            diagnostics=dict(data.get("diagnostics", {})),
        )


@dataclass(frozen=True)
class DimeZeroInputProposal:
    """Zero-Torque Counterfactual (ZTCF) forward proposal with acceleration decomposition."""

    t_start: float
    t_end: float
    times: np.ndarray
    states: tuple[DimeCompleteState, ...]
    accelerations: np.ndarray
    decompositions: tuple[AccelerationDecomposition, ...]
    source_intervention: str = "ZTCF"

    def __post_init__(self) -> None:
        raw_times = np.asarray(self.times, dtype=np.float64)
        raw_a = np.asarray(self.accelerations, dtype=np.float64)
        require(
            len(self.states) == len(raw_times), "States count must match times length"
        )
        require(
            len(self.decompositions) == len(raw_times),
            "Decompositions count must match times length",
        )
        object.__setattr__(self, "times", _make_readonly_array(raw_times))
        object.__setattr__(self, "accelerations", _make_readonly_array(raw_a))

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_start": float(self.t_start),
            "t_end": float(self.t_end),
            "times": self.times.tolist(),
            "states": [s.to_dict() for s in self.states],
            "accelerations": self.accelerations.tolist(),
            "decompositions": [
                {
                    "a_grav": d.a_grav.tolist(),
                    "a_drift": d.a_drift.tolist(),
                    "a_ctrl": d.a_ctrl.tolist(),
                }
                for d in self.decompositions
            ],
            "source_intervention": self.source_intervention,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeZeroInputProposal:
        decomps = tuple(
            AccelerationDecomposition(
                a_grav=np.array(d["a_grav"], dtype=np.float64),
                a_drift=np.array(d["a_drift"], dtype=np.float64),
                a_ctrl=np.array(d["a_ctrl"], dtype=np.float64),
            )
            for d in data["decompositions"]
        )
        return cls(
            t_start=float(data["t_start"]),
            t_end=float(data["t_end"]),
            times=np.array(data["times"], dtype=np.float64),
            states=tuple(DimeCompleteState.from_dict(s) for s in data["states"]),
            accelerations=np.array(data["accelerations"], dtype=np.float64),
            decompositions=decomps,
            source_intervention=str(data.get("source_intervention", "ZTCF")),
        )


@dataclass(frozen=True)
class DimeEstimationResult:
    """Standardized receipt and estimation output conforming to DIME contracts."""

    contract_version: str
    trajectory_states: tuple[DimeCompleteState, ...]
    estimated_controls: np.ndarray | None
    residuals: Mapping[str, np.ndarray]
    uncertainty_kind: UncertaintyKind
    uncertainty_summary: Mapping[str, Any]
    model_hash: str
    qualification_status: CapabilityStatus
    alignment_metric: float | None = None
    cancellation_metric: float | None = None
    receipt: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "estimated_controls",
            _make_readonly_array(self.estimated_controls)
            if self.estimated_controls is not None
            else None,
        )
        res_map = {k: _make_readonly_array(v) for k, v in self.residuals.items()}
        object.__setattr__(self, "residuals", MappingProxyType(res_map))
        object.__setattr__(
            self,
            "uncertainty_summary",
            MappingProxyType(dict(self.uncertainty_summary)),
        )
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "trajectory_states": [s.to_dict() for s in self.trajectory_states],
            "estimated_controls": (
                self.estimated_controls.tolist()
                if self.estimated_controls is not None
                else None
            ),
            "residuals": {k: v.tolist() for k, v in self.residuals.items()},
            "uncertainty_kind": self.uncertainty_kind,
            "uncertainty_summary": dict(self.uncertainty_summary),
            "model_hash": self.model_hash,
            "qualification_status": self.qualification_status,
            "alignment_metric": self.alignment_metric,
            "cancellation_metric": self.cancellation_metric,
            "receipt": dict(self.receipt),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DimeEstimationResult:
        u_raw = data.get("estimated_controls")
        return cls(
            contract_version=str(data["contract_version"]),
            trajectory_states=tuple(
                DimeCompleteState.from_dict(s) for s in data["trajectory_states"]
            ),
            estimated_controls=np.array(u_raw, dtype=np.float64)
            if u_raw is not None
            else None,
            residuals={
                str(k): np.array(v, dtype=np.float64)
                for k, v in data.get("residuals", {}).items()
            },
            uncertainty_kind=data["uncertainty_kind"],
            uncertainty_summary=dict(data.get("uncertainty_summary", {})),
            model_hash=str(data["model_hash"]),
            qualification_status=data["qualification_status"],
            alignment_metric=float(data["alignment_metric"])
            if data.get("alignment_metric") is not None
            else None,
            cancellation_metric=(
                float(data["cancellation_metric"])
                if data.get("cancellation_metric") is not None
                else None
            ),
            receipt=dict(data.get("receipt", {})),
        )


# ==============================================================================
# Runtime Exclusivity Contract
# ==============================================================================


@dataclass(frozen=True)
class EstimationIntervalFactor:
    """Declared factor contribution over a specific time window."""

    name: str
    factor_type: str
    t_start: float
    t_end: float
    contributes_to_objective: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(
            self.t_start <= self.t_end,
            "t_start must be <= t_end",
            (self.t_start, self.t_end),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class RuntimeExclusivityContract:
    """Enforces runtime mutual exclusivity between marginalized and explicit input factors."""

    def __init__(self) -> None:
        self._factors: list[EstimationIntervalFactor] = []

    def register_factor(self, factor: EstimationIntervalFactor) -> None:
        """Register factor and enforce runtime exclusivity invariants."""
        if factor.factor_type == "diagnostic":
            require(
                not factor.contributes_to_objective,
                "Permitted diagnostics cannot contribute a duplicate factor to the estimation objective",
                factor.name,
            )

        if factor.contributes_to_objective:
            for existing in self._factors:
                if not existing.contributes_to_objective:
                    continue
                # Check interval overlap
                overlap = max(
                    0.0,
                    min(factor.t_end, existing.t_end)
                    - max(factor.t_start, existing.t_start),
                )
                if overlap > 0.0:
                    incompatible_pairs = {
                        ("marginalized_input_transition", "explicit_input_likelihood"),
                        ("explicit_input_likelihood", "marginalized_input_transition"),
                        ("reduced_subspace_dynamics", "explicit_input_likelihood"),
                        ("explicit_input_likelihood", "reduced_subspace_dynamics"),
                        ("reduced_subspace_dynamics", "marginalized_input_transition"),
                        ("marginalized_input_transition", "reduced_subspace_dynamics"),
                        ("reduced_subspace_dynamics", "reduced_subspace_dynamics"),
                    }
                    if (factor.factor_type, existing.factor_type) in incompatible_pairs:
                        raise PreconditionError(
                            f"Runtime exclusivity violation: factor ('{factor.name}', type='{factor.factor_type}') "
                            f"and existing factor ('{existing.name}', type='{existing.factor_type}') cannot co-occur "
                            f"on overlapping interval [{max(factor.t_start, existing.t_start)}, {min(factor.t_end, existing.t_end)}]."
                        )

        self._factors.append(factor)

    @property
    def factors(self) -> tuple[EstimationIntervalFactor, ...]:
        return tuple(self._factors)


# ==============================================================================
# Dynamics Provider Protocol & Implementations
# ==============================================================================


@runtime_checkable
class DimeDynamicsProvider(Protocol):
    """Protocol for stateful dynamics providers supporting DIME contracts."""

    @property
    def model_hash(self) -> str: ...

    @property
    def capability(self) -> ProviderCapability: ...

    def get_state(self) -> DimeCompleteState: ...

    def set_state(self, state: DimeCompleteState) -> None: ...

    def snapshot(self) -> ProviderSnapshot: ...

    def restore(self, snapshot: ProviderSnapshot) -> None: ...

    def step(self, request: DimeFullStepRequest) -> DimeFullStepResult: ...

    def compute_zero_input_proposal(
        self, state: DimeCompleteState, duration: float, dt: float
    ) -> DimeZeroInputProposal: ...

    def compute_acceleration_decomposition(
        self, state: DimeCompleteState, controls: np.ndarray
    ) -> AccelerationDecomposition: ...


# Re-export concrete providers from dime_providers
from src.shared.python.estimation.dime_providers import (
    AnalyticPendulumProvider,
    DeterministicFakeProvider,
    UnderactuatedAnalyticProvider,
)

__all__ = [
    "AnalyticPendulumProvider",
    "ContactPolicy",
    "ControlChannelSpec",
    "ControlPhysicalType",
    "DIME_CONTRACTS_VERSION",
    "DeterministicFakeProvider",
    "DimeCompleteState",
    "DimeDynamicsProvider",
    "DimeEstimationResult",
    "DimeFullStepRequest",
    "DimeFullStepResult",
    "DimeObservationWindow",
    "DimeZeroInputProposal",
    "DynamicsProvider",  # noqa: F822 - deprecated alias via module __getattr__
    "EstimationIntervalFactor",
    "ManifoldContract",
    "PassiveLoadSpec",
    "ProviderCapability",
    "ProviderSnapshot",
    "QuaternionManifold",
    "RuntimeExclusivityContract",
    "SE3Manifold",
    "UncertaintyKind",
    "UnderactuatedAnalyticProvider",
    "VectorSpaceManifold",
    "check_qualification_rules",
]


__getattr__ = deprecated_alias_getattr(
    __name__, {"DynamicsProvider": DimeDynamicsProvider}
)
