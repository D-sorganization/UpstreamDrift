"""Backend-agnostic simulation contracts for the golf double-pendulum model.

This module defines the *frozen interface* shared by every simulation backend
(ODE reference, MuJoCo CPU, MuJoCo Warp GPU). Concrete backends live in sibling
modules and are constructed through :mod:`simulation_backends.factory`.

Design notes
------------
* **Program to an interface.** ``SimulationBackend`` is a structural
  :class:`typing.Protocol`; the ODE and MuJoCo backends are interchangeable
  implementations. See ADR-0023.
* **Interface segregation (LOD).** Not every backend can supply every service.
  ``DynamicsProvider`` (mass matrix / bias forces) is implemented only by the
  CPU backends that expose MuJoCo/analytical primitives; ``BatchedBackend`` is
  implemented only by the GPU backend. Callers ``isinstance``-check the *exact*
  capability they need rather than assuming a god-object.
* **One trace schema.** :class:`Trace` / :class:`BatchTrace` are the common
  output of every backend so the analysis layer is backend-agnostic. The HDF5
  serialisation of these lives in :mod:`simulation_backends.trace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityRef,
    capability_level_supported,
    normalize_capability,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Versioned schema stamped into every serialised trace. Bump on any breaking
#: change to the on-disk layout (see :mod:`simulation_backends.trace`).
#: v2.0.0 adds optional /torques, /wrench, /markers, /contacts groups.
#: v2.1.0 adds optional MyoSuite muscle-output datasets.
#: v1.x files are auto-migrated by :func:`simulation_backends.trace_io.read_trace`.
SCHEMA_VERSION = "2.1.0"


@dataclass(frozen=True)
class BackendCapabilities:
    """Static description of what a backend can and cannot do.

    Capability flags let callers branch *without* importing optional GPU
    dependencies or relying on ``hasattr`` probing.

    Attributes:
        name: Stable backend identifier (e.g. ``"ode"``, ``"mujoco"``).
        device: Compute device, ``"cpu"`` or ``"cuda"``.
        supports_batched: Whether ``rollout_batch`` runs many envs in parallel.
        is_differentiable: Whether gradients can flow through ``rollout``.
        provides_dynamics: Whether ``mass_matrix`` / ``bias_forces`` are exposed.
    """

    name: str
    device: str = "cpu"
    supports_batched: bool = False
    is_differentiable: bool = False
    provides_dynamics: bool = False

    def level_for(self, capability: CapabilityRef) -> CapabilityLevel:
        """Return support level for a canonical capability query.

        Legacy boolean flags remain the storage format for backend descriptors;
        this method adapts them to the engine-core taxonomy.
        """
        normalized = normalize_capability(capability)
        if normalized is Capability.FORWARD_SIM:
            return CapabilityLevel.FULL
        if normalized in (Capability.DYNAMICS_PRIMITIVES, Capability.MASS_MATRIX):
            return (
                CapabilityLevel.FULL if self.provides_dynamics else CapabilityLevel.NONE
            )
        if normalized is Capability.BATCHED_ROLLOUT:
            return (
                CapabilityLevel.FULL if self.supports_batched else CapabilityLevel.NONE
            )
        if normalized is Capability.DIFFERENTIABLE_ROLLOUT:
            return (
                CapabilityLevel.FULL if self.is_differentiable else CapabilityLevel.NONE
            )
        return CapabilityLevel.NONE

    def supports(
        self,
        capability: CapabilityRef,
        *,
        minimum: CapabilityLevel = CapabilityLevel.PARTIAL,
    ) -> bool:
        """Return whether ``capability`` is supported at ``minimum`` level."""
        return capability_level_supported(
            self.level_for(capability),
            minimum=minimum,
        )

    def to_capability_map(self) -> dict[Capability, CapabilityLevel]:
        """Return a canonical capability-to-level mapping for this backend."""
        return {capability: self.level_for(capability) for capability in Capability}


@dataclass
class SimState:
    """Generalised configuration of the mechanism at one instant.

    Attributes:
        q: Joint positions, shape ``(nq,)`` [rad].
        v: Joint velocities, shape ``(nv,)`` [rad/s].
        time: Simulation time [s].
    """

    q: np.ndarray
    v: np.ndarray
    time: float = 0.0

    def __post_init__(self) -> None:
        """Coerce to float arrays and validate the precondition ``len(q)>0``."""
        self.q = np.asarray(self.q, dtype=float).reshape(-1)
        self.v = np.asarray(self.v, dtype=float).reshape(-1)
        if self.q.size == 0:
            raise ValueError("SimState.q must be non-empty")
        if self.q.shape != self.v.shape:
            raise ValueError(
                f"q and v must share shape; got {self.q.shape} vs {self.v.shape}"
            )

    @property
    def dim(self) -> int:
        """Number of generalised coordinates."""
        return int(self.q.size)

    def copy(self) -> SimState:
        """Return a deep copy (arrays duplicated) of this state."""
        return SimState(q=self.q.copy(), v=self.v.copy(), time=self.time)


@dataclass
class Trace:
    """Time history of a single rollout — the common backend output.

    Attributes:
        t: Sample times, shape ``(T,)`` [s].
        q: Positions, shape ``(T, nq)`` [rad].
        v: Velocities, shape ``(T, nv)`` [rad/s].
        u: Applied controls, shape ``(T, nu)`` [N*m], or ``None`` if passive.
        dt: Integration step [s].
        backend: Name of the backend that produced the trace.
        meta: Free-form provenance metadata (scalars/strings only).
        torques: Generalised joint forces, shape ``(T, nu)`` [N*m], or ``None``.
        wrench: Contact wrench [fx,fy,fz,tx,ty,tz], shape ``(T, 6)``, or ``None``.
        markers: Predicted marker positions, shape ``(T, n_markers, 3)`` [m], or
            ``None``.
        contacts: Contact point positions, shape ``(T, n_contacts, 3)`` [m], or
            ``None``.
        muscle_names: Names corresponding to muscle-output columns.
        muscle_activations: Muscle activations, shape ``(T, n_muscles)``, or
            ``None``.
        muscle_forces: Muscle forces, shape ``(T, n_muscles)`` [N], or ``None``.
        muscle_lengths: Muscle-tendon lengths, shape ``(T, n_muscles)`` [m], or
            ``None``.
        muscle_velocities: Muscle contraction velocities, shape
            ``(T, n_muscles)`` [m/s], or ``None``.
    """

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    u: np.ndarray | None = None
    dt: float = 0.0
    backend: str = "unknown"
    meta: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    torques: np.ndarray | None = None
    wrench: np.ndarray | None = None
    markers: np.ndarray | None = None
    contacts: np.ndarray | None = None
    muscle_names: tuple[str, ...] = ()
    muscle_activations: np.ndarray | None = None
    muscle_forces: np.ndarray | None = None
    muscle_lengths: np.ndarray | None = None
    muscle_velocities: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate array shapes are mutually consistent (postcondition guard)."""
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.q = np.atleast_2d(np.asarray(self.q, dtype=float))
        self.v = np.atleast_2d(np.asarray(self.v, dtype=float))
        n = self.t.shape[0]
        if self.q.shape[0] != n or self.v.shape[0] != n:
            raise ValueError(
                "Trace arrays disagree on number of timesteps: "
                f"t={n}, q={self.q.shape[0]}, v={self.v.shape[0]}"
            )
        if self.u is not None:
            self.u = np.atleast_2d(np.asarray(self.u, dtype=float))
            if self.u.shape[0] != n:
                raise ValueError(
                    f"control history has {self.u.shape[0]} rows, expected {n}"
                )
        if self.torques is not None:
            self.torques = np.atleast_2d(np.asarray(self.torques, dtype=float))
            if self.torques.shape[0] != n:
                raise ValueError(
                    f"torques has {self.torques.shape[0]} rows, expected {n}"
                )
        if self.wrench is not None:
            self.wrench = np.asarray(self.wrench, dtype=float)
            if self.wrench.ndim != 2 or self.wrench.shape != (n, 6):
                raise ValueError(
                    f"wrench must have shape ({n}, 6), got {self.wrench.shape}"
                )
        if self.markers is not None:
            self.markers = np.asarray(self.markers, dtype=float)
            if (
                self.markers.ndim != 3
                or self.markers.shape[0] != n
                or self.markers.shape[2] != 3
            ):
                raise ValueError(
                    f"markers must have shape ({n}, n_markers, 3), "
                    f"got {self.markers.shape}"
                )
        if self.contacts is not None:
            self.contacts = np.asarray(self.contacts, dtype=float)
            if (
                self.contacts.ndim != 3
                or self.contacts.shape[0] != n
                or self.contacts.shape[2] != 3
            ):
                raise ValueError(
                    f"contacts must have shape ({n}, n_contacts, 3), "
                    f"got {self.contacts.shape}"
                )
        self.muscle_names = tuple(str(name) for name in self.muscle_names)
        self._validate_muscle_history("muscle_activations", n)
        self._validate_muscle_history("muscle_forces", n)
        self._validate_muscle_history("muscle_lengths", n)
        self._validate_muscle_history("muscle_velocities", n)
        muscle_arrays = [
            arr
            for arr in (
                self.muscle_activations,
                self.muscle_forces,
                self.muscle_lengths,
                self.muscle_velocities,
            )
            if arr is not None
        ]
        if muscle_arrays:
            n_muscles = muscle_arrays[0].shape[1]
            if any(arr.shape[1] != n_muscles for arr in muscle_arrays):
                raise ValueError("all muscle-output arrays must have same columns")
            if self.muscle_names and len(self.muscle_names) != n_muscles:
                raise ValueError(
                    f"muscle_names has {len(self.muscle_names)} entries but "
                    f"muscle histories have {n_muscles} columns"
                )

    def _validate_muscle_history(self, attr_name: str, n: int) -> None:
        """Validate an optional ``(T, n_muscles)`` muscle-output array."""
        value = getattr(self, attr_name)
        if value is None:
            return
        array = np.atleast_2d(np.asarray(value, dtype=float))
        if array.ndim != 2 or array.shape[0] != n:
            raise ValueError(
                f"{attr_name} must have shape ({n}, n_muscles), got {array.shape}"
            )
        setattr(self, attr_name, array)

    @property
    def num_steps(self) -> int:
        """Number of recorded timesteps ``T``."""
        return int(self.t.shape[0])

    def final_state(self) -> SimState:
        """Return the last recorded state as a :class:`SimState`."""
        return SimState(q=self.q[-1], v=self.v[-1], time=float(self.t[-1]))


@dataclass
class BatchTrace:
    """Time history of ``N`` rollouts evaluated in parallel.

    Attributes:
        t: Shared sample times, shape ``(T,)`` [s].
        q: Positions, shape ``(N, T, nq)`` [rad].
        v: Velocities, shape ``(N, T, nv)`` [rad/s].
        u: Controls, shape ``(N, T, nu)`` [N*m], or ``None`` if passive.
        dt: Integration step [s].
        backend: Name of the backend that produced the batch.
        meta: Free-form provenance metadata.
    """

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    u: np.ndarray | None = None
    dt: float = 0.0
    backend: str = "unknown"
    meta: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate ``q``/``v`` are rank-3 and agree on ``(N, T)``."""
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.q = np.asarray(self.q, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        if self.q.ndim != 3 or self.v.ndim != 3:
            raise ValueError(
                "BatchTrace q and v must be rank-3 (N, T, dim); "
                f"got {self.q.ndim}D and {self.v.ndim}D"
            )
        if self.q.shape[:2] != self.v.shape[:2]:
            raise ValueError("BatchTrace q and v must agree on (N, T)")
        if self.q.shape[1] != self.t.shape[0]:
            raise ValueError("BatchTrace time axis disagrees with t length")

    @property
    def num_envs(self) -> int:
        """Number of parallel environments ``N``."""
        return int(self.q.shape[0])

    @property
    def num_steps(self) -> int:
        """Number of recorded timesteps ``T``."""
        return int(self.t.shape[0])

    def env(self, index: int) -> Trace:
        """Extract a single environment's history as a :class:`Trace`.

        Args:
            index: Environment index in ``[0, num_envs)``.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < self.num_envs:
            raise IndexError(f"env index {index} out of range [0, {self.num_envs})")
        u_i = None if self.u is None else self.u[index]
        return Trace(
            t=self.t,
            q=self.q[index],
            v=self.v[index],
            u=u_i,
            dt=self.dt,
            backend=self.backend,
            meta=dict(self.meta),
        )


@runtime_checkable
class SimulationBackend(Protocol):
    """Structural contract every simulation backend satisfies.

    Implementations are interchangeable: the ODE reference backend and the
    MuJoCo/MJWarp backends all honour this Protocol so the analysis layer never
    depends on a concrete class.
    """

    @property
    def capabilities(self) -> BackendCapabilities:
        """Static capability description for this backend instance."""
        ...

    def reset(self, state: SimState | None = None) -> None:
        """Reset to ``state`` (or the canonical initial state if ``None``)."""
        ...

    def step(self, dt: float | None = None) -> None:
        """Advance the simulation by one step of size ``dt``."""
        ...

    def get_state(self) -> SimState:
        """Return the current :class:`SimState`."""
        ...

    def set_control(self, u: np.ndarray) -> None:
        """Set the generalised control/torque vector for subsequent steps."""
        ...

    def get_time(self) -> float:
        """Return the current simulation time [s]."""
        ...

    def forward_dynamics(
        self, q: np.ndarray, v: np.ndarray, u: np.ndarray | None = None
    ) -> np.ndarray:
        """Return joint accelerations ``qacc`` for state ``(q, v)`` under ``u``."""
        ...

    def rollout(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
    ) -> Trace:
        """Integrate ``horizon`` steps and return the recorded :class:`Trace`.

        Args:
            controls: Prescribed control history ``(horizon, nu)``, or ``None``
                for a passive (zero-torque) rollout.
            horizon: Number of steps to integrate (``> 0``).
            dt: Integration step size [s] (``> 0``).
        """
        ...


@runtime_checkable
class DynamicsProvider(Protocol):
    """Optional contract for backends exposing analytical dynamics primitives.

    Implemented by the ODE reference backend and the MuJoCo CPU backend, whose
    ``mass_matrix`` and ``bias_forces`` provide an *independent derivation* of
    the equations of motion for cross-validation (see M5/M7 in the epic).
    """

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return the dense joint-space inertia matrix ``M(q)``, shape ``(n, n)``."""
        ...

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return bias forces ``C(q,v)v + g(q) (+ damping)``, shape ``(n,)``."""
        ...


@runtime_checkable
class BatchedBackend(Protocol):
    """Optional contract for backends that evaluate many rollouts in parallel."""

    def rollout_batch(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
        num_envs: int,
    ) -> BatchTrace:
        """Integrate ``num_envs`` rollouts and return a :class:`BatchTrace`.

        Args:
            controls: Either ``None`` (passive), a shared ``(horizon, nu)``
                history applied to every env, or a per-env
                ``(num_envs, horizon, nu)`` history.
            horizon: Number of steps to integrate (``> 0``).
            dt: Integration step size [s] (``> 0``).
            num_envs: Number of parallel environments (``> 0``).
        """
        ...
