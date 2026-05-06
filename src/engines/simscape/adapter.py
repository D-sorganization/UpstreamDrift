"""Protocol-compliant skeleton for :class:`SimscapeAdapter`.

This module implements every method of
:class:`src.shared.python.engine_core.interfaces.PhysicsEngine`, satisfying
each composing sub-protocol (``Loadable``, ``Steppable``, ``Queryable``,
``DynamicsComputable``, ``CounterfactualComputable``, ``Recordable``,
``Checkpointable``).

Design notes
------------
- **No MATLAB at import time.** Module import is side-effect free; this
  file can be imported on hosts without a MATLAB licence and without
  ``matlabengine`` installed. Only :meth:`SimscapeAdapter.step` and
  :meth:`SimscapeAdapter.simulate_with_coefficients` would ever require
  the engine, and those methods raise ``NotImplementedError`` in this
  skeleton (deferred to issue #4006).
- **Lifecycle.** State transitions are guarded by
  :class:`src.engines.simscape._lifecycle.LifecycleGuard`. Methods that
  read state require ``LOADED`` or ``RUNNING``; methods that mutate state
  perform an explicit transition through the guard.
- **Metadata-only load.** ``load_from_path`` confirms the .slx file
  exists, confirms the sibling ``PolynomialInputValues.mat`` metadata
  file exists, and synthesises joint metadata from a hard-coded constant
  list (the GolfSwing3D_Kinetic model has 16 polynomial joints; cross-
  referenced with ``getPolynomialParameterInfo.m`` and
  ``option4_python_bridge/TESTING.md``). The real MATLAB-backed metadata
  query lands with #4006.
- **DRY.** Method bodies that genuinely depend on MATLAB delegate to the
  private :meth:`_deferred` helper, which raises the canonical
  ``NotImplementedError`` with a stable message format.
- **Law of Demeter.** The adapter exposes delegating helpers
  (``model_loaded``, ``state_summary``) rather than letting callers reach
  into ``_lifecycle``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.simscape._errors import (
    SimscapeModelNotFoundError,
    SimscapeStateError,
)
from src.engines.simscape._lifecycle import AdapterState, LifecycleGuard
from src.shared.python.core.contracts import (
    invariant,
    postcondition,
    precondition,
)
from src.shared.python.engine_core.checkpoint import StateCheckpoint
from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "SimscapeAdapter",
]


# ---------------------------------------------------------------------------
# Model-metadata constants
#
# Source: ``getPolynomialParameterInfo.m`` reads ``PolynomialInputValues.mat``
# and groups variables ending in A..G into joint blocks of 7 coefficients.
# The Simulink ``GolfSwing3D_Kinetic`` model carries 16 such polynomial
# joints, confirmed by ``option4_python_bridge/TESTING.md`` line 409.
# ---------------------------------------------------------------------------
_GOLFSWING3D_JOINT_NAMES: tuple[str, ...] = (
    "Hip",
    "Spine",
    "Torso",
    "LScap",
    "LShoulder",
    "LElbow",
    "LForearm",
    "LWrist",
    "RScap",
    "RShoulder",
    "RElbow",
    "RForearm",
    "RWrist",
    "Neck",
    "LHand",
    "RHand",
)
"""Canonical 16-joint name list for GolfSwing3D_Kinetic (placeholder until #4006)."""

_DEFERRED_MESSAGE = "deferred to #4006"

_METADATA_FILENAME = "PolynomialInputValues.mat"


@invariant(
    lambda self: not self.model_loaded or self.model_name != "",
    "if a model is loaded, model_name must be non-empty",
)
@invariant(
    lambda self: self._cache_max_entries >= 0,
    "cache_max_entries must be non-negative (0 = disabled)",
)
class SimscapeAdapter:
    """Protocol-compliant skeleton for the Simscape Multibody adapter.

    The adapter satisfies the full
    :class:`src.shared.python.engine_core.interfaces.PhysicsEngine` protocol
    (which composes ``SimulationInterface``, ``DynamicsInterface``, and
    ``Checkpointable``). Methods that require MATLAB to do useful work
    raise :class:`NotImplementedError` with the message
    ``"deferred to #4006"`` after a successful lifecycle check, so the
    state-machine and protocol-conformance tests pass while the
    simulation-bound work documents itself as outstanding.

    Args:
        rng_seed: Non-negative seed propagated to the MATLAB-side model on
            every simulate call once #4006 lands. Stored only.
        cache_enabled: Whether to honour the result cache once it is wired
            up. Stored only.
        cache_max_entries: LRU cache capacity (``0`` disables caching).
        startup_timeout_s: Engine-startup deadline; stored only.

    Raises:
        TypeError: If any argument has the wrong type.
        ValueError: If ``rng_seed`` or ``cache_max_entries`` is negative.

    Example:
        >>> adapter = SimscapeAdapter()
        >>> adapter.model_name
        ''
        >>> adapter.close()
    """

    @precondition(
        lambda self,
        rng_seed=42,
        cache_enabled=True,
        cache_max_entries=1024,
        startup_timeout_s=60.0: (isinstance(rng_seed, int) and rng_seed >= 0),
        "rng_seed must be a non-negative int",
    )
    @precondition(
        lambda self,
        rng_seed=42,
        cache_enabled=True,
        cache_max_entries=1024,
        startup_timeout_s=60.0: (
            isinstance(cache_max_entries, int) and cache_max_entries >= 0
        ),
        "cache_max_entries must be a non-negative int",
    )
    def __init__(
        self,
        rng_seed: int = 42,
        cache_enabled: bool = True,
        cache_max_entries: int = 1024,
        startup_timeout_s: float = 60.0,
    ) -> None:
        self._rng_seed: int = int(rng_seed)
        self._cache_enabled: bool = bool(cache_enabled)
        self._cache_max_entries: int = int(cache_max_entries)
        self._startup_timeout_s: float = float(startup_timeout_s)

        self._lifecycle: LifecycleGuard = LifecycleGuard()
        self._model_name: str = ""
        self._model_path: Path | None = None
        self._joint_names: tuple[str, ...] = ()
        self._dof: int = 0
        self._sim_time: float = 0.0
        self._control: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Convenience / introspection (LoD-friendly delegators)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Stable engine identifier used by registries and tests."""
        return "simscape_3d"

    @property
    def dof(self) -> int:
        """Number of generalised coordinates (``0`` until a model is loaded).

        Raises:
            SimscapeStateError: If queried before ``load_from_path``.
        """
        if not self.model_loaded:
            raise SimscapeStateError(
                "dof",
                current_state=self._lifecycle.state.value,
                required_state="loaded|running",
            )
        return self._dof

    @property
    def model_loaded(self) -> bool:
        """Return ``True`` if a model has been loaded successfully."""
        return self._lifecycle.is_loaded()

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Return the joint-name tuple discovered at load time."""
        return self._joint_names

    def state_summary(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the adapter's lifecycle state.

        Used by tests, debug tooling, and the eventual loader integration in
        issue #038. Does not leak filesystem paths.
        """
        return {
            "name": self.name,
            "lifecycle": self._lifecycle.state.value,
            "model_loaded": self.model_loaded,
            "dof": self._dof,
            "sim_time": self._sim_time,
            "rng_seed": self._rng_seed,
            "cache_enabled": self._cache_enabled,
            "cache_max_entries": self._cache_max_entries,
        }

    # ------------------------------------------------------------------
    # SimulationInterface / Loadable
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the loaded model name, or ``""`` if no model is loaded."""
        return self._model_name

    @precondition(
        lambda self, path: isinstance(path, str) and path != "",
        "path must be a non-empty string",
    )
    @postcondition(
        lambda result: result is None,
        "load_from_path returns None",
    )
    def load_from_path(self, path: str) -> None:
        """Load a Simulink ``.slx`` model and read joint metadata.

        The skeleton does **not** start a MATLAB Engine. It instead
        verifies that the .slx and its sibling
        ``PolynomialInputValues.mat`` are present on disk and synthesises
        the joint-name list from a constant validated against
        ``getPolynomialParameterInfo.m``. The real metadata round-trip
        ships with #4006.

        Args:
            path: Path to a Simulink ``.slx`` model.

        Raises:
            ValueError: If the file extension is not ``.slx``.
            SimscapeModelNotFoundError: If the model file or its metadata
                sibling is missing.
            SimscapeStateError: If called after ``close``.
        """
        slx_path = Path(path)
        if slx_path.suffix.lower() != ".slx":
            raise ValueError(
                f"Simscape adapter only loads .slx models, got '{slx_path.suffix}'"
            )

        if not slx_path.exists():
            raise SimscapeModelNotFoundError(slx_path.name, reason="file not found")

        metadata_path = slx_path.parent / _METADATA_FILENAME
        if not metadata_path.exists():
            raise SimscapeModelNotFoundError(
                metadata_path.name,
                reason="metadata sibling required for skeleton load",
            )

        # Skeleton metadata: hard-coded joint list, validated by tests.
        self._joint_names = _GOLFSWING3D_JOINT_NAMES
        self._dof = len(self._joint_names)
        self._model_name = slx_path.stem
        self._model_path = slx_path
        self._sim_time = 0.0

        self._lifecycle.transition(AdapterState.LOADED, operation="load_from_path")
        logger.info("simscape adapter loaded model %s (skeleton)", self._model_name)

    def load_from_string(  # noqa: ARG002 - signature mandated by protocol
        self,
        content: str,
        extension: str | None = None,
    ) -> None:
        """Not supported. .slx is a binary archive; raises ``NotImplementedError``.

        We deliberately do not gate this with ``@precondition`` because we
        always want callers to see the explicit ``NotImplementedError`` —
        a precondition violation would obscure the intent.
        """
        raise NotImplementedError(
            "SimscapeAdapter cannot load from a string — .slx is binary. "
            "Use load_from_path with an .slx file on disk."
        )

    # ------------------------------------------------------------------
    # Steppable
    # ------------------------------------------------------------------

    @postcondition(
        lambda result: result is None,
        "reset returns None",
    )
    def reset(self) -> None:
        """Reset the simulation clock to zero.

        Raises:
            SimscapeStateError: If called before a model is loaded.
        """
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="reset",
        )
        self._sim_time = 0.0
        self._control = None
        self._lifecycle.transition(AdapterState.LOADED, operation="reset")

    @precondition(
        lambda self, dt=None: dt is None or (isinstance(dt, float) and dt > 0),
        "dt must be a positive float or None",
    )
    def step(self, dt: float | None = None) -> None:  # noqa: ARG002
        """Advance the simulation by ``dt`` seconds (deferred to #4006).

        The lifecycle check still runs so that
        ``test_step_before_load_raises_state_error`` can verify the
        state-machine guard fires before any MATLAB call would have
        happened.

        Raises:
            SimscapeStateError: If no model is loaded.
            NotImplementedError: After lifecycle checks pass — the MATLAB
                Engine call is deferred to issue #4006.
        """
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="step",
        )
        self._lifecycle.transition(AdapterState.RUNNING, operation="step")
        self._deferred("step")

    def forward(self) -> None:
        """Compute kinematics/dynamics without advancing time (deferred)."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="forward",
        )
        self._deferred("forward")

    # ------------------------------------------------------------------
    # Queryable
    # ------------------------------------------------------------------

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return zero-valued (q, v) of length ``dof`` (deferred to #4006).

        The skeleton returns trivial zero arrays of the correct shape so
        that downstream code can wire the protocol; the real Simscape
        state-extraction lands with #4006.
        """
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="get_state",
        )
        return np.zeros(self._dof, dtype=np.float64), np.zeros(
            self._dof, dtype=np.float64
        )

    @precondition(
        lambda self, q, v: bool(
            isinstance(q, np.ndarray)
            and isinstance(v, np.ndarray)
            and q.ndim == 1
            and v.ndim == 1
            and np.all(np.isfinite(q))
            and np.all(np.isfinite(v))
        ),
        "q and v must be finite 1-D numpy arrays",
    )
    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:  # noqa: ARG002
        """Set the simulation state (deferred to #4006)."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="set_state",
        )
        self._deferred("set_state")

    @precondition(
        lambda self, u: bool(
            isinstance(u, np.ndarray) and u.ndim == 1 and np.all(np.isfinite(u))
        ),
        "u must be a finite 1-D numpy array",
    )
    def set_control(self, u: np.ndarray) -> None:
        """Store control vector ``u`` for the next ``step`` call."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="set_control",
        )
        self._control = np.asarray(u, dtype=np.float64).copy()

    @postcondition(lambda result: result >= 0.0, "time is non-negative")
    def get_time(self) -> float:
        """Return current simulation time (always ``0.0`` in the skeleton)."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="get_time",
        )
        return float(self._sim_time)

    def get_joint_names(self) -> list[str]:
        """Return joint names discovered at load time (or empty before load)."""
        return list(self._joint_names)

    def get_full_state(self) -> dict[str, Any]:
        """Return q, v, t, M in a single batched call.

        ``M`` is ``None`` until #4006 wires the MATLAB-side mass matrix.
        """
        q, v = self.get_state()
        return {"q": q, "v": v, "t": self.get_time(), "M": None}

    def get_capabilities(self) -> Any:
        """Report capabilities; the skeleton declares NONE for everything.

        The lazy import keeps ``capabilities`` out of the module-import
        graph so this file stays free of optional heavy deps.
        """
        from src.shared.python.engine_core.capabilities import (
            CapabilityLevel,
            EngineCapabilities,
        )

        return EngineCapabilities(
            engine_name=self.name,
            mass_matrix=CapabilityLevel.NONE,
            jacobian=CapabilityLevel.NONE,
            contact_forces=CapabilityLevel.NONE,
            inverse_dynamics=CapabilityLevel.NONE,
            drift_acceleration=CapabilityLevel.NONE,
        )

    # ------------------------------------------------------------------
    # DynamicsInterface / DynamicsComputable
    # ------------------------------------------------------------------

    def compute_mass_matrix(self) -> np.ndarray:
        """Mass matrix M(q) — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="compute_mass_matrix",
        )
        return self._deferred_array("compute_mass_matrix")

    def compute_bias_forces(self) -> np.ndarray:
        """Bias forces C(q,v)v + g(q) — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="compute_bias_forces",
        )
        return self._deferred_array("compute_bias_forces")

    def compute_gravity_forces(self) -> np.ndarray:
        """Gravity g(q) — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="compute_gravity_forces",
        )
        return self._deferred_array("compute_gravity_forces")

    @precondition(
        lambda self, qacc: isinstance(qacc, np.ndarray) and qacc.ndim == 1,
        "qacc must be a 1-D numpy array",
    )
    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Inverse dynamics — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="compute_inverse_dynamics",
        )
        return self._deferred_array("compute_inverse_dynamics")

    @precondition(
        lambda self, body_name: isinstance(body_name, str) and body_name != "",
        "body_name must be a non-empty string",
    )
    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:  # noqa: ARG002
        """Body Jacobian — returns ``None`` for unknown bodies (skeleton)."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="compute_jacobian",
        )
        # Skeleton: no body table populated yet.
        return None

    def compute_drift_acceleration(self) -> np.ndarray:
        """Section F drift acceleration — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="compute_drift_acceleration",
        )
        return self._deferred_array("compute_drift_acceleration")

    @precondition(
        lambda self, tau: isinstance(tau, np.ndarray) and tau.ndim == 1,
        "tau must be a 1-D numpy array",
    )
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Section F control acceleration — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="compute_control_acceleration",
        )
        return self._deferred_array("compute_control_acceleration")

    def compute_contact_forces(self) -> np.ndarray:
        """Return ground-reaction force vector (skeleton: zero vector)."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="compute_contact_forces",
        )
        return np.zeros(3, dtype=np.float64)

    def set_shaft_properties(  # noqa: ARG002
        self,
        length: float,
        EI_profile: np.ndarray,
        mass_profile: np.ndarray,
        damping_ratio: float = 0.02,
    ) -> bool:
        """Flexible-shaft configuration (not supported by Simscape skeleton)."""
        return False

    def get_shaft_state(self) -> dict[str, np.ndarray] | None:
        """Flexible-shaft state (not supported by Simscape skeleton)."""
        return None

    # ------------------------------------------------------------------
    # CounterfactualComputable
    # ------------------------------------------------------------------

    @precondition(
        lambda self, q, v: (
            isinstance(q, np.ndarray)
            and isinstance(v, np.ndarray)
            and q.ndim == 1
            and v.ndim == 1
        ),
        "q and v must be 1-D numpy arrays",
    )
    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Zero-Torque Counterfactual — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="compute_ztcf",
        )
        return self._deferred_array("compute_ztcf")

    @precondition(
        lambda self, q: isinstance(q, np.ndarray) and q.ndim == 1,
        "q must be a 1-D numpy array",
    )
    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Zero-Velocity Counterfactual — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="compute_zvcf",
        )
        return self._deferred_array("compute_zvcf")

    # ------------------------------------------------------------------
    # Recordable
    # ------------------------------------------------------------------

    @precondition(
        lambda self, field_name: isinstance(field_name, str) and field_name != "",
        "field_name must be a non-empty string",
    )
    def get_time_series(  # noqa: ARG002
        self, field_name: str
    ) -> tuple[np.ndarray, np.ndarray | list[Any]]:
        """Time-series getter — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="get_time_series",
        )
        self._deferred("get_time_series")
        # Unreachable; satisfies mypy return-type analysis.
        return np.zeros(0), np.zeros(0)

    def get_induced_acceleration_series(  # noqa: ARG002
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Induced-acceleration getter — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="get_induced_acceleration_series",
        )
        self._deferred("get_induced_acceleration_series")
        return np.zeros(0), np.zeros(0)

    @precondition(
        lambda self, config: isinstance(config, dict),
        "config must be a dict",
    )
    def set_analysis_config(self, config: dict[str, Any]) -> None:  # noqa: ARG002
        """Toggle Simscape logging fields — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="set_analysis_config",
        )
        self._deferred("set_analysis_config")

    # ------------------------------------------------------------------
    # System-identification convenience surface (sim2real consumer)
    # ------------------------------------------------------------------

    def get_link_masses(self) -> np.ndarray:
        """Return current link masses — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="get_link_masses",
        )
        return self._deferred_array("get_link_masses")

    @precondition(
        lambda self, masses: (
            isinstance(masses, np.ndarray)
            and masses.ndim == 1
            and bool(np.all(masses > 0))
        ),
        "masses must be a strictly-positive 1-D numpy array",
    )
    def set_link_masses(self, masses: np.ndarray) -> None:  # noqa: ARG002
        """Set link masses — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="set_link_masses",
        )
        self._deferred("set_link_masses")

    def get_joint_damping(self) -> np.ndarray:
        """Return per-joint damping — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="get_joint_damping",
        )
        return self._deferred_array("get_joint_damping")

    @precondition(
        lambda self, damping: (
            isinstance(damping, np.ndarray)
            and damping.ndim == 1
            and bool(np.all(damping >= 0))
        ),
        "damping must be a non-negative 1-D numpy array",
    )
    def set_joint_damping(self, damping: np.ndarray) -> None:  # noqa: ARG002
        """Set per-joint damping — deferred to #4006."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="set_joint_damping",
        )
        self._deferred("set_joint_damping")

    # ------------------------------------------------------------------
    # Headline motion-matching method
    # ------------------------------------------------------------------

    @precondition(
        lambda self, coeffs: (
            isinstance(coeffs, np.ndarray)
            and coeffs.ndim == 1
            and coeffs.size > 0
            and coeffs.size % 7 == 0
            and bool(np.all(np.isfinite(coeffs)))
        ),
        "coeffs must be a finite 1-D numpy array of length n_joints*7",
    )
    def simulate_with_coefficients(self, coeffs: np.ndarray) -> Any:  # noqa: ARG002
        """Run one Simscape simulation — deferred to #4006.

        After a successful lifecycle check, raises ``NotImplementedError``
        so that callers can verify the protocol contract today and the
        full implementation lands cleanly with the next issue.
        """
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="simulate_with_coefficients",
        )
        self._deferred("simulate_with_coefficients")

    # ------------------------------------------------------------------
    # Checkpointable
    # ------------------------------------------------------------------

    @property
    def engine_type(self) -> str:
        """Engine identifier matching :class:`EngineType.MATLAB_3D`."""
        return EngineType.MATLAB_3D.value

    def save_checkpoint(self) -> StateCheckpoint:
        """Return a ``StateCheckpoint`` of the current adapter state."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="save_checkpoint",
        )
        q, v = self.get_state()
        return StateCheckpoint.create(
            engine_type=self.engine_type,
            engine_state={
                "model_name": self._model_name,
                "lifecycle": self._lifecycle.state.value,
            },
            q=q,
            v=v,
            timestamp=self._sim_time,
        )

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Restore from a ``StateCheckpoint`` (skeleton: time + sentinel only)."""
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="restore_checkpoint",
        )
        if checkpoint.engine_type != self.engine_type:
            raise ValueError(
                f"checkpoint engine_type '{checkpoint.engine_type}' "
                f"does not match adapter '{self.engine_type}'"
            )
        self._sim_time = float(checkpoint.timestamp)

    # ------------------------------------------------------------------
    # Lifecycle close + context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the adapter; idempotent."""
        if self._lifecycle.is_stopped():
            return
        self._lifecycle.transition(AdapterState.STOPPED, operation="close")
        self._control = None

    def __enter__(self) -> SimscapeAdapter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Dunder + private helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a path-free representation safe for logs.

        ``test_repr_does_not_leak_paths`` asserts that this never embeds
        the absolute model path — only the model basename — to keep logs
        portable across hosts and CI runners.
        """
        model = self._model_name or "<unloaded>"
        return (
            f"SimscapeAdapter(name={self.name!r}, "
            f"model={model!r}, "
            f"state={self._lifecycle.state.value!r}, "
            f"dof={self._dof})"
        )

    @staticmethod
    def _deferred(operation: str) -> None:
        """Raise the canonical deferred-work error for ``operation``."""
        raise NotImplementedError(f"SimscapeAdapter.{operation} {_DEFERRED_MESSAGE}")

    @staticmethod
    def _deferred_array(operation: str) -> np.ndarray:
        """Like :meth:`_deferred` but typed as returning ``np.ndarray``.

        Always raises; the return annotation exists so mypy's flow
        analysis on dynamics methods stays happy.
        """
        SimscapeAdapter._deferred(operation)
        # Unreachable — satisfies mypy.
        return np.zeros(0)


def _matlab_engine_available() -> bool:
    """Return ``True`` if ``matlab.engine`` is importable.

    Used by tests and by the eventual loader integration in #038. Kept at
    module scope to avoid pulling MATLAB into ``__init__``; the function
    itself is import-time safe.
    """
    if os.environ.get("UD_SIMSCAPE_FORCE_NO_MATLAB") == "1":
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("matlab") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False
