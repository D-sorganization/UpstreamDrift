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

from src.engines.simscape._cache import _ResultCache, make_cache_key
from src.engines.simscape._errors import (
    SimscapeModelNotFoundError,
    SimscapeNotInstalledError,
    SimscapeSimulationError,
    SimscapeStateError,
)
from src.engines.simscape._lifecycle import AdapterState, LifecycleGuard
from src.engines.simscape._output import SimscapeOutput
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

        # Lazy MATLAB-engine handle and result cache. The cache may
        # legitimately have ``capacity == 0`` (caching disabled).
        self._engine: Any | None = None
        self._cache: _ResultCache[SimscapeOutput] = _ResultCache(
            capacity=cache_max_entries if cache_enabled else 0,
        )
        self._matlab_version: str = ""

    # ------------------------------------------------------------------
    # Concurrency / pool note
    # ------------------------------------------------------------------
    # **Single MATLAB engine per process.** ``_engine`` is initialised
    # via :func:`src.engines.simscape._engine_pool.get_shared_engine`
    # on the first MATLAB-bound call. A pool with multiple engines and
    # work-stealing is tracked in issue #4008 / #039. Until then, every
    # adapter in the same process shares the singleton, and the adapter
    # itself does not run MATLAB calls in parallel.

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

        On the first call this also lazily starts the shared MATLAB
        engine via
        :func:`src.engines.simscape._engine_pool.get_shared_engine`.
        When MATLAB is unavailable the adapter falls back to skeleton
        behaviour (existence-check on the .slx + metadata sibling, plus
        the hard-coded :data:`_GOLFSWING3D_JOINT_NAMES`) so that purely
        offline workflows continue to function.

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

        self._model_name = slx_path.stem
        self._model_path = slx_path
        self._sim_time = 0.0

        # Try to start the shared MATLAB engine and pull live joint
        # metadata. If MATLAB is unavailable we still load the model in
        # skeleton mode using the hard-coded joint list.
        from src.engines.simscape._engine_pool import (
            get_shared_engine,
            is_matlab_available,
        )

        if is_matlab_available():
            self._engine = get_shared_engine(startup_timeout_s=self._startup_timeout_s)
            self._load_matlab_model(slx_path)
            self._joint_names, self._dof = self._fetch_joint_metadata()
        else:
            self._joint_names = _GOLFSWING3D_JOINT_NAMES
            self._dof = len(self._joint_names)

        self._lifecycle.transition(AdapterState.LOADED, operation="load_from_path")
        logger.info("simscape adapter loaded model %s", self._model_name)

    def _load_matlab_model(self, slx_path: Path) -> None:
        """Load the .slx into the shared MATLAB engine workspace.

        Adds the model directory to the MATLAB path (so sibling
        ``getPolynomialParameterInfo.m`` resolves) and invokes
        ``load_system`` on the model name.

        Raises:
            SimscapeSimulationError: If the MATLAB-side load fails.
        """
        if self._engine is None:  # pragma: no cover - defensive
            raise SimscapeSimulationError(
                "internal error: _engine is None inside _load_matlab_model"
            )
        eng = self._engine
        try:
            eng.addpath(str(slx_path.parent), nargout=0)
            eng.load_system(self._model_name, nargout=0)
            self._matlab_version = str(eng.version(nargout=1))
        except Exception as exc:  # noqa: BLE001 - wrap MATLAB error
            err_id = getattr(exc, "MatlabError", "") or ""
            raise SimscapeSimulationError(
                f"MATLAB failed to load model '{self._model_name}': {exc}",
                matlab_error_id=err_id,
            ) from exc

    def _fetch_joint_metadata(self) -> tuple[tuple[str, ...], int]:
        """Query ``getPolynomialParameterInfo`` for joint names + dof.

        The MATLAB function returns a struct with field ``JointNames``
        (cellstr) and ``NumJoints`` (double). We coerce them to
        Python-native types here.

        Returns:
            ``(joint_names, dof)`` tuple.

        Raises:
            SimscapeSimulationError: If the MATLAB call fails or the
                returned struct is malformed.
        """
        if self._engine is None:  # pragma: no cover - defensive
            raise SimscapeSimulationError(
                "internal error: _engine is None inside _fetch_joint_metadata"
            )
        try:
            info = self._engine.getPolynomialParameterInfo(nargout=1)
        except Exception as exc:  # noqa: BLE001 - wrap MATLAB error
            err_id = getattr(exc, "MatlabError", "") or ""
            raise SimscapeSimulationError(
                f"getPolynomialParameterInfo failed: {exc}",
                matlab_error_id=err_id,
            ) from exc

        try:
            names_raw = info["JointNames"]
            num_joints = int(info["NumJoints"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SimscapeSimulationError(
                f"unexpected getPolynomialParameterInfo struct: {exc}"
            ) from exc

        joint_names = tuple(str(n) for n in names_raw)
        return joint_names, num_joints

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
    def step(self, dt: float | None = None) -> None:
        """Advance the simulation by ``dt`` seconds.

        The MATLAB engine is required. We invoke ``sim()`` over a short
        horizon (``dt`` seconds, default 1 ms) and update the internal
        clock from the returned logsout.

        Raises:
            SimscapeStateError: If no model is loaded.
            SimscapeNotInstalledError: If MATLAB is unavailable.
            SimscapeSimulationError: On any MATLAB-side failure.
        """
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="step",
        )
        self._lifecycle.transition(AdapterState.RUNNING, operation="step")

        from src.engines.simscape._engine_pool import (
            get_shared_engine,
            is_matlab_available,
        )

        if self._engine is None:
            if not is_matlab_available():
                raise SimscapeNotInstalledError("step requires MATLAB Engine")
            self._engine = get_shared_engine(startup_timeout_s=self._startup_timeout_s)

        horizon = float(dt) if dt is not None else 1e-3
        try:
            self._engine.set_param(
                self._model_name,
                "StopTime",
                str(self._sim_time + horizon),
                nargout=0,
            )
            self._engine.sim(self._model_name, nargout=0)
        except Exception as exc:  # noqa: BLE001 - wrap MATLAB error
            err_id = getattr(exc, "MatlabError", "") or ""
            raise SimscapeSimulationError(
                f"step({horizon}) failed: {exc}",
                matlab_error_id=err_id,
            ) from exc
        self._sim_time += horizon

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
        """Return current ``(q, v)`` from the Simscape state vector.

        When MATLAB is unavailable (skeleton mode) we return zeros of
        the correct shape. With MATLAB present we read the model
        operating point via ``Simulink.SimulationData.Dataset``.
        """
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="get_state",
        )
        if self._engine is None:
            return (
                np.zeros(self._dof, dtype=np.float64),
                np.zeros(self._dof, dtype=np.float64),
            )
        try:
            op = self._engine.eval(
                f"Simulink.BlockDiagram.getInitialState('{self._model_name}')",
                nargout=1,
            )
            q = np.asarray(self._engine.getfield(op, "q"), dtype=np.float64)
            v = np.asarray(self._engine.getfield(op, "v"), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - wrap MATLAB error
            err_id = getattr(exc, "MatlabError", "") or ""
            raise SimscapeSimulationError(
                f"get_state failed: {exc}",
                matlab_error_id=err_id,
            ) from exc
        return q.reshape(-1), v.reshape(-1)

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
    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set the simulation state on the Simscape model.

        Builds a ``Simulink.SimulationData.Dataset`` whose two elements
        carry ``q`` and ``v`` respectively, then writes it back to the
        model workspace via ``setVariable``.

        Raises:
            ValueError: If ``q`` or ``v`` has length other than ``dof``.
            SimscapeNotInstalledError: If MATLAB is unavailable.
            SimscapeSimulationError: On any MATLAB-side failure.
        """
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="set_state",
        )
        if q.shape != (self._dof,) or v.shape != (self._dof,):
            raise ValueError(
                f"q and v must have shape ({self._dof},); got q={q.shape}, v={v.shape}"
            )
        if self._engine is None:
            raise SimscapeNotInstalledError("set_state requires MATLAB Engine")

        from src.engines.simscape._simscape_io import _to_matlab_double

        try:
            self._engine.set_param(
                self._model_name, "LoadInitialState", "on", nargout=0
            )
            self._engine.assignin("base", "ud_q", _to_matlab_double(q), nargout=0)
            self._engine.assignin("base", "ud_v", _to_matlab_double(v), nargout=0)
            self._engine.eval(
                "ud_state = Simulink.SimulationData.Dataset; "
                "ud_state = ud_state.addElement(timeseries(ud_q,0),'q'); "
                "ud_state = ud_state.addElement(timeseries(ud_v,0),'v');",
                nargout=0,
            )
            self._engine.set_param(
                self._model_name, "InitialState", "ud_state", nargout=0
            )
        except Exception as exc:  # noqa: BLE001 - wrap MATLAB error
            err_id = getattr(exc, "MatlabError", "") or ""
            raise SimscapeSimulationError(
                f"set_state failed: {exc}",
                matlab_error_id=err_id,
            ) from exc

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
    @postcondition(
        lambda result: isinstance(result, SimscapeOutput),
        "simulate_with_coefficients must return a SimscapeOutput",
    )
    def simulate_with_coefficients(
        self,
        coeffs: np.ndarray,
        *,
        opts: dict[str, Any] | None = None,
    ) -> SimscapeOutput:
        """Run one Simscape simulation with the given polynomial coefficients.

        Behaviour:
            1. Hash ``(coeffs, model_params, matlab_version)`` and check cache.
            2. On cache hit, return the cached :class:`SimscapeOutput`.
            3. On miss, build a ``Simulink.SimulationInput``, call ``sim()``
               via the MATLAB Engine, extract logsout into a
               :class:`SimscapeOutput`, and cache it.

        Latency: ~50-200 ms warm, ~10-30 s on the first call (engine startup).

        Args:
            coeffs: Flat 1-D float64 vector of length ``n_joints * 7``.
            opts: Optional simulation options forwarded to the MATLAB
                helper (e.g. integration tolerances). Treated as opaque
                model parameters for cache-key purposes.

        Returns:
            Frozen :class:`SimscapeOutput` with consistent ``N`` across
            all arrays.

        Raises:
            SimscapeStateError: If no model is loaded.
            SimscapeNotInstalledError: If the MATLAB Engine is required
                (cache miss) but unavailable.
            SimscapeSimulationError: If the MATLAB-side simulation fails.
        """
        self._lifecycle.require(
            AdapterState.LOADED,
            AdapterState.RUNNING,
            operation="simulate_with_coefficients",
        )

        key = make_cache_key(
            coeffs,
            model_params=self._serialise_model_params(opts),
            matlab_version=self._matlab_version,
        )
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("simscape cache hit for key=%s", key[:12])
            return cached

        result = self._simulate_uncached(coeffs, opts=opts)
        self._cache.put(key, result)
        return result

    def _serialise_model_params(self, opts: dict[str, Any] | None) -> bytes:
        """Return a stable bytes representation of opts + tunable params.

        Used as one component of the cache key. We sort keys so
        ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` collide.
        """
        import json

        payload = {
            "opts": opts or {},
            "model_name": self._model_name,
            "rng_seed": self._rng_seed,
        }
        return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")

    def _simulate_uncached(
        self,
        coeffs: np.ndarray,
        *,
        opts: dict[str, Any] | None = None,  # noqa: ARG002 - reserved for #4008
    ) -> SimscapeOutput:
        """Perform the actual MATLAB ``sim()`` call (no caching).

        Split out from :meth:`simulate_with_coefficients` so tests can
        mock this method to verify cache behaviour without a live MATLAB
        engine.

        Raises:
            SimscapeNotInstalledError: If MATLAB is unavailable.
            SimscapeSimulationError: On any MATLAB-side failure.
        """
        from src.engines.simscape._engine_pool import (
            get_shared_engine,
            is_matlab_available,
        )
        from src.engines.simscape._simscape_io import (
            build_simulation_input,
            logsout_to_simscape_output,
        )

        if self._engine is None:
            if not is_matlab_available():
                raise SimscapeNotInstalledError(
                    "simulate_with_coefficients requires MATLAB Engine"
                )
            self._engine = get_shared_engine(startup_timeout_s=self._startup_timeout_s)

        sim_input = build_simulation_input(
            self._engine,
            model_name=self._model_name,
            coeffs=coeffs,
            n_joints=self._dof,
        )
        try:
            logsout = self._engine.simulate_with_coefficients(sim_input, nargout=1)
        except Exception as exc:  # noqa: BLE001 - wrap MATLAB error
            err_id = getattr(exc, "MatlabError", "") or ""
            raise SimscapeSimulationError(
                f"sim() failed for model '{self._model_name}': {exc}",
                matlab_error_id=err_id,
            ) from exc
        return logsout_to_simscape_output(logsout)

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
        """Shut down the adapter; idempotent.

        Releases this adapter's reference to the shared MATLAB engine
        (the engine itself is process-scoped — see
        :func:`src.engines.simscape._engine_pool.shutdown_shared_engine`).
        Calling ``close`` twice is a no-op.
        """
        if self._lifecycle.is_stopped():
            return
        # Best-effort close_system on the loaded model so the engine
        # workspace stays clean for subsequent adapters in this process.
        if self._engine is not None and self._model_name:
            try:
                self._engine.close_system(self._model_name, 0, nargout=0)
            except Exception:  # noqa: BLE001 - shutdown best-effort
                logger.exception("close_system failed (ignored)")
        self._engine = None
        self._cache.clear()
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

        return importlib.util.find_spec("matlab.engine") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False
