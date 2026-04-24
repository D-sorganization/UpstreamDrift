"""Engine Interchange / Adapter Layer  (Issue #3051).

Purpose
-------
This module defines the **BaseEngineAdapter** — the concrete base class that
third-party or internal engine wrappers should subclass when bridging an
external physics solver to the UpstreamDrift ``PhysicsEngine`` Protocol.

Why an Adapter Layer?
---------------------
The ``PhysicsEngine`` Protocol in
``src/shared/python/engine_core/interfaces.py`` defines the *structural*
contract that every engine must satisfy.  That protocol works well for
duck-typing checks at runtime, but it does not give implementors:

1. A place to put shared bookkeeping (model path, time counter, config).
2. A clear extension point with meaningful ``NotImplementedError`` messages
   that tell the implementor *exactly* which methods are missing.
3. Lifecycle hooks (``on_load``, ``on_reset``) that wrapper subclasses can
   override without re-implementing the full state-machine logic.

The ``BaseEngineAdapter`` here addresses all three.

Intended Usage
--------------
To wrap a new physics solver, subclass ``BaseEngineAdapter`` and override the
``_`` (private) hook methods::

    from src.engines.adapter import BaseEngineAdapter

    class MyEngineAdapter(BaseEngineAdapter):
        ENGINE_NAME = "my_engine"

        def _do_load(self, path: str) -> None:
            self._solver = MySolver.load(path)

        def _do_reset(self) -> None:
            self._solver.reset()

        def _do_step(self, dt: float | None) -> None:
            self._solver.step(dt or self._default_dt)

        def _do_get_state(self):
            return self._solver.q, self._solver.v

        # … implement remaining abstract methods …

Relationship to the Protocol
-----------------------------
``BaseEngineAdapter`` is declared as implementing ``PhysicsEngine`` via a
``# type: ignore`` comment to avoid a circular import, but all sub-protocols
(``Loadable``, ``Steppable``, ``Queryable``, ``DynamicsComputable``,
``CounterfactualComputable``, ``Recordable``) are represented here as abstract
methods with ``NotImplementedError`` stubs.

Status
------
This file contains the skeleton / scaffolding.  Concrete engine adapters are
located under ``src/engines/physics_engines/<name>/``.  The existing engines
(MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) pre-date this adapter layer and
implement the Protocol directly; they should be progressively migrated to
extend ``BaseEngineAdapter`` in a follow-up refactor sprint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEngineAdapter(ABC):
    """Abstract base class for UpstreamDrift engine adapters.

    Subclass this to bridge an external physics solver to the ``PhysicsEngine``
    Protocol.  Override the ``_do_*`` hook methods; do not override the public
    methods directly unless you need to change the bookkeeping logic.

    Attributes:
        ENGINE_NAME: Class-level string identifier used by ``EngineManager``.
        _model_path: Path of the most recently loaded model (or ``None``).
        _time: Current simulation time in seconds.
        _is_loaded: Whether a model has been successfully loaded.
        _default_dt: Default time step in seconds (override in subclass).
    """

    ENGINE_NAME: str = "base"

    def __init__(self) -> None:
        self._model_path: str | None = None
        self._time: float = 0.0
        self._is_loaded: bool = False
        self._default_dt: float = 0.002  # 500 Hz default

    # ------------------------------------------------------------------
    # Public API — implements Loadable sub-protocol
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the engine identifier and loaded model path."""
        if self._model_path is None:
            return self.ENGINE_NAME
        return f"{self.ENGINE_NAME}:{self._model_path}"

    def load_from_path(self, path: str) -> None:
        """Load a model from *path* and transition to INITIALIZED state.

        Args:
            path: Filesystem path to the model file (URDF, MJCF, .osim, etc.)

        Raises:
            FileNotFoundError: If the path does not exist.
            RuntimeError: If the underlying solver fails to load.
        """
        self._do_load(path)
        self._model_path = path
        self._is_loaded = True
        self._time = 0.0

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Load a model from an in-memory *content* string.

        Args:
            content: Model file content as a string.
            extension: Optional file extension hint (e.g. ``".urdf"``).

        Raises:
            RuntimeError: If the underlying solver fails to parse.
        """
        self._do_load_string(content, extension)
        self._model_path = "<in-memory>"
        self._is_loaded = True
        self._time = 0.0

    # ------------------------------------------------------------------
    # Public API — implements Steppable sub-protocol
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset simulation to t=0 with zero velocities and default pose."""
        self._assert_loaded("reset")
        self._do_reset()
        self._time = 0.0

    def step(self, dt: float | None = None) -> None:
        """Advance simulation by one time step.

        Args:
            dt: Time step in seconds.  Defaults to ``self._default_dt``.
        """
        self._assert_loaded("step")
        effective_dt = dt if dt is not None else self._default_dt
        self._do_step(effective_dt)
        self._time += effective_dt

    def forward(self) -> None:
        """Compute forward kinematics without advancing time."""
        self._assert_loaded("forward")
        self._do_forward()

    # ------------------------------------------------------------------
    # Public API — implements Queryable sub-protocol
    # ------------------------------------------------------------------

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (q, v) — generalised positions and velocities."""
        self._assert_loaded("get_state")
        return self._do_get_state()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set generalised positions *q* and velocities *v*."""
        self._assert_loaded("set_state")
        self._do_set_state(q, v)

    def set_control(self, u: np.ndarray) -> None:
        """Apply generalised control forces / torques *u*."""
        self._assert_loaded("set_control")
        self._do_set_control(u)

    def get_time(self) -> float:
        """Return current simulation time in seconds."""
        return self._time

    def get_configuration(self) -> np.ndarray:
        """Convenience shortcut — return only generalised positions *q*."""
        q, _ = self.get_state()
        return q

    # ------------------------------------------------------------------
    # Public API — implements DynamicsComputable sub-protocol
    # ------------------------------------------------------------------

    def compute_mass_matrix(self) -> np.ndarray:
        """Return the (nv x nv) joint-space mass matrix M(q)."""
        self._assert_loaded("compute_mass_matrix")
        return self._do_compute_mass_matrix()

    def compute_bias_forces(self) -> np.ndarray:
        """Return the (nv,) bias force vector C(q,v)."""
        self._assert_loaded("compute_bias_forces")
        return self._do_compute_bias_forces()

    def compute_gravity_forces(self) -> np.ndarray:
        """Return the (nv,) gravity generalised force vector g(q)."""
        self._assert_loaded("compute_gravity_forces")
        return self._do_compute_gravity_forces()

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Return joint torques tau for a given joint acceleration *qacc*."""
        self._assert_loaded("compute_inverse_dynamics")
        return self._do_compute_inverse_dynamics(qacc)

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Return spatial Jacobian for body *body_name* or ``None`` if unavailable."""
        self._assert_loaded("compute_jacobian")
        return self._do_compute_jacobian(body_name)

    def compute_drift_acceleration(self) -> np.ndarray:
        """Return free-floating (uncontrolled) acceleration under gravity/contact."""
        self._assert_loaded("compute_drift_acceleration")
        return self._do_compute_drift_acceleration()

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Return acceleration component due to control torques *tau*."""
        self._assert_loaded("compute_control_acceleration")
        return self._do_compute_control_acceleration(tau)

    # ------------------------------------------------------------------
    # Public API — implements CounterfactualComputable sub-protocol
    # ------------------------------------------------------------------

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute zero-torque counterfactual acceleration at (q, v)."""
        self._assert_loaded("compute_ztcf")
        return self._do_compute_ztcf(q, v)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Compute zero-velocity counterfactual acceleration at *q*."""
        self._assert_loaded("compute_zvcf")
        return self._do_compute_zvcf(q)

    # ------------------------------------------------------------------
    # Abstract hook methods — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _do_load(self, path: str) -> None:
        """Load model from file path.  Called by ``load_from_path``."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _do_load(path)"
        )

    def _do_load_string(self, content: str, extension: str | None) -> None:
        """Load model from string content.  Override if the solver supports it."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support load_from_string. "
            "Override _do_load_string to add support."
        )

    @abstractmethod
    def _do_reset(self) -> None:
        """Reset solver state to t=0.  Called by ``reset``."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _do_reset()"
        )

    @abstractmethod
    def _do_step(self, dt: float) -> None:
        """Advance solver by *dt* seconds.  Called by ``step``."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _do_step(dt)"
        )

    def _do_forward(self) -> None:
        """Compute forward kinematics without time advance.  Override if supported."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _do_forward. "
            "Override to add forward-kinematics support."
        )

    @abstractmethod
    def _do_get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (q, v).  Called by ``get_state``."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _do_get_state()"
        )

    @abstractmethod
    def _do_set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set (q, v).  Called by ``set_state``."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _do_set_state(q, v)"
        )

    @abstractmethod
    def _do_set_control(self, u: np.ndarray) -> None:
        """Apply control input *u*.  Called by ``set_control``."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _do_set_control(u)"
        )

    def _do_compute_mass_matrix(self) -> np.ndarray:
        """Return mass matrix M(q)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_mass_matrix. "
            "Override _do_compute_mass_matrix to add support."
        )

    def _do_compute_bias_forces(self) -> np.ndarray:
        """Return bias force vector C(q,v)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_bias_forces. "
            "Override _do_compute_bias_forces to add support."
        )

    def _do_compute_gravity_forces(self) -> np.ndarray:
        """Return gravity force vector g(q)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_gravity_forces. "
            "Override _do_compute_gravity_forces to add support."
        )

    def _do_compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Return inverse-dynamics torques for *qacc*."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_inverse_dynamics. "
            "Override _do_compute_inverse_dynamics to add support."
        )

    def _do_compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Return spatial Jacobian for *body_name*."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_jacobian. "
            "Override _do_compute_jacobian to add support."
        )

    def _do_compute_drift_acceleration(self) -> np.ndarray:
        """Return drift (uncontrolled) acceleration."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_drift_acceleration. "
            "Override _do_compute_drift_acceleration to add support."
        )

    def _do_compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Return control-induced acceleration for *tau*."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_control_acceleration. "
            "Override _do_compute_control_acceleration to add support."
        )

    def _do_compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return zero-torque counterfactual acceleration."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_ztcf. "
            "Override _do_compute_ztcf to add support."
        )

    def _do_compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Return zero-velocity counterfactual acceleration."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_zvcf. "
            "Override _do_compute_zvcf to add support."
        )

    def set_analysis_config(self, config: dict[str, Any]) -> None:
        """Apply analysis configuration dictionary (optional capability).

        Args:
            config: Key-value configuration for the solver's analysis mode.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement set_analysis_config. "
            "Override to add analysis-config support."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_loaded(self, method_name: str) -> None:
        """Raise ``RuntimeError`` if no model has been loaded yet."""
        if not self._is_loaded:
            raise RuntimeError(
                f"Cannot call {method_name}() before loading a model. "
                "Call load_from_path() or load_from_string() first."
            )

    def __repr__(self) -> str:
        status = f"model={self._model_path!r}" if self._is_loaded else "no model loaded"
        return f"<{self.__class__.__name__} engine={self.ENGINE_NAME!r} {status}>"
