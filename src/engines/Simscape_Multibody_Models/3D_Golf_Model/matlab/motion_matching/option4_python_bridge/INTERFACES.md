# Interfaces — Option 4

The Python contracts. Implementations under issues #036–#040 must match these signatures and DbC annotations exactly.

All decorators come from `src.shared.python.core.contracts` per [shared/CODING_STANDARDS.md § DbC](../shared/CODING_STANDARDS.md#dbc-design-by-contract).

## Module layout

```
option4_python_bridge/
├── simscape_adapter.py        # SimscapeAdapter
├── simscape_adapter_pool.py   # SimscapeAdapterPool
├── simscape_output.py         # SimscapeOutput dataclass + logsout converter
├── simscape_errors.py         # SimulationError, EngineStartupError, LicenseError, ModelLoadError
├── cache.py                   # _ResultCache (private, used by SimscapeAdapter)
├── loader.py                  # load_matlab_3d_engine — wired into src/engines/loaders.py
└── tests/
    ├── .gitkeep
    └── (test files per TESTING.md)
```

## `SimscapeOutput` dataclass

`option4_python_bridge/simscape_output.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimscapeOutput:
    """Flat numpy view of one Simscape simulation run.

    Mirrors ClubTarget from shared/CLUB_IK_SPEC.md so the cost function
    can subtract a SimscapeOutput from a ClubTarget directly.

    Invariants (DbC):
        - all arrays share the same N along axis 0
        - time is strictly increasing, time[0] == 0
        - club_quat rows are unit-norm to 1e-6
        - 0 <= impact_idx < N
    """

    time:       np.ndarray   # (N,)             float64, seconds, monotonic, dt = 1/sample_rate
    butt:       np.ndarray   # (N, 3)           float64, metres, world frame
    clubhead:   np.ndarray   # (N, 3)           float64, metres, world frame
    club_quat:  np.ndarray   # (N, 4)           float64, [w, x, y, z], unit-norm
    q:          np.ndarray   # (N, n_q)         float64, generalized coordinates
    v:          np.ndarray   # (N, n_v)         float64, generalized velocities
    tau:        np.ndarray   # (N, n_joints)    float64, applied joint torques (N·m)
    omega:      np.ndarray   # (N, n_joints)    float64, joint angular velocities (rad/s)
    impact_idx: int                            # index of max clubhead speed
```

The MATLAB-side helper that builds a flat-double view of `logsout` (so we don't marshal a `Simulink.SimulationData.Dataset` across the bridge) is part of issue #018 (`simulate_with_coefficients.m`).

## Errors

`option4_python_bridge/simscape_errors.py`:

```python
from src.shared.python.data_io.common_utils import GolfModelingError


class SimulationError(GolfModelingError):
    """Raised when a Simscape simulation fails (integrator divergence,
    missing block, MATLAB-side exception)."""

    def __init__(self, message: str, *, matlab_error_id: str = "",
                 matlab_traceback: str = "") -> None:
        super().__init__(message)
        self.matlab_error_id = matlab_error_id
        self.matlab_traceback = matlab_traceback


class EngineStartupError(SimulationError):
    """Raised when matlab.engine cannot start, the engine process dies
    mid-call, or import of the matlabengine package fails."""


class LicenseError(EngineStartupError):
    """Subclass of EngineStartupError for recognizable license-checkout
    failures (matlab_error_id starts with 'MATLAB:license:')."""


class ModelLoadError(SimulationError):
    """Raised when load_system / load_from_path fails on the MATLAB side."""
```

All inherit from `GolfModelingError` so the existing `loaders.py` error path continues to work.

## `SimscapeAdapter`

`option4_python_bridge/simscape_adapter.py`:

```python
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from src.shared.python.core.contracts import precondition, postcondition, invariant
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger

from .simscape_output import SimscapeOutput
from .simscape_errors import (
    SimulationError, EngineStartupError, LicenseError, ModelLoadError,
)

logger = get_logger(__name__)


@invariant(
    lambda self: self._engine is None or self._model_name != "",
    "if engine is started, a model name must be set",
)
@invariant(
    lambda self: self._cache_max_entries >= 0,
    "cache_max_entries must be non-negative (0 = disabled)",
)
class SimscapeAdapter(PhysicsEngine):
    """PhysicsEngine implementation backed by MATLAB Simscape Multibody.

    Wraps GolfSwing3D_Kinetic.slx via the MATLAB Engine API for Python.

    See ASSUMPTIONS.md and APPROACH.md for the full contract.

    Lifecycle:
        SimscapeAdapter()                  # cheap, no MATLAB started
        adapter.load_from_path(slx_path)   # starts engine, ~10-30 s
        adapter.simulate_with_coefficients(theta)  # ~50-200 ms
        adapter.close()                    # quits engine

    Thread safety:
        Not thread-safe. One adapter per process. Use SimscapeAdapterPool
        for concurrency.

    Design by Contract:
        Preconditions on each method (see decorators).
        Class invariants enforced by @invariant.
        Error contract: any MATLAB-side failure is wrapped in SimulationError.
    """

    @precondition(
        lambda self, rng_seed=42, cache_enabled=True, cache_max_entries=1024,
              startup_timeout_s=60.0:
            isinstance(rng_seed, int) and rng_seed >= 0,
        "rng_seed must be a non-negative int",
    )
    @precondition(
        lambda self, rng_seed=42, cache_enabled=True, cache_max_entries=1024,
              startup_timeout_s=60.0:
            isinstance(cache_max_entries, int) and cache_max_entries >= 0,
        "cache_max_entries must be a non-negative int",
    )
    def __init__(
        self,
        rng_seed: int = 42,
        cache_enabled: bool = True,
        cache_max_entries: int = 1024,
        startup_timeout_s: float = 60.0,
    ) -> None:
        """Construct adapter without starting MATLAB.

        The engine starts lazily on the first call that needs it.
        """

    # ------------------------------------------------------------------
    # PhysicsEngineProtocol.Loadable
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the loaded model name, or '' if no model loaded."""

    @precondition(
        lambda self, path: isinstance(path, str) and path != "",
        "path must be a non-empty string",
    )
    @postcondition(
        lambda self, path, _result: self.model_name != "",
        "after load_from_path, model_name must be non-empty",
    )
    def load_from_path(self, path: str) -> None:
        """Start the engine if not started and load the .slx model.

        Raises:
            FileNotFoundError: path does not exist.
            ValueError: file is not a .slx.
            EngineStartupError: MATLAB Engine cannot start.
            ModelLoadError: load_system failed on the MATLAB side.
        """

    @precondition(
        lambda self, content, extension=None: False,
        "load_from_string is not supported by SimscapeAdapter — .slx is binary",
    )
    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Not supported. Raises NotImplementedError."""

    # ------------------------------------------------------------------
    # PhysicsEngineProtocol.Steppable
    # ------------------------------------------------------------------

    @postcondition(
        lambda self, _result: self.get_time() == 0.0,
        "after reset, time must be zero",
    )
    def reset(self) -> None:
        """Reset the simulation to t=0 with default initial conditions."""

    @precondition(
        lambda self, dt=None: dt is None or (isinstance(dt, float) and dt > 0),
        "dt must be a positive float or None",
    )
    @postcondition(
        lambda self, dt, _result: True,  # time advances; checked by caller in tests
        "after step, time has advanced by approximately dt",
    )
    def step(self, dt: float | None = None) -> None:
        """Advance simulation by dt seconds (slow path; ~10-20 ms per call).

        Implementation detail: each step calls sim() with the current state
        injected via Simulink.SimulationInput.setInitialState. RL inner-loop
        training is infeasible at this rate — use Option 2's surrogate.
        """

    def forward(self) -> None:
        """Compute kinematics/dynamics without advancing time.

        Implemented by re-running the model with sim duration 0; cheap on
        a warm engine but not free (~5-10 ms).
        """

    # ------------------------------------------------------------------
    # PhysicsEngineProtocol.Queryable
    # ------------------------------------------------------------------

    @postcondition(
        lambda self, result: (
            isinstance(result, tuple) and len(result) == 2
            and result[0].ndim == 1 and result[1].ndim == 1
        ),
        "get_state returns (q, v) as 1-D numpy arrays",
    )
    def get_state(self) -> tuple[np.ndarray, np.ndarray]: ...

    @precondition(
        lambda self, q, v: (
            isinstance(q, np.ndarray) and isinstance(v, np.ndarray)
            and q.ndim == 1 and v.ndim == 1
            and np.all(np.isfinite(q)) and np.all(np.isfinite(v))
        ),
        "q and v must be finite 1-D numpy arrays",
    )
    def set_state(self, q: np.ndarray, v: np.ndarray) -> None: ...

    @precondition(
        lambda self, u: (
            isinstance(u, np.ndarray) and u.ndim == 1 and np.all(np.isfinite(u))
        ),
        "u must be a finite 1-D numpy array",
    )
    def set_control(self, u: np.ndarray) -> None: ...

    @postcondition(lambda self, result: result >= 0.0, "time is non-negative")
    def get_time(self) -> float: ...

    # ------------------------------------------------------------------
    # PhysicsEngineProtocol.DynamicsComputable
    # ------------------------------------------------------------------

    def compute_mass_matrix(self) -> np.ndarray: ...
    def compute_bias_forces(self) -> np.ndarray: ...
    def compute_gravity_forces(self) -> np.ndarray: ...

    @precondition(
        lambda self, qacc: isinstance(qacc, np.ndarray) and qacc.ndim == 1,
        "qacc must be a 1-D numpy array",
    )
    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray: ...

    @precondition(
        lambda self, body_name: isinstance(body_name, str) and body_name != "",
        "body_name must be a non-empty string",
    )
    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """May return None if the body is not present in the .slx hierarchy."""

    def compute_drift_acceleration(self) -> np.ndarray: ...

    @precondition(
        lambda self, tau: isinstance(tau, np.ndarray) and tau.ndim == 1,
        "tau must be a 1-D numpy array",
    )
    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray: ...

    # ------------------------------------------------------------------
    # PhysicsEngineProtocol.CounterfactualComputable
    # ------------------------------------------------------------------

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray: ...
    def compute_zvcf(self, q: np.ndarray) -> np.ndarray: ...

    # ------------------------------------------------------------------
    # PhysicsEngineProtocol.Recordable
    # ------------------------------------------------------------------

    def get_time_series(
        self, field_name: str
    ) -> tuple[np.ndarray, np.ndarray | list[Any]]: ...

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def set_analysis_config(self, config: dict[str, Any]) -> None: ...

    # ------------------------------------------------------------------
    # System-identification convenience surface (used by sim2real)
    # ------------------------------------------------------------------

    def get_link_masses(self) -> np.ndarray:
        """Return current link masses as (n_links,) numpy array."""

    @precondition(
        lambda self, masses: (
            isinstance(masses, np.ndarray) and masses.ndim == 1
            and np.all(masses > 0)
        ),
        "masses must be a strictly-positive 1-D numpy array",
    )
    def set_link_masses(self, masses: np.ndarray) -> None:
        """Side effect: clears the simulation cache."""

    def get_joint_damping(self) -> np.ndarray: ...

    @precondition(
        lambda self, damping: (
            isinstance(damping, np.ndarray) and damping.ndim == 1
            and np.all(damping >= 0)
        ),
        "damping must be a non-negative 1-D numpy array",
    )
    def set_joint_damping(self, damping: np.ndarray) -> None:
        """Side effect: clears the simulation cache."""

    # ------------------------------------------------------------------
    # The motion-matching headline method
    # ------------------------------------------------------------------

    @precondition(
        lambda self, coeffs: (
            isinstance(coeffs, np.ndarray)
            and coeffs.ndim == 1
            and coeffs.size > 0
            and coeffs.size % 7 == 0
            and np.all(np.isfinite(coeffs))
        ),
        "coeffs must be a finite 1-D numpy array of length n_joints*7",
    )
    @postcondition(
        lambda self, coeffs, result: (
            isinstance(result, SimscapeOutput)
            and result.time.shape[0] == result.butt.shape[0]
            == result.clubhead.shape[0] == result.club_quat.shape[0]
        ),
        "result is a SimscapeOutput with consistent N across all arrays",
    )
    def simulate_with_coefficients(self, coeffs: np.ndarray) -> SimscapeOutput:
        """Run one Simscape simulation with the given polynomial coefficients.

        This is the headline method used by every motion-matching consumer.

        Behaviour:
            1. Hash (coeffs, model_params) and check cache.
            2. On cache hit, return the cached SimscapeOutput.
            3. On miss, build a Simulink.SimulationInput, call sim(),
               extract logsout into a SimscapeOutput, cache it.

        Latency: ~50-200 ms warm, ~10-30 s on the first call (engine startup).

        Raises:
            SimulationError: any MATLAB-side failure (integrator divergence,
                model error, license loss mid-call).
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Quit the underlying MATLAB Engine. Idempotent."""

    def __enter__(self) -> "SimscapeAdapter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
```

## Protocol-method coverage matrix

For PR review and CI, every method must declare its support level:

| Method                                                       | Status                  | Latency                              | Notes                                             |
| ------------------------------------------------------------ | ----------------------- | ------------------------------------ | ------------------------------------------------- |
| `model_name`                                                 | full                    | <1 ms                                | property, no engine call                          |
| `load_from_path`                                             | full                    | 10–30 s first time, 1–3 s thereafter | engine startup amortized                          |
| `load_from_string`                                           | **NotImplementedError** | —                                    | .slx is binary; no string form                    |
| `reset`                                                      | full                    | ~10 ms                               | resets `Simulink.SimulationInput.setInitialState` |
| `step`                                                       | full but slow           | ~10–20 ms                            | RL inner-loop infeasible — use surrogate          |
| `forward`                                                    | full                    | ~5–10 ms                             | sim with duration 0                               |
| `get_state`, `set_state`                                     | full                    | ~5–10 ms                             | flat double marshalling                           |
| `set_control`, `get_time`                                    | full                    | ~5–10 ms                             |                                                   |
| `compute_mass_matrix`                                        | full                    | ~10–20 ms                            | uses Simscape `mass_matrix` API                   |
| `compute_bias_forces`, `compute_gravity_forces`              | full                    | ~10–20 ms                            |                                                   |
| `compute_inverse_dynamics`                                   | full                    | ~15–25 ms                            |                                                   |
| `compute_jacobian`                                           | partial                 | ~10 ms                               | returns `None` for bodies not in the .slx         |
| `compute_drift_acceleration`, `compute_control_acceleration` | full                    | ~15–25 ms                            | per Section F superposition                       |
| `compute_ztcf`, `compute_zvcf`                               | full                    | ~15–25 ms                            | per Section G                                     |
| `get_time_series`, `get_induced_acceleration_series`         | full                    | ~5–15 ms                             | reads from cached `logsout` of last sim           |
| `set_analysis_config`                                        | full                    | <1 ms                                | toggles which fields the .slx logs                |
| `simulate_with_coefficients`                                 | full (headline)         | ~50–200 ms                           | the motion-matching path                          |

`test_protocol_compliance` (see [TESTING.md](TESTING.md)) iterates every method on `PhysicsEngine` and asserts it is either implemented or raises `NotImplementedError` with a clear message.

## `SimscapeAdapterPool`

`option4_python_bridge/simscape_adapter_pool.py`:

```python
from __future__ import annotations
from collections.abc import Iterable, Sequence
import numpy as np
from src.shared.python.core.contracts import precondition

from .simscape_output import SimscapeOutput


class SimscapeAdapterPool:
    """Pool of SimscapeAdapter instances for parallel inference.

    Each pool worker is a separate Python process owning one MATLAB Engine.
    Pool size is bounded above by the host's MATLAB license count.

    Usage:
        with SimscapeAdapterPool(pool_size=4, model_path=slx_path) as pool:
            outs = pool.map_simulate(thetas)

    Cache: each worker has its own in-process cache. Cross-worker caching
    is an explicit non-goal for v1.
    """

    @precondition(
        lambda self, pool_size, model_path:
            isinstance(pool_size, int) and pool_size >= 1,
        "pool_size must be a positive int",
    )
    @precondition(
        lambda self, pool_size, model_path:
            isinstance(model_path, str) and model_path.endswith(".slx"),
        "model_path must end in .slx",
    )
    def __init__(self, pool_size: int, model_path: str) -> None: ...

    @precondition(
        lambda self, thetas: all(
            isinstance(t, np.ndarray) and t.ndim == 1 and t.size % 7 == 0
            for t in thetas
        ),
        "every theta must be a 1-D numpy array with size multiple of 7",
    )
    def map_simulate(
        self, thetas: Sequence[np.ndarray]
    ) -> list[SimscapeOutput]:
        """Distribute simulations across pool workers.

        Order of results matches order of inputs.
        """

    def imap_simulate(
        self, thetas: Iterable[np.ndarray]
    ) -> Iterable[SimscapeOutput]:
        """Streaming variant. Yields results as they complete (any order)."""

    def close(self) -> None:
        """Quit every worker engine. Idempotent."""

    def __enter__(self) -> "SimscapeAdapterPool":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
```

## Loader function

`option4_python_bridge/loader.py`:

```python
from __future__ import annotations
from pathlib import Path
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.data_io.common_utils import GolfModelingError

logger = get_logger(__name__)

DEFAULT_SLX_RELPATH = Path(
    "engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/"
    "GolfSwing3D_Kinetic.slx"
)


def load_matlab_3d_engine(suite_root: Path) -> PhysicsEngine:
    """Factory for SimscapeAdapter wired into the registry as MATLAB_3D.

    Postcondition (DbC): returned engine is non-None.

    Raises:
        GolfModelingError: matlabengine is not installed, license missing,
            or the default .slx is not on disk.
    """
    try:
        from src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.\
            motion_matching.option4_python_bridge.simscape_adapter import (
            SimscapeAdapter,
        )
    except ImportError as e:
        raise GolfModelingError(
            "MATLAB Engine for Python not installed. Run "
            "`python -m pip install matlabengine`. See "
            "option4_python_bridge/INSTALLATION.md for the full procedure."
        ) from e

    adapter = SimscapeAdapter()
    slx_path = suite_root / DEFAULT_SLX_RELPATH
    if not slx_path.exists():
        raise GolfModelingError(
            f"GolfSwing3D_Kinetic.slx not found at expected path: {slx_path}"
        )
    adapter.load_from_path(str(slx_path))
    logger.info("MATLAB_3D (Simscape) engine loaded successfully")
    return adapter
```

(The exact import path inside the `try` block depends on the final Python package name — Simscape's underscore-prefixed dirs need `_3D_Golf_Model` style aliasing. The implementing agent should confirm against the actual `__init__.py` files when they wire this up.)

## Registration patch for `src/engines/loaders.py`

Conceptual diff — **do not apply yet** (Phase 2 work, after issues #036–#039 close).

```diff
--- a/src/engines/loaders.py
+++ b/src/engines/loaders.py
@@ -41,9 +41,11 @@ __all__ = [
     "load_mujoco_engine",
     "load_drake_engine",
     "load_pinocchio_engine",
     "load_opensim_engine",
     "load_myosim_engine",
+    "load_matlab_3d_engine",
     "load_pendulum_engine",
     "load_golf_swing_pendulum_engine",
     "load_putting_green_engine",
     "LOADER_MAP",
 ]

@@ -380,6 +382,17 @@ def load_putting_green_engine(suite_root: Path) -> PhysicsEngine:  # noqa: ARG00
         raise GolfModelingError("Putting Green engine not found.") from e


+def load_matlab_3d_engine(suite_root: Path) -> PhysicsEngine:
+    """Load Simscape Multibody (MATLAB_3D) engine via the Python bridge.
+
+    Postcondition: returned engine is non-None (DbC).
+    """
+    from src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.\
+        motion_matching.option4_python_bridge.loader import (
+        load_matlab_3d_engine as _load,
+    )
+    return _load(suite_root)
+
+
 # Helper for loaders map
 LOADER_MAP: dict[EngineType, Callable[[Path], PhysicsEngine]] = {
     EngineType.MUJOCO: load_mujoco_engine,
     EngineType.DRAKE: load_drake_engine,
     EngineType.PINOCCHIO: load_pinocchio_engine,
     EngineType.OPENSIM: load_opensim_engine,
     EngineType.MYOSIM: load_myosim_engine,
+    EngineType.MATLAB_3D: load_matlab_3d_engine,
     EngineType.PENDULUM: load_pendulum_engine,
     EngineType.GOLF_SWING_PENDULUM: load_golf_swing_pendulum_engine,
     EngineType.PUTTING_GREEN: load_putting_green_engine,
 }
```

`EngineType.MATLAB_3D` already exists in [`engine_registry.py`](../../../../../../shared/python/engine_core/engine_registry.py) (line 23), so no enum change is needed.

`EngineType.MATLAB_2D` is reserved for a future 2D pendulum bridge; Option 4 does **not** wire it up.
