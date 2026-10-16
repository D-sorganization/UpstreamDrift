# UpstreamDrift User Manual -- Part II: Physics Engine Architecture

---

# Chapter 5: Physics Engine Framework

## 5.1 Overview

The UpstreamDrift Golf Modeling Suite is built on a **multi-engine architecture** that
allows users to swap between different physics backends without changing the higher-level
simulation, analytics, or GUI code. The framework achieves this through four
cooperating components:

1. **`PhysicsEngine` Protocol** -- the unified abstract interface every backend must satisfy.
2. **`BasePhysicsEngine`** -- a convenience base class that factors out common behaviour
   and enforces Design by Contract (DbC).
3. **`EngineRegistry` / `EngineManager`** -- the discovery, registration, and runtime
   switching layer.
4. **`EngineProbes`** -- lightweight readiness checks that detect whether a particular
   backend (MuJoCo, Drake, Pinocchio, ...) is installed, configured, and functional.

All source files discussed in this chapter reside under:

```
src/shared/python/
    interfaces.py
    base_physics_engine.py
    engine_manager.py
    engine_registry.py
    engine_probes.py
    contracts.py
```

---

## 5.2 The `PhysicsEngine` Protocol

**File:** `src/shared/python/interfaces.py`

`PhysicsEngine` is defined as a `typing.Protocol` decorated with `@runtime_checkable`.
Any class that structurally conforms to this protocol can be used anywhere a
`PhysicsEngine` is expected -- no explicit inheritance is required. The protocol
also extends `Checkpointable`, enabling state serialisation and rollback.

### 5.2.1 State Machine

Every engine implementation must respect the following lifecycle:

```
UNINITIALIZED  --[load_from_path / load_from_string]--> INITIALIZED
INITIALIZED    --[reset]------------------------------>  INITIALIZED  (t = 0)
INITIALIZED    --[step(dt)]--------------------------->  INITIALIZED  (t += dt)
```

### 5.2.2 Core State Methods

| Method             | Signature                             | Description                                                                                                                          |
| ------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `get_state()`      | `() -> tuple[np.ndarray, np.ndarray]` | Returns $(q, v)$ where $q \in \mathbb{R}^{n_q}$ are generalized coordinates and $v \in \mathbb{R}^{n_v}$ are generalized velocities. |
| `set_state(q, v)`  | `(np.ndarray, np.ndarray) -> None`    | Sets state and triggers `forward()` to recompute derived quantities.                                                                 |
| `set_control(u)`   | `(np.ndarray) -> None`                | Stores a control vector $u \in \mathbb{R}^{n_u}$ for the next `step()`.                                                              |
| `get_time()`       | `() -> float`                         | Returns the current simulation time $t \geq 0$.                                                                                      |
| `get_full_state()` | `() -> dict[str, Any]`                | Batched query returning `{'q', 'v', 't', 'M'}` in one call (performance optimisation).                                               |

### 5.2.3 Dynamics Interface

All engines expose the following dynamics quantities:

**Mass Matrix.** The joint-space inertia matrix $M(q) \in \mathbb{R}^{n_v \times n_v}$ is
symmetric positive definite:

$$M(q) = M(q)^T, \quad \lambda_{\min}(M) > 0$$

```python
def compute_mass_matrix(self) -> np.ndarray:
    """Returns M(q), shape (n_v, n_v)."""
```

**Bias Forces.** The combined Coriolis, centrifugal, and gravitational force vector:

$$b(q, v) = C(q, v)\,v + g(q) \in \mathbb{R}^{n_v}$$

```python
def compute_bias_forces(self) -> np.ndarray:
    """Returns b = C(q,v)*v + g(q), shape (n_v,)."""
```

**Gravity Forces.** The configuration-dependent gravity vector alone:

$$g(q) \in \mathbb{R}^{n_v}$$

```python
def compute_gravity_forces(self) -> np.ndarray:
    """Returns g(q), shape (n_v,)."""
```

**Inverse Dynamics.** Given a desired acceleration $\ddot{q}$, compute the required
generalised forces:

$$\tau = M(q)\,\ddot{q} + C(q, v)\,v + g(q)$$

```python
def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
    """Returns tau = ID(q, v, qacc), shape (n_v,)."""
```

**Jacobian.** The spatial Jacobian for a named body frame maps joint velocities
to linear and angular body-frame velocities:

$$J = \frac{\partial x}{\partial q} \in \mathbb{R}^{6 \times n_v}$$

```python
def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
    """Returns {'linear': (3, n_v), 'angular': (3, n_v), 'spatial': (6, n_v)}."""
```

### 5.2.4 Section F -- Drift-Control Decomposition

A central design requirement of UpstreamDrift is the **superposition principle**:

$$\ddot{q}_{\text{full}} = \ddot{q}_{\text{drift}} + \ddot{q}_{\text{control}}$$

where

$$\ddot{q}_{\text{drift}} = M^{-1}\bigl(-C(q,v)\,v - g(q)\bigr)$$
$$\ddot{q}_{\text{control}} = M^{-1}\,\tau$$

```python
def compute_drift_acceleration(self) -> np.ndarray:
    """Passive acceleration with zero control inputs, shape (n_v,)."""

def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
    """Control-attributed acceleration M^{-1} tau, shape (n_v,)."""
```

This decomposition answers the fundamental biomechanics question: _How much of
the observed motion is due to active muscle effort versus passive dynamics?_

### 5.2.5 Section G -- Counterfactual Experiments

Two counterfactual methods enable causal reasoning about the swing:

**ZTCF (Zero-Torque Counterfactual, G1):** "What happens if all actuators turn off?"

$$\ddot{q}_{\text{ZTCF}} = M(q)^{-1}\bigl(-C(q,v)\,v - g(q) + J^T\lambda\bigr)$$

```python
def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray
```

**ZVCF (Zero-Velocity Counterfactual, G2):** "What happens if motion freezes?"

$$\ddot{q}_{\text{ZVCF}} = M(q)^{-1}\bigl(-g(q) + \tau + J^T\lambda\bigr)$$

```python
def compute_zvcf(self, q: np.ndarray) -> np.ndarray
```

The difference $\Delta a_{\text{velocity}} = a_{\text{full}} - a_{\text{ZVCF}}$
isolates Coriolis/centrifugal contributions.

### 5.2.6 Section B5 -- Flexible Shaft Interface (Optional)

Engines may optionally model a flexible club shaft via Euler-Bernoulli beam theory:

```python
def set_shaft_properties(
    self,
    length: float,              # Shaft length [m]
    EI_profile: np.ndarray,     # Bending stiffness [N*m^2]
    mass_profile: np.ndarray,   # Linear mass density [kg/m]
    damping_ratio: float = 0.02 # Modal damping ratio
) -> bool:
    """Returns True if shaft flexibility is supported and configured."""

def get_shaft_state(self) -> dict[str, np.ndarray] | None:
    """Returns {'deflection', 'rotation', 'velocity', 'modal_amplitudes'}."""
```

### 5.2.7 Recorder Interface

`RecorderInterface` is a separate protocol for capturing time-series data:

```python
@runtime_checkable
class RecorderInterface(Protocol):
    engine: Any

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray]: ...
    def get_induced_acceleration_series(self, source_name: str | int) -> tuple[np.ndarray, np.ndarray]: ...
    def set_analysis_config(self, config: dict[str, Any]) -> None: ...
```

---

## 5.3 Design by Contract -- `contracts.py`

**File:** `src/shared/python/contracts.py`

UpstreamDrift enforces Design by Contract (DbC) principles through decorators and a
mixin class. Contracts can be globally toggled via the `CONTRACTS_ENABLED` flag for
production performance.

### 5.3.1 Exception Hierarchy

```
ContractViolationError
  |-- PreconditionError     (caller violated input requirements)
  |-- PostconditionError    (implementation violated output guarantees)
  |-- InvariantError        (object state inconsistency)
  |-- StateError            (operation in wrong lifecycle state)
```

Each exception carries structured diagnostic information:

```python
class ContractViolationError(Exception):
    def __init__(
        self,
        contract_type: str,
        message: str,
        function_name: str | None = None,
        details: dict[str, Any] | None = None,
    ): ...
```

### 5.3.2 Decorator API

**Preconditions** guard method entry:

```python
@precondition(lambda self: self._is_initialized, "Engine must be initialized")
def step(self, dt: float) -> None: ...
```

The condition callable receives the same arguments as the decorated function. If it
returns `False`, a `PreconditionError` is raised before the method body executes.

**Postconditions** guard method exit:

```python
@postcondition(check_finite, "Mass matrix must contain finite values")
def compute_mass_matrix(self) -> np.ndarray: ...
```

The condition callable receives the **return value**. If it returns `False`, a
`PostconditionError` is raised (the original return value is discarded).

**State requirements** are a specialised precondition for lifecycle checks:

```python
@require_state(lambda self: self._is_initialized, "initialized")
def get_state(self) -> EngineState: ...
```

**Invariant checking** is available via the `ContractChecker` mixin and the
`@invariant_checked` decorator:

```python
class MyEngine(ContractChecker, BasePhysicsEngine):
    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        return [
            (lambda: self._is_initialized or self.model is None,
             "Uninitialised engine must not have a model"),
        ]

    @invariant_checked
    def set_mass(self, mass: float) -> None:
        self._mass = mass  # Invariants checked after this returns
```

### 5.3.3 Utility Predicates

| Predicate                 | Signature                     | Purpose                        |
| ------------------------- | ----------------------------- | ------------------------------ |
| `check_finite`            | `(np.ndarray) -> bool`        | All values finite (no NaN/Inf) |
| `check_shape`             | `(np.ndarray, tuple) -> bool` | Shape matches expected         |
| `check_positive`          | `(value) -> bool`             | All values $> 0$               |
| `check_non_negative`      | `(value) -> bool`             | All values $\geq 0$            |
| `check_symmetric`         | `(matrix, tol) -> bool`       | $\|A - A^T\| < \text{tol}$     |
| `check_positive_definite` | `(matrix) -> bool`            | All eigenvalues $> 0$          |

### 5.3.4 Convenience Aliases

```python
requires_initialized = require_state(
    lambda self: getattr(self, "_is_initialized", False), "initialized")

requires_model_loaded = require_state(
    lambda self: getattr(self, "model", None) is not None, "model loaded")

@finite_result       # Postcondition: result has no NaN/Inf
@non_empty_result    # Postcondition: result.size > 0
```

---

## 5.4 `BasePhysicsEngine`

**File:** `src/shared/python/base_physics_engine.py`

`BasePhysicsEngine` is an abstract base class that factors out behaviour common
to all engine implementations:

```python
class BasePhysicsEngine(ContractChecker, PhysicsEngine):
    def __init__(self, allowed_dirs: list[Path] | None = None):
        self.model: Any = None
        self.data: Any = None
        self.state: EngineState | None = None
        self._is_initialized = False
```

### 5.4.1 `EngineState` Data Class

```python
class EngineState:
    def __init__(self, nq: int = 0, nv: int = 0):
        self.q: np.ndarray = np.zeros(nq)    # Positions
        self.v: np.ndarray = np.zeros(nv)    # Velocities
        self.a: np.ndarray = np.zeros(nv)    # Accelerations
        self.tau: np.ndarray = np.zeros(nv)  # Torques/forces
        self.time: float = 0.0
```

### 5.4.2 Template Method Pattern

`BasePhysicsEngine` uses the Template Method pattern. Concrete engines implement
the private `_impl` hooks while the base class handles validation:

```python
# Base class (handles preconditions, invariants, logging)
@log_errors("Failed to load model from path", reraise=True)
@invariant_checked
def load_from_path(self, path: str) -> None:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if self.allowed_dirs:
        validated_path = validate_path(path, self.allowed_dirs, strict=True)
    self._load_from_path_impl(path)    # <-- concrete engine
    self._is_initialized = True

# Concrete engine must implement:
@abstractmethod
def _load_from_path_impl(self, path: str) -> None: ...

@abstractmethod
def _load_from_string_impl(self, content: str, extension: str | None) -> None: ...
```

### 5.4.3 Invariants

The base class defines three invariants that all engines must maintain:

1. **Model consistency:** If `_is_initialized` is `True`, then `model` is not `None`.
2. **Dimension compatibility:** `len(q) == len(v)` or `len(q) == len(v) + 1`
   (quaternion representation).
3. **Non-negative time:** `state.time >= 0.0`.

### 5.4.4 Checkpointing

`BasePhysicsEngine` provides default `save_checkpoint()` / `restore_checkpoint()`
implementations that serialise the `EngineState` into a `StateCheckpoint` object.

---

## 5.5 Engine Registry

**File:** `src/shared/python/engine_registry.py`

The registry decouples engine _discovery_ from engine _instantiation_.

### 5.5.1 `EngineType` Enumeration

```python
class EngineType(Enum):
    MUJOCO    = "mujoco"
    DRAKE     = "drake"
    PINOCCHIO = "pinocchio"
    OPENSIM   = "opensim"
    MYOSIM    = "myosim"
    MATLAB_2D = "matlab_2d"
    MATLAB_3D = "matlab_3d"
    PENDULUM  = "pendulum"
```

### 5.5.2 `EngineStatus` Enumeration

```python
class EngineStatus(Enum):
    AVAILABLE   = "available"
    UNAVAILABLE = "unavailable"
    LOADING     = "loading"
    LOADED      = "loaded"
    ERROR       = "error"
```

### 5.5.3 Registration Data

```python
@dataclass
class EngineRegistration:
    engine_type: EngineType
    factory: EngineFactory        # Callable[[], PhysicsEngine]
    registration_path: Path | None = None
    requires_binary: list[str] = field(default_factory=list)
    probe_class: type | None = None
```

### 5.5.4 Global Singleton

```python
_registry = EngineRegistry()

def get_registry() -> EngineRegistry:
    return _registry
```

The `EngineRegistry` class exposes three methods:

- `register(registration)` -- add an engine.
- `get(engine_type) -> EngineRegistration | None` -- look up by type.
- `all_types() -> list[EngineType]` -- enumerate registered types.

---

## 5.6 Engine Manager

**File:** `src/shared/python/engine_manager.py`

The `EngineManager` is the user-facing facade that ties the registry, probes, and
engine factories together.

```python
class EngineManager:
    def __init__(self, suite_root: Path | None = None):
        self.current_engine: EngineType | None = None
        self.active_physics_engine: PhysicsEngine | None = None
        self.engine_status: dict[EngineType, EngineStatus] = {}
```

### 5.6.1 Initialisation Sequence

1. Compute engine paths relative to `suite_root`.
2. Instantiate all probes (MuJoCoProbe, DrakeProbe, PinocchioProbe, ...).
3. Register factories from the `LOADER_MAP` into the global `EngineRegistry`.
4. Call `_discover_engines()` to populate `engine_status`.

### 5.6.2 Key Methods

| Method                        | Description                                                   |
| ----------------------------- | ------------------------------------------------------------- |
| `get_available_engines()`     | Returns `list[EngineType]` of engines with status AVAILABLE.  |
| `switch_engine(engine_type)`  | Loads and activates a different backend. Returns `bool`.      |
| `get_active_physics_engine()` | Returns the current `PhysicsEngine` instance or `None`.       |
| `probe_all_engines()`         | Runs readiness probes for every engine and caches results.    |
| `get_diagnostic_report()`     | Returns a human-readable string report with fix instructions. |
| `cleanup()`                   | Shuts down MATLAB engines, releases references.               |

### 5.6.3 Engine Switching

```python
def switch_engine(self, engine_type: EngineType) -> bool:
    if self.engine_status[engine_type] != EngineStatus.AVAILABLE:
        return False
    self._load_engine(engine_type)
    self.current_engine = engine_type
    return True
```

Standard engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSim) are loaded through
the registry factory. Special cases (MATLAB, Pendulum) use dedicated loaders.

---

## 5.7 Engine Probes

**File:** `src/shared/python/engine_probes.py`

Each engine has a corresponding probe class that checks installation status at runtime.

### 5.7.1 `ProbeStatus` Enumeration

```python
class ProbeStatus(Enum):
    AVAILABLE            = "available"
    MISSING_BINARY       = "missing_binary"
    MISSING_ASSETS       = "missing_assets"
    VERSION_MISMATCH     = "version_mismatch"
    NOT_INSTALLED        = "not_installed"
    CONFIGURATION_ERROR  = "configuration_error"
```

### 5.7.2 `EngineProbeResult`

```python
@dataclass
class EngineProbeResult:
    engine_name: str
    status: ProbeStatus
    version: str | None
    missing_dependencies: list[str]
    diagnostic_message: str
    details: dict[str, Any] | None = None

    def is_available(self) -> bool: ...
    def get_fix_instructions(self) -> str: ...
```

### 5.7.3 Probe Implementations

| Probe            | Checks                                                                    |
| ---------------- | ------------------------------------------------------------------------- |
| `MuJoCoProbe`    | `import mujoco`, engine dir, Python modules, XML model files              |
| `DrakeProbe`     | `import pydrake`, `pydrake.multibody`, MeshCat port 7000-7010, engine dir |
| `PinocchioProbe` | `import pinocchio`, engine dir, `pinocchio_golf` module                   |
| `OpenSimProbe`   | `import opensim`, engine dir                                              |
| `MyoSimProbe`    | `import mujoco` (dependency), engine dir, `myosim_physics_engine.py`      |
| `MatlabProbe`    | `import matlab.engine`, model directory (2D or 3D), `.slx`/`.m` files     |
| `PendulumProbe`  | Engine dir, `constants.py`, `pendulum_solver.py`                          |

The `DrakeProbe` additionally tests MeshCat port availability by attempting to bind
to ports 7000 through 7010, reporting `CONFIGURATION_ERROR` if all are blocked.

---

---
