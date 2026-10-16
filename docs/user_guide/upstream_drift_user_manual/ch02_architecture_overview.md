# Chapter 2: Architecture Overview

## 2.1 High-Level System Architecture

UpstreamDrift is structured as a layered, modular system. The following diagram
illustrates the primary architectural layers from user-facing interfaces down to
the physics engine backends:

```
+===========================================================================+
|                        USER INTERFACE LAYER                                |
|  +-------------------+  +---------------------+  +---------------------+  |
|  | PyQt6 Classic GUI |  | Tauri/React Web UI  |  | REST API (FastAPI)  |  |
|  +-------------------+  +---------------------+  +---------------------+  |
+===========================================================================+
                                    |
+===========================================================================+
|                        LAUNCHER & ORCHESTRATION                            |
|  +--------------------+  +------------------+  +----------------------+   |
|  | upstream_drift_launcher    |  | launcher services  |  | engine dashboards    |   |
|  +--------------------+  +------------------+  +----------------------+   |
+===========================================================================+
                                    |
+===========================================================================+
|                        SHARED SERVICES LAYER                               |
|  +-----------+ +----------+ +---------+ +----------+ +---------+          |
|  | Analysis  | | Plotting | | Optim.  | | Signal   | | UI Lib  |          |
|  +-----------+ +----------+ +---------+ +----------+ +---------+          |
|  +-----------+ +----------+ +---------+ +----------+ +---------+          |
|  | Contracts | | Logging  | | Config  | | Security | | Injury  |          |
|  +-----------+ +----------+ +---------+ +----------+ +---------+          |
+===========================================================================+
                                    |
+===========================================================================+
|                        ENGINE ABSTRACTION LAYER                            |
|  +--------------------------------------------------------------------+  |
|  |            Unified Engine Interface (PhysicsEngine Protocol)        |  |
|  +--------------------------------------------------------------------+  |
|  |  EngineManager  |  EngineRegistry  |  CrossEngineValidator          |  |
|  +--------------------------------------------------------------------+  |
+===========================================================================+
                                    |
+===========================================================================+
|                        PHYSICS ENGINE BACKENDS                             |
|  +---------+ +-------+ +-----------+ +---------+ +----------+            |
|  | MuJoCo  | | Drake | | Pinocchio | | OpenSim | | MyoSuite |            |
|  | (3.3+)  | | (1.22)| | (2.6+)    | |         | |          |            |
|  +---------+ +-------+ +-----------+ +---------+ +----------+            |
+===========================================================================+
                                    |
+===========================================================================+
|                        DOMAIN MODULES                                      |
|  +----------+ +-----------+ +------------+ +------------+                 |
|  | Robotics | | Learning  | | Deployment | | Research   |                 |
|  | contact  | | RL        | | realtime   | | MPC        |                 |
|  | control  | | imitation | | digital    | | diff.phys  |                 |
|  | locomot. | | sim2real  | |   twin     | | deformable |                 |
|  | sensing  | | retarget  | | safety     | | multi-robot|                 |
|  | planning | |           | | teleop     | |            |                 |
|  +----------+ +-----------+ +------------+ +------------+                 |
+===========================================================================+
```

### Layer Responsibilities

**User Interface Layer**: Provides three distinct interfaces for interacting with
the platform. The PyQt6 GUI offers a native desktop experience with rich widget
support. The Tauri/React application provides a modern web-based desktop experience.
The FastAPI REST API enables programmatic access, automation, and integration
with external systems.

**Launcher and Orchestration Layer**: Manages application startup, engine selection,
model loading, and inter-process communication. The `upstream_drift_launcher.py`
module serves as the primary orchestrator for the PyQt6 interface, while
`launch_upstream_drift.py` provides the CLI entry point supporting all launch modes.

**Shared Services Layer**: Houses reusable libraries consumed by all engines and
modules, including biomechanical analysis routines, plotting utilities, optimization
solvers, signal processing tools, UI components, configuration management,
Design by Contract infrastructure, structured logging, and security utilities.

**Engine Abstraction Layer**: Defines the `PhysicsEngine` protocol (a Python
structural typing interface) and provides the `EngineManager` and `EngineRegistry`
for dynamic engine discovery, loading, and lifecycle management. The
`CrossEngineValidator` enables comparison of results across engines.

**Physics Engine Backends**: The five physics engines, each wrapped in an
adapter that conforms to the `PhysicsEngine` protocol. Each engine has its own
sub-directory under `src/engines/physics_engines/` containing Python bindings,
models, configuration files, and engine-specific utilities.

**Domain Modules**: Specialized modules for robotics (contact, control, locomotion,
sensing, planning), learning (RL, imitation, sim-to-real, retargeting), deployment
(real-time control, digital twin, safety, teleoperation), and research (MPC,
differentiable physics, deformable objects, multi-robot coordination).

## 2.2 Multi-Engine Design

The central architectural innovation of UpstreamDrift is its multi-engine design.
Rather than committing to a single physics simulation library, the platform
abstracts engine-specific details behind a common protocol and provides mechanisms
for running the same model across multiple engines.

### The PhysicsEngine Protocol

All engines conform to the `PhysicsEngine` protocol defined in
`src/shared/python/base_physics_engine.py`. This protocol specifies the minimum
interface that any engine adapter must implement:

```python
class PhysicsEngine(Protocol):
    """Protocol defining the interface for all physics engine adapters."""

    def load_model(self, model_path: str, **kwargs) -> None:
        """Load a model file (URDF, MJCF, XML, etc.)."""
        ...

    def step(self, dt: float) -> None:
        """Advance the simulation by one time step."""
        ...

    def get_state(self) -> dict[str, np.ndarray]:
        """Return the current simulation state (q, qdot, etc.)."""
        ...

    def set_control(self, torques: np.ndarray) -> None:
        """Apply joint torques or control inputs."""
        ...

    def get_dynamics(self) -> dict[str, np.ndarray]:
        """Return dynamics matrices (M, C, g, etc.)."""
        ...

    def reset(self) -> None:
        """Reset the simulation to initial conditions."""
        ...
```

Each engine adapter (e.g., `MuJoCoAdapter`, `DrakeAdapter`, `PinocchioAdapter`)
implements this protocol, translating generic calls into engine-specific API
invocations. This allows analysis code to remain engine-agnostic:

```python
from src.shared.python.engine_core.engine_manager import EngineManager

manager = EngineManager()
engine = manager.get_engine("mujoco")  # or "drake", "pinocchio", etc.
engine.load_model("models/humanoid_28dof.urdf")
engine.step(dt=0.001)
state = engine.get_state()
```

### Engine Capabilities Matrix

Not all engines support all features. The following matrix summarizes the
capabilities of each engine:

| Capability               | MuJoCo       | Drake   | Pinocchio  | OpenSim | MyoSuite |
| ------------------------ | ------------ | ------- | ---------- | ------- | -------- |
| Forward dynamics         | Yes          | Yes     | Yes        | Yes     | Yes      |
| Inverse dynamics         | Yes          | Yes     | Yes        | Yes     | Limited  |
| Inverse kinematics       | Yes          | Yes     | Yes        | Yes     | No       |
| Contact dynamics         | Yes          | Yes     | Limited    | Limited | Yes      |
| Trajectory optimization  | Limited      | Yes     | Via CasADi | No      | No       |
| Muscle models            | Via MyoSuite | No      | No         | Yes     | Yes      |
| URDF loading             | Via MJCF     | Yes     | Yes        | Via XML | Via MJCF |
| Real-time visualization  | Yes          | Meshcat | Meshcat    | GUI     | MuJoCo   |
| Differentiable dynamics  | Limited      | Yes     | Via CasADi | No      | No       |
| Maximum DOF (tested)     | 28           | 28      | 28         | 37      | 28       |
| Maximum muscles (tested) | 290          | 0       | 0          | 290     | 290      |

### Engine Selection Logic

The `EngineManager` implements a selection heuristic that considers:

1. **Requested features**: If the user needs muscle dynamics, engines without
   muscle support are excluded.
2. **Model format**: URDF models are natively supported by Drake and Pinocchio;
   MJCF models are natively supported by MuJoCo.
3. **Performance requirements**: Pinocchio is the fastest for pure rigid-body
   computations; MuJoCo offers the best balance of features and speed.
4. **Availability**: Only engines that are installed and importable are offered.

The `engine_availability.py` module probes the system at startup to determine
which engines are available, using lazy imports to avoid loading heavy native
libraries until they are actually needed.

## 2.3 Module Structure and Package Organization

The repository is organized into the following top-level directories:

```
UpstreamDrift/
├── src/                          # Main source code
│   ├── api/                      # FastAPI REST API server
│   │   ├── auth/                 #   Authentication (JWT, bcrypt)
│   │   ├── middleware/           #   Security headers, upload limits
│   │   ├── models/               #   Pydantic request/response models
│   │   ├── routes/               #   Endpoint definitions
│   │   ├── services/             #   Business logic services
│   │   └── utils/                #   Tracing, error codes, path validation
│   ├── config/                   # Global configuration
│   ├── deployment/               # Deployment modules
│   │   ├── digital_twin/         #   Digital twin estimator and twin
│   │   ├── realtime/             #   Real-time controller and state
│   │   ├── safety/               #   Collision detection, safety monitor
│   │   └── teleoperation/        #   Devices, teleoperation interface
│   ├── engines/                  # Physics engine backends
│   │   ├── common/               #   Shared engine utilities
│   │   ├── pendulum_models/      #   Educational pendulum models
│   │   ├── physics_engines/      #   Engine adapters
│   │   │   ├── drake/            #     Drake engine
│   │   │   ├── mujoco/           #     MuJoCo engine
│   │   │   ├── myosuite/         #     MyoSuite engine
│   │   │   ├── opensim/          #     OpenSim engine
│   │   │   ├── pinocchio/        #     Pinocchio engine
│   │   │   ├── pendulum/         #     Pendulum engine wrapper
│   │   │   └── putting_green/    #     Putting simulation
│   │   └── Simscape_Multibody_Models/  # MATLAB models
│   ├── launchers/                # GUI launchers and dashboards
│   ├── learning/                 # Machine learning modules
│   │   ├── imitation/            #   Imitation learning
│   │   ├── retargeting/          #   Motion retargeting
│   │   ├── rl/                   #   Reinforcement learning
│   │   └── sim2real/             #   Sim-to-real transfer
│   ├── research/                 # Research modules
│   │   ├── deformable/           #   Deformable object simulation
│   │   ├── differentiable/       #   Differentiable physics
│   │   ├── mpc/                  #   Model Predictive Control
│   │   └── multi_robot/          #   Multi-robot coordination
│   ├── robotics/                 # Robotics modules
│   │   ├── contact/              #   Contact models
│   │   ├── control/              #   Control algorithms
│   │   ├── core/                 #   Core robotics utilities
│   │   ├── locomotion/           #   Locomotion planning
│   │   ├── planning/             #   Motion planning
│   │   └── sensing/              #   Sensing and estimation
│   ├── shared/                   # Shared libraries
│   │   ├── matlab/               #   MATLAB utilities
│   │   ├── models/               #   Model files (URDF, MJCF, XML)
│   │   ├── python/               #   Python shared modules (~120 files)
│   │   ├── tools/                #   Development and analysis tools
│   │   └── urdf/                 #   URDF generation utilities
│   ├── tools/                    # CLI tools and utilities
│   └── unreal_integration/       # Unreal Engine integration
├── tests/                        # Test suite
│   ├── unit/                     #   Unit tests
│   ├── integration/              #   Integration tests
│   ├── cross_engine/             #   Cross-engine comparison tests
│   ├── physics_validation/       #   Physics accuracy tests
│   ├── security/                 #   Security tests
│   ├── benchmarks/               #   Performance benchmarks
│   ├── deployment/               #   Deployment tests
│   ├── learning/                 #   Learning module tests
│   ├── research/                 #   Research module tests
│   ├── analytical/               #   Analytical solution tests
│   ├── acceptance/               #   Acceptance tests
│   ├── headless/                 #   Headless (no-GUI) tests
│   └── fixtures/                 #   Test fixtures and data
├── ui/                           # Tauri/React web desktop app
│   ├── src/                      #   React source code
│   ├── src-tauri/                #   Tauri (Rust) backend
│   └── public/                   #   Static assets
├── docs/                         # Documentation
├── examples/                     # Example scripts
├── scripts/                      # Development and maintenance scripts
├── data/                         # Data files
├── assets/                       # Branding and static assets
├── installer/                    # OS-specific installers
├── vendor/                       # Vendored third-party code
├── launch_upstream_drift.py          # Main CLI entry point
├── start_api_server.py           # API server entry point
├── pyproject.toml                # Project metadata and dependencies
├── environment.yml               # Conda environment specification
├── Makefile                      # Development task automation
├── Dockerfile                    # Container image definition
├── Dockerfile.unified            # Unified container image
└── conftest.py                   # Root pytest configuration
```

### The `src/shared/python/` Module

The `shared/python/` directory is the largest single module, containing
approximately 120 Python files that provide reusable functionality consumed across
all engines and analysis modules. Key components include:

- **`base_physics_engine.py`**: The `PhysicsEngine` protocol definition.
- **`engine_manager.py`**: Dynamic engine loading and lifecycle management.
- **`engine_registry.py`**: Engine capability registration and discovery.
- **`contracts.py`**: Design by Contract decorators and utilities.
- **`logging_config.py`**: Centralized structured logging via `structlog`.
- **`analysis/`**: Biomechanical analysis routines.
- **`plotting/`**: Visualization and chart generation (10+ plot types).
- **`optimization/`**: Numerical optimization solvers.
- **`signal_toolkit/`**: Signal processing utilities for time-series data.
- **`ui/`**: Reusable UI widgets and adapters (PyQt6).
- **`injury/`**: Injury risk assessment algorithms.
- **`spatial_algebra/`**: Spatial vector algebra for rigid-body computations.
- **`aerodynamics.py`**: Ball flight aerodynamic models.
- **`impact_model.py`**: Club-ball impact mechanics.
- **`hill_muscle.py`**: Hill-type muscle model implementation.
- **`multi_muscle.py`**: Multi-muscle coordination solver.
- **`flexible_shaft.py`**: Club shaft flexibility model.
- **`grip_contact_model.py`**: Two-handed grip contact analysis.
- **`ground_reaction_forces.py`**: Ground reaction force computation.
- **`terrain.py`** / **`terrain_engine.py`**: Terrain and topography models.

## 2.4 Design by Contract (DbC)

UpstreamDrift employs Design by Contract as a central quality assurance mechanism.
The `src/shared/python/contracts.py` module provides three decorator types that
enforce formal specifications at function and class boundaries.

### Preconditions

Preconditions specify what must be true **before** a function or method executes.
They validate input parameters and object state:

```python
from src.shared.python.contracts import precondition

@precondition(lambda self: self._is_initialized, "Engine must be initialized")
@precondition(lambda self, dt: dt > 0, "Time step must be positive")
def step(self, dt: float) -> None:
    """Advance the simulation by dt seconds."""
    ...
```

If a precondition fails, a `PreconditionViolationError` is raised with a clear
diagnostic message indicating which condition was violated and in which function.

### Postconditions

Postconditions specify what must be true **after** a function completes. They
validate return values and output state:

```python
from src.shared.python.contracts import postcondition

@postcondition(lambda result: result.shape[0] > 0, "Result must be non-empty")
@postcondition(
    lambda result: np.all(np.isfinite(result)),
    "Result must contain only finite values"
)
def compute_acceleration(self) -> np.ndarray:
    """Compute joint-space acceleration vector."""
    ...
```

### Class Invariants

Class invariants specify conditions that must hold true for every instance of a
class, checked after every public method call:

```python
from src.shared.python.contracts import invariant

@invariant(lambda self: self.mass > 0, "Mass must be positive")
@invariant(lambda self: len(self.joints) == self.n_dof, "Joint count must match DOF")
class RigidBody:
    """Represents a rigid body in the simulation."""
    ...
```

### Performance Considerations

Contract checking introduces runtime overhead proportional to the complexity of
the contract predicates. For performance-critical production deployments, contracts
can be globally disabled:

```python
from src.shared.python.contracts import CONTRACTS_ENABLED
import src.shared.python.contracts as contracts_module

# Disable all contract checking
contracts_module.CONTRACTS_ENABLED = False
```

When disabled, the contract decorators become zero-cost pass-throughs. In
development and testing, contracts should always remain enabled to catch
violations early.

## 2.5 Dependency Graph

The following simplified dependency graph shows the major inter-module
relationships. Arrows indicate "depends on" relationships:

```
launch_upstream_drift.py
    ├── src/api/local_server
    │     └── src/api/server (FastAPI app)
    │           ├── src/api/routes/* (endpoints)
    │           ├── src/api/services/* (business logic)
    │           ├── src/api/auth/* (JWT, bcrypt)
    │           ├── src/api/middleware/* (security, upload)
    │           └── src/shared/python/engine_manager
    ├── src/launchers/upstream_drift_launcher (PyQt6)
    │     ├── src/launchers/ui_components
    │     ├── src/shared/python/config/model_registry
    │     └── src/shared/python/engine_availability
    └── src/engines/physics_engines/*/
          └── (engine-specific modules)

src/shared/python/ (consumed by all modules above)
    ├── base_physics_engine (Protocol)
    ├── engine_manager / engine_registry
    ├── contracts (DbC)
    ├── logging_config (structlog)
    ├── analysis/ plotting/ optimization/
    ├── biomechanics: hill_muscle, multi_muscle, grip_contact_model, ...
    ├── physics: aerodynamics, impact_model, flexible_shaft, terrain, ...
    └── security_utils, env_validator, config_utils, ...

src/robotics/ (contact, control, locomotion, sensing, planning)
    └── src/shared/python/

src/learning/ (rl, imitation, sim2real, retargeting)
    ├── src/engines/physics_engines/*/
    └── src/shared/python/

src/deployment/ (realtime, digital_twin, safety, teleoperation)
    ├── src/engines/physics_engines/*/
    └── src/shared/python/

src/research/ (mpc, differentiable, deformable, multi_robot)
    ├── src/engines/physics_engines/*/
    └── src/shared/python/
```

### External Dependencies

The platform depends on the following major external libraries:

| Category             | Library         | Version    | Purpose                           |
| -------------------- | --------------- | ---------- | --------------------------------- |
| Scientific Computing | NumPy           | >= 1.26.4  | Array computation, linear algebra |
| Scientific Computing | SciPy           | >= 1.13.1  | Optimization, integration, signal |
| Scientific Computing | pandas          | >= 2.2.3   | Data manipulation and analysis    |
| Scientific Computing | SymPy           | >= 1.12    | Symbolic mathematics              |
| Physics Engine       | MuJoCo          | >= 3.3.0   | Primary physics engine            |
| Physics Engine       | Drake           | >= 1.22.0  | Trajectory optimization engine    |
| Physics Engine       | Pinocchio (pin) | >= 2.6.0   | Fast rigid-body algorithms        |
| Web Framework        | FastAPI         | >= 0.126.0 | REST API framework                |
| Web Server           | Uvicorn         | >= 0.30.0  | ASGI server                       |
| Data Validation      | Pydantic        | >= 2.5.0   | Request/response model validation |
| Database             | SQLAlchemy      | >= 2.0.0   | ORM and database abstraction      |
| Authentication       | PyJWT           | >= 2.10.1  | JWT token handling                |
| Password Hashing     | bcrypt          | >= 4.1.0   | Secure password/key hashing       |
| Rate Limiting        | SlowAPI         | >= 0.1.9   | API rate limiting                 |
| GUI                  | PyQt6           | >= 6.5.0   | Desktop GUI framework             |
| Visualization        | Matplotlib      | >= 3.8.0   | Plotting and chart generation     |
| Computer Vision      | OpenCV          | >= 4.8.0   | Image/video processing            |
| Logging              | structlog       | >= 24.1.0  | Structured logging                |
| Data Formats         | PyYAML          | >= 6.0.1   | YAML configuration parsing        |
| Data Formats         | defusedxml      | >= 0.7.1   | Secure XML parsing                |
| Data Formats         | ezc3d           | >= 1.4.0   | C3D motion capture file support   |
| Security             | cryptography    | >= 44.0.1  | Cryptographic operations          |
| Expression Eval      | simpleeval      | >= 1.0.0   | Safe expression evaluation        |
| HTTP Client          | httpx           | >= 0.27.0  | Async HTTP client for testing     |

## 2.6 Data Flow and Communication Patterns

### Simulation Data Flow

A typical simulation run follows this data flow:

1. **Model Loading**: The user selects an engine and model. The `EngineManager`
   loads the appropriate model file (URDF, MJCF, or XML) into the selected
   physics engine via the `PhysicsEngine.load_model()` method.

2. **Initialization**: Initial conditions (joint positions $q_0$, velocities
   $\dot{q}_0$) are set. Muscle activation levels, control parameters, and
   environmental conditions are configured.

3. **Simulation Loop**: For each time step $\Delta t$:

   - Control inputs (torques $\tau$ or muscle activations $a$) are computed
     by the control module or provided by the user.
   - The engine advances the state: $q_{k+1}, \dot{q}_{k+1} = f(q_k, \dot{q}_k, \tau_k, \Delta t)$.
   - Contact forces $F_{\text{ext}}$ are resolved by the engine's contact solver.
   - State data is recorded by the analysis pipeline.

4. **Post-Processing**: After simulation, the analysis module computes derived
   quantities (energies, forces, injury risk metrics) and generates plots.

5. **Export**: Results are saved to CSV, JSON, or the SQLite database
   (`golf_modeling_suite.db`).

### API Communication Pattern

When accessed via the REST API, the data flow is:

```
Client (Browser/Script)
    |
    | HTTP/WebSocket (JSON)
    v
FastAPI Server (src/api/server.py)
    |
    | JWT Authentication + Rate Limiting
    v
Route Handler (src/api/routes/*.py)
    |
    | Pydantic Validation
    v
Service Layer (src/api/services/*.py)
    |
    | Engine Abstraction
    v
EngineManager -> PhysicsEngine Adapter -> Engine Backend
    |
    | Results (numpy arrays)
    v
Response Serialization (Pydantic) -> JSON -> Client
```

## 2.7 Security Architecture

The security architecture implements defense in depth:

**Authentication**: JWT (JSON Web Tokens) with configurable expiration.
Tokens are signed using the `SECRET_KEY` environment variable and verified
on every authenticated request via the `src/api/auth/dependencies.py` module.

**Password/Key Hashing**: bcrypt with automatic salt generation. The platform
previously used SHA256 (fast hash, vulnerable to brute-force); this was upgraded
in v2.1.0 as a critical security fix.

**Input Validation**: All API inputs are validated through Pydantic models with
strict type checking and field constraints. File uploads are subject to size
limits enforced by `src/api/middleware/upload_limits.py`. File paths are validated
against directory traversal attacks via `src/api/utils/path_validation.py`.

**Security Headers**: The `src/api/middleware/security_headers.py` module adds
standard security headers (X-Content-Type-Options, X-Frame-Options,
Content-Security-Policy, etc.) to all responses.

**Rate Limiting**: The SlowAPI middleware limits request rates per IP address
to prevent abuse and denial-of-service attacks.

**CORS**: Cross-Origin Resource Sharing is restricted to explicitly configured
origins via the `get_cors_origins()` configuration function.

**Trusted Hosts**: The `TrustedHostMiddleware` ensures that requests are only
accepted from configured hostnames, preventing host header injection.

**Safe Expression Evaluation**: The `simpleeval` library replaces Python's
built-in `eval()` for any user-provided mathematical expressions, preventing
arbitrary code execution (CWE-94 mitigation).

**XML Security**: All XML parsing uses `defusedxml` to prevent XML external
entity (XXE) attacks and billion-laughs denial-of-service attacks.

---
