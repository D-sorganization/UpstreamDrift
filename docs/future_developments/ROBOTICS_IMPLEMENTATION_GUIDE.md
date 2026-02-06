# UpstreamDrift Robotics Implementation Guide

## By Robots, For Robots

**Version**: 1.0
**Date**: 2026-02-03
**Status**: Implementation Guide

---

## Core Philosophy

> "All implementations must be engine-agnostic and available to adapt to anything."

This guide establishes the mandatory principles, patterns, and practices for implementing the robotics expansion. Every contributor must follow these guidelines without exception.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [Architecture Overview](#architecture-overview)
3. [Design by Contract Requirements](#design-by-contract-requirements)
4. [Test-Driven Development Protocol](#test-driven-development-protocol)
5. [Code Quality Standards](#code-quality-standards)
6. [Implementation Phases](#implementation-phases)
7. [API Integration Requirements](#api-integration-requirements)
8. [Engine Agnosticism](#engine-agnosticism)

---

## Design Principles

### The Non-Negotiables

#### 1. DRY (Don't Repeat Yourself)

- **Every piece of knowledge must have a single, unambiguous, authoritative representation.**
- If you find yourself copying code, extract it into a shared function/class.
- Use mixins for cross-cutting concerns.
- Configuration belongs in one place.

#### 2. Orthogonality

- **Components must be independent and interchangeable.**
- Changing one module should not require changes in unrelated modules.
- Each function/class does ONE thing well.
- Dependencies flow downward through clean interfaces.

#### 3. Reversibility

- **Design for change.**
- Abstract away engine-specific details behind protocols.
- Use dependency injection.
- Avoid hard-coded assumptions about specific implementations.

#### 4. Reusability

- **Build components as libraries, not applications.**
- Every class should be usable in contexts you haven't imagined.
- Prefer composition over inheritance.
- Design for extension without modification (Open/Closed Principle).

### The SOLID Principles

| Principle                 | Application in UpstreamDrift                       |
| ------------------------- | -------------------------------------------------- |
| **S**ingle Responsibility | Each class has one reason to change                |
| **O**pen/Closed           | Extend via protocols, not modification             |
| **L**iskov Substitution   | All engines interchangeable via Protocol           |
| **I**nterface Segregation | Small, focused protocols                           |
| **D**ependency Inversion  | Depend on abstractions (Protocol), not concretions |

---

## Architecture Overview

### Directory Structure for Robotics Modules

```
src/
├── robotics/                          # NEW: Main robotics module
│   ├── __init__.py
│   ├── core/                          # Core abstractions
│   │   ├── __init__.py
│   │   ├── types.py                   # Shared type definitions
│   │   ├── protocols.py               # Protocol definitions
│   │   └── exceptions.py              # Robotics-specific exceptions
│   ├── contact/                       # Phase 1: Contact dynamics
│   │   ├── __init__.py
│   │   ├── contact_state.py           # ContactState dataclass
│   │   ├── contact_manager.py         # ContactManager class
│   │   ├── friction_cone.py           # Friction cone utilities
│   │   └── grasp_analysis.py          # Grasp quality metrics
│   ├── sensing/                       # Phase 1: Force/torque sensing
│   │   ├── __init__.py
│   │   ├── force_torque_sensor.py     # F/T sensor simulation
│   │   ├── imu_sensor.py              # IMU simulation
│   │   └── noise_models.py            # Sensor noise models
│   ├── control/                       # Phase 1: Control systems
│   │   ├── __init__.py
│   │   ├── whole_body/                # Whole-body control
│   │   │   ├── __init__.py
│   │   │   ├── task.py                # Task descriptors
│   │   │   ├── qp_solver.py           # QP-based solver
│   │   │   └── wbc_controller.py      # WholeBodyController
│   │   └── operational_space/         # Task-space control
│   │       ├── __init__.py
│   │       └── osc_controller.py
│   ├── locomotion/                    # Phase 1: Locomotion
│   │   ├── __init__.py
│   │   ├── gait/                      # Gait generation
│   │   │   ├── __init__.py
│   │   │   ├── zmp_planner.py         # ZMP-based planning
│   │   │   ├── capture_point.py       # DCM/ICP
│   │   │   └── cpg.py                 # Central pattern generators
│   │   ├── balance/                   # Balance control
│   │   │   ├── __init__.py
│   │   │   ├── lipm.py                # Linear inverted pendulum
│   │   │   └── centroidal.py          # Centroidal dynamics
│   │   └── footstep/                  # Footstep planning
│   │       ├── __init__.py
│   │       └── footstep_planner.py
│   ├── planning/                      # Phase 2: Motion planning
│   │   └── ...
│   ├── perception/                    # Phase 2: Perception
│   │   └── ...
│   ├── learning/                      # Phase 3: Learning
│   │   └── ...
│   ├── deployment/                    # Phase 4: Industrial
│   │   └── ...
│   └── analysis/                      # Analysis mixins
│       ├── __init__.py
│       ├── locomotion_metrics.py
│       └── manipulation_metrics.py
├── api/
│   └── routes/
│       └── robotics.py                # NEW: Robotics API endpoints
└── shared/
    └── python/
        └── interfaces.py              # EXTEND: Add robotics protocols
```

### Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer                                │
│   (FastAPI routes, Pydantic models, dependency injection)        │
├─────────────────────────────────────────────────────────────────┤
│                      Service Layer                               │
│   (RoboticsService, SimulationService, AnalysisService)          │
├─────────────────────────────────────────────────────────────────┤
│                     Robotics Layer                               │
│   (Control, Locomotion, Contact, Sensing, Planning, Learning)    │
├─────────────────────────────────────────────────────────────────┤
│                    Protocol Layer                                │
│   (PhysicsEngine, HumanoidEngine, ManipulationEngine)            │
├─────────────────────────────────────────────────────────────────┤
│                     Engine Layer                                 │
│   (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design by Contract Requirements

### Every Function MUST Document

```python
def compute_contact_force(
    self,
    contact: ContactState,
    compliance: float,
) -> np.ndarray:
    """Compute contact force using compliance model.

    Design by Contract:
        Preconditions:
            - contact.penetration >= 0 (contact exists)
            - compliance > 0 (valid compliance parameter)
            - self._is_initialized (engine ready)

        Postconditions:
            - result.shape == (3,)
            - np.all(np.isfinite(result))
            - result @ contact.normal >= 0 (force in normal direction)

        Invariants:
            - Does not modify contact state
            - Pure computation (no side effects)

    Args:
        contact: Contact state information.
        compliance: Surface compliance [m/N].

    Returns:
        Contact force vector [fx, fy, fz] in world frame [N].

    Raises:
        PreconditionError: If preconditions violated.
        PostconditionError: If result invariants violated.
    """
```

### Contract Decorators Usage

```python
from src.shared.python.contracts import (
    precondition,
    postcondition,
    require_state,
    check_finite,
    check_positive,
    ContractChecker,
    invariant_checked,
)

class ContactManager(ContractChecker):
    """Manages multi-contact scenarios with contract enforcement."""

    def _get_invariants(self) -> list[tuple[Callable[[], bool], str]]:
        return [
            (lambda: len(self._contacts) >= 0, "Contact count non-negative"),
            (lambda: all(c.is_valid() for c in self._contacts), "All contacts valid"),
        ]

    @precondition(
        lambda self, q: check_finite(q),
        "Configuration must be finite"
    )
    @postcondition(
        lambda result: all(isinstance(c, ContactState) for c in result),
        "Result must contain ContactState objects"
    )
    @invariant_checked
    def detect_contacts(self, q: np.ndarray) -> list[ContactState]:
        """Detect contacts at configuration q."""
        ...
```

### Contract Verification Levels

| Level      | When to Use                                    | Performance Impact    |
| ---------- | ---------------------------------------------- | --------------------- |
| **Always** | Safety-critical (contact forces, joint limits) | Accept overhead       |
| **Debug**  | Development validation                         | Disable in production |
| **Test**   | Test assertions                                | Only in test suite    |

---

## Test-Driven Development Protocol

### The TDD Cycle

```
1. RED:    Write a failing test that defines desired behavior
2. GREEN:  Write minimal code to pass the test
3. REFACTOR: Clean up while keeping tests green
```

### Test File Organization

```
tests/
├── unit/
│   └── robotics/
│       ├── test_contact_state.py      # Unit tests for ContactState
│       ├── test_contact_manager.py    # Unit tests for ContactManager
│       ├── test_force_torque_sensor.py
│       ├── test_whole_body_controller.py
│       └── test_locomotion.py
├── integration/
│   └── robotics/
│       ├── test_contact_cross_engine.py   # Cross-engine contact tests
│       ├── test_wbc_with_engines.py       # WBC integration
│       └── test_locomotion_integration.py
├── acceptance/
│   └── robotics/
│       ├── test_humanoid_walking.py   # End-to-end walking
│       └── test_manipulation_pick.py  # End-to-end manipulation
└── fixtures/
    └── robotics/
        ├── humanoid_urdfs.py          # Test humanoid models
        └── contact_scenarios.py       # Contact test scenarios
```

### Test Naming Convention

```python
def test_<unit>_<scenario>_<expected_outcome>():
    """Test that <unit> <does expected thing> when <scenario>."""

# Examples:
def test_contact_manager_detects_single_point_contact():
    """Test that ContactManager detects point contact when sphere touches plane."""

def test_wbc_solver_respects_joint_limits_under_high_torque():
    """Test that WBC solver enforces limits when desired torque exceeds bounds."""

def test_zmp_planner_generates_stable_trajectory_on_flat_ground():
    """Test that ZMP planner produces stable trajectory when walking on flat surface."""
```

### Fixture Pattern for Engine Agnosticism

```python
import pytest
from src.shared.python.engine_availability import (
    MUJOCO_AVAILABLE,
    PINOCCHIO_AVAILABLE,
    DRAKE_AVAILABLE,
)

# Parametrize tests to run on all available engines
def get_available_engines():
    """Return list of available engine types for testing."""
    engines = []
    if MUJOCO_AVAILABLE:
        engines.append("mujoco")
    if PINOCCHIO_AVAILABLE:
        engines.append("pinocchio")
    if DRAKE_AVAILABLE:
        engines.append("drake")
    return engines

@pytest.fixture(params=get_available_engines())
def physics_engine(request, humanoid_urdf):
    """Fixture providing initialized physics engine for each available backend."""
    engine_type = request.param
    engine = create_engine(engine_type)
    engine.load_from_path(humanoid_urdf)
    yield engine
    # Cleanup if needed

@pytest.fixture
def humanoid_urdf(tmp_path):
    """Create test humanoid URDF."""
    # Standard test humanoid definition
    ...
```

### Test Coverage Requirements

| Module         | Required Coverage | Critical Paths              |
| -------------- | ----------------- | --------------------------- |
| Core Protocols | 100%              | All protocol methods        |
| Contact        | 95%               | Force computation, friction |
| Sensing        | 90%               | Measurement, noise models   |
| Control        | 95%               | QP solver, constraints      |
| Locomotion     | 90%               | Gait generation, balance    |

---

## Code Quality Standards

### Naming Conventions

```python
# Classes: PascalCase, noun phrases
class ContactManager:
class WholeBodyController:
class ForceTorqueSensor:

# Functions/Methods: snake_case, verb phrases
def detect_contacts() -> list[ContactState]:
def compute_friction_cone() -> np.ndarray:
def solve_qp() -> tuple[np.ndarray, bool]:

# Constants: SCREAMING_SNAKE_CASE
MAX_CONTACT_FORCE = 1000.0  # N
DEFAULT_FRICTION_COEFFICIENT = 0.5
GRAVITY_VECTOR = np.array([0, 0, -9.81])

# Private: leading underscore
def _validate_input():
_cached_jacobian: np.ndarray | None

# Type variables: Single uppercase or descriptive
T = TypeVar("T")
EngineT = TypeVar("EngineT", bound=PhysicsEngine)
```

### Function Size Guidelines

```python
# GOOD: Small, focused function
def compute_normal_force(penetration: float, compliance: float) -> float:
    """Compute normal contact force from penetration depth.

    Uses linear compliance model: f = penetration / compliance
    """
    return penetration / compliance

# BAD: Doing too much
def process_contact(contact, engine, state, params, config, ...):
    # 200 lines of mixed concerns
    ...

# BETTER: Decompose into focused functions
def process_contact(contact: ContactState, engine: PhysicsEngine) -> ContactResult:
    """Process contact through pipeline."""
    force = compute_contact_force(contact)
    friction = compute_friction_force(contact, force)
    return ContactResult(normal=force, friction=friction)
```

### Type Hints (Mandatory)

```python
from typing import Protocol, TypeVar, Generic
from collections.abc import Callable, Sequence
import numpy as np
from numpy.typing import NDArray

# All functions must have complete type hints
def compute_jacobian(
    engine: PhysicsEngine,
    body_name: str,
    point: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute spatial Jacobian.

    Returns:
        Tuple of (linear_jacobian, angular_jacobian).
    """
    ...

# Use Protocol for duck typing
@runtime_checkable
class Steppable(Protocol):
    def step(self, dt: float) -> None: ...

# Generic types for reusability
T = TypeVar("T")

class StateHistory(Generic[T]):
    def __init__(self, max_length: int) -> None:
        self._buffer: list[T] = []
        self._max_length = max_length
```

### Docstring Format (Google Style)

```python
def solve_inverse_kinematics(
    engine: PhysicsEngine,
    target_pose: np.ndarray,
    seed: np.ndarray | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """Solve inverse kinematics for target end-effector pose.

    Uses damped least-squares with nullspace optimization for
    redundancy resolution.

    Args:
        engine: Physics engine with loaded model.
        target_pose: Target pose as 7D vector [x, y, z, qw, qx, qy, qz].
        seed: Initial configuration guess. Uses current state if None.
        max_iterations: Maximum solver iterations.
        tolerance: Convergence tolerance for position error [m].

    Returns:
        Tuple containing:
            - solution: Joint configuration (n_q,) achieving target.
            - success: True if converged within tolerance.
            - info: Dictionary with solver statistics:
                - 'iterations': Number of iterations used
                - 'final_error': Final position error [m]
                - 'converged': Whether tolerance was met

    Raises:
        ValueError: If target_pose is not 7D.
        StateError: If engine not initialized.

    Example:
        >>> target = np.array([0.5, 0.0, 0.8, 1, 0, 0, 0])
        >>> q, success, info = solve_inverse_kinematics(engine, target)
        >>> if success:
        ...     engine.set_state(q, np.zeros_like(q))
    """
```

### Linting Configuration

```toml
# pyproject.toml

[tool.ruff]
line-length = 100
target-version = "py311"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "ARG", # flake8-unused-arguments
    "SIM", # flake8-simplify
]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

---

## Implementation Phases

### Phase 1: Humanoid Foundation (CURRENT)

| Component                     | Priority | Status | Dependencies |
| ----------------------------- | -------- | ------ | ------------ |
| Core Types & Protocols        | Critical | TODO   | None         |
| ContactState & ContactManager | Critical | TODO   | Core         |
| ForceTorqueSensor             | High     | TODO   | Core         |
| Whole-Body Control (HQP)      | Critical | TODO   | Contact      |
| Bipedal Locomotion (LIPM)     | Critical | TODO   | WBC          |
| API Integration               | High     | TODO   | All above    |
| Test Suite                    | Critical | TODO   | All above    |

### Phase 2: Perception & Planning

| Component               | Priority | Dependencies |
| ----------------------- | -------- | ------------ |
| Collision Checker       | Critical | Phase 1      |
| RRT/RRT\* Planners      | High     | Collision    |
| Trajectory Optimization | High     | Collision    |
| Perception Interface    | Medium   | Phase 1      |
| Scene Manager           | Medium   | Phase 1      |

### Phase 3: Learning & Adaptation

| Component              | Priority | Dependencies |
| ---------------------- | -------- | ------------ |
| Gymnasium Environments | Critical | Phase 1-2    |
| Domain Randomization   | High     | Gym Envs     |
| Imitation Learning     | High     | Gym Envs     |
| Motion Retargeting     | Medium   | Phase 1      |

### Phase 4: Industrial Deployment

| Component           | Priority | Dependencies |
| ------------------- | -------- | ------------ |
| Real-Time Interface | Critical | Phase 1      |
| Digital Twin        | High     | RT Interface |
| Safety System       | Critical | Phase 1-2    |
| Teleoperation       | High     | Phase 1      |

### Phase 5: Advanced Research

| Component              | Priority | Dependencies |
| ---------------------- | -------- | ------------ |
| MPC Framework          | High     | Phase 1      |
| Differentiable Physics | Medium   | Phase 1      |
| Multi-Robot            | Medium   | Phase 1-2    |
| Deformable Objects     | Low      | Phase 1      |

---

## API Integration Requirements

### Route Structure

```python
# src/api/routes/robotics.py

from fastapi import APIRouter, Depends, HTTPException
from src.api.dependencies import get_engine_manager
from src.api.models.robotics import (
    ContactQueryRequest,
    ContactQueryResponse,
    WBCRequest,
    WBCResponse,
    LocomotionCommandRequest,
)

router = APIRouter(prefix="/robotics", tags=["robotics"])

@router.post("/contacts/detect", response_model=ContactQueryResponse)
async def detect_contacts(
    request: ContactQueryRequest,
    engine_manager: EngineManager = Depends(get_engine_manager),
) -> ContactQueryResponse:
    """Detect contacts at current or specified configuration."""
    ...

@router.post("/control/wbc/solve", response_model=WBCResponse)
async def solve_whole_body_control(
    request: WBCRequest,
    engine_manager: EngineManager = Depends(get_engine_manager),
) -> WBCResponse:
    """Solve whole-body control problem."""
    ...

@router.post("/locomotion/command")
async def send_locomotion_command(
    request: LocomotionCommandRequest,
    engine_manager: EngineManager = Depends(get_engine_manager),
) -> dict:
    """Send velocity command to locomotion controller."""
    ...
```

### Pydantic Models

```python
# src/api/models/robotics.py

from pydantic import BaseModel, Field
from typing import Literal

class ContactState(BaseModel):
    """API model for contact state."""
    contact_id: int
    body_a: str
    body_b: str
    position: list[float] = Field(..., min_length=3, max_length=3)
    normal: list[float] = Field(..., min_length=3, max_length=3)
    penetration: float = Field(..., ge=0)
    normal_force: float = Field(..., ge=0)
    friction_force: list[float] = Field(..., min_length=3, max_length=3)

class TaskDescriptor(BaseModel):
    """API model for WBC task."""
    name: str
    priority: int = Field(..., ge=0)
    task_type: Literal["equality", "inequality", "soft"]
    weight: float = Field(1.0, gt=0)
    target: list[float]

class WBCRequest(BaseModel):
    """Request for whole-body control solve."""
    tasks: list[TaskDescriptor]
    contact_bodies: list[str] = []
    joint_limits_enabled: bool = True

class WBCResponse(BaseModel):
    """Response from whole-body control solve."""
    success: bool
    joint_accelerations: list[float]
    joint_torques: list[float]
    contact_forces: dict[str, list[float]]
    solver_time_ms: float
```

---

## Engine Agnosticism

### The Golden Rule

> **Every robotics component must work with ANY physics engine that implements the required Protocol.**

### Protocol-Based Design

```python
# src/robotics/core/protocols.py

from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class RoboticsCapable(Protocol):
    """Minimum protocol for robotics functionality."""

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Get (q, v) state."""
        ...

    def compute_mass_matrix(self) -> np.ndarray:
        """Get mass matrix M(q)."""
        ...

    def compute_bias_forces(self) -> np.ndarray:
        """Get bias forces C(q,v) + g(q)."""
        ...

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Get Jacobian for body."""
        ...


@runtime_checkable
class ContactCapable(Protocol):
    """Protocol for engines with contact support."""

    def get_contacts(self) -> list[dict]:
        """Get active contacts."""
        ...

    def get_contact_jacobian(self, contact_id: int) -> np.ndarray:
        """Get contact Jacobian."""
        ...


@runtime_checkable
class HumanoidCapable(RoboticsCapable, Protocol):
    """Protocol for humanoid robot capabilities."""

    def get_com_position(self) -> np.ndarray:
        """Get center of mass position."""
        ...

    def get_com_velocity(self) -> np.ndarray:
        """Get center of mass velocity."""
        ...

    def compute_centroidal_momentum_matrix(self) -> np.ndarray:
        """Get centroidal momentum matrix."""
        ...
```

### Engine Adapter Pattern

```python
# src/robotics/adapters/base.py

from abc import ABC, abstractmethod
from src.robotics.core.protocols import RoboticsCapable

class RoboticsAdapter(ABC):
    """Base adapter for adding robotics capabilities to engines."""

    def __init__(self, engine: RoboticsCapable):
        self._engine = engine

    @property
    def engine(self) -> RoboticsCapable:
        return self._engine

    @abstractmethod
    def compute_com_position(self) -> np.ndarray:
        """Compute center of mass position."""
        ...


# src/robotics/adapters/mujoco_adapter.py

class MuJoCoRoboticsAdapter(RoboticsAdapter):
    """MuJoCo-specific robotics adapter."""

    def compute_com_position(self) -> np.ndarray:
        # MuJoCo-specific implementation using mj_subtree_com
        ...


# src/robotics/adapters/pinocchio_adapter.py

class PinocchioRoboticsAdapter(RoboticsAdapter):
    """Pinocchio-specific robotics adapter."""

    def compute_com_position(self) -> np.ndarray:
        # Pinocchio-specific implementation using pin.centerOfMass
        ...
```

### Factory for Engine-Agnostic Components

```python
# src/robotics/factory.py

from src.shared.python.engine_manager import EngineType
from src.robotics.adapters.base import RoboticsAdapter
from src.robotics.adapters.mujoco_adapter import MuJoCoRoboticsAdapter
from src.robotics.adapters.pinocchio_adapter import PinocchioRoboticsAdapter
from src.robotics.adapters.drake_adapter import DrakeRoboticsAdapter

ADAPTER_REGISTRY: dict[EngineType, type[RoboticsAdapter]] = {
    EngineType.MUJOCO: MuJoCoRoboticsAdapter,
    EngineType.PINOCCHIO: PinocchioRoboticsAdapter,
    EngineType.DRAKE: DrakeRoboticsAdapter,
}

def create_robotics_adapter(
    engine: PhysicsEngine,
    engine_type: EngineType,
) -> RoboticsAdapter:
    """Create appropriate robotics adapter for engine type."""
    adapter_cls = ADAPTER_REGISTRY.get(engine_type)
    if adapter_cls is None:
        raise ValueError(f"No robotics adapter for engine type: {engine_type}")
    return adapter_cls(engine)
```

---

## Quality Gates

Before any PR is merged:

### Automated Checks (CI)

- [ ] `ruff check .` passes with zero errors
- [ ] `black --check .` passes
- [ ] `mypy --strict src/robotics` passes
- [ ] All unit tests pass: `pytest tests/unit/robotics -v`
- [ ] All integration tests pass: `pytest tests/integration/robotics -v`
- [ ] Coverage >= 90%: `pytest --cov=src/robotics --cov-fail-under=90`

### Manual Review Checklist

- [ ] Design by Contract: All public methods documented with pre/post/invariants
- [ ] Engine Agnostic: Works with MuJoCo, Pinocchio, Drake (where applicable)
- [ ] Small Functions: No function exceeds 50 lines
- [ ] Good Names: Names are descriptive and follow conventions
- [ ] DRY: No duplicated code
- [ ] Tests First: Tests written before implementation
- [ ] API Integration: New functionality exposed via REST API

---

## Quick Reference

### Creating a New Component

```bash
# 1. Create directory structure
mkdir -p src/robotics/newcomponent

# 2. Create __init__.py with public API
touch src/robotics/newcomponent/__init__.py

# 3. Create test file FIRST
touch tests/unit/robotics/test_newcomponent.py

# 4. Write failing tests
# 5. Implement to pass tests
# 6. Refactor

# 7. Run quality checks
ruff check src/robotics/newcomponent
black src/robotics/newcomponent
mypy --strict src/robotics/newcomponent
pytest tests/unit/robotics/test_newcomponent.py -v --cov
```

### Import Style

```python
# Standard library
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

# Third-party
import numpy as np
from numpy.typing import NDArray

# Local - absolute imports only
from src.shared.python.contracts import precondition, postcondition
from src.shared.python.interfaces import PhysicsEngine
from src.robotics.core.protocols import RoboticsCapable
from src.robotics.core.types import ContactState
```

---

## Conclusion

This guide is not optional. Every line of code in the robotics expansion must adhere to these principles. The goal is a maintainable, extensible, testable codebase that can evolve with the field of robotics.

**By Robots, For Robots.**

---

_Last updated: 2026-02-03_
