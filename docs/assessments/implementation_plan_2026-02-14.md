# UpstreamDrift — Implementation Plan (2026-02-14)

> **Reference Assessment**: `docs/assessments/comprehensive_assessment_2026-02-14.md`
> **Principles**: TDD, DbC, DRY, Orthogonality, Reversibility, Decoupled Code

---

## Phase 1: Error Handling & Type Safety (Quick Wins)

**Target Issues**: PP-ERR-001, PP-ERR-002, PP-ORTH-005

### 1.1 Replace print() with logging

**Files to modify (20+ occurrences):**

- `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/kinematic_forces.py` — Replace print() with logging
- `src/launchers/launcher_simulation.py` — Replace print() with logging + structured error types
- `src/shared/models/myosuite/examples/train_elbow_policy.py` — Replace print() with logging

**Pattern:**

```python
import logging
logger = logging.getLogger(__name__)

# Before:
print(f"Time: {result.time:.3f} s")
# After:
logger.info("Simulation result: time=%.3f s", result.time)
```

### 1.2 Add return type annotations to AIP methods

**Files to modify:**

- `src/api/aip/dispatcher.py` — Add `-> dict[str, Any]` and similar return types to all 3 functions
- `src/api/aip/methods.py` — Add return types to all 16 method functions
- `src/api/auth/dependencies.py` — Add return types to 4 auth dependency functions

### 1.3 Clean up stale root files

**Delete:**

- `=0.3.2` (stale pip artifact)
- `=4.1.0` (stale pip artifact)
- `failures.log`, `test_error.log`, `test_failure.log`, `test_failures.log`, `test_output.txt` (stale test output)
- `launch_status.txt` (stale launcher output)

### 1.4 Resolve TODO/FIXME markers

Review and resolve the 12 TODO/FIXME markers in source code.

---

## Phase 2: DRY Consolidation

**Target Issues**: PP-DRY-001, PP-DRY-002, PP-DRY-003, PP-DRY-004

### 2.1 Create shared engine adapter base (relates to #1372)

**Create:** `src/shared/python/engine_core/adapter_base.py`

Extract common engine adapter boilerplate:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SimulationState:
    """Canonical simulation state returned by all adapters."""
    time: float
    positions: "np.ndarray"
    velocities: "np.ndarray"

class EngineAdapterBase(ABC):
    """Base class for physics engine adapters.

    Enforces DbC: preconditions on init/step, postconditions on state queries.
    """

    @abstractmethod
    def initialize(self, model_path: str) -> None:
        """Load model. Precondition: model_path exists."""
        ...

    @abstractmethod
    def step(self, dt: float) -> SimulationState:
        """Advance simulation. Precondition: dt > 0. Postcondition: time increases."""
        ...

    @abstractmethod
    def reset(self) -> SimulationState:
        """Reset to initial state. Postcondition: time == 0."""
        ...
```

**TDD requirement:** Write `tests/unit/test_adapter_base.py` FIRST with contract tests.

### 2.2 Create shared launcher factory (relates to #1372)

**Create:** `src/shared/python/launcher_factory.py`

Extract common launcher patterns from:

- `src/launchers/launcher_simulation.py`
- `launch_golf_suite.py`
- `start_api_server.py`

### 2.3 Consolidate test fixtures

**Create:** `tests/conftest_engines.py`

Consolidate mock engine setup patterns used across `tests/unit/`, `tests/integration/`, `tests/parity/`.

### 2.4 Extract shared plotting base

**Create:** `src/shared/python/plotting/base_renderer.py`

Extract matplotlib configuration and axis setup boilerplate from:

- `src/shared/python/plotting/renderers/kinematics.py`
- `src/shared/python/plotting/renderers/signal.py`
- `src/shared/python/plotting/renderers/stability.py`

---

## Phase 3: Orthogonality & Decomposition (Critical)

**Target Issues**: PP-ORTH-001, PP-ORTH-002, PP-ORTH-003, PP-ORTH-004

### 3.1 Decompose `drake_gui_app.py` (2,177 lines → relates to #1390)

Split into sub-package `src/engines/physics_engines/drake/python/src/drake_gui/`:

- `__init__.py` — Re-exports
- `app.py` — Main application orchestrator (< 200 lines)
- `simulation_controller.py` — Simulation start/stop/step with DbC contracts
- `visualization.py` — 3D rendering and camera control
- `data_panel.py` — Data display and export
- `file_operations.py` — Model loading and saving

### 3.2 Decompose `pinocchio gui.py` (2,007 lines → relates to #1390)

Same pattern as 3.1 — split into sub-package `pinocchio_golf/gui/`.

### 3.3 Decompose `humanoid_launcher.py` (1,583 lines → relates to #1371)

Split into:

- `humanoid_launcher/model_loader.py`
- `humanoid_launcher/physics_setup.py`
- `humanoid_launcher/gui_renderer.py`
- `humanoid_launcher/launcher.py` — Orchestrator

### 3.4 Decompose `workflow_engine.py` (1,185 lines)

Split into:

- `ai/workflow/engine.py` — Core orchestration
- `ai/workflow/steps.py` — Step definitions and execution
- `ai/workflow/integrations.py` — AI provider integrations

---

## Phase 4: DbC & Contract Enforcement

**Target Issues**: CQ-DBC-001, CQ-DBC-002, CQ-DBC-003, relates to #1375

### 4.1 Add runtime contracts to engine adapters

For each engine adapter (MuJoCo, Drake, Pinocchio, OpenSim):

- Add precondition assertions on `initialize()`, `step()`, `reset()`
- Add postcondition assertions on state queries
- Add invariant checks on model state

### 4.2 Add contracts to API boundary

For each API method in `aip/methods.py`:

- Add Pydantic models for request/response
- Add precondition validation on parameters
- Add postcondition checks on return values

### 4.3 Add dependency direction tests (relates to #1370)

**Create:** `tests/architecture/test_dependency_direction.py`

Verify that:

- `src/shared/` does not import from `src/engines/`
- `src/engines/` does not import from `src/api/`
- `src/api/` does not import from `src/launchers/`

---

## Phase 5: Testing Uplift

**Target Issues**: PP-TEST-001, PP-TEST-002, CQ-TDD-001

### 5.1 Add tests for decomposed GUI modules

After Phase 3 decompositions, create test files for each new module:

- `tests/unit/test_drake_gui_simulation_controller.py`
- `tests/unit/test_drake_gui_visualization.py`
- etc.

### 5.2 Increase coverage toward 35% target

Focus on untested areas:

- Launcher modules
- AIP methods
- Workflow engine

---

## Cross-Repository Dependencies

- **Tools**: UpstreamDrift depends on `ud-tools` via vendor submodule. The `EngineAdapterBase` pattern (Phase 2.1) should be available as a shared library IF it proves reusable across repos.
- **AffineDrift**: Assessment utilities in AffineDrift's `assessment_utils.py` may need updates if assessment script interfaces change.
- **Gasification_Model**: Gasification_Model also uses `ud-tools` — any shared library changes must maintain backward compatibility.

---

## Success Criteria

After all phases:

- [ ] 0 print() calls in production code (excluding examples with noqa)
- [ ] All AIP methods have return type annotations
- [ ] Stale root files cleaned up
- [ ] No file exceeds 500 lines (down from 2,177)
- [ ] Engine adapters implement `EngineAdapterBase` with DbC contracts
- [ ] All API methods have Pydantic request/response models
- [ ] Dependency direction tests pass in CI
- [ ] Test coverage ≥ 35%
- [ ] CI/CD passes on all changes
