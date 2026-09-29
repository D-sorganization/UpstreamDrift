# Orthogonality Review Report

**Date:** 2026-02-02
**Scope:** Full codebase analysis of component independence and coupling

## Executive Summary

This report analyzes the orthogonality of the UpstreamDrift codebase - the principle that components should be independent so changes to one don't affect others. Overall, the architecture demonstrates **good high-level separation** (engines, API, UI, shared utilities) but has **specific coupling issues** that should be addressed.

| Area                     | Score | Summary                                     |
| ------------------------ | ----- | ------------------------------------------- |
| Engine Interface Layer   | 6/10  | Leaky abstractions, inconsistent signatures |
| Shared Python Modules    | 7/10  | Good separation with notable exceptions     |
| API Layer                | 5/10  | Business logic in routes, inconsistent DI   |
| Cross-Layer Dependencies | 8/10  | Clean boundaries, minimal violations        |

---

## 1. Engine Interface Layer

### 1.1 Leaky Abstractions (HIGH)

**Location:** `src/shared/python/interfaces.py:590-625`

The `PhysicsEngine` protocol documents engine-specific implementation details:

```python
def set_shaft_properties(self, length: float, ...) -> bool:
    """Configure flexible shaft properties.

    Note:
        The shaft model is engine-dependent:
        - MuJoCo: Composite body chain with torsional joints
        - Drake: Multibody with compliant elements
        - Pinocchio: Modal representation
    """
```

**Problem:** Clients shouldn't know that MuJoCo uses "composite body chains" or Pinocchio uses "modal representation." This violates abstraction principles.

**Recommendation:** Remove engine-specific documentation from the protocol. Each engine should document its implementation internally.

### 1.2 Inconsistent Return Types (HIGH)

| Method                     | Protocol Returns                | Base Implementation Returns                  |
| -------------------------- | ------------------------------- | -------------------------------------------- |
| `get_state()`              | `tuple[np.ndarray, np.ndarray]` | `EngineState \| None`                        |
| `compute_jacobian()`       | `dict[str, np.ndarray] \| None` | Keys vary: `linear`, `angular`, OR `spatial` |
| `compute_contact_forces()` | `np.ndarray`                    | Shape (3,) OR (6,) undefined                 |

**Location:**

- `interfaces.py:167` vs `base_physics_engine.py:304`
- `interfaces.py:372-393`
- `interfaces.py:395-410`

**Recommendation:** Define explicit return types using dataclasses or TypedDict to ensure consistency.

### 1.3 Engine Manager Tight Coupling (HIGH)

**Location:** `src/shared/python/engine_manager.py:174-189`

```python
# Handle special cases that don't conform to standard PhysicsEngine yet
if engine_type in (EngineType.MATLAB_2D, EngineType.MATLAB_3D):
    self._load_matlab_engine(engine_type)
elif engine_type == EngineType.PENDULUM:
    self._load_pendulum_engine()
else:
    # Standard Registry Loading
    registry = get_registry()
```

**Problems:**

1. MATLAB/PENDULUM engines bypass the registry pattern
2. Different storage attributes: `_matlab_engine` vs `active_physics_engine`
3. Hard-coded validation paths per engine type (lines 287-308)

**Recommendation:** Implement strategy pattern - each engine should self-describe its configuration paths and conform to the standard interface.

### 1.4 Unenforceable Contracts (HIGH)

**Location:** `src/shared/python/interfaces.py:414-478`

The protocol documents a CRITICAL CONTRACT:

```python
@abstractmethod
def compute_drift_acceleration(self) -> np.ndarray:
    """CRITICAL CONTRACT: a_drift + a_control = a_full (superposition)"""
```

**Problem:** The contract is documented but not enforced. An engine could violate `a_drift + a_control ≈ a_full` without detection.

**Recommendation:** Add contract validation decorators or automated tests that verify superposition.

---

## 2. Shared Python Modules

### 2.1 God Module (CRITICAL)

**Location:** `src/shared/python/statistical_analysis.py` (2,236 lines)

The `StatisticalAnalyzer` class has 35 methods and inherits from 6 mixins:

- Energy analysis
- Stability analysis
- Phase detection
- Ground reaction force analysis
- Angular momentum analysis
- Swing metrics

**Problems:**

- Single class handling 6 different analysis domains
- Difficult to test in isolation
- Changes to one analysis type risk affecting others

**Recommendation:** Split into focused analyzers: `SwingPhaseAnalyzer`, `EnergyAnalyzer`, `CoordinationAnalyzer`, etc.

### 2.2 Engine-Specific Code in Shared Module (CRITICAL)

**Location:** `src/shared/python/physics_validation.py`

Despite being in `shared/python/`, this module is MuJoCo-specific:

```python
# Line 25
if TYPE_CHECKING:
    import mujoco

# Lines 87-117: Constructor expects mujoco.MjModel and mujoco.MjData
# Lines 131, 151, 192: Direct MuJoCo API calls (mj_forward, mj_fullM, mj_step)
```

**Recommendation:** Move to `src/engines/mujoco/` or create an abstract `PhysicsValidator` with engine-specific implementations.

### 2.3 MuJoCo Imports in Shared Modules (MEDIUM)

Multiple shared modules import MuJoCo directly:

| File                                | Type          | Issue                                           |
| ----------------------------------- | ------------- | ----------------------------------------------- |
| `physics_validation.py:25`          | TYPE_CHECKING | **CRITICAL** - entire module is MuJoCo-specific |
| `marker_mapping.py:21,107`          | Runtime       | Used for model inspection                       |
| `engine_loaders.py:17`              | Conditional   | Acceptable - factory pattern                    |
| `engine_availability.py:17,219,575` | Conditional   | Acceptable - availability checking              |
| `engine_probes.py:106,567`          | Conditional   | Acceptable - probing                            |
| `provenance.py:121`                 | Conditional   | Version tracking                                |
| `error_decorators.py:274`           | Conditional   | Error type handling                             |
| `error_utils.py:350`                | Conditional   | Error handling                                  |
| `standard_models.py:325`            | Conditional   | Model validation                                |

**Acceptable Pattern:** Most use try/except for optional functionality.
**Problematic:** `physics_validation.py` and `marker_mapping.py` require MuJoCo at runtime.

### 2.4 Mixin Coupling (MEDIUM)

**Location:** `src/shared/python/analysis/swing_metrics.py:140-146`

```python
def find_club_head_speed_peak(self) -> PeakInfo | None:
    """Find peak club head speed - delegates to BasicStatsMixin."""
    return super().find_club_head_speed_peak()  # CALLS PARENT IN MRO
```

**Problem:** `SwingMetricsMixin` hard-depends on `BasicStatsMixin` being in the Method Resolution Order. These mixins cannot be reused independently.

**Recommendation:** Use composition instead of inheritance delegation.

### 2.5 Import Path Inconsistency (LOW)

**Location:** `src/shared/python/statistical_analysis.py:340,365,2000,2076`

```python
from shared.python import signal_processing  # WRONG - should be src.shared.python
```

Inconsistent with module-level imports that use `src.shared.python`.

---

## 3. API Layer

### 3.1 Direct Engine Access in Routes (HIGH)

**Location:** `src/api/routes/engines.py:95-104`, `src/api/routes/simulation_ws.py:50-142`

Routes directly manipulate the physics engine:

```python
engine = engine_manager.get_active_physics_engine()
engine.load_from_path(validated_path)  # Direct engine call
engine.step(timestep)  # Direct engine call in WebSocket handler
engine.get_state()  # Direct engine call
```

**Problem:** Routes should delegate to services. Business logic (simulation stepping) is in the route layer.

**Recommendation:** Create `EngineService` to encapsulate engine operations.

### 3.2 Business Logic in Route Handlers (HIGH)

**Location:** `src/api/routes/video.py:78-109`

```python
@router.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), ...):
    # Pipeline instantiation in route:
    config = VideoProcessingConfig(...)
    pipeline = VideoPosePipeline(config)
    result = pipeline.process_video(temp_path)  # Should be in VideoService
```

**Recommendation:** Move pipeline logic to a dedicated `VideoService`.

### 3.3 Inconsistent Dependency Injection (MEDIUM)

Most routes use global variables with `configure()` functions:

```python
# Pattern in simulation.py, analysis.py, video.py, core.py, export.py:
_simulation_service: SimulationService | None = None

def configure(simulation_service: SimulationService | None, ...):
    global _simulation_service  # Anti-pattern
    _simulation_service = simulation_service
```

**Contrast with correct pattern in `engines.py`:**

```python
async def get_engines(
    engine_manager: EngineManager = Depends(get_engine_manager),  # Correct
):
```

**Recommendation:** Standardize on FastAPI's `Depends()` pattern for all routes.

### 3.4 Service Responsibilities Too Broad (MEDIUM)

**Location:** `src/api/services/simulation_service.py`

`SimulationService` handles:

1. Engine management (loading, checking)
2. Model loading
3. State management
4. Recorder setup
5. Simulation execution
6. Data extraction
7. Analysis execution

**Recommendation:** Extract `EngineService` and `DataExtractionService`.

### 3.5 Data Model Mismatch (LOW)

**Location:** `src/api/services/analysis_service.py:147-154`

```python
if hasattr(request, "data") and request.data:  # No "data" field in AnalysisRequest!
```

The service expects a `data` field that doesn't exist in `AnalysisRequest`.

---

## 4. Cross-Layer Dependencies

### 4.1 Positive Findings

| Check                                     | Result                       |
| ----------------------------------------- | ---------------------------- |
| Shared modules import from `src.engines`? | **No violations**            |
| Shared modules import from `src.api`?     | **No violations**            |
| UI imports Python directly?               | **No** - Uses HTTP/WebSocket |
| Engines import from API?                  | **No violations**            |

The high-level layer boundaries are well-maintained.

### 4.2 Proper Isolation

These areas demonstrate good orthogonality:

- **Injury modules** (`src/shared/python/injury/`): Isolated subpackage with minimal external dependencies
- **Signal processing** (`src/shared/python/signal_processing.py`): 13 standalone functions with clear single responsibilities
- **Constants/Parameters separation**: Clean dependency chain with no circular imports
- **UI layer**: Properly communicates through API, no direct Python imports

---

## 5. Summary of Issues

### Critical Priority

| Issue                          | Location                         | Impact                                 |
| ------------------------------ | -------------------------------- | -------------------------------------- |
| God Module                     | `statistical_analysis.py`        | Hard to maintain, test, extend         |
| Engine-specific code in shared | `physics_validation.py`          | False abstraction, hidden dependencies |
| Direct engine access in routes | `engines.py`, `simulation_ws.py` | Tight coupling, bypasses services      |

### High Priority

| Issue                         | Location                           | Impact                           |
| ----------------------------- | ---------------------------------- | -------------------------------- |
| Inconsistent return types     | `interfaces.py` vs implementations | Runtime errors, type confusion   |
| Engine manager special-casing | `engine_manager.py`                | Violates Open-Closed Principle   |
| Unenforceable contracts       | `interfaces.py`                    | Silent correctness violations    |
| Business logic in routes      | `video.py`                         | Mixed concerns, duplication risk |

### Medium Priority

| Issue                    | Location                | Impact                          |
| ------------------------ | ----------------------- | ------------------------------- |
| Mixin coupling           | `swing_metrics.py`      | Limited reusability             |
| Inconsistent DI pattern  | Multiple route files    | Testing difficulty              |
| Service too broad        | `simulation_service.py` | Single responsibility violation |
| MuJoCo in marker_mapping | `marker_mapping.py`     | Hidden engine dependency        |

### Low Priority

| Issue                     | Location                  | Impact               |
| ------------------------- | ------------------------- | -------------------- |
| Import path inconsistency | `statistical_analysis.py` | Maintenance burden   |
| Data model mismatch       | `analysis_service.py`     | Defensive code smell |

---

## 6. Recommendations

### Immediate Actions

1. **Move `physics_validation.py`** to `src/engines/mujoco/validation.py`
2. **Fix return type signatures** in `interfaces.py` - use explicit types
3. **Standardize DI** - Replace global `configure()` with `Depends()`

### Short-Term Refactoring

1. **Split `statistical_analysis.py`** into focused analyzer classes
2. **Create `EngineService`** to encapsulate engine operations in API
3. **Create `VideoService`** to move video processing from routes
4. **Make MATLAB/PENDULUM** conform to standard engine interface

### Long-Term Architecture

1. **Strategy pattern** for engine configuration (self-describing engines)
2. **Contract validation framework** for physics invariants
3. **Composition over inheritance** for analysis mixins
4. **Event-driven architecture** for cross-cutting concerns (task management)

---

## 7. Orthogonality Metrics

To track improvement over time:

| Metric                           | Current   | Target |
| -------------------------------- | --------- | ------ |
| Max lines per module             | 2,236     | < 500  |
| Engine-specific code in shared/  | 2 modules | 0      |
| Routes with direct engine access | 2         | 0      |
| Routes using global DI           | 5         | 0      |
| Inconsistent interface methods   | 3         | 0      |

---

## Appendix: Files Analyzed

### Core Interface Layer

- `src/shared/python/interfaces.py`
- `src/shared/python/base_physics_engine.py`
- `src/shared/python/engine_manager.py`
- `src/shared/python/engine_registry.py`
- `src/shared/python/mock_engine.py`

### Shared Modules

- `src/shared/python/statistical_analysis.py`
- `src/shared/python/physics_validation.py`
- `src/shared/python/physics_constants.py`
- `src/shared/python/physics_parameters.py`
- `src/shared/python/signal_processing.py`
- `src/shared/python/analysis/*.py`
- `src/shared/python/injury/*.py`

### API Layer

- `src/api/server.py`
- `src/api/local_server.py`
- `src/api/routes/*.py`
- `src/api/services/*.py`
