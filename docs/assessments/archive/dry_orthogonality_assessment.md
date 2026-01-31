# DRY and Orthogonality Assessment
## Golf Modeling Suite - Pragmatic Programmer Principles Analysis

**Date:** 2026-01-26
**Scope:** Full codebase analysis of `/src` directory
**Methodology:** Automated code pattern analysis and manual review

---

## Executive Summary

This assessment evaluates the Golf Modeling Suite against two core principles from "The Pragmatic Programmer":

1. **DRY (Don't Repeat Yourself)**: "Every piece of knowledge must have a single, unambiguous, authoritative representation within a system."

2. **Orthogonality**: "Eliminate effects between unrelated things. Design components that are self-contained, independent, and have a single, well-defined purpose."

### Overall Findings

| Principle | Grade | Summary |
|-----------|-------|---------|
| **DRY** | C | Significant code duplication across physics engines, especially spatial algebra and analysis modules |
| **Orthogonality** | C+ | Good protocol-based design, but launcher and API server have excessive coupling |

### Key Statistics

- **Duplicated Modules:** 13 major module groups with near-identical implementations
- **Duplicated Functions:** 20+ functions repeated across engines
- **Files with Mixed Concerns:** 3 critical files (1689, 864, 354 lines respectively)
- **Global State Instances:** 5+ module-level singletons creating hidden dependencies
- **Cascading Change Points:** Adding a new engine requires changes in 5+ files

---

## Part 1: DRY Violations

### Critical Severity (Immediate Action Required)

#### 1.1 Spatial Algebra Module Duplication

**Impact:** Foundational mathematical code is duplicated with trivial differences.

| Module | Drake Location | MuJoCo Location | Duplication |
|--------|---------------|-----------------|-------------|
| Inertia | `drake/python/src/spatial_algebra/inertia.py` | `mujoco/python/mujoco_humanoid_golf/spatial_algebra/inertia.py` | ~90% |
| Joints | `drake/python/src/spatial_algebra/joints.py` | `mujoco/python/mujoco_humanoid_golf/spatial_algebra/joints.py` | ~85% |
| Transforms | `drake/python/src/spatial_algebra/transforms.py` | `mujoco/python/mujoco_humanoid_golf/spatial_algebra/transforms.py` | ~95% |
| Spatial Vectors | `drake/python/src/spatial_algebra/spatial_vectors.py` | `mujoco/python/mujoco_humanoid_golf/spatial_algebra/spatial_vectors.py` | ~90% |

**Example of Duplicated Validation:**
```python
# Drake lines 48-63, MuJoCo lines 45-53 - IDENTICAL
if e_rot.shape != (3, 3):
    msg = f"E must be 3x3 rotation matrix, got shape {e_rot.shape}"
    raise ValueError(msg)

if not np.isclose(det_e, 1.0, atol=1e-6):
    msg = f"E may not be a valid rotation matrix (det={det_e})"
    raise ValueError(msg)
```

**Recommendation:** Create shared `src/shared/python/spatial_algebra/` module and have both engines import from it.

---

#### 1.2 Physics Engine Analysis Modules

**Impact:** Core analysis algorithms reimplemented 3-5 times.

##### Manipulability Analysis
- `drake/python/src/manipulability.py` (197 lines)
- `mujoco/python/mujoco_humanoid_golf/manipulability.py` (100+ lines)
- `pinocchio/python/pinocchio_golf/manipulability.py`

All three independently compute:
```python
mobility_matrix = J @ J.T
eigenvalues, eigenvectors = np.linalg.eigh(mobility_matrix)
# Identical ellipsoid computation follows
```

##### Induced Acceleration Analysis
- `drake/python/src/induced_acceleration.py` (60+ lines)
- `mujoco/python/mujoco_humanoid_golf/rigid_body_dynamics/induced_acceleration.py` (60+ lines)
- `pinocchio/python/pinocchio_golf/induced_acceleration.py` (60+ lines)

All decompose dynamics identically: `M(q)q̈ + C(q,q̇)q̇ + G(q) = τ`

**Recommendation:** Create `src/shared/python/analysis/` with base analyzer classes using Template Method pattern.

---

#### 1.3 Counterfactual Methods (ZTCF/ZVCF)

**Impact:** Every physics engine implements the same save-compute-restore pattern.

```
Pattern followed by all 5 engines:
1. Validate initialization
2. Save current state (q, v, control)
3. Set counterfactual state
4. Compute forward dynamics
5. Extract acceleration
6. Restore original state
7. Return result
```

**Lines duplicated per engine:** ~50 lines × 5 engines = 250 lines

**Recommendation:** Extract `CounterfactualMixin` or Template Method:
```python
class CounterfactualMixin:
    def compute_counterfactual(self, q, v, zero_control=False, zero_velocity=False):
        saved_state = self._save_state()
        try:
            self._apply_counterfactual(q, v, zero_control, zero_velocity)
            return self._compute_acceleration()
        finally:
            self._restore_state(saved_state)
```

---

### High Severity

#### 1.4 Quality Check Scripts

Three identical scripts with same banned patterns and magic number detection:
- `drake/scripts/quality_check.py` (80+ lines)
- `pinocchio/scripts/quality_check.py` (80+ lines)
- `mujoco/scripts/quality_check.py` (80+ lines)

**Recommendation:** Single `src/tools/quality_check.py` with engine-specific config files.

---

#### 1.5 Muscle Analysis Implementations

| File | Lines | Engine |
|------|-------|--------|
| `opensim/python/muscle_analysis.py` | 560 | OpenSim |
| `myosuite/python/muscle_analysis.py` | 560 | MyoSuite |

Nearly identical dataclasses, constants, and analysis logic:
```python
# Both files:
@dataclass
class MuscleAnalysis / MyoSuiteMuscleState:
    muscle_forces: dict[str, float]
    moment_arms: dict[str, dict[str, float]]
    activation_levels: dict[str, float]
```

**Recommendation:** Create `src/shared/python/analysis/muscle_analysis_base.py`.

---

#### 1.6 Jacobian Computation

MuJoCo and MyoSuite have 95% identical Jacobian code:

```python
# MuJoCo (lines 311-330):
body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
jacp = np.zeros((3, self.model.nv))
jacr = np.zeros((3, self.model.nv))
mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

# MyoSuite (lines 272-292):
body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
jacp = np.zeros((3, self.sim.model.nv))
jacr = np.zeros((3, self.sim.model.nv))
mujoco.mj_jacBody(self.sim.model, self.sim.data, jacp, jacr, body_id)
```

**Recommendation:** Extract shared MuJoCo Jacobian utility function.

---

### Medium Severity

#### 1.7 Initialization Guards

Every method in every engine repeats:
```python
if self.model is None or self.data is None:
    logger.warning("...")
    return default_value
```

**Count:** ~50 duplicate guard clauses across engines (13+ methods × 5 engines)

**Recommendation:** Use `@require_initialized` decorator from base class.

---

#### 1.8 Logger Utilities

Exact duplicate re-exports:
- `drake/python/src/logger_utils.py` (29 lines)
- `pinocchio/python/src/logger_utils.py` (29 lines)

Both simply import from `src.shared.python.logger_utils`.

**Recommendation:** Remove these files; import directly from shared module.

---

#### 1.9 Constants Redefinition

```python
# Defined in multiple places:
DEFAULT_TIME_STEP = float(constants.DEFAULT_TIME_STEP)  # 3 engines
GRAVITY_M_S2 = 9.81  # 4+ locations
NINETY_DEGREES_RAD = 1.5708  # OpenSim AND MyoSuite
```

**Recommendation:** Single source of truth in `src/shared/python/constants.py`.

---

## Part 2: Orthogonality Violations

### Critical Severity

#### 2.1 GolfLauncher Monolith

**File:** `src/launchers/golf_launcher.py` (1689 lines)

**Mixed Concerns (count: 8+):**
- UI layout and rendering (~400 lines)
- Model selection and management (~300 lines)
- Process launching and management (~500 lines)
- Docker environment management (~300 lines)
- AI assistant integration (~60 lines)
- Toast notifications (~100 lines)
- Layout persistence (~200 lines)
- Keyboard shortcuts and window geometry (~200 lines)

**Coupling Evidence:**
```python
# Lines 1280-1313: Long if-elif chain coupling model types to launch methods
if model_type == "mujoco":
    self._custom_launch_mujoco_humanoid(model_path, ...)
elif model_type == "drake":
    self._custom_launch_drake_gui(model_path, ...)
# ... 10+ branches
```

**Impact:** Testing individual features requires mocking the entire 1689-line window class.

**Recommendation:** Extract:
- `ProcessLauncher` class
- `LayoutManager` class
- `DockerManager` class
- `NotificationService` class

---

#### 2.2 API Server Mixed Concerns

**File:** `src/api/server.py` (864 lines)

**Mixed Concerns:**
- FastAPI app configuration (lines 68-99)
- Security middleware (lines 136-191)
- Path validation (lines 226-295)
- Task management (lines 305-399)
- Service initialization (lines 402-439)
- Engine endpoints (lines 469-536)
- Simulation endpoints (lines 541-595)
- Video analysis endpoints (lines 600-803)
- Analysis endpoints (lines 809-820)
- Export endpoints (lines 826-846)

**Recommendation:** Split into router modules:
- `routes/engines.py`
- `routes/simulation.py`
- `routes/video.py`
- `routes/analysis.py`
- `middleware/security.py`

---

#### 2.3 Engine Type Change Cascades

**Adding a new physics engine requires changes in:**

1. `src/shared/python/engine_registry.py` - Define enum
2. `src/shared/python/engine_manager.py` - Add path mapping + probe class
3. `src/shared/python/engine_loaders.py` - Create loader
4. `src/launchers/golf_launcher.py` - Add launch method + mapping
5. `src/config/models.yaml` - Register models

**Recommendation:** Plugin-based architecture with auto-discovery.

---

### High Severity

#### 2.4 Global State Dependencies

**API Server Globals (`server.py` lines 298-302, 399):**
```python
engine_manager: EngineManager | None = None
simulation_service: SimulationService | None = None
analysis_service: AnalysisService | None = None
video_pipeline: VideoPosePipeline | None = None
active_tasks = TaskManager()  # Line 399
```

**Impact:**
- Every endpoint implicitly depends on globals
- Tests share state between cases
- Cannot run multiple instances safely

**Recommendation:** Dependency injection via FastAPI's `Depends()`.

---

#### 2.5 Global Registry Singleton

**File:** `src/shared/python/engine_registry.py` (lines 72-78)

```python
_registry = EngineRegistry()

def get_registry() -> EngineRegistry:
    return _registry
```

**Impact:** Hidden dependency in all engine-loading code.

**Recommendation:** Pass registry explicitly or use FastAPI/pytest fixtures.

---

#### 2.6 Hardcoded Configuration

**Scattered across files:**

| Configuration | File | Lines |
|---------------|------|-------|
| API ports | `api/server.py` | 79-81, 858-863 |
| CORS origins | `api/server.py` | 82-94 |
| Launcher paths | `launchers/golf_launcher.py` | 1358-1493 |
| MeshCat port | `launchers/golf_launcher.py` | 1540 |
| Layout config | `launchers/golf_launcher.py` | 145-149 |

**Example - Hardcoded CORS:**
```python
allow_origins=[
    "http://localhost:3000",   # React development
    "http://localhost:8080",   # Vue development
    "https://app.golfmodelingsuite.com",  # Production
]
```

**Recommendation:** Centralize in `src/config/` with environment variable overrides.

---

### Medium Severity

#### 2.7 EngineManager Mixed Responsibilities

**File:** `src/shared/python/engine_manager.py` (354 lines)

**Distinct Concerns:**
- Engine discovery (lines 147-166)
- General engine loading (lines 167-209)
- MATLAB special-case loading (lines 210-246)
- Pendulum special-case loading (lines 248-254)
- Diagnostics and probing (lines 310-354)

**Impact:** Cannot use discovery without loading capabilities.

**Recommendation:**
- `EngineDiscovery` - finding available engines
- `EngineLoader` - instantiating engines
- `EngineDiagnostics` - health checks

---

#### 2.8 Circular Dependency Indicators

**Lazy imports in `golf_launcher.py` (lines 75-94):**
```python
def _lazy_load_engine_manager() -> tuple[Any, Any]:
    global _EngineManager, _EngineType
    if _EngineManager is None:
        from src.shared.python.engine_manager import EngineManager as _EM
        # ...
```

**Impact:** Indicates problematic module dependencies.

**Recommendation:** Review import graph and break cycles.

---

## Part 3: Positive Findings

### Good DRY Practices

#### 3.1 Engine Availability Module
`src/shared/python/engine_availability.py` (653 lines) properly centralizes all engine import checking, eliminating duplicate try/except patterns.

#### 3.2 Base Physics Engine Protocol
`src/shared/python/interfaces.py` defines a clear `PhysicsEngine` protocol that all engines implement consistently.

#### 3.3 Constants Module
`src/shared/python/constants.py` centralizes many (though not all) constants.

---

### Good Orthogonality Practices

#### 3.4 MuJoCo Flexible Shaft Implementation
Lines 478-630 in the MuJoCo engine show proper separation:
- `set_shaft_properties()` - Configuration interface
- `_compute_shaft_modes()` - Physics calculation (isolated)
- `get_shaft_state()` - State extraction

Completely decoupled from core dynamics.

#### 3.5 MyoSuite Muscle Delegation
Properly delegates to separate analyzer:
```python
def get_muscle_analyzer(self):
    return MyoSuiteMuscleAnalyzer(self.sim)
```

#### 3.6 Drake Lazy Finalization
Clean state machine pattern:
```python
def _ensure_finalized(self) -> None:
    if not self._is_finalized:
        self.plant.Finalize()
        self._is_finalized = True
```

---

## Part 4: Recommendations Summary

### Immediate Actions (Week 1-2)

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P0 | Extract shared spatial algebra module | Eliminates ~400 lines duplication | Medium |
| P0 | Create CounterfactualMixin for ZTCF/ZVCF | Eliminates ~200 lines duplication | Low |
| P1 | Consolidate quality check scripts | Reduces maintenance burden | Low |
| P1 | Remove duplicate logger_utils files | Cleaner imports | Low |

### Short-term Actions (Month 1)

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P1 | Split GolfLauncher into focused classes | Enables testing, reduces coupling | High |
| P1 | Split API server into route modules | Cleaner architecture | Medium |
| P2 | Create base analyzer classes | Eliminates ~300 lines duplication | Medium |
| P2 | Consolidate muscle analysis modules | Eliminates ~500 lines duplication | Medium |

### Medium-term Actions (Quarter 1)

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P2 | Implement dependency injection in API | Improves testability | Medium |
| P2 | Create plugin architecture for engines | Eliminates cascading changes | High |
| P3 | Centralize all configuration | Reduces scattered config | Medium |
| P3 | Break circular dependencies | Cleaner module graph | Medium |

---

## Appendix: File References

### High-Priority Files for Refactoring

| File | Lines | Issues |
|------|-------|--------|
| `src/launchers/golf_launcher.py` | 1689 | Monolithic, mixed concerns, hardcoded paths |
| `src/api/server.py` | 864 | Global state, mixed concerns, hardcoded config |
| `src/shared/python/engine_manager.py` | 354 | Multiple responsibilities |
| `src/engines/physics_engines/drake/python/src/spatial_algebra/*.py` | ~400 | Duplicated with MuJoCo |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/spatial_algebra/*.py` | ~400 | Duplicated with Drake |

### Physics Engine Implementations

| Engine | File | Lines |
|--------|------|-------|
| Drake | `drake/python/drake_physics_engine.py` | 558 |
| Pinocchio | `pinocchio/python/pinocchio_physics_engine.py` | 440 |
| MuJoCo | `mujoco/python/mujoco_humanoid_golf/physics_engine.py` | 630 |
| MyoSuite | `myosuite/python/myosuite_physics_engine.py` | 602 |
| OpenSim | `opensim/python/opensim_physics_engine.py` | 649 |
| Pendulum | `pendulum/python/pendulum_physics_engine.py` | 315 |

---

## Conclusion

The Golf Modeling Suite demonstrates competent protocol-based design and good use of shared utilities in some areas. However, significant technical debt exists in:

1. **Duplicated spatial algebra and analysis code** across physics engines
2. **Monolithic UI and API classes** mixing multiple concerns
3. **Global state** creating hidden dependencies
4. **Hardcoded configuration** scattered throughout the codebase

Addressing these issues will improve maintainability, testability, and make it easier to extend the system with new physics engines or analysis capabilities.

The recommended refactoring priority is:
1. Extract shared mathematical modules (high impact, medium effort)
2. Split monolithic classes (high impact, high effort)
3. Implement dependency injection (medium impact, medium effort)
4. Centralize configuration (medium impact, low effort)
