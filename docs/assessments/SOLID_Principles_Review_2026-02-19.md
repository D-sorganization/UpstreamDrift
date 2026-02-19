# SOLID Principles Review — UpstreamDrift (2026-02-19)

## Executive Summary

This assessment evaluates UpstreamDrift's adherence to the five SOLID principles across its ~288K-line Python codebase. The physics engine abstraction layer demonstrates **strong** OCP, LSP, and DIP design. The primary weaknesses lie in **SRP violations in launcher/GUI modules** and a **moderately fat PhysicsEngine protocol** (ISP). Overall SOLID maturity is above average for a codebase of this size and complexity.

| Principle | Score | Verdict |
|-----------|-------|---------|
| **S** — Single Responsibility | 5.5/10 | Significant violations in launcher layer |
| **O** — Open/Closed | 8.5/10 | Excellent protocol + registry design |
| **L** — Liskov Substitution | 8.0/10 | Strong protocol compliance; minor gaps |
| **I** — Interface Segregation | 7.0/10 | Cohesive but large PhysicsEngine protocol |
| **D** — Dependency Inversion | 7.5/10 | Good factory/registry pattern; launcher layer violations |
| **Overall** | **7.3/10** | |

---

## S — Single Responsibility Principle

**Score: 5.5/10**

SRP is the weakest SOLID dimension. Several launcher and GUI modules accumulate 4–6 distinct responsibilities per class, creating "God class" anti-patterns.

### Critical Violations

#### 1. `src/launchers/settings_dialog.py` — `SettingsDialog` (524 LOC, 21 methods)

Mixes five distinct responsibilities:

| Responsibility | Methods | Lines |
|---------------|---------|-------|
| Tab UI construction | `_create_layout_tab`, `_create_configuration_tab`, `_create_diagnostics_tab` | 90–274 |
| Diagnostic HTML rendering | `_render_diag_summary`, `_render_diag_checks`, `_render_diag_engines`, `_render_diag_runtime`, `_render_diag_recommendations` | 336–438 |
| Log file I/O | `_load_app_log`, `_load_process_log` | 276–314 |
| Docker build orchestration | `_start_build`, `_on_build_log`, `_on_build_finished`, `_cancel_build` | 471–518 |
| Timer event handling | `timerEvent` | 520–524 |

**Example** (`settings_dialog.py:374–415`):
```python
def _render_diag_engines(self, checks: list) -> str:
    # Data navigation + extraction + HTML rendering + business logic
    engine_check = next((c for c in checks if c["name"] == "engine_availability"), None)
    engines = engine_check.get("details", {}).get("engines", [])
    html = "<h3>Physics Engines</h3>"
    for eng in engines:
        installed = eng.get("installed", False)
        color = "#2da44e" if installed else "#f85149"
        # ... more mixed concerns
```

**Remediation**: Extract `DiagnosticsRenderer`, `LogManager`, and `DockerBuildController` classes.

#### 2. `src/launchers/launcher_ui_setup.py` — `LauncherUISetupMixin` (523 LOC, 25 methods)

Six responsibilities in a single mixin:

| Responsibility | Method count | Lines |
|---------------|-------------|-------|
| Menu bar construction | 5 methods | 113–206 |
| Top bar widget assembly | 4 methods | 208–318 |
| Search interaction | 3 methods | 368–385 |
| Process console management | 3 methods | 389–438 |
| AI panel integration | 2 methods | 446–496 |
| Dock/overlay management | 3 methods | 500–523 |

**Remediation**: Extract `MenuBuilder`, `TopBarBuilder`, `ConsoleManager`, and `AIIntegrationController`.

#### 3. `src/launchers/launcher_diagnostics.py` — `LauncherDiagnostics` (709 LOC, 12 methods)

Combines seven checker responsibilities in one class: Python environment, YAML validation, YAML completeness, model registry, layout config, asset files, PyQt availability, engine availability, and recommendation generation.

**Example** (`launcher_diagnostics.py:226–282`): `check_models_yaml()` performs file existence checking, YAML parsing, content validation, completeness checking, and multi-type exception handling — all in a single method.

**Remediation**: Extract individual `*Checker` classes (e.g., `YAMLChecker`, `RegistryChecker`, `EngineChecker`) implementing a common `DiagnosticCheck` protocol, and have `LauncherDiagnostics` act as an orchestrator.

#### 4. `src/launchers/launcher_simulation.py` — `LauncherSimulationMixin` (588 LOC, 16 methods)

Mixes launch routing, dependency checking, UI error display, and five app-specific launchers (URDF, C3D, shot tracer, MATLAB, MJCF) in a single class.

**Example** (`launcher_simulation.py:222–256`): `launch_simulation()` chains 8 sequential decision paths (model selection, special app routing, model lookup, MATLAB handling, Docker orchestration, dependency checking, local execution, error handling).

**Remediation**: Apply the Strategy or Factory pattern for app-specific launchers.

#### 5. `src/launchers/launcher_process_manager.py` — `ProcessManager` (550 LOC, 15 methods)

Combines environment management, logging, process launching (two execution modes), WSL integration (4 methods), and process lifecycle management.

**Remediation**: Use Strategy pattern for execution modes (`SeparateTerminalStrategy`, `UnifiedConsoleStrategy`) and extract `WSLLauncher`.

### Positive Examples

| Module | Assessment |
|--------|-----------|
| `src/shared/python/control_features_registry.py` (509 LOC, 3 classes) | **Good**. `FeatureDescriptor` is a pure data class; `ControlFeaturesRegistry` has a single responsibility: feature discovery and execution. |
| `src/api/server.py` (262 LOC) | **Good**. Focused solely on FastAPI application setup. Business logic properly delegated to services. Contains an explicit SRP-motivated extraction comment: `"Extracted from startup_event for SRP and testability."` |
| `src/shared/python/engine_core/base_physics_engine.py` | **Good**. Provides shared engine infrastructure (state management, path validation, checkpointing) without mixing domain logic. |

---

## O — Open/Closed Principle

**Score: 8.5/10**

The engine abstraction layer is the strongest SOLID dimension. Adding a new physics engine requires no modification to existing dispatcher code.

### Strengths

1. **Protocol-based interface** (`src/shared/python/engine_core/interfaces.py:37`): `PhysicsEngine` is a `@runtime_checkable Protocol` with comprehensive Design by Contract documentation. All seven engines conform to this protocol.

2. **Registry + factory loader pattern** (`src/engines/loaders.py:236–244`):
   ```python
   LOADER_MAP = {
       EngineType.MUJOCO: load_mujoco_engine,
       EngineType.DRAKE: load_drake_engine,
       EngineType.PINOCCHIO: load_pinocchio_engine,
       EngineType.OPENSIM: load_opensim_engine,
       EngineType.MYOSIM: load_myosim_engine,
       EngineType.PENDULUM: load_pendulum_engine,
       EngineType.PUTTING_GREEN: load_putting_green_engine,
   }
   ```
   Each loader uses lazy imports, returns `PhysicsEngine`, and is self-contained. Extending = adding one function + one map entry.

3. **Engine Registry** (`src/shared/python/engine_core/engine_registry.py`): Type-safe registrations with `EngineFactory: TypeAlias = Callable[[], PhysicsEngine]`. The `EngineManager` loads engines through the registry, never by direct instantiation.

4. **Model generation plugins** (`src/tools/model_generation/plugins/__init__.py:16`): ABC-based plugin architecture where new generators are added without modifying the dispatcher.

5. **YAML-driven model registry** (`src/shared/python/config/model_registry.py`): Models are defined in configuration, not hardcoded.

### Violations

| Location | Severity | Issue |
|----------|----------|-------|
| `src/shared/python/engine_core/engine_manager.py:218` | Minor | MATLAB engines require special-case `if engine_type in (EngineType.MATLAB_2D, EngineType.MATLAB_3D)` instead of going through the standard registry path. |
| `src/shared/python/engine_core/engine_manager.py:333–343` | Minor | `validation_paths` dictionary requires a manual entry for each engine type — should auto-derive from registry metadata. |

**Remediation**: Migrate MATLAB engines to the standard `PhysicsEngine` protocol or create an `ExternalEngineAdapter` abstraction, removing the special-case branch.

---

## L — Liskov Substitution Principle

**Score: 8.0/10**

Engine implementations generally honor parent contracts. The protocol defines clear pre/postconditions, and the `BasePhysicsEngine` consolidates shared behavior.

### Strengths

1. **Explicit contracts**: The `PhysicsEngine` protocol (`interfaces.py:1–21`) documents a state machine (`UNINITIALIZED -> INITIALIZED`), global invariants (superposition: `a_drift + a_control = a_full`), and per-method pre/postconditions with documented exception types.

2. **BasePhysicsEngine** (`base_physics_engine.py:74`): Template method pattern — subclasses implement `_load_from_path_impl()` and `_load_from_string_impl()`, while the base class handles validation, state transitions, and error wrapping. This enforces consistent behavior across engines.

3. **Runtime protocol checking** (`checkpoint.py:352–354`):
   ```python
   if not isinstance(engine, Checkpointable):
       raise TypeError("Engine must implement Checkpointable protocol")
   ```
   Prevents silent contract violations at runtime.

4. **Optional methods with safe defaults**: `compute_contact_forces()` returns `np.zeros(3)`, `set_shaft_properties()` returns `False`, `get_shaft_state()` returns `None`. Subclasses can override without breaking substitutability.

5. **Complete abstract method implementation**: All seven engines implement all required abstract methods (verified: `load_from_path`, `step`, `reset`, `get_state`, `set_state`, `compute_mass_matrix`, `compute_drift_acceleration`, `compute_ztcf`, `compute_zvcf`, etc.). No missing implementations detected.

### Violations

| Location | Severity | Issue |
|----------|----------|-------|
| `src/engines/physics_engines/mujoco/.../physics_engine.py:38` | Medium | MuJoCo implements the protocol directly without inheriting `BasePhysicsEngine`, duplicating checkpoint/loading logic. This creates risk of behavioral divergence. |
| `src/engines/physics_engines/drake/.../drake_physics_engine.py` | Medium | Same pattern — Drake also bypasses `BasePhysicsEngine`. |

**Remediation**: Refactor MuJoCo and Drake to inherit from `BasePhysicsEngine`, delegating to `_load_from_path_impl()` etc. This ensures consistent state transitions, error handling, and contract enforcement across all engines.

---

## I — Interface Segregation Principle

**Score: 7.0/10**

The `PhysicsEngine` protocol is cohesive (all methods relate to physics simulation) but large (~22 methods). Secondary protocols demonstrate good segregation.

### PhysicsEngine Protocol Breakdown

| Section | Methods | Required? | Description |
|---------|---------|-----------|-------------|
| Core simulation | `load_from_path`, `load_from_string`, `reset`, `step`, `forward` | Yes | Model loading and time-stepping |
| State access | `get_state`, `set_state`, `get_time`, `set_control` | Yes | State manipulation |
| Dynamics | `compute_mass_matrix`, `compute_bias_forces`, `compute_gravity_forces`, `compute_inverse_dynamics`, `compute_jacobian` | Yes | Dynamic quantities |
| Drift-Control (Section F) | `compute_drift_acceleration`, `compute_control_acceleration` | Yes | Causal decomposition |
| Counterfactual (Section G) | `compute_ztcf`, `compute_zvcf` | Yes | What-if experiments |
| Optional | `get_full_state`, `get_joint_names`, `get_capabilities`, `compute_contact_forces`, `set_shaft_properties`, `get_shaft_state` | No | Default implementations provided |

**Assessment**: The 16 required abstract methods span core simulation, dynamics computation, and domain-specific analysis (Section F/G). While all are used by the analytics layer, a simple engine (e.g., Pendulum) must implement ZTCF/ZVCF even if those counterfactual features are irrelevant to its use case.

### Well-Segregated Secondary Protocols

| Protocol | File | Methods | Assessment |
|----------|------|---------|-----------|
| `RecorderInterface` | `interfaces.py:673` | 3 | Focused on data recording |
| `PoseEstimator` | `pose_estimation/interface.py` | 3 | Focused on pose estimation |
| `PhysicsEngineProtocol` (terrain) | `physics/terrain_engine.py` | 1 | Single method |
| `TopographyProvider` | `physics/topography.py` | 2 properties | Minimal surface |
| `BallFlightModel` | `physics/flight_models.py` | 4 | Focused on ball flight |
| `ShaftModel` | `physics/flexible_shaft.py` | 3 | Focused on shaft simulation |
| `ImpactModel` | `physics/impact_model.py` | 1 | Single method |

These demonstrate the team's ability to design focused interfaces. The `PhysicsEngine` protocol is the exception.

### Recommendation

Consider splitting `PhysicsEngine` into role-based sub-protocols:

```python
class SimulationEngine(Protocol):          # load, step, reset, state access (9 methods)
class DynamicsEngine(Protocol):            # mass matrix, forces, ID, jacobian (5 methods)
class CausalAnalysisEngine(Protocol):      # drift/control, ZTCF, ZVCF (4 methods)
class FlexibleShaftEngine(Protocol):       # shaft properties/state (2 methods, already optional)
```

Engines that only need simulation (e.g., quick prototyping) wouldn't need to implement `CausalAnalysisEngine`. The existing `get_capabilities()` method could report which sub-protocols are supported.

---

## D — Dependency Inversion Principle

**Score: 7.5/10**

The engine infrastructure layer demonstrates exemplary DIP. The launcher layer has notable violations where high-level modules import concrete engine classes.

### Strengths

1. **Factory loader pattern** (`src/engines/loaders.py:1–10`): The module docstring explicitly acknowledges the DIP fix:
   > *"Previously lived in `src.shared.python.engine_loaders` which created an inverted dependency (shared -> engines). Now lives in `src.engines.loaders` which is the correct dependency direction (engines layer)."*

   Each loader uses lazy imports and returns `PhysicsEngine`, keeping the dependency arrow pointing from concrete to abstract.

2. **Engine Registry** (`engine_registry.py`): `EngineFactory: TypeAlias = Callable[[], PhysicsEngine]` — the registry stores factories returning the protocol type, not concrete classes.

3. **FastAPI dependency injection** (`src/api/dependencies.py`): Proper IoC via `Depends()`:
   ```python
   async def get_engines(
       engine_manager: EngineManager = Depends(get_engine_manager), ...
   )
   ```

4. **Service layer constructor injection** (`src/api/services/simulation_service.py:25–31`):
   ```python
   class SimulationService:
       def __init__(self, engine_manager: EngineManager) -> None:
           self.engine_manager = engine_manager
   ```

5. **Correct dependency direction**: `src/shared/python/engine_core/` depends on `interfaces.py` (abstract). Concrete engines in `src/engines/` depend on `interfaces.py`. The `src/engines/loaders.py` bridge module lives in the engines layer, keeping the dependency arrow correct.

### Violations

| File | Line | Issue | Severity |
|------|------|-------|----------|
| `src/launchers/mujoco_dashboard.py` | 7–9 | `from src.engines.physics_engines.mujoco...physics_engine import MuJoCoPhysicsEngine` — high-level launcher directly imports concrete engine | High |
| `src/launchers/drake_dashboard.py` | (similar) | Same pattern for Drake | High |
| `src/engines/.../mujoco/.../sim_widget.py` | 27, 103 | Widget imports and directly instantiates `MuJoCoPhysicsEngine` | High |
| `src/shared/python/engine_core/engine_manager.py` | 218 | MATLAB special-case branch creates implicit coupling to `EngineType.MATLAB_*` constants | Low |

**Impact**: The launcher DIP violations mean these files cannot be tested without the concrete engine dependencies installed. They also cannot be reused with alternative engines.

**Remediation**:
- Engine-specific dashboards should receive the engine instance via constructor injection or the registry/factory pattern.
- `sim_widget.py` should accept a `PhysicsEngine` protocol instance rather than importing `MuJoCoPhysicsEngine`.
- Use the existing `LOADER_MAP` or `EngineManager` to obtain engine instances in launcher code.

---

## Cross-Cutting Observations

### Architectural Strengths
- **Protocol-first design**: The `PhysicsEngine` protocol with DbC documentation is a model for how to define cross-implementation contracts in Python.
- **Layered dependency direction**: The `src/engines/loaders.py` migration explicitly fixed a `shared -> engines` inversion.
- **Template Method in BasePhysicsEngine**: Separates invariant behavior (state transitions, validation) from engine-specific logic (`_load_from_path_impl`).

### Systemic Issues
- **Mixin overuse in launchers**: `LauncherUISetupMixin`, `LauncherSimulationMixin` etc. accumulate unrelated responsibilities that would be better served by composition.
- **God class pattern in GUI**: The launcher/dashboard modules consistently concentrate 500–2000+ LOC per class. This is the primary source of SRP, ISP, and maintainability debt.
- **Inconsistent BasePhysicsEngine adoption**: MuJoCo and Drake bypass the base class, undermining the template method pattern and creating LSP risk.

---

## Prioritized Remediation Plan

### Priority 1 — High Impact, Moderate Effort

| ID | Principle | Action | Files |
|----|-----------|--------|-------|
| SOLID-001 | SRP | Extract `DiagnosticsRenderer` from `SettingsDialog` | `settings_dialog.py` |
| SOLID-002 | SRP | Create `LauncherFactory` for app-specific launchers | `launcher_simulation.py` |
| SOLID-003 | DIP | Inject engine instances into dashboard launchers via registry | `mujoco_dashboard.py`, `drake_dashboard.py` |
| SOLID-004 | LSP | Migrate MuJoCo and Drake to inherit from `BasePhysicsEngine` | `physics_engine.py`, `drake_physics_engine.py` |

### Priority 2 — Medium Impact

| ID | Principle | Action | Files |
|----|-----------|--------|-------|
| SOLID-005 | SRP | Extract `MenuBuilder`, `TopBarBuilder` from `LauncherUISetupMixin` | `launcher_ui_setup.py` |
| SOLID-006 | SRP | Create individual `*Checker` classes for diagnostics | `launcher_diagnostics.py` |
| SOLID-007 | SRP | Apply Strategy pattern for execution modes in `ProcessManager` | `launcher_process_manager.py` |
| SOLID-008 | OCP | Migrate MATLAB engines to standard `PhysicsEngine` protocol | `engine_manager.py` |

### Priority 3 — Lower Impact, Future Consideration

| ID | Principle | Action | Files |
|----|-----------|--------|-------|
| SOLID-009 | ISP | Split `PhysicsEngine` into role-based sub-protocols | `interfaces.py` |
| SOLID-010 | DIP | Refactor `sim_widget.py` to accept protocol, not concrete class | `sim_widget.py` |
| SOLID-011 | SRP | Extract `WSLLauncher` from `ProcessManager` | `launcher_process_manager.py` |

---

## Methodology

This review examined:
- All files in `src/shared/python/engine_core/` (interfaces, base class, registry, manager)
- All engine loader and implementation entry points in `src/engines/`
- All launcher modules in `src/launchers/` (6 files, ~3,500 LOC)
- API layer in `src/api/` (server, dependencies, services, routes)
- Plugin architecture in `src/tools/model_generation/`
- Existing assessments in `docs/assessments/`

Analysis techniques: class/method counting, responsibility enumeration, import graph analysis, protocol completeness verification, and pattern detection (factory, registry, template method, strategy).
