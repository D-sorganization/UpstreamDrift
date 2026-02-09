# UpstreamDrift Decoupling Plan

## Executive Summary

This document identifies coupling issues in the UpstreamDrift codebase and proposes a phased plan to improve modularity, testability, and maintainability. The analysis covers 1,236 Python files across 12 top-level source modules (`src/`), with particular focus on the `shared` monolith (103 root-level files, 45,614 lines), cross-module circular dependencies, and god objects in the launcher and GUI layers.

**Key findings:**
- The `src/shared/python/` directory is a monolithic "grab bag" of 103+ root-level modules spanning physics, UI, data I/O, AI, logging, security, and domain logic
- Circular dependencies exist between `shared <-> engines`, `shared <-> robotics`, `api -> launchers`, and `deployment -> engines/learning`
- The `logging_config` module is imported by 215 files, making it a central coupling point
- Several files exceed 1,500 lines (launcher: 2,557, ui_components: 1,332, GUI files: 1,900+)
- The API layer uses legacy `configure()` patterns alongside modern DI, creating dual coupling paths
- Engine loaders in `shared/` import directly from `engines/`, inverting the dependency direction

---

## 1. Architecture Overview (Current State)

### 1.1 Module Dependency Graph

```
                  ┌──────────────┐
                  │   launchers  │
                  └──────┬───────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         ┌────────┐ ┌────────┐ ┌──────┐
         │  api   │ │engines │ │ tools│
         └───┬────┘ └───┬────┘ └──┬───┘
             │          │         │
             ▼          ▼         ▼
         ┌──────────────────────────────┐
         │         shared (monolith)    │
         │  103 root files, 45K lines  │
         └──────────────┬───────────────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
         ┌────────┐ ┌────────┐ ┌────────────┐
         │robotics│ │learning│ │ deployment │
         └────────┘ └────────┘ └────────────┘
```

### 1.2 Circular Dependencies (Current)

| From | To | Files | Root Cause |
|------|----|-------|------------|
| `shared` | `engines` | 4 imports | `interfaces.py` imports `engines.common.capabilities`; `engine_loaders.py` imports engine implementations |
| `engines` | `shared` | 195 imports | Normal (correct direction), but overwhelming breadth |
| `shared` | `robotics` | 1 import | `control_interface.py` imports `robotics.control` |
| `robotics` | `shared` | 2 imports | `contracts` usage |
| `api` | `launchers` | 2 imports | `local_server.py` imports launcher handlers |
| `deployment` | `engines` | 3 imports | Protocol imports |
| `deployment` | `learning` | 2 imports | Imitation learning imports |

### 1.3 Top Coupling Hotspots

| Module | Imported By | Concern |
|--------|-------------|---------|
| `shared.python.logging_config` | 215 files | Cross-cutting concern deeply embedded |
| `shared.python.ai` | 73 files | AI subsystem too entangled with core |
| `shared.python.plotting` | 44 files | Visualization coupled to physics/analysis |
| `shared.python.engine_availability` | 31 files | Engine detection spread everywhere |
| `shared.python.signal_toolkit` | 29 files | Signal processing mixed with domain logic |
| `shared.python.interfaces` | 23 files | Protocol definition in wrong location |
| `shared.python.theme` | 23 files | UI theming in shared physics library |
| `shared.python.engine_manager` | 20 files | Manager class coupled to all engines |

---

## 2. Identified Decoupling Opportunities

### 2.1 The `shared/python/` Monolith (Critical Priority)

**Problem:** 103 root-level Python files in a single flat namespace covering unrelated concerns:
- Physics domain: `aerodynamics.py`, `ball_flight_physics.py`, `flexible_shaft.py`, `impact_model.py`, `terrain.py`, `hill_muscle.py`, `ground_reaction_forces.py`, ...
- UI/GUI: `gui_utils.py`, `draggable_tabs.py`, `viewpoint_controls.py`, `help_system.py`, `help_content.py`, ...
- Data I/O: `data_utils.py`, `io_utils.py`, `export.py`, `import_utils.py`, ...
- Analysis: `statistical_analysis.py` (2,236 lines), `comparative_analysis.py`, ...
- AI subsystem: `ai/` package (8,142 lines, 10 files)
- Infrastructure: `logging_config.py`, `security_utils.py`, `subprocess_utils.py`, ...
- Configuration: `config_utils.py`, `configuration_manager.py`, `env_validator.py`, ...

**Impact:** Any import from `shared` pulls in a conceptual dependency on the entire monolith. Module boundaries are meaningless because everything lives in the same package.

**Proposed split into focused packages:**

| New Package | Current Files | Purpose |
|-------------|--------------|---------|
| `core/` | `core.py`, `exceptions.py`, `constants.py`, `numerical_constants.py`, `physics_constants.py`, `contracts.py`, `type_utils.py`, `version.py` | Minimal foundation with zero heavy deps |
| `logging/` | `logging_config.py`, `logger_utils.py` | Logging infrastructure (cross-cutting) |
| `physics/` | `aerodynamics.py`, `ball_flight_physics.py`, `flexible_shaft.py`, `flight_models.py`, `flight_model_options.py`, `ground_reaction_forces.py`, `impact_model.py`, `terrain.py`, `terrain_engine.py`, `terrain_mixin.py`, `topography.py`, `grip_contact_model.py`, `equipment.py` | Golf physics domain models |
| `biomechanics/` | `hill_muscle.py`, `multi_muscle.py`, `muscle_analysis.py`, `muscle_equilibrium.py`, `activation_dynamics.py`, `biomechanics_data.py`, `kinematic_sequence.py`, `swing_plane_analysis.py`, `swing_plane_visualization.py`, `injury/` package | Biomechanics and muscle modeling |
| `engine_core/` | `interfaces.py`, `base_physics_engine.py`, `engine_registry.py`, `engine_manager.py`, `engine_loaders.py`, `engine_probes.py`, `engine_availability.py`, `unified_engine_interface.py`, `mock_engine.py` | Engine abstraction layer |
| `analysis/` | `statistical_analysis.py`, `analysis/` package, `comparative_analysis.py`, `comparative_plotting.py`, `data_fitting.py`, `physics_validation.py`, `validation.py`, `validation_data.py`, `validation_helpers.py`, `validation_utils.py`, `kaggle_validation.py` | Analysis and validation |
| `data_io/` | `data_utils.py`, `io_utils.py`, `export.py`, `import_utils.py`, `common_utils.py`, `dataset_generator.py`, `output_manager.py`, `path_utils.py`, `swing_capture_import.py`, `marker_mapping.py` | Data loading, saving, and conversion |
| `ui/` | `ui/` package, `gui_utils.py`, `draggable_tabs.py`, `viewpoint_controls.py`, `help_system.py`, `help_content.py`, `image_utils.py`, `theme/` package, `pose_editor/` package, `ellipsoid_visualization.py` | All UI-related code |
| `signal/` | `signal_processing.py`, `signal_toolkit/` package | Signal processing |
| `spatial/` | `spatial_algebra/` package, `reference_frames.py`, `indexed_acceleration.py`, `manipulability.py` | Spatial math and coordinate frames |
| `security/` | `security_utils.py`, `secure_subprocess.py`, `subprocess_utils.py`, `env_validator.py` | Security and subprocess management |
| `config/` | `config_utils.py`, `configuration_manager.py`, `environment.py`, `handedness_support.py`, `standard_models.py`, `model_registry.py` | Configuration management |

### 2.2 `interfaces.py` Location (High Priority)

**Problem:** `src/shared/python/interfaces.py` defines the `PhysicsEngine` Protocol but imports from `src.engines.common.capabilities` (via TYPE_CHECKING), creating a circular dependency between `shared` and `engines`.

**Solution:** Move `PhysicsEngine` Protocol and `RecorderInterface` to a new `src/engine_core/protocols.py` package that sits between `shared` and `engines`. The capabilities module should also move here since it defines the engine contract.

**Dependency flow after fix:**
```
engines -> engine_core -> core
shared  -> engine_core -> core
```

### 2.3 `engine_loaders.py` Inverted Dependencies (High Priority)

**Problem:** `src/shared/python/engine_loaders.py` imports concrete engine implementations from `src/engines/physics_engines/*/`. This means `shared` depends on `engines`, inverting the intended dependency direction.

**Solution:** Move `engine_loaders.py` to `src/engines/loaders.py` or have each engine register itself via a plugin/entry-point pattern. The `EngineManager` should discover engines through the registry, not through a hardcoded loader map in `shared`.

### 2.4 `control_interface.py` -> `robotics` Circular Dependency (Medium Priority)

**Problem:** `src/shared/python/control_interface.py` imports from `src.robotics.control`, while `robotics` imports from `shared`. This creates a circular dependency.

**Solution:** Either:
1. Move `control_interface.py` to `src/robotics/` (it's a robotics concern), or
2. Extract a shared `control_protocols.py` with just the Protocol definitions, imported by both

### 2.5 API -> Launchers Coupling (Medium Priority)

**Problem:** `src/api/local_server.py` imports from `src.launchers.launcher_model_handlers` and `src.launchers.launcher_process_manager`. The API layer should not depend on the GUI launcher.

**Solution:** Extract the shared logic (model handler registry, process management) into a service layer that both `api` and `launchers` can import. Likely candidates for extraction:
- Model handler registry -> `src/engine_core/model_handlers.py`
- Process management -> `src/services/process_manager.py`

### 2.6 API Legacy `configure()` Pattern (Medium Priority)

**Problem:** `src/api/server.py` uses both `app.state` for DI and legacy `configure()` calls on route modules:
```python
core_routes.configure(engine_manager)
engine_routes.configure(engine_manager, logger)
simulation_routes.configure(app.state.simulation_service, active_tasks, logger)
```

This creates two coupling paths - one through FastAPI's dependency injection and one through module-level state.

**Solution:** Complete the migration to FastAPI `Depends()`. Remove all `configure()` calls and have routes use `Depends()` to access services from `app.state`.

### 2.7 God Objects / Oversized Files (Medium Priority)

| File | Lines | Problem |
|------|-------|---------|
| `launchers/golf_launcher.py` | 2,557 | Monolithic launcher combining UI, process management, engine discovery, Docker integration |
| `shared/python/statistical_analysis.py` | 2,236 | Mixin aggregator importing 6+ analysis submodules, re-exporting everything |
| `engines/drake/python/src/drake_gui_app.py` | 2,050 | GUI + physics + plotting in one file |
| `engines/pinocchio/python/pinocchio_golf/gui.py` | 1,904 | Same pattern |
| `engines/mujoco/python/mujoco_humanoid_golf/sim_widget.py` | 1,792 | Same pattern |
| `shared/python/ui/qt/widgets/signal_toolkit_widget.py` | 1,767 | Single widget file too large |
| `shared/python/ai/glossary_data_core.py` | 1,747 | Static data as Python code |
| `shared/python/ai/glossary_data_extended.py` | 2,339 | Static data as Python code |
| `launchers/ui_components.py` | 1,332 | Multiple unrelated UI components in one file |
| `tools/model_generation/editor/frankenstein_editor.py` | 1,399 | Monolithic editor |
| `unreal_integration/data_models.py` | 1,362 | Data models too large for single file |
| `engines/mujoco/python/mujoco_humanoid_golf/linkage_mechanisms/__init__.py` | 1,362 | Too much code in __init__ |

### 2.8 Engine GUI Duplication (Lower Priority)

**Problem:** Each engine (MuJoCo, Drake, Pinocchio) has its own 1,500-2,500 line GUI application with significant overlap in:
- Tab management
- Plot rendering
- Simulation controls
- Data export

**Solution:** Create a shared `SimulationGUIBase` in `src/shared/python/ui/` that handles common patterns, with engine-specific subclasses only overriding engine-specific behavior. The dashboard module (`src/shared/python/dashboard/`) already partially addresses this but hasn't been fully adopted.

### 2.9 `deployment` Module Dependencies (Lower Priority)

**Problem:** `deployment` imports from both `engines.protocols` and `learning.imitation`, creating a dependency on two separate domains.

**Solution:** Define deployment-facing protocols in a `deployment/protocols.py` that `engines` and `learning` implement, inverting the dependency via the Dependency Inversion Principle.

### 2.10 Glossary Data as Python Code (Lower Priority)

**Problem:** `ai/glossary_data_core.py` (1,747 lines) and `ai/glossary_data_extended.py` (2,339 lines) contain static data structures embedded as Python code.

**Solution:** Move this data to JSON or YAML files in a `data/` directory and load them at runtime. This reduces import overhead and makes the data editable without touching Python code.

---

## 3. Phased Implementation Plan

### Phase 1: Break Circular Dependencies (Foundation)

**Goal:** Eliminate all circular imports between top-level modules.

| Step | Action | Files Changed | Risk |
|------|--------|---------------|------|
| 1.1 | Move `PhysicsEngine` protocol and `EngineCapabilities` to `src/engine_core/` | `interfaces.py`, `capabilities.py`, imports in ~23 files | Low - pure refactoring |
| 1.2 | Move `engine_loaders.py` from `shared/` to `src/engines/` | `engine_loaders.py`, `engine_manager.py`, imports in ~5 files | Low |
| 1.3 | Move `control_interface.py` to `src/robotics/` | 1 file move, ~8 import updates | Low |
| 1.4 | Extract model handler logic from `launchers/` into `src/services/` for API use | `local_server.py`, `launcher_model_handlers.py` | Medium - API behavior change |

**Validation:** After Phase 1, run `python3 -c "import ast; ..."` dependency analysis script and confirm zero circular dependencies between top-level modules.

### Phase 2: Split the `shared/python/` Monolith

**Goal:** Break the 103-file monolith into focused packages.

| Step | Action | Estimated Files | Risk |
|------|--------|-----------------|------|
| 2.1 | Create `src/shared/python/core/` with exceptions, constants, contracts, types | ~8 files, ~200 import updates | Medium - high fan-out |
| 2.2 | Create `src/shared/python/physics/` with golf physics models | ~13 files, ~30 import updates | Low |
| 2.3 | Create `src/shared/python/biomechanics/` with muscle/injury models | ~12 files, ~20 import updates | Low |
| 2.4 | Create `src/shared/python/data_io/` with data loading/export | ~10 files, ~25 import updates | Low |
| 2.5 | Create `src/shared/python/security/` with security utilities | ~4 files, ~15 import updates | Low |
| 2.6 | Create `src/shared/python/config/` with configuration management | ~6 files, ~15 import updates | Low |
| 2.7 | Provide backward-compatible re-exports in old locations | N/A | None |

**Strategy:** Each sub-package gets an `__init__.py` that re-exports its public API. Old import paths are preserved via re-exports in the original module locations (deprecation warnings optional). This allows incremental migration.

### Phase 3: Complete API Dependency Injection

**Goal:** Remove all `configure()` patterns from the API layer.

| Step | Action | Files Changed | Risk |
|------|--------|---------------|------|
| 3.1 | Create `src/api/dependencies.py` with `Depends()` providers for all services | 1 new file | Low |
| 3.2 | Migrate `core_routes` to use `Depends()` | 1 file | Low |
| 3.3 | Migrate `engine_routes` to use `Depends()` | 1 file | Low |
| 3.4 | Migrate `simulation_routes` to use `Depends()` | 1 file | Low |
| 3.5 | Migrate remaining routes (`video`, `analysis`, `export`) | 3 files | Low |
| 3.6 | Remove legacy `configure()` calls from `server.py` | 1 file | Low |

**Validation:** All API tests pass. No module-level mutable state in route modules.

### Phase 4: Reduce God Objects

**Goal:** Break down files >1,500 lines into focused modules.

| Step | Action | Estimated Impact |
|------|--------|------------------|
| 4.1 | Split `golf_launcher.py` into: launcher core, engine panel, docker panel, settings | 4 new files from 1 |
| 4.2 | Split `ui_components.py` into: cards, dialogs, workers, splash | 4 new files from 1 |
| 4.3 | Split each engine GUI into: tabs/, controls/, renderer/ | ~3 files per engine |
| 4.4 | Move glossary data to JSON files | 2 Python files -> 2 JSON files |
| 4.5 | Split `linkage_mechanisms/__init__.py` into sub-modules | 3-4 new files from 1 |

### Phase 5: Engine GUI Unification (Optional)

**Goal:** Reduce duplication across engine GUIs.

| Step | Action |
|------|--------|
| 5.1 | Create `SimulationGUIBase` with common tab management, plotting, export |
| 5.2 | Refactor MuJoCo GUI to extend `SimulationGUIBase` |
| 5.3 | Refactor Drake GUI to extend `SimulationGUIBase` |
| 5.4 | Refactor Pinocchio GUI to extend `SimulationGUIBase` |

---

## 4. Dependency Direction Rules (Target State)

After all phases, the dependency graph should follow these rules:

```
Rule 1: Dependencies flow downward only (no cycles)
Rule 2: UI layers never imported by non-UI code
Rule 3: Engine implementations never imported by shared code
Rule 4: Protocol/interface definitions live in engine_core, not shared

Target dependency order (top depends on bottom):
  launchers, tools
      │
  api, deployment
      │
  engines (implementations)
      │
  engine_core (protocols, registry)
      │
  shared/physics, shared/biomechanics, shared/analysis
      │
  shared/data_io, shared/signal, shared/spatial
      │
  shared/core (exceptions, constants, contracts)
      │
  shared/logging (standalone cross-cutting concern)
```

UI packages (`shared/ui`, `shared/theme`, `shared/plotting`) sit alongside their respective layers and are never imported by non-UI code.

---

## 5. Migration Strategy

### Backward Compatibility

For each moved module, the original location retains a thin re-export file:

```python
# src/shared/python/interfaces.py (after move)
"""Backward compatibility - imports moved to engine_core."""
from src.engine_core.protocols import PhysicsEngine, RecorderInterface

__all__ = ["PhysicsEngine", "RecorderInterface"]
```

### Testing Strategy

1. **Before each phase:** Run full test suite, record baseline
2. **During migration:** Use import compatibility shims
3. **After each phase:** Run full test suite, confirm no regressions
4. **Validation script:** Automated circular dependency checker as a CI check

### CI Integration

Add a CI step that runs the dependency direction checker:
```yaml
- name: Check dependency direction
  run: python scripts/check_dependency_direction.py
```

This script should fail if any circular dependency is detected between top-level modules.

---

## 6. Metrics and Success Criteria

| Metric | Current | Phase 1 Target | Phase 2 Target | Final Target |
|--------|---------|----------------|----------------|--------------|
| Circular deps (top-level pairs) | 5 | 0 | 0 | 0 |
| Max file size (lines) | 3,162 | 3,162 | 2,000 | 1,000 |
| `shared/python/` root files | 103 | 99 | ~30 | ~20 |
| Modules imported by >100 files | 1 | 1 | 1 | 1 (logging only) |
| API `configure()` calls | 6 | 6 | 6 | 0 |

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Import breakage during refactoring | High | Medium | Backward-compatible re-exports, comprehensive test suite |
| Merge conflicts with ongoing work | Medium | Medium | Coordinate with team, do phases in focused PRs |
| Over-engineering the split | Low | Medium | Follow "one package = one concept" rule, resist premature abstraction |
| Performance regression from deeper imports | Low | Low | Python caches imports; measure if concerned |
| Test discovery issues | Medium | Low | Update `pyproject.toml` testpaths as needed |

---

## 8. Recommended Execution Order

1. **Phase 1** (circular deps) - Do first, as it unblocks clean architecture reasoning
2. **Phase 3** (API DI) - Quick win, isolated to API layer, low risk
3. **Phase 2** (monolith split) - Largest effort, highest impact
4. **Phase 4** (god objects) - Can be done incrementally per file
5. **Phase 5** (GUI unification) - Optional, only if actively developing GUIs

Each phase should be submitted as one or more focused pull requests with clear descriptions of what moved and why.
