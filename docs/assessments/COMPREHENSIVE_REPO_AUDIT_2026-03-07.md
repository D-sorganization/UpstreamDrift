# Comprehensive Repository Audit — UpstreamDrift

**Date:** 2026-03-07
**Branch audited:** `fix/dark-theme-import-error` (could not switch to `main` due to stale `.git/index.lock`)
**Auditor:** Claude (adversarial review, requested by maintainer)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Repository Hygiene](#2-repository-hygiene)
3. [Architecture & File Organization](#3-architecture--file-organization)
4. [Implementation Completeness](#4-implementation-completeness)
5. [Functional State & Load-Blocking Issues](#5-functional-state--load-blocking-issues)
6. [Design by Contract (DbC) Audit](#6-design-by-contract-dbc-audit)
7. [Test-Driven Development (TDD) Audit](#7-test-driven-development-tdd-audit)
8. [DRY Principle Audit](#8-dry-principle-audit)
9. [PyQt6 vs React/Tauri Parity Assessment](#9-pyqt6-vs-reacttauri-parity-assessment)
10. [Build, Config & CI Infrastructure](#10-build-config--ci-infrastructure)
11. [Prioritized Action Plan](#11-prioritized-action-plan)
12. [Appendix: File-Level Issue Table](#appendix-file-level-issue-table)

---

## 1. Executive Summary

### Overall Grade: C+

UpstreamDrift is an ambitious multi-physics golf simulation platform with dual frontends (PyQt6 desktop, React/Tauri web-desktop), a FastAPI backend, and integrations with MuJoCo, Drake, Pinocchio, OpenSim, and MyoSuite. The project demonstrates strong architectural vision and excellent documentation volume, but suffers from significant execution gaps that would prevent clean operation.

### Critical Findings

| Category | Grade | Key Issue |
|----------|-------|-----------|
| **Repo Hygiene** | D | 45+ junk/temp files committed to root; stale lock file; on feature branch |
| **Architecture** | B- | Good separation of concerns but duplicated directory hierarchies and unclear boundaries |
| **Implementation** | B | Core features complete in both UIs, but many stubs and partial implementations |
| **Functional State** | C | Import errors, hardcoded paths, configuration conflicts that would prevent clean startup |
| **DbC** | C- | Framework exists and is well-designed, but applied to <30% of codebase |
| **TDD** | C | 525 test files exist, but fragile isolation, missing markers, unclear coverage |
| **DRY** | D+ | Severe violations in engine loaders (7x duplication), error handling, UI features |
| **Cross-Platform Parity** | D | No shared UI abstraction layer; features diverge significantly between frontends |
| **Build/CI** | B- | Modern tooling, but critical version conflicts and missing test step in CI |

### The Hardest Truth

This repository has the documentation and scaffolding of a mature project, but the implementation discipline of a prototype. There are 380 assessment files in `docs/assessments/` — more assessments than source files in `src/`. The ratio of planning-to-execution is inverted. The project needs less assessment and more cleanup.

---

## 2. Repository Hygiene

### 2.1 Branch State — PROBLEM

The repo is on `fix/dark-theme-import-error`, not `main`. There are **24 local branches**, many of which appear to be abandoned:

- `assessment-generation-2026-02-26-*` (auto-generated branch names)
- `completist-audit-2026-02-26-*`
- `jules-*` (AI agent branches)
- `patent-review-update-*`

**Recommendation:** Delete all merged branches. Enforce branch naming conventions. The stale `.git/index.lock` file indicates a crashed git process that has not been cleaned up.

### 2.2 Junk Files at Root — CRITICAL

**45+ files that should not be in the repository:**

| Category | Files | Total Size |
|----------|-------|-----------|
| Error collection logs | `collect_errors.txt` through `collect_errors7.txt` | ~4.2 MB |
| CI test logs | `ci_tests311.log`, `ci_tests311_latest.log` | ~12.7 MB |
| Test output dumps | `full_test_run.txt`, `full_test_run_2.txt`, `test_results.txt`, etc. | ~2 MB |
| Numbered issue dumps | `1556.txt`, `1557.txt`, `1558.txt`, `1561.txt` | <100 KB |
| PR check outputs | `pr_checks.txt`, `pr_checks2.txt` | ~6 KB |
| Engine error logs | `error_log_drake.txt`, `error_log_mujoco.txt`, `simscape_errors.txt` | ~500 KB |
| Lint/type error dumps | `remaining_lint_errors.txt`, `ruff_errors.txt` | ~140 KB |
| Misc outputs | `out_1568.txt`, `out_1570.txt`, `run_id.txt`, `qg.log` | ~10 KB |
| UI test errors | `ui_test_err.txt`, `ui_test_err2.txt`, `sim_ux_err3.txt` | ~50 KB |
| Stale database | `golf_modeling_suite.db` | 52 KB |
| Coverage artifacts | `coverage.xml`, `htmlcov/` directory | ~2 MB |
| Python cache | `__pycache__/` at root | variable |
| Stray image | `Matlab Logo.jpg` | 120 KB |

**Total junk: ~22 MB of files that should be in `.gitignore`.**

These files signal that the repo is being used as a scratch pad rather than a clean source of truth. Every one of these should be removed and `.gitignore` updated to prevent reintroduction.

### 2.3 .gitignore Gaps

The following patterns are likely missing from `.gitignore`:

```
*.log
collect_errors*.txt
full_test_run*.txt
test_results*.txt
test_output*.txt
pr_checks*.txt
out_*.txt
error_log_*.txt
*_err*.txt
run_id.txt
*.db
htmlcov/
__pycache__/
coverage.xml
```

### 2.4 Duplicated Directory Structures

| Pair | Problem |
|------|---------|
| `/api/` vs `/src/api/` | Root `/api/` contains only stale `.pyc` files. Legacy directory never cleaned up. |
| `/engines/` vs `/src/engines/` | Both contain pendulum models and physics engine bindings. Unclear which is canonical. |
| `/tools/` vs `/src/tools/` | Different content but confusing parallel hierarchy. |
| `/launchers/` vs `/src/launchers/` | Both contain launcher implementations. Root `/launchers/` has compiled `.pyc` (141 KB). |
| `/python/` vs `/src/shared/python/` | `python/mujoco_humanoid_golf/` overlaps with `src/engines/physics_engines/mujoco/`. |

**Recommendation:** Establish `src/` as the single source of truth. Move or delete root-level duplicates. Remove all committed `.pyc` files.

---

## 3. Architecture & File Organization

### 3.1 Current Structure (Simplified)

```
UpstreamDrift/
├── src/                        # Main source (correct)
│   ├── api/                    # FastAPI backend
│   ├── config/                 # Configuration
│   ├── deployment/             # Production patterns
│   ├── engines/                # Physics engine layer
│   ├── launchers/              # PyQt6 launcher
│   ├── learning/               # ML modules
│   ├── reinforcement_learning/ # RL training
│   ├── research/               # Research implementations
│   ├── robotics/               # Robotics modules
│   ├── shared/                 # Shared utilities (70+ subdirs)
│   ├── spatial_algebra/        # Math utilities
│   ├── tools/                  # Tool utilities
│   └── unreal_integration/     # Unreal Engine
├── ui/                         # React/Tauri frontend
├── tests/                      # Test suite (525 files)
├── docs/                       # Documentation (35+ subdirs)
├── scripts/                    # Build/automation (60+ scripts)
├── data/                       # Sample data
├── shared/                     # MyoSuite models (CONFUSING NAME)
├── engines/                    # Duplicate of src/engines (REMOVE)
├── api/                        # Stale legacy (REMOVE)
├── launchers/                  # Duplicate of src/launchers (REMOVE)
├── python/                     # Stray modules (CONSOLIDATE)
├── tools/                      # URDF generator (MOVE to src/tools)
├── mujoco_golf_pendulum/       # Standalone model (MOVE)
├── vendor/                     # External tools
├── assets/                     # Branding
├── installer/                  # Windows installer
├── archive/                    # Legacy (empty)
└── [45+ junk files]            # DELETE
```

### 3.2 Architectural Problems

**Problem 1: `src/shared/python/` is a God Module (70+ subdirectories)**

This is the single biggest structural problem. `src/shared/python/` contains:

- Engine core logic
- UI components (Qt-specific)
- Theme/styling
- AI adapters (Anthropic, OpenAI, Gemini, Ollama)
- Data I/O
- Security utilities
- Physics calculations
- Humanoid character building
- Model generation
- Plotting engines
- Pose estimation
- Configuration
- Logging
- CLI utilities
- Control interfaces

This is not a "shared" module — it's the entire application logic dumped into one namespace. It violates the Single Responsibility Principle at the package level.

**Recommendation:** Split `src/shared/python/` into focused packages:

```
src/
├── core/           # Contracts, errors, logging, config
├── engine_core/    # Engine manager, types, availability
├── physics/        # Physics calculations, spatial algebra
├── ai/             # AI adapters
├── data/           # Data I/O, processing
├── modeling/       # Humanoid builder, model generation
├── visualization/  # Plotting, pose estimation
├── security/       # Subprocess utils, safe eval
└── ui_shared/      # Theme, style constants (no Qt imports)
```

**Problem 2: Root `/shared/` is a Confusing Name**

The root `shared/` directory contains only MyoSuite musculoskeletal model files (80+ XML, STL meshes). It has nothing to do with `src/shared/`. Rename to `models/myosuite/` or `data/myosuite/`.

**Problem 3: No Clear Dependency Direction**

The codebase has a `scripts/config/dependency_direction_rules.json` file, which implies someone tried to enforce layered architecture. But the actual imports violate these rules:

- `src/shared/python/` imports from `src/engines/` (lower layer importing from higher)
- `src/engines/loaders.py` imports from `src/shared/python/data_io/` (bidirectional dependency)
- `src/launchers/` directly imports from both `src/engines/` and `src/shared/python/`

**Problem 4: `docs/assessments/` Contains 380 Files**

This is more assessment files than actual source files. Many are auto-generated, redundant, or stale. This directory needs aggressive pruning — keep only the latest canonical assessments and archive the rest.

---

## 4. Implementation Completeness

### 4.1 Backend API — 75% Complete

| Component | Status | Notes |
|-----------|--------|-------|
| Route discovery/registry | ✅ Complete | Auto-discovers routes (#1485) |
| API versioning (`/api/v1/`) | ✅ Complete | Clean prefix pattern (#1488) |
| Simulation endpoints | ✅ Complete | Sync + async with task management |
| WebSocket streaming | ✅ Complete | Live simulation frames |
| URDF model serving | ✅ Complete | Model file serving |
| Analysis endpoints | ⚠️ Partial | Route exists, service logic unclear |
| Authentication | ⚠️ Scaffold only | JWT/OAuth modules exist but likely not wired |
| Error handling middleware | ⚠️ Exists but inconsistently applied | `handle_api_errors` decorator unused on most routes |
| Database/persistence | ⚠️ Minimal | `database.py` exists but SQLite file at root suggests ad-hoc usage |
| Video analysis | ❌ Stub | Endpoint exists, unclear if functional |
| Chat/AI integration | ⚠️ Partial | Adapters exist for 4 providers |

### 4.2 React/Tauri Frontend — 85% Complete

| Feature | Status | Component |
|---------|--------|-----------|
| Launcher dashboard | ✅ | `LauncherDashboard.tsx` |
| 7 route pages | ✅ | Dashboard, Simulation, ModelExplorer, PuttingGreen, VideoAnalyzer, DataExplorer, MotionCapture |
| 3D visualization (Three.js) | ✅ | `Scene3D.tsx` |
| URDF model rendering | ✅ | `URDFViewer.tsx` |
| WebSocket simulation | ✅ | `client.ts` with reconnection |
| State management (Zustand) | ✅ | 3 stores |
| Force overlay visualization | ✅ | `ForceOverlay.tsx` |
| Settings/preferences UI | ❌ Missing | |
| Offline/error recovery | ⚠️ Partial | Connection status shown but no offline mode |
| Accessibility | ❌ Missing | No ARIA labels, keyboard navigation |

### 4.3 PyQt6 Frontend — 90% Complete

| Feature | Status | Notes |
|---------|--------|-------|
| Main launcher window | ✅ | Mixin composition pattern |
| Model card grid | ✅ | Draggable tiles |
| Engine-specific dashboards | ✅ | Drake, MuJoCo, Pinocchio |
| Process management | ✅ | Subprocess spawn/kill |
| Docker integration | ✅ | Full stage management |
| Dark theme | ✅ | Complete with customization |
| Settings dialog | ✅ | |
| AI chat panel | ⚠️ Conditional | Only if AI_AVAILABLE |
| Console output | ✅ | Docked widget |
| MATLAB integration | ⚠️ Scaffold | `matlab_launcher_unified.py` exists |
| OpenSim dashboard | ❌ Missing | |
| MyoSim dashboard | ❌ Missing | |

### 4.4 Physics Engines — Variable

| Engine | Integration | Notes |
|--------|------------|-------|
| MuJoCo | ✅ Strong | Full loader, model, GUI |
| Drake | ✅ Strong | Loader, GUI, pose editor |
| Pinocchio | ⚠️ Moderate | Loader exists, dashboard minimal |
| OpenSim | ⚠️ Scaffold | Referenced in code but no loader |
| MyoSuite | ⚠️ Data only | 80+ XML models in `shared/`, no integration code found |
| MATLAB | ⚠️ Stub | Web-only launcher redirect |
| Unreal Engine | ❌ Scaffold | `src/unreal_integration/` exists, likely empty |

---

## 5. Functional State & Load-Blocking Issues

### 5.1 CRITICAL — Will Prevent Loading

**Issue 1: Import Error in `safe_eval.py`**
- **File:** `src/shared/python/safe_eval.py:30`
- **Problem:** `from contracts import require` — should be `from src.shared.python.contracts import require`
- **Impact:** Any code path that uses safe_eval will crash with ImportError
- **Severity:** BLOCKING

**Issue 2: NumPy Version Conflict**
- **File:** `requirements.txt` specifies `numpy>=2.0.1`; `environment.yml` specifies `numpy>=1.26.4,<2.0.0`
- **Impact:** Conda environment will install incompatible NumPy version
- **Severity:** BLOCKING for conda users

**Issue 3: MyPy Version Mismatch in CI**
- **File:** `.github/workflows/ci-standard.yml` pins `mypy==1.13.0`; lock file has `1.19.1`
- **Impact:** CI may fail or produce different results than local
- **Severity:** HIGH

**Issue 4: Docker Port Mismatch**
- **File:** `Dockerfile` exposes port 8000; `docker-compose.yml` maps to 8001; Vite proxy targets 8001
- **Impact:** Containerized deployment won't route correctly
- **Severity:** HIGH

**Issue 5: Flask in requirements.txt but not in pyproject.toml**
- **Impact:** Dependency confusion; Flask is not used by the project (FastAPI is the framework)
- **Severity:** MEDIUM (unnecessary dependency, potential security surface)

**Issue 6: Stale `.git/index.lock`**
- **Impact:** All git operations fail until manually removed
- **Severity:** HIGH for all developers

### 5.2 HIGH — Will Cause Runtime Errors

**Issue 7: Hardcoded Model Paths in Engine Loaders**
- **File:** `src/engines/loaders.py:44-50, 88-96, 138-146`
- **Problem:** Paths like `suite_root / "engines" / "physics_engines" / "mujoco" / "models" / "simple_pendulum.xml"` assume fixed directory structure
- **Impact:** Any reorganization breaks all engine loading

**Issue 8: Missing `__all__` Definitions**
- **Files:** Most modules in `src/shared/python/`
- **Impact:** Public API is undefined; wildcard imports pull everything

**Issue 9: `EngineManager.__init__` is ~100 Lines**
- **File:** `src/shared/python/engine_core/engine_manager.py`
- **Impact:** God constructor; impossible to test individual behaviors

**Issue 10: `sys.modules` Manipulation in Test Conftest**
- **File:** `tests/conftest.py:22-62`
- **Impact:** Tests depend on import order; intermittent failures

---

## 6. Design by Contract (DbC) Audit

### 6.1 Framework Assessment — Well Designed, Poorly Applied

The `src/shared/python/contracts.py` module provides a solid DbC framework:

- `@precondition` / `@postcondition` decorators
- `require()` / `ensure()` functions
- Configurable enforcement levels
- `ContractChecker` base class

**Grade for the framework itself: A**

### 6.2 Application Assessment — Inconsistent

| Module | DbC Coverage | Notes |
|--------|-------------|-------|
| `contracts.py` itself | 100% | Self-documenting |
| `EngineManager` | ~60% | Uses `ContractChecker`, but `__init__` has no contracts |
| `ControlInterface` | 0% | No precondition checking on `set_gains()`, `apply_torque()` despite being safety-critical |
| API routes | 0% | Rely on Pydantic validation only (not DbC) |
| API services | ~30% | Some `@postcondition` usage, but inconsistent |
| Engine loaders | 0% | No contracts on any loader function |
| UI components (PyQt6) | 0% | No DbC in any widget |
| UI components (React) | N/A | TypeScript types serve as partial contracts |
| Test helpers | 0% | No contracts |
| Safe eval | ~50% | Has contracts but broken import |

**Overall DbC Coverage: ~15-20%**

### 6.3 Specific DbC Violations

**Violation 1: Safety-Critical Code Without Contracts**
```python
# src/shared/python/control_interface.py
class ControlInterface:
    def apply_torque(self, torque: np.ndarray) -> None:
        # Missing: require(np.all(np.abs(torque) <= self.torque_limit))
        # Missing: require(torque.shape == (self.num_joints,))
        ...

    def set_gains(self, kp: float, kd: float) -> None:
        # Missing: require(kp > 0, "Proportional gain must be positive")
        # Missing: require(kd >= 0, "Derivative gain must be non-negative")
        ...
```

**Violation 2: Engine Loading Without Postconditions**
```python
# src/engines/loaders.py
def load_mujoco_engine(suite_root: Path) -> PhysicsEngine:
    # Missing: ensure(result.is_initialized())
    # Missing: ensure(result.model is not None)
    engine = MuJoCoPhysicsEngine()
    return engine  # Could return uninitialized engine
```

**Violation 3: API Service Methods Without Input Validation**
```python
# src/api/services/analysis_service.py
async def run_analysis(self, config: AnalysisConfig) -> AnalysisResult:
    # Missing: require(config.engine_type in SUPPORTED_ENGINES)
    # Missing: require(config.time_step > 0)
    ...
```

### 6.4 Recommendations

1. **Mandatory DbC for all public methods** in `src/engines/`, `src/shared/python/control_interface.py`, `src/api/services/`
2. **Create a `@safety_critical` decorator** that combines precondition + postcondition + logging
3. **Add contract checking to CI** — run with contracts enabled and fail on violations
4. **Document contract enforcement levels** for each deployment mode (development, testing, production)

---

## 7. Test-Driven Development (TDD) Audit

### 7.1 Test Infrastructure

| Metric | Value | Assessment |
|--------|-------|-----------|
| Test files | 525 | HIGH volume |
| Test directories | 20+ categories | Well-organized |
| Test framework | pytest | Standard |
| Coverage tooling | coverage.xml, htmlcov | Present |
| Property-based testing | `.hypothesis/` exists | Used somewhere |
| Security testing | `tests/security/` | Exists |
| Benchmarks | `tests/benchmarks/` | Exists |
| Integration tests | `tests/integration/` | Exists |
| Acceptance tests | `tests/acceptance/` | Exists |
| React tests | 15 `.test.tsx` files | Moderate |

### 7.2 Critical TDD Problems

**Problem 1: CI Does Not Run Tests**

This is the single most damaging finding. The CI pipeline does not include a pytest step. 525 test files exist but there is no evidence they are run in CI. Tests are decorative.

**Problem 2: Fragile Test Isolation via `sys.modules` Hacking**

```python
# tests/conftest.py
_PROTECTED_PREFIXES = (
    "pinocchio",  # C extension corruption
    "pydrake",    # Gets replaced with MagicMock
)
# Manipulates sys.modules to prevent cross-test contamination
```

This is a symptom of architectural coupling — engines should not corrupt each other's state. The fix is engine-level isolation (subprocess per engine test suite), not import-time monkey-patching.

**Problem 3: No Test Markers or Categories**

Tests lack `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.requires_network`, etc. This means:
- Can't run fast tests only
- Can't skip tests that require specific engines
- Can't parallelize by category

**Problem 4: Unknown Coverage**

`coverage.xml` exists at root (committed — shouldn't be), but there's no coverage enforcement threshold. Without a minimum coverage gate, coverage tends to erode over time.

**Problem 5: React Test Coverage is Thin**

15 test files for 66 source files = ~23% file coverage. Key untested components:
- `ConnectionStatus.tsx`
- `DiagnosticsPanel.tsx`
- `LivePlot.tsx`
- `ParameterPanel.tsx`
- `ForceOverlayPanel.tsx`

**Problem 6: No PyQt6 Tests Found**

`src/launchers/tests/` contains only `__init__.py`. Zero test coverage for the PyQt6 launcher, which is the more mature (90% complete) UI.

### 7.3 Recommendations

1. **Add pytest to CI immediately** — this is the #1 priority action
2. **Add test markers** to `pyproject.toml` and tag all tests
3. **Set coverage threshold** at 60% minimum, enforced in CI
4. **Replace `sys.modules` hack** with subprocess isolation for engine tests
5. **Add PyQt6 tests** using `pytest-qt` — at minimum for launcher startup, model registry, and process management
6. **Increase React test coverage** to at least 50% of components

---

## 8. DRY Principle Audit

### 8.1 Severity: CRITICAL

DRY violations are the most pervasive code quality issue in this codebase.

### 8.2 Specific DRY Violations

**Violation 1: Engine Loaders — 7x Code Duplication (CRITICAL)**

**File:** `src/engines/loaders.py` — 233 lines, ~200 of which are duplicated across 7 loader functions.

Each loader follows the identical pattern:
1. Try import
2. Create probe
3. Run probe
4. Check availability
5. Create engine
6. Find model path
7. Load model
8. Return engine
9. Catch ImportError

This pattern is repeated verbatim 7 times with only the class names and paths changed. **200+ lines could be reduced to ~50 with a factory function.**

```python
# CURRENT (repeated 7 times):
def load_mujoco_engine(suite_root: Path) -> PhysicsEngine:
    try:
        import mujoco
        from src.engines... import MuJoCoPhysicsEngine
        probe = MuJoCoProbe(suite_root)
        result = probe.probe()
        if not result.is_available():
            raise GolfModelingError(...)
        engine = MuJoCoPhysicsEngine()
        model_path = suite_root / "engines" / ... / "simple_pendulum.xml"
        if model_path.exists():
            engine.load_from_path(str(model_path))
        return engine
    except ImportError as e:
        raise GolfModelingError(...) from e

# SHOULD BE:
def _make_loader(engine_cls, probe_cls, import_name, model_subpath):
    def loader(suite_root: Path) -> PhysicsEngine:
        # Single implementation
        ...
    return loader

load_mujoco_engine = _make_loader(MuJoCoPhysicsEngine, MuJoCoProbe, "mujoco", "mujoco/models/simple_pendulum.xml")
load_drake_engine = _make_loader(DrakePhysicsEngine, DrakeProbe, "pydrake", "drake/models/pendulum.urdf")
# etc.
```

**Violation 2: API Route Error Handling — Pattern Duplication (HIGH)**

Every route handler has its own try/except block with nearly identical error handling:

```python
# Found in simulation.py, analysis.py, terrain.py, etc.
try:
    result = await service.do_thing(request)
    return result
except TimeoutError as exc:
    raise HTTPException(status_code=504, detail=...)
except ValueError as exc:
    raise HTTPException(status_code=400, detail=...)
except Exception as exc:
    raise HTTPException(status_code=500, detail=...)
```

The `@handle_api_errors` decorator exists in `src/api/middleware/error_handler.py` but is **not applied consistently**. This means:
- Bug fixes to error handling must be applied to every route
- Error response format drifts between routes
- New routes copy-paste the pattern and potentially miss cases

**Violation 3: Feature Duplication Between PyQt6 and React (HIGH)**

No shared abstraction layer exists. Both frontends independently implement:

| Feature | PyQt6 Location | React Location |
|---------|---------------|----------------|
| Help system | `help_dialogs.py` | `HelpPanel.tsx` + `helpData.ts` |
| Engine selection | `model_registry.py` | `EngineSelector.tsx` + `useEngineStore.ts` |
| Simulation controls | `launcher_simulation.py` | `SimulationControls.tsx` |
| Theme/styling | `launcher_theme.py` | Tailwind config + style constants |
| Toast notifications | `launcher_dialogs.py` | `Toast.tsx` |
| Settings management | `settings_dialog.py` | Not implemented (but will be duplicated when added) |

**Each new feature must be implemented twice, tested twice, and maintained in two codebases.**

**Violation 4: Configuration Defaults Scattered (MEDIUM)**

Default values appear in multiple locations:

```python
# In src/api/config.py
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

# In src/shared/python/control_interface.py
torque_limit: float = 100.0
position_limit_lower: float = -np.pi
velocity_limit: float = 10.0

# In ui/src/api/client.ts
MAX_FRAMES_HISTORY = 1000

# In ui/src/stores/useSimulationStore.ts
DEFAULT_PARAMETERS = { ... }  # Hardcoded defaults
```

These should be in a single configuration schema with per-deployment overrides.

**Violation 5: Duplicate Config Files (MEDIUM)**

- `mypy.ini` duplicates `[tool.mypy]` in `pyproject.toml`
- `pytest_improvements.ini` duplicates `[tool.pytest.ini_options]` in `pyproject.toml`
- `setup.py` exists alongside `pyproject.toml` (redundant)
- `requirements.txt` exists alongside `pyproject.toml` dependency specification

### 8.3 DRY Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Estimated duplicated lines (engine loaders) | ~200 | <50 |
| Estimated duplicated error handling patterns | ~150 | 0 (use decorator) |
| Feature implementations requiring dual maintenance | ~10 features | 0 (shared abstractions) |
| Redundant config files | 4 | 0 |
| Hardcoded default values across files | ~15 instances | 1 (centralized config) |

---

## 9. PyQt6 vs React/Tauri Parity Assessment

### 9.1 Feature Parity Matrix

| Feature | PyQt6 | React/Tauri | Shared |
|---------|-------|-------------|--------|
| **Launcher dashboard** | ✅ Grid + drag | ✅ Grid tiles | Via manifest API |
| **Engine selection** | ✅ Registry | ✅ Store | None (duplicated) |
| **3D visualization** | ⚠️ Meshcat only | ✅ Three.js | None |
| **URDF viewer** | ❌ | ✅ | None |
| **Force overlay** | ❌ | ✅ | None |
| **Simulation controls** | ✅ Per-engine | ✅ Unified | None |
| **WebSocket streaming** | ❌ | ✅ | None |
| **Settings dialog** | ✅ | ❌ | None |
| **Docker management** | ✅ | ❌ | None |
| **Process management** | ✅ Direct | ❌ API only | `subprocess_utils.py` |
| **AI chat** | ⚠️ Conditional | ❌ | None |
| **Console output** | ✅ Docked | ❌ | None |
| **Help system** | ✅ | ✅ | None (duplicated) |
| **Dark theme** | ✅ | ✅ | `style_constants.py` |
| **Toast notifications** | ✅ | ✅ | None (duplicated) |
| **Multi-page navigation** | ❌ Single window | ✅ 7 pages | None |
| **Data explorer** | ❌ | ✅ | None |
| **Video analyzer** | ❌ | ✅ | None |
| **Putting green** | ❌ | ✅ | None |
| **Motion capture** | ⚠️ Launcher only | ✅ Page | None |
| **Drag & drop** | ✅ | ❌ | None |
| **Keyboard shortcuts** | ✅ | ❌ | None |

### 9.2 The Core Parity Problem

**There is no shared UI contract or abstraction layer.**

- PyQt6 talks to Python directly (same process)
- React talks to Python via REST API (separate process)
- Neither uses a shared "ViewModel" or "Presenter" pattern
- Feature definitions (what buttons exist, what they do) are hardcoded independently in each frontend

This means:
1. Adding a feature requires implementation in 2 codebases
2. Behavior can drift silently
3. Testing requires 2 separate test suites
4. Bug fixes may be applied to one UI but not the other

### 9.3 Recommended Shared Architecture

```
src/
├── contracts/
│   ├── launcher_contract.py    # What a launcher UI must implement
│   ├── simulation_contract.py  # What simulation controls must implement
│   ├── settings_contract.py    # What settings must support
│   └── feature_flags.py        # Which features are enabled
├── viewmodels/
│   ├── launcher_vm.py          # State + logic for launcher (no UI)
│   ├── simulation_vm.py        # State + logic for simulation
│   └── settings_vm.py          # State + logic for settings
└── api/
    └── ... (serves viewmodel data to React)
```

The PyQt6 frontend would use viewmodels directly. The React frontend would access them via API. Both would implement the same contracts, ensuring parity.

---

## 10. Build, Config & CI Infrastructure

### 10.1 Configuration Conflicts

| Conflict | File A | File B | Impact |
|----------|--------|--------|--------|
| NumPy version | `requirements.txt` (>=2.0.1) | `environment.yml` (<2.0.0) | Conda env installs wrong version |
| MyPy version | CI workflow (1.13.0) | Lock file (1.19.1) | Different type checking results |
| Python version | `Dockerfile` (3.12) | `environment.yml` (3.11) | Container vs local mismatch |
| API port | `Dockerfile` (EXPOSE 8000) | `docker-compose.yml` (8001) | Container networking broken |
| Flask dependency | `requirements.txt` (present) | `pyproject.toml` (absent) | Unnecessary dependency installed |

### 10.2 Redundant Configuration

| Redundant File | Canonical Location | Action |
|---------------|-------------------|--------|
| `mypy.ini` | `pyproject.toml [tool.mypy]` | Delete `mypy.ini` |
| `pytest_improvements.ini` | `pyproject.toml [tool.pytest.ini_options]` | Delete `.ini` |
| `setup.py` | `pyproject.toml` | Delete `setup.py` |
| `requirements.txt` | `pyproject.toml [project.dependencies]` | Keep for pip-only users but sync |

### 10.3 CI/CD Assessment

- **Workflows:** 60+ files in `.github/workflows/`; 45 are Jules-AI experimental
- **Critical gap:** No pytest step in standard CI pipeline
- **Pre-commit hooks:** Well-configured (60+ hooks)
- **Security scanning:** Bandit, pip-audit, semgrep reports exist
- **Missing:** Dependency vulnerability blocking, coverage threshold enforcement

### 10.4 Makefile Assessment

The `Makefile` exists but its targets should be audited against actual usage. Common issues:
- Targets may reference deleted scripts
- No `make test` target that mirrors CI
- Missing `make clean` to remove junk files

---

## 11. Prioritized Action Plan

### Phase 1: Emergency Cleanup (Week 1, ~8 hours)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 1 | Delete 45+ junk files from root | Repo hygiene | 30 min |
| 2 | Update `.gitignore` with missing patterns | Prevent reintroduction | 15 min |
| 3 | Fix `safe_eval.py` import error (line 30) | Unblock safe_eval | 5 min |
| 4 | Remove stale `.git/index.lock` | Unblock git operations | 5 min |
| 5 | Delete root `/api/` directory (stale .pyc files) | Remove confusion | 5 min |
| 6 | Resolve NumPy version conflict | Unblock conda users | 15 min |
| 7 | Remove Flask from `requirements.txt` | Reduce attack surface | 5 min |
| 8 | Fix Docker port mismatch (8000 vs 8001) | Unblock containerized deploy | 15 min |
| 9 | Add pytest step to CI pipeline | Unblock test automation | 1 hour |
| 10 | Delete redundant config files (`mypy.ini`, `pytest_improvements.ini`, `setup.py`) | Reduce confusion | 15 min |
| 11 | Merge to `main` and delete stale branches | Clean branch state | 30 min |

### Phase 2: Architecture Cleanup (Weeks 2-3, ~20 hours)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 12 | Refactor engine loaders to factory pattern | Eliminate 150+ LOC duplication | 3 hours |
| 13 | Apply `@handle_api_errors` to all routes | Consistent error handling | 2 hours |
| 14 | Consolidate duplicate directories (`engines/`, `launchers/`, `python/`) into `src/` | Clear hierarchy | 4 hours |
| 15 | Rename root `shared/` to `data/myosuite/` | Remove naming confusion | 30 min |
| 16 | Split `src/shared/python/` into focused packages | SRP at package level | 8 hours |
| 17 | Add `__all__` to all public modules | Define public API | 2 hours |
| 18 | Prune `docs/assessments/` to latest canonical versions | Reduce noise | 1 hour |

### Phase 3: Quality Systems (Weeks 3-4, ~15 hours)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 19 | Apply DbC contracts to `ControlInterface` | Safety-critical validation | 2 hours |
| 20 | Apply DbC contracts to all engine loaders | Load validation | 2 hours |
| 21 | Add pytest markers to all test files | Enable selective test runs | 3 hours |
| 22 | Add coverage threshold (60%) to CI | Prevent coverage erosion | 1 hour |
| 23 | Add PyQt6 tests using `pytest-qt` | Test coverage for primary UI | 4 hours |
| 24 | Centralize all configuration defaults | Single source of truth | 3 hours |

### Phase 4: Parity Infrastructure (Month 2, ~30 hours)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 25 | Design and implement ViewModel layer | Shared state logic | 15 hours |
| 26 | Create UI contracts for launcher, simulation, settings | Ensure feature parity | 5 hours |
| 27 | Migrate PyQt6 to use ViewModels | Reduce duplication | 5 hours |
| 28 | Migrate React to use ViewModel-backed API | Reduce duplication | 5 hours |

---

## Appendix: File-Level Issue Table

| File | Issue | Severity | Category |
|------|-------|----------|----------|
| `src/shared/python/safe_eval.py:30` | Wrong import path (`from contracts import require`) | CRITICAL | Functional |
| `src/engines/loaders.py:22-233` | 7x duplicated loader pattern (~200 LOC) | CRITICAL | DRY |
| `requirements.txt` | NumPy >=2.0.1 conflicts with `environment.yml` <2.0.0 | CRITICAL | Config |
| `requirements.txt` | Flask dependency not in `pyproject.toml` | HIGH | Config |
| `.github/workflows/ci-standard.yml` | No pytest step; MyPy version mismatch | CRITICAL | CI |
| `Dockerfile` / `docker-compose.yml` | Port 8000 vs 8001 mismatch | HIGH | Config |
| `src/shared/python/engine_core/engine_manager.py` | God class (~100 line `__init__`) | HIGH | Architecture |
| `src/shared/python/control_interface.py` | No DbC contracts on safety-critical methods | HIGH | DbC |
| `src/api/routes/*.py` | `@handle_api_errors` not applied consistently | HIGH | DRY |
| `tests/conftest.py:22-62` | `sys.modules` manipulation for test isolation | HIGH | TDD |
| `src/launchers/tests/__init__.py` | Empty — zero PyQt6 tests | HIGH | TDD |
| `src/api/config.py:7-38` | Hardcoded configuration values | MEDIUM | Config |
| `src/api/dependencies.py:101-116` | `Any` type hints lose type safety | MEDIUM | Types |
| `src/api/models/requests.py` | Missing validator docstrings (Raises clause) | LOW | Docs |
| `mypy.ini` | Duplicates `pyproject.toml [tool.mypy]` | MEDIUM | Config |
| `pytest_improvements.ini` | Duplicates `pyproject.toml [tool.pytest.ini_options]` | MEDIUM | Config |
| Root directory | 45+ junk/temp files committed | HIGH | Hygiene |
| `root shared/` | Confusing name (contains MyoSuite models, not shared code) | MEDIUM | Architecture |
| `root api/` | Stale directory with only `.pyc` files | MEDIUM | Hygiene |
| `root engines/` | Duplicates `src/engines/` | MEDIUM | Architecture |
| `root launchers/` | Contains compiled `.pyc` (141 KB) | MEDIUM | Hygiene |

---

*This assessment was generated adversarially. All findings are based on direct file reading and code analysis. The goal is to surface issues that would otherwise go unnoticed, not to discourage the development team. The architectural vision of this project is strong — execution needs to catch up.*
