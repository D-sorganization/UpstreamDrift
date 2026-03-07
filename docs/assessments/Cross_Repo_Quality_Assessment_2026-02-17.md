# Cross-Repository Quality Assessment — 2026-02-17

**Assessment Date**: 2026-02-17
**Methodology**: Pragmatic Programmer Framework + Automated AST Analysis
**Scope**: AffineDrift, UpstreamDrift, Gasification_Model, Tools
**Assessed By**: Automated + Manual Code Review

---

## Executive Summary

| Metric                   | AffineDrift            | UpstreamDrift | Gasification_Model | Tools      |
| ------------------------ | ---------------------- | ------------- | ------------------ | ---------- |
| **Python Files**         | 2,097 (non-venv: ~120) | 1,396         | 1,805              | 1,038      |
| **Total Functions**      | 1,058                  | 17,052        | 18,447             | 8,841      |
| **Total Classes**        | 124                    | 3,160         | 3,207              | 1,826      |
| **Functions > 50 Lines** | 54 (5.1%)              | 1,001 (5.9%)  | 1,561 (8.5%)       | 633 (7.2%) |
| **Files > 500 Lines**    | 5                      | 245           | 305                | 144        |
| **Docstring Coverage**   | 72.2%                  | 82.5%         | 76.8%              | 73.5%      |
| **Type Hint Coverage**   | 70.7%                  | 72.0%         | 58.8%              | 67.3%      |
| **print() Calls**        | 85                     | 424           | 970                | 140        |
| **Bare Excepts**         | 0                      | 0             | 0                  | 0          |
| **Wildcard Imports**     | 0                      | 0             | 7                  | 0          |
| **TODO/FIXME Markers**   | 12                     | 36            | 35                 | 36         |
| **Tests Passed**         | 421                    | 5,264         | 4,330              | 1,996      |
| **Tests Failed**         | 1                      | 34            | 64                 | 1          |
| **Test Pass Rate**       | 99.8%                  | 99.4%         | 95.6%              | 99.9%      |

---

## 1. AffineDrift

### Overall Score: 7.2/10

#### 1.1 DRY (Don't Repeat Yourself) — 6/10

- **Positive**: Clean module separation between content, scripts, and src packages
- **Issue**: `Universal_Joint_Model_Enhanced.py` is duplicated verbatim across `content/` and `docs/content/` (exact same 484-line initUI + 333-line update_diagram). This is the single most severe DRY violation
- **Issue**: Assessment scripts (`run_assessment.py`, `generate_assessment_summary.py`) contain overlapping report-generation logic that could be extracted into a shared utility
- **TODO**: 12 markers remaining

#### 1.2 Design by Contract (DbC) — 4/10

- **Critical Gap**: Only 1 assertion in non-test production code; 11 precondition raises
- **Issue**: For a physics/mathematics-oriented codebase, there should be far more input validation on numerical parameters (angle ranges, matrix dimensions, physical constraints)
- **Recommendation**: Add preconditions to all public calculation functions (e.g., validate angle ranges before torque transmission calculations)

#### 1.3 Orthogonality / Decoupling — 7/10

- **Positive**: Clean separation between articles, content analysis tools, and the wrist universal joint app
- **Issue**: The Streamlit app (`wrist_universal_joint/streamlit_app.py`) has a 128-line `_render_sidebar` that mixes UI layout, parameter validation, and physics computation. These responsibilities should be decomposed

#### 1.4 Function Size / Single Responsibility — 4/10

- **Critical**: 54 functions exceed 50 lines (5.1% of total)
- **Worst Offender**: `initUI` (484 lines) — pure UI setup monolith that should be decomposed into `_create_menu_bar()`, `_create_parameter_panel()`, `_create_visualization_panel()`, etc.
- **Worst Offender**: `update_diagram` (333 lines) — mixed rendering, computation, and layout logic
- **Note**: These are duplicated files, so fixing one fixes two

#### 1.5 Reusability — 6/10

- **Positive**: `constants.py` provides shared physical constants
- **Issue**: Utility functions in scripts are not importable as a library (they rely on CLI argparse patterns)
- **Issue**: Report generation templates are embedded in scripts rather than externalized

#### 1.6 Reversibility / Changeability — 7/10

- **Positive**: Streamlit app uses configurable parameters
- **Issue**: The Universal Joint model has hardcoded rendering configurations that should be data-driven

#### 1.7 Comment Quality — 7/10

- **Positive**: 72.2% docstring coverage is above average
- **Issue**: Many docstrings are auto-generated boilerplate without meaningful parameter descriptions
- **Gap**: Inline comments explaining physics formulas are sparse

#### 1.8 Test Quality — 7/10

- **Positive**: 421 tests with 99.8% pass rate (only 1 failing)
- **Positive**: Good coverage of CLI tools and report generation
- **Issue**: No integration tests for the Streamlit app
- **Issue**: Physics calculations (torque transmission, universal joint kinematics) have limited edge-case testing

---

## 2. UpstreamDrift

### Overall Score: 8.0/10

#### 2.1 DRY — 7/10

- **Positive**: Shared Python packages (`validation_pkg`, `logging_pkg`, `theme`) avoid cross-module duplication
- **Positive**: Centralized error handling via `core/error_utils.py`
- **Issue**: 424 `print()` calls remaining — many should be replaced with the logging framework
- **Issue**: Some validation patterns are reimplemented across `validation_pkg` and individual module-level validators

#### 2.2 Design by Contract (DbC) — 8/10

- **Strong**: 94 assertions + 557 precondition raises in production code
- **Strong**: Dedicated `validation_pkg` with physics-aware validators (`validate_mass`, `validate_inertia_matrix`, `validate_joint_limits`)
- **Strong**: `validate_physical_bounds` decorator pattern for automatic DbC
- **Issue**: Not all public API functions use the validation framework consistently

#### 2.3 Orthogonality / Decoupling — 8/10

- **Strong**: Clean API/engine/shared architecture with well-defined package boundaries
- **Strong**: Engine abstraction allows multiple physics backends (Pinocchio, Drake, MATLAB/Simscape)
- **Issue**: `launch_golf_suite.py` main function is 100 lines — couples launcher logic with engine discovery
- **Issue**: Some theme/UI code imports from engine-specific packages

#### 2.4 Function Size / Single Responsibility — 6/10

- **Concerning**: 1,001 functions exceed 50 lines (5.9%), with the worst being 119 lines
- **Note**: Many of the worst offenders (99-line functions) are in test fixtures or UI setup
- **Issue**: `_setup_ui` (100 lines), `_build_engine_profiles` (100 lines), `run_agent` (105 lines) all need decomposition

#### 2.5 Reusability — 9/10

- **Excellent**: Shared library architecture (`src/shared/python/`) is well-designed
- **Excellent**: Physics validation utilities are generic and reusable across engines
- **Excellent**: Spatial algebra, biomechanics, and data_io modules are independently importable

#### 2.6 Reversibility / Changeability — 8/10

- **Strong**: Multi-engine architecture makes physics backend swappable
- **Strong**: Configuration via dataclasses (`GolferModel`, `ClubModel`, `OptimizationConfig`)
- **Issue**: MATLAB/Simscape integration has some hardcoded paths

#### 2.7 Comment Quality — 8/10

- **Strong**: 82.5% docstring coverage — highest of all repos
- **Strong**: Physics formulas include mathematical context (e.g., "KE = 0.5·ω^T·I·ω > 0")
- **Issue**: Some auto-generated docstrings lack meaningful parameter descriptions

#### 2.8 Test Quality — 7/10

- **Strong**: 5,264 tests with 99.4% pass rate
- **Strong**: Multi-layer test strategy (unit, integration, physics_validation)
- **Issue**: 34 test failures, primarily in `ball_flight_physics` (Numba compilation) and GUI tests
- **Issue**: Test coverage at ~40.6% (from CI report) — significant uncovered code in `unreal_integration` and UI modules
- **Gap**: Integration tests for launcher are fragile (2 errors)

---

## 3. Gasification_Model

### Overall Score: 6.5/10

#### 3.1 DRY — 5/10

- **Critical**: 970 `print()` calls — the highest of all repos, indicating widespread logging-vs-print confusion
- **Critical**: `vendor/ud-tools/` contains full copies of shared code that also exists in the Tools repo. The `create_plotting_tab` (904 lines) and `create_plot_left_content` (732 lines) in vendored code are extreme monoliths
- **Issue**: 7 wildcard imports (`from module import *`) — these make dependency tracing impossible
- **Issue**: Assessment scripts duplicate logic found in the Tools repo

#### 3.2 Design by Contract (DbC) — 6/10

- **Moderate**: 75 assertions + 436 precondition raises
- **Issue**: For a thermodynamic/engineering calculation codebase, the level of input validation is inadequate
- **Issue**: Gibbs minimizer and enthalpy calculator lack systematic boundary checks
- **Gap**: No systematic use of validation decorators (unlike UpstreamDrift)

#### 3.3 Orthogonality / Decoupling — 5/10

- **Issue**: The `electrode_advisor` module has `_create_visual_controls_panel` at 447 lines and `_create_input_panel` at 339 lines — massive UI/logic coupling
- **Issue**: Vendor directory creates a shadow dependency that's hard to track
- **Issue**: Thermodynamic calculator mixes computation, file I/O, and UI presentation
- **Positive**: Process calculators are reasonably modular

#### 3.4 Function Size / Single Responsibility — 3/10

- **Critical**: 1,561 functions exceed 50 lines (8.5%) — the worst ratio of all repos
- **Extreme**: Top offenders are 904, 732, 623, 497, 447 lines. These are "God functions" that need immediate decomposition
- **Critical**: Many of these are in UI setup code that mixes layout, event binding, validation, and business logic
- **Note**: Some worst offenders are in `vendor/` (external copies), but even excluding those, the pattern persists

#### 3.5 Reusability — 5/10

- **Issue**: Thermodynamic calculation utilities are embedded in application code rather than exposed as a library
- **Issue**: Process calculators could be a standalone package but are tightly coupled to the UI
- **Positive**: Some utility modules (`optimization.py`, `model_generation/`) show good library design

#### 3.6 Reversibility / Changeability — 5/10

- **Issue**: Direct dependencies on specific solver backends without abstraction layers
- **Issue**: UI framework (PyQt6) is deeply coupled with business logic
- **Positive**: Configuration files are used for process parameters

#### 3.7 Comment Quality — 6/10

- **Moderate**: 76.8% docstring coverage
- **Gap**: Many thermodynamic formulas lack explanatory comments
- **Issue**: Engineering-specific terminology is used without context for maintainers unfamiliar with gasification processes

#### 3.8 Test Quality — 5/10

- **Concerning**: 4,330 passed but 64 failures + 109 errors = 95.6% pass rate (lowest)
- **Issue**: Verification benchmarks failing (`test_benchmarks.py`) indicates regression in core calculation accuracy
- **Issue**: 65 tests skipped — may mask underlying issues
- **Issue**: Memory leak tests (`test_tab_manager_memory_leaks.py`) are erroring
- **Gap**: xfailed tests (Gibbs minimizer convergence) indicate known unresolved bugs

---

## 4. Tools

### Overall Score: 7.5/10

#### 4.1 DRY — 7/10

- **Positive**: Shared utility modules (`upstream_drift_tools`, `theme`, `assessment_utils`) centralize common logic
- **Issue**: 140 `print()` calls remaining
- **Issue**: Some tool-specific scripts duplicate validation patterns available in shared modules
- **Issue**: 36 TODO/FIXME markers indicate unfinished consolidation

#### 4.2 Design by Contract (DbC) — 7/10

- **Good**: 61 assertions + 380 precondition raises
- **Positive**: Better than AffineDrift and on par per-function with Gasification_Model
- **Gap**: URDF/model editors lack systematic input validation on XML/model parameters

#### 4.3 Orthogonality / Decoupling — 6/10

- **Issue**: `test_shared_does_not_import_tool_packages` test is failing — shared library is importing tool-specific code (3 violations). This is a layer boundary violation
- **Issue**: `generate_stylesheet` (556 lines) in theme module mixes stylesheet generation with theme logic
- **Positive**: Data processor has reasonable separation between UI and processing logic

#### 4.4 Function Size / Single Responsibility — 5/10

- **Concerning**: 633 functions exceed 50 lines (7.2%)
- **Worst**: `create_help_tab` (296 lines), `_create_visual_controls_panel` (379 lines), `_validate_urdf` (252 lines)
- **Issue**: `run_adam_optimization` (243 lines) mixes optimization loop, convergence checking, and results formatting

#### 4.5 Reusability — 8/10

- **Strong**: Shared python packages are well-structured for cross-repo use
- **Strong**: Assessment utilities, theme system, and process calculators are importable libraries
- **Issue**: Some tools have hardcoded file paths

#### 4.6 Reversibility / Changeability — 7/10

- **Positive**: Theme system is configurable
- **Positive**: Model generation uses template patterns
- **Issue**: Some direct file system assumptions would break on different OS configurations

#### 4.7 Comment Quality — 6/10

- **Moderate**: 73.5% docstring coverage — lowest of the four, but still reasonable
- **Gap**: Complex algorithms (character builder, optimization) lack inline explanations
- **Issue**: MATLAB quality analysis code has minimal documentation

#### 4.8 Test Quality — 8/10

- **Strong**: 1,996 tests with 99.9% pass rate (only 1 failure)
- **Strong**: Architecture-level tests catch layer boundary violations
- **Issue**: The single failing test (`test_shared_does_not_import_tool_packages`) indicates an actual code issue, not a test issue
- **Gap**: Some UI components lack test coverage

---

## Comparative Radar Scores (1-10)

| Dimension               | AffineDrift | UpstreamDrift | Gasification_Model |  Tools  |
| ----------------------- | :---------: | :-----------: | :----------------: | :-----: |
| **DRY**                 |      6      |       7       |         5          |    7    |
| **DbC**                 |      4      |       8       |         6          |    7    |
| **Orthogonality**       |      7      |       8       |         5          |    6    |
| **Function Size / SRP** |      4      |       6       |         3          |    5    |
| **Decoupling**          |      7      |       8       |         5          |    6    |
| **Reusability**         |      6      |       9       |         5          |    8    |
| **Reversibility**       |      7      |       8       |         5          |    7    |
| **Changeability**       |      7      |       8       |         5          |    7    |
| **Comment Quality**     |      7      |       8       |         6          |    6    |
| **Test Quality**        |      7      |       7       |         5          |    8    |
| **Test Coverage**       |      7      |       6       |         5          |    7    |
| **OVERALL**             |   **6.3**   |    **7.5**    |      **5.0**       | **6.7** |

---

## Critical Findings (Must Fix)

### P0 — Immediate Action Required

1. **[Gasification_Model] God Functions**: 5 functions exceed 400 lines. These are untestable, unmaintainable monoliths
2. **[Gasification_Model] 64 Test Failures + 109 Errors**: Verification benchmarks failing indicates potential calculation regression
3. **[Gasification_Model] 970 print() calls**: Violates organizational logging standards
4. **[Tools] Layer Boundary Violation**: Shared library importing tool-specific code (3 violations detected by architecture test)

### P1 — High Priority

5. **[AffineDrift] Duplicate Files**: `Universal_Joint_Model_Enhanced.py` exists identically in two locations
6. **[UpstreamDrift] 424 print() calls**: Should use the established logging framework
7. **[UpstreamDrift] 34 Test Failures**: Ball flight physics and GUI tests need attention
8. **[Gasification_Model] 7 Wildcard Imports**: Makes dependency tracking impossible

### P2 — Medium Priority

9. **[AffineDrift] Weak DbC**: Only 1 assertion in production code for a math-heavy codebase
10. **[All Repos] Function Size**: Combined 3,249 functions exceed 50 lines across all repos
11. **[Gasification_Model] Vendor Shadow Copies**: `vendor/ud-tools/` duplicates code from Tools repo
12. **[Tools] generate_stylesheet at 556 lines**: Theme logic needs decomposition

---

## Remediation Plan

### Wave 1 — Critical Fixes (Week 1)

| #   | Repo               | Task                                            | Impact            |
| --- | ------------------ | ----------------------------------------------- | ----------------- |
| 1   | Gasification_Model | Fix 64 test failures + 109 errors               | Test reliability  |
| 2   | Gasification_Model | Decompose top 5 God functions (>400 lines)      | SRP, Testability  |
| 3   | Tools              | Fix shared→tool import boundary violations      | Architecture      |
| 4   | Gasification_Model | Replace `from x import *` with explicit imports | DRY, Traceability |

### Wave 2 — print() → logging Migration (Week 2)

| #   | Repo               | Task                          | Impact               |
| --- | ------------------ | ----------------------------- | -------------------- |
| 5   | Gasification_Model | Convert 970 print() → logging | Standards compliance |
| 6   | UpstreamDrift      | Convert 424 print() → logging | Standards compliance |
| 7   | Tools              | Convert 140 print() → logging | Standards compliance |
| 8   | AffineDrift        | Convert 85 print() → logging  | Standards compliance |

### Wave 3 — Function Decomposition (Week 3-4)

| #   | Repo               | Task                                                                     | Impact           |
| --- | ------------------ | ------------------------------------------------------------------------ | ---------------- |
| 9   | Gasification_Model | Decompose all functions > 200 lines                                      | SRP, Testability |
| 10  | Tools              | Decompose generate_stylesheet, \_validate_urdf, run_adam_optimization    | SRP              |
| 11  | AffineDrift        | Decompose initUI (484 lines) and update_diagram (333 lines)              | SRP              |
| 12  | UpstreamDrift      | Decompose functions > 100 lines (setup_logging, \_build_engine_profiles) | SRP              |

### Wave 4 — DbC Hardening (Week 4-5)

| #   | Repo               | Task                                                   | Impact          |
| --- | ------------------ | ------------------------------------------------------ | --------------- |
| 13  | AffineDrift        | Add preconditions to all physics calculation functions | Correctness     |
| 14  | Gasification_Model | Add validation decorators to thermodynamic calculators | Correctness     |
| 15  | Tools              | Add input validation to URDF/model editors             | Robustness      |
| 16  | All                | Increase type hint coverage to ≥85% across all repos   | Static analysis |

### Wave 5 — DRY & Architecture Cleanup (Week 5-6)

| #   | Repo               | Task                                                      | Impact           |
| --- | ------------------ | --------------------------------------------------------- | ---------------- |
| 17  | AffineDrift        | Remove duplicate Universal_Joint_Model_Enhanced.py        | DRY              |
| 18  | Gasification_Model | Evaluate vendor/ for removal or git submodule replacement | DRY, Maintenance |
| 19  | All                | Eliminate remaining TODO/FIXME markers (119 total)        | Completeness     |
| 20  | UpstreamDrift      | Raise test coverage from 40.6% to ≥60%                    | Quality          |

---

## Methodology Notes

- **Function size analysis**: AST-based, counts from `def` to `end_lineno`
- **Docstring coverage**: Percentage of functions with `ast.get_docstring()` returning non-None
- **Type hint coverage**: Percentage of `def` lines containing `->` return type annotation
- **print() detection**: Line-based heuristic (lines starting with `print(` after strip)
- **Test execution**: `pytest tests/ --tb=no -q` for each repo
- **Excludes**: `.venv`, `.git`, `__pycache__`, `.mypy_cache`, `env`, `site-packages`
