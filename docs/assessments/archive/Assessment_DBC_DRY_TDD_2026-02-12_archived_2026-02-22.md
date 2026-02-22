# UpstreamDrift — DBC / DRY / TDD Quality Assessment

**Assessment Date:** 2026-02-12
**Repository:** UpstreamDrift (Golf Modeling Suite)
**Total Source Files:** 979 Python files (src/)
**Overall Grade:** 5.5/10

---

## Executive Summary

UpstreamDrift is the largest repository with 979 source files. It has **the strongest contract adoption** among the three repos (56 preconditions, 41 postconditions), but still leaves vast swaths of the codebase unprotected. DRY violations are severe — 174 functions >100 lines, two 2,000+ line glossary files, 347 `print()` statements, and 298 inline stylesheets. The test suite is large (4,032 tests) but has 62 pre-existing failures and relies heavily on mocking (2,445 mock patterns), suggesting brittle test design.

---

## 1. Design by Contract (DbC) — Grade: 6.5/10

### 1.1 Infrastructure (8/10)

One unified contract module:

- `src/shared/python/core/contracts.py` — Full decorator suite with precondition, postcondition, invariant

### 1.2 Adoption Metrics

| Metric                         | Count | Assessment                         |
| ------------------------------ | ----- | ---------------------------------- |
| `@precondition` decorators     | 56    | Moderate — best of the three repos |
| `@postcondition` decorators    | 41    | Moderate                           |
| `@invariant` decorators        | 4     | Low                                |
| `raise` statements             | 1,150 | Strong defensive coding            |
| `assert` statements (non-test) | 40    | Appropriate                        |
| `validate_*` functions         | 96    | Strong validation library          |

### 1.3 Adoption Analysis

**Where contracts ARE used (good examples):**

- `physics_engine.py` — `@precondition(lambda self: self.is_initialized)` on `step()`, `reset()`, `forward()`
- `rigid_body_dynamics/` — Pre/postconditions on dynamics computations
- `model_generation/` — Validation on builder operations

**Where contracts are MISSING (risk):**

- `drake_gui_app.py` (2,111 lines) — Large GUI with no contracts
- `pinocchio_golf/gui.py` (1,942 lines) — No input validation
- `humanoid_launcher.py` (1,817 lines) — No contracts
- `golf_gui_application.py` (1,639 lines) — Tab-building without validation
- `head_models.py` (1,634 lines) — Complex model generation without bounds checking
- API endpoints in `local_server.py` — No input sanitization contracts

**Adoption density:** 56 preconditions / 979 files = 5.7% coverage — Only the core physics engine is well-protected.

### 1.4 Scoring Breakdown

| Component                 | Score      | Notes                        |
| ------------------------- | ---------- | ---------------------------- |
| Infrastructure            | 8/10       | Unified contracts.py         |
| Adoption (preconditions)  | 6/10       | 56 uses, best of three repos |
| Adoption (postconditions) | 5/10       | 41 uses                      |
| Adoption (invariants)     | 2/10       | Only 4 uses                  |
| Validation utilities      | 8/10       | 96 validate\_ functions      |
| Physics engine coverage   | 9/10       | Excellent DbC on core engine |
| GUI/API coverage          | 2/10       | Large GUI files unprotected  |
| **Average**               | **6.5/10** |                              |

---

## 2. Don't Repeat Yourself (DRY) — Grade: 4.0/10

### 2.1 Monolith Files (Critical)

| File                        | Lines | Issue                          |
| --------------------------- | ----- | ------------------------------ |
| `glossary_data_extended.py` | 2,339 | Data file masquerading as code |
| `drake_gui_app.py`          | 2,111 | Monolithic GUI application     |
| `pinocchio_golf/gui.py`     | 1,942 | Monolithic GUI                 |
| `humanoid_launcher.py`      | 1,817 | Single-file launcher           |
| `glossary_data_core.py`     | 1,747 | Data file masquerading as code |
| `golf_gui_application.py`   | 1,639 | Monolithic tab builder         |
| `head_models.py`            | 1,634 | Model definitions              |

### 2.2 Function-Level Duplication

| Metric                               | Count                                | Target               |
| ------------------------------------ | ------------------------------------ | -------------------- |
| Functions >100 lines                 | 174                                  | < 30                 |
| Largest function                     | 2,330 lines (`get_extended_entries`) | < 100                |
| `sys.path` manipulations             | 67                                   | 0                    |
| `print()` in production code         | 347                                  | 0 (use logging)      |
| Inline `setStyleSheet()` calls       | 298                                  | 0 (use theme system) |
| Duplicate JSON/YAML loading patterns | 84                                   | Centralize           |
| Magic numbers                        | 317                                  | Extract to constants |

### 2.3 Top 5 DRY Hot Spots

1. **`sys.path` hacks** — 67 instances! Worst of all three repos. Shows systemic packaging issues.
2. **`print()` statements** — 347 in production code. Must migrate to `logging` module.
3. **Inline stylesheets** — 298 `setStyleSheet()` calls despite theme package existing in `src/shared/python/theme/`
4. **Glossary data-as-code** — Two files (2,339 + 1,747 = 4,086 lines) storing dictionary data as Python functions. Should be JSON/YAML data files.
5. **GUI monoliths** — Four GUI files >1,600 lines, all following the same pattern of mixing UI layout, event handling, and business logic.

### 2.4 `sys.path` Analysis

67 `sys.path` manipulations indicate that the package structure is broken:

- Modules can't find each other through normal Python imports
- Each script independently modifies `sys.path` to locate shared code
- This creates fragile runtime behavior and makes refactoring dangerous

### 2.5 Scoring Breakdown

| Component                  | Score      | Notes                                  |
| -------------------------- | ---------- | -------------------------------------- |
| Module decomposition       | 3/10       | 7 files >1,600 lines                   |
| Function granularity       | 2/10       | 174 functions >100, one is 2,330 lines |
| Style/theme centralization | 3/10       | 298 inline styles despite theme pkg    |
| Logging consistency        | 2/10       | 347 print() calls (worst)              |
| Magic number hygiene       | 4/10       | 317 bare literals                      |
| Path management            | 1/10       | 67 sys.path hacks (worst)              |
| Data separation            | 3/10       | Glossary data stored as code           |
| Infrastructure reuse       | 7/10       | Good shared/ packages exist            |
| **Average**                | **4.0/10** |                                        |

---

## 3. Test-Driven Development (TDD) — Grade: 5.5/10

### 3.1 Coverage Metrics

| Metric                    | Count                         | Assessment         |
| ------------------------- | ----------------------------- | ------------------ |
| Test files                | 260                           | Strong count       |
| Test functions            | 4,032                         | Very large suite   |
| Source modules (non-init) | 780                           | Large surface      |
| Test-to-source ratio      | 0.33 files, 5.17 tests/module | Good ratio         |
| Mock/Patch usage          | 2,445                         | Very heavy mocking |
| Fixture usage             | 401                           | Moderate           |
| Parametrize usage         | 24                            | Low                |

### 3.2 Critical Issues

**62 pre-existing test failures on main:**
These are NOT flaky tests — they are consistently broken:

- `test_drag_drop_functionality.py` — Broken mock patterns (JSON serialization)
- `test_launcher_fixes.py` — Same mock issues
- `test_gui_coverage.py` — Metaclass conflicts in mock setup
- `test_process_worker.py` — Incorrect mock attribute access
- `test_pose_estimation_gui.py` — Missing dependency handling
- `test_ci_infrastructure.py` — False dependency assertions
- `test_data_fitting.py` — Shape mismatch in numerical computation

**Excessive mocking (2,445 instances):**

- Tests are tightly coupled to implementation details
- Many tests will break on any refactoring regardless of behavior preservation
- Mock objects don't validate method signatures, leading to false positives
- Example: `MagicMock` used where a JSON string is expected → `TypeError`

### 3.3 Test Brittleness Indicators

| Indicator                        | Count    | Risk                  |
| -------------------------------- | -------- | --------------------- |
| Pre-existing failures            | 62       | 🔴 Critical           |
| Mock/Patch per test file         | 9.4 avg  | 🔴 Over-mocked        |
| Parametrize per test file        | 0.09 avg | 🟡 Under-parametrized |
| `MagicMock` type errors in tests | 11+      | 🔴 Wrong mock types   |
| Metaclass conflicts              | 6+       | 🔴 Mock setup broken  |

### 3.4 Scoring Breakdown

| Component                  | Score      | Notes                  |
| -------------------------- | ---------- | ---------------------- |
| Coverage breadth           | 7/10       | 260 test files, good   |
| Coverage depth             | 6/10       | 4,032 tests            |
| Test reliability           | 3/10       | 62 failures on main    |
| Test isolation             | 4/10       | Over-mocked, brittle   |
| Test reuse (fixtures)      | 5/10       | 401 fixtures, moderate |
| Parametrize/property-based | 3/10       | Only 24 parametrize    |
| CI enforcement             | 4/10       | Tests run but 62 fail  |
| **Average**                | **5.5/10** |                        |

---

## 4. Remediation Priority

### Phase 1: Quick Wins (1-2 days each)

| #   | Action                                                        | Impact   | Effort |
| --- | ------------------------------------------------------------- | -------- | ------ |
| 1   | Fix 62 pre-existing test failures                             | TDD +2   | Medium |
| 2   | Replace 347 `print()` with `logging`                          | DRY +1.5 | Medium |
| 3   | Remove 67 `sys.path` hacks (proper package install)           | DRY +1.5 | Medium |
| 4   | Move glossary data to JSON files (eliminate 4,086 code lines) | DRY +1   | Low    |

### Phase 2: Structural (3-5 days each)

| #   | Action                                                         | Impact   | Effort |
| --- | -------------------------------------------------------------- | -------- | ------ |
| 5   | Decompose 4 GUI monoliths (>1,600 lines each)                  | DRY +2   | High   |
| 6   | Centralize 298 inline stylesheets to theme system              | DRY +1.5 | High   |
| 7   | Add `@precondition` to 30 more critical APIs (GUI, API, model) | DbC +2   | Medium |
| 8   | Replace brittle MagicMock patterns with spec-based mocks       | TDD +1.5 | Medium |

### Phase 3: Strategic (1-2 weeks)

| #   | Action                                        | Impact | Effort |
| --- | --------------------------------------------- | ------ | ------ |
| 9   | Extract 317 magic numbers to named constants  | DRY +1 | Medium |
| 10  | Add `@invariant` to 10+ core classes          | DbC +1 | Medium |
| 11  | Add property-based tests for physics/dynamics | TDD +1 | Medium |
| 12  | Enforce pytest-cov 70% minimum gate           | TDD +1 | Low    |

---

## 5. Target Grades After Remediation

| Dimension   | Current | Phase 1 | Phase 2 | Phase 3 |
| ----------- | ------- | ------- | ------- | ------- |
| DbC         | 6.5     | 7.0     | 8.5     | 9.0     |
| DRY         | 4.0     | 6.5     | 8.5     | 9.0     |
| TDD         | 5.5     | 7.5     | 8.5     | 9.0     |
| **Overall** | **5.5** | **7.0** | **8.5** | **9.0** |

---

_Assessment conducted 2026-02-12 using AST analysis, grep heuristics, and manual code review._
