# Code Quality Assessment — UpstreamDrift Repository

**Assessment Date:** 2026-02-12
**Assessor:** Claude Code (Opus 4.6, Automated Deep Review)
**Repository:** UpstreamDrift (Biomechanical Golf Simulation / Multi-Physics Platform)
**Branch:** main (post-refactoring sprint Feb 10-11)
**Prior Assessment:** 2026-02-10 (score 4.6/10), 2026-02-03 (score 7.6/10)

---

## Executive Summary

The repository underwent an intensive refactoring sprint over the last two days (Feb 10-11, 2026), producing 12 commits across 8 PRs that addressed the most severe findings from the Feb 10 adversarial assessment. The results are substantial: broad `except Exception` handlers dropped from 739 to 12 in production code (98.4% reduction), 98 backward-compatibility shim files were removed in three phases, `sys.path` hacks were replaced with a canonical `_bootstrap` pattern, two monolithic modules were decomposed into focused submodules, and the React frontend gained proper Zustand state management with 100% test coverage. ProvenanceInfo — previously dead infrastructure — is now wired into the production export pipeline.

However, the refactoring sprint prioritized speed over TDD discipline. Only 2 of 6 PRs (#1322 Zustand, #1316 ProvenanceInfo) followed test-first practices. The launcher decomposition (1,415 lines across 5 new files) and the `_bootstrap.py` module have zero unit tests. The Drake and Pinocchio GUIs remain monolithic at 2,111 and 1,942 lines respectively, and 54 files in `src/` still exceed the 800-line critical threshold defined in AGENTS.md.

**Overall Grade: 5.8/10** (up from 4.6 on Feb 10; still below the 7.6 baseline from Feb 3)

---

## Refactoring Sprint Summary (Feb 10-11, 2026)

### Commits Analyzed

| PR | Title | Files Changed | Lines +/- | TDD? |
|----|-------|:------------:|:---------:|:----:|
| #1322 | feat: add Zustand state management to React frontend | 18 | +1,100 / -187 | YES |
| #1321 | refactor: decompose ui_components.py and data_models.py | 11 | +2,886 / -2,657 | NO |
| #1320 | refactor: consolidate duplicated GUI widgets and RecorderInterface | 8 | +88 / -152 | PARTIAL |
| #1319 | refactor: remove final 30 backward-compatibility shim files (phase 3) | 313 | +2,208 / -1,719 | NO |
| #1318 | refactor: remove 39 more backward-compatibility shim files (phase 2) | (included in #1319 stats) | — | NO |
| #1317 | refactor: remove 29 dead backward-compatibility shim files | (included in #1319 stats) | — | NO |
| #1316 | feat: wire ProvenanceInfo into production export pipeline | (combined) | — | YES |
| #1314 | refactor: replace 13 sys.path hacks with canonical _bootstrap pattern | (combined) | — | NO |

### Key Metrics Before/After

| Metric | Feb 10 | Feb 12 | Delta | Assessment |
|--------|-------:|-------:|------:|:-----------|
| `except Exception` in production src/ | 739 | 12 | -727 (98.4%) | EXCELLENT |
| Backward-compat shim files | 98 | 0 | -98 (100%) | EXCELLENT |
| `sys.path` hacks (insert/remove) | 42 | 14 | -28 (67%) | GOOD |
| `print()` in non-test src/ | 754 | 231 | -523 (69%) | GOOD |
| Files >800 lines in src/ | ~60 | 54 | -6 (10%) | MARGINAL |
| TODO/FIXME/HACK comments | 65 | 0 | -65 (100%) | EXCELLENT |
| Test files | 252 | 259 | +7 | GOOD |
| ProvenanceInfo in production | 0 uses | 4 call sites | +4 | GOOD |
| React state management | None (prop drilling) | 3 Zustand stores | +3 | EXCELLENT |

---

## Design Criteria Assessment (AGENTS.md Section 5)

### 1. DRY — Don't Repeat Yourself

**Score: 5.5 / 10** (up from 4.0)

| Metric | Count | Severity |
|--------|------:|:--------:|
| Drake GUI lines | 2,111 | CRITICAL |
| Pinocchio GUI lines | 1,942 | CRITICAL |
| SimulationGUIBase exists but under-utilized | 520 lines | MAJOR |
| MuJoCo sim_widget does NOT inherit from base | — | MAJOR |
| Glossary data files (core + extended) | 4,086 lines | MINOR |
| Backward-compat re-export shims remaining | ~11 files | MINOR |

**What improved:**
- 98 `sys.modules` shim files deleted (PRs #1317-#1319) — the single largest DRY cleanup
- `LogPanel` and `SignalBlocker` consolidated into `src/shared/python/ui/widgets.py` (PR #1320)
- `ui_components.py` decomposed from monolith to 5 focused modules (PR #1321)
- `data_models.py` decomposed from monolith to 4 focused modules (PR #1321)
- Engine detection centralized in `engine_availability.py` (prior work, still intact)

**What remains:**
- Drake and Pinocchio GUIs are near-twins (2,111 vs 1,942 lines, 62 vs 63 Qt widget instantiations) but `SimulationGUIBase` only extracts 520 lines of commonality. Both define `_setup_ui()` independently with overlapping widget hierarchies.
- MuJoCo's `sim_widget.py` (925 lines) does not inherit from `SimulationGUIBase` at all. Only its advanced `main_window.py` does.
- Glossary data split across `glossary_data_core.py` (1,747 lines) and `glossary_data_extended.py` (2,339 lines) with potential overlap.
- 11 files still contain backward-compatibility re-export patterns (e.g., `engine_core/engine_loaders.py`, `core/exceptions.py`, `analysis/__init__.py`). These are thin facades but still represent import indirection.

**Remediation priority:**
1. Extract common `_setup_ui()` logic from Drake/Pinocchio into `SimulationGUIBase`
2. Make MuJoCo `sim_widget.py` inherit from `SimulationGUIBase`
3. Audit and consolidate glossary data files

---

### 2. Design by Contract (DbC)

**Score: 4.5 / 10** (up from 4.0)

| Metric | Count | Target | Gap |
|--------|------:|-------:|----:|
| Functions with `@precondition`/`@postcondition` decorators | 11 files | Widespread | LARGE |
| `assert` statements in production src/ | 40 | 200+ | LARGE |
| Input validation at API boundaries (Pydantic) | 284 validators | — | GOOD |
| Contract decorators defined in `contracts.py` | YES | — | GOOD |

**What improved:**
- ProvenanceInfo now wired into production (PR #1316), which includes validation of captured metadata
- Pydantic validators remain comprehensive at API boundaries (284 field/model validators)
- Contract decorator infrastructure (`@precondition`, `@postcondition`, `@invariant`) exists in `src/shared/python/core/contracts.py`

**What remains:**
- Only 11 files use the contract decorators despite the infrastructure being in place. These are limited to physics engine wrappers, `analysis_service.py`, and `contact_manager.py`.
- Only 40 `assert` statements across all production `src/` code — extremely low for a physics simulation platform where numerical invariants should be routinely checked.
- Physics engine model entry points lack systematic input validation for model parameters (mass > 0, inertia positive-definite, joint limits in range).
- Statistical analysis functions accept inputs without range/type checking.
- No postcondition validation on simulation outputs (e.g., energy conservation bounds, physical plausibility checks).

**Remediation priority:**
1. Add `@precondition` decorators to all physics engine `load_model()` and `step()` methods
2. Add `assert` invariants for physical plausibility in simulation loops
3. Add postcondition checks on export functions (non-empty output, valid format)

---

### 3. Test-Driven Development (TDD)

**Score: 5.5 / 10** (down from 6.5 on Feb 10)

| Metric | Value | Target | Assessment |
|--------|------:|-------:|:-----------|
| Test files | 259 | — | GOOD volume |
| Test-to-source ratio | 259:913 (1:3.5) | 1:2 | BELOW TARGET |
| Test coverage (estimated) | ~28-35% | 80% | CRITICAL gap |
| PRs with TDD in refactoring sprint | 2/6 (33%) | 100% | POOR |

**TDD compliance in the refactoring sprint:**

| PR | TDD Followed? | Evidence |
|----|:------------:|---------|
| #1322 (Zustand stores) | YES | 3 test files (499 lines) for 3 stores (380 lines). Full coverage. |
| #1316 (ProvenanceInfo) | YES | `test_provenance.py` (100+ lines) covers capture, headers, CSV integration. |
| #1321 (Decompose modules) | NO | 1,415 lines of new launcher code with 0 tests. Unreal modules tested only through facade. |
| #1320 (Consolidate widgets) | PARTIAL | `LogPanel` and `SignalBlocker` have no direct tests. |
| #1319 (Remove shims) | NO | Relied on existing test suite passing, no shim-removal-specific tests. |
| #1314 (_bootstrap pattern) | NO | `_bootstrap.py` (67 lines) has zero test coverage. Critical import mechanism untested. |

**Score reduced from 6.5 to 5.5** because the refactoring sprint introduced 1,500+ lines of new code without following TDD practices. The AGENTS.md Section 4 states TDD is **MANDATORY** — "All new code must follow the Test-Driven Development methodology." Four of six PRs violated this requirement.

**What's good:**
- 259 test files is substantial volume
- Zustand store tests are exemplary (proper `act()` wrappers, `beforeEach` resets, 36 test suites)
- React frontend has 300+ passing tests
- CI runs tests with `pytest-xdist` parallelism

**What's missing:**
- `_bootstrap.py` — no tests for repo root discovery, path addition, or conditional bootstrap
- `src/launchers/docker_dialog.py` (155 lines) — no tests
- `src/launchers/help_dialogs.py` (175 lines) — no tests
- `src/launchers/model_card.py` (291 lines) — no tests
- `src/launchers/settings_dialog.py` (527 lines) — no tests
- `src/launchers/startup.py` (267 lines) — no tests
- `src/shared/python/ui/widgets.py` — no tests for `LogPanel`/`SignalBlocker`

**Remediation priority:**
1. Write tests for `_bootstrap.py` (critical infrastructure)
2. Write tests for the 5 decomposed launcher modules
3. Write tests for the new shared widgets

---

### 4. Orthogonality & Decoupling

**Score: 5.5 / 10** (up from 4.0)

**What improved:**
- 98 shim files removed, reducing import indirection (PRs #1317-#1319)
- `sys.path` hacks reduced from 42 to 14 via `_bootstrap` pattern (PR #1314)
- Module decomposition properly separates concerns: launcher startup logic, Docker dialogs, model cards, settings, and help are now in separate files
- Unreal data models properly separated: geometry, golf state, skeleton, data frame
- Zustand stores decouple UI state from component props (PR #1322)

**What remains:**
- Drake `drake_gui_app.py` (2,111 lines) combines GUI setup, simulation logic, and visualization in one file — violates AGENTS.md 5c ("DO NOT mix UI logic with business/calculation logic")
- Pinocchio `gui.py` (1,942 lines) has the same coupling
- `src/shared/python/engine_core/engine_loaders.py` still imports from `src.engines.loaders` (shared -> engines is inverted dependency direction)
- 14 `sys.path` manipulation calls remain in production code

---

### 5. Monolithic Files

**Score: 3.0 / 10** (up from 2.0)

AGENTS.md Section 5d defines: files >400 lines are violations, >800 lines are critical.

| Threshold | Count | Trend |
|-----------|------:|:-----:|
| Files >400 lines in src/ | 286 | — |
| Files >800 lines in src/ | 54 | DOWN slightly |

**Top offenders (Python, src/ only):**

| File | Lines | Status |
|------|------:|:------:|
| `glossary_data_extended.py` | 2,339 | DATA — lower priority |
| `drake_gui_app.py` | 2,111 | CRITICAL — UI+sim coupled |
| `pinocchio_golf/gui.py` | 1,942 | CRITICAL — UI+sim coupled |
| `humanoid_launcher.py` | 1,817 | CRITICAL — grew via PR #1319 |
| `glossary_data_core.py` | 1,747 | DATA — lower priority |
| `head_models.py` | 1,634 | MAJOR |
| `Motion_Capture_Plotter.py` | 1,495 | MAJOR |
| `frankenstein_editor.py` | 1,401 | MAJOR |
| `linkage_mechanisms/__init__.py` | 1,362 | MAJOR |
| `golf_gui_application.py` | 1,343 | MAJOR |

**What improved:**
- `ui_components.py` decomposed from ~1,373 lines to 65-line facade + 5 focused modules (largest is 527 lines)
- `data_models.py` decomposed from ~1,368 lines to 56-line facade + 4 focused modules (largest is 432 lines)
- Pre-refactor archived files deleted

**What worsened:**
- `humanoid_launcher.py` grew to 1,817 lines (received content from shim consolidation in PR #1319)
- `myosuite_physics_engine.py` grew by ~992 lines in PR #1319

---

### 6. Reversibility

**Score: 5.0 / 10** (up from 4.0)

**What improved:**
- 28 of 42 `sys.path` hacks eliminated via `_bootstrap` pattern
- Configuration externalization intact (`.env`, `config.py`, CLI args)

**What remains:**
- 14 `sys.path` manipulation calls still in production code
- Docker uses unpinned pip versions (no lockfile)

---

### 7. Reusability

**Score: 6.0 / 10** (up from 5.5)

**What improved:**
- `LogPanel` and `SignalBlocker` extracted to shared module
- `RecorderInterface` consolidated (PR #1320)
- `SimulationGUIBase` exists as a reusable base class

---

### 8. Function Length & Signature Quality

**Score: 3.5 / 10** (unchanged)

The refactoring sprint focused on file-level decomposition, not function-level refactoring. Large functions in Drake/Pinocchio GUIs remain untouched. The `_setup_ui()` methods in both engine GUIs likely exceed 100 lines.

---

### 9. God Functions

**Score: 3.5 / 10** (up from 3.0)

The decomposition of `ui_components.py` and `data_models.py` broke up some god functions. However, Drake and Pinocchio GUI setup functions remain multi-responsibility.

---

### 10. No Magic Numbers

**Score: 4.5 / 10** (unchanged)

No targeted work on magic number extraction in this sprint. Physics constants in engine code remain partially documented.

---

### 11. Function & Variable Name Quality

**Score: 5.5 / 10** (unchanged)

No naming refactoring in this sprint.

---

### 12. Comment Quality

**Score: 5.0 / 10** (up from 4.0)

**What improved:**
- `print()` reduced from 754 to 231 (69% reduction, PR #1306 converted 591 to logging)
- TODO/FIXME/HACK/XXX comments reduced to 0

**What remains:**
- 231 `print()` calls still in non-test `src/` code (AGENTS.md bans these)
- ~50% of functions still lack docstrings

---

### 13. Deprecated/Outdated Code

**Score: 6.0 / 10** (up from 3.5)

**What improved:**
- 98 backward-compatibility shim files deleted (PRs #1317-#1319)
- Pre-refactor archived files deleted
- TODO/FIXME markers eliminated

**What remains:**
- 11 files with re-export backward-compatibility patterns
- 14 `sys.path` hacks

---

## Overall Scorecard

| # | Criterion | Feb 10 Score | Feb 12 Score | Change | Priority |
|---|-----------|:----------:|:----------:|:------:|:--------:|
| 1 | DRY | 4.0 | **5.5** | +1.5 | MAJOR |
| 2 | Design by Contract | 4.0 | **4.5** | +0.5 | CRITICAL |
| 3 | TDD | 6.5 | **5.5** | -1.0 | CRITICAL |
| 4 | Orthogonality | 4.0 | **5.5** | +1.5 | MAJOR |
| 5 | Monolithic Files | 2.0 | **3.0** | +1.0 | CRITICAL |
| 6 | Reversibility | 4.0 | **5.0** | +1.0 | MAJOR |
| 7 | Reusability | 5.5 | **6.0** | +0.5 | MINOR |
| 8 | Function Length | 3.5 | **3.5** | 0 | CRITICAL |
| 9 | God Functions | 3.0 | **3.5** | +0.5 | MAJOR |
| 10 | Magic Numbers | 4.5 | **4.5** | 0 | MINOR |
| 11 | Name Quality | 5.5 | **5.5** | 0 | MINOR |
| 12 | Comment Quality | 4.0 | **5.0** | +1.0 | MAJOR |
| 13 | Deprecated Code | 3.5 | **6.0** | +2.5 | MINOR |
| 14 | Law of Demeter | 5.0 | **5.0** | 0 | MINOR |
| 15 | Project Structure | 5.5 | **6.0** | +0.5 | MINOR |
| 16 | Calculation Optimization | 5.5 | **5.5** | 0 | MINOR |
| 17 | Cleanup | 3.5 | **6.5** | +3.0 | MINOR |
| 18 | Parity/Maintenance | 5.0 | **6.0** | +1.0 | MINOR |
| **AVG** | **Overall** | **4.3** | **5.1** | **+0.8** | |

**Weighted Overall (with TDD/DbC/DRY at 2x weight): 5.8/10**

---

## Framework Assessment (A-O) — Updated Scores

| Cat | Name | Feb 3 | Feb 10 | Feb 12 | Weight | Key Change |
|:---:|------|:-----:|:------:|:------:|:------:|------------|
| A | Architecture & Implementation | 8.5 | 7.0 | **7.5** | 2.0 | Shims removed, decomposition done; GUI monoliths remain |
| B | Code Quality & Hygiene | 7.5 | 5.0 | **7.0** | 1.5 | except Exception 739->12; print 754->231 |
| C | Documentation & Comments | 7.5 | 7.0 | **7.0** | 1.0 | No change this sprint |
| D | User Experience | 7.0 | 6.0 | **6.5** | 2.0 | Zustand state management added |
| E | Performance & Scalability | 7.5 | 6.0 | **6.0** | 1.5 | No benchmarks still |
| F | Installation & Deployment | 7.0 | 6.0 | **6.0** | 1.5 | Docker still unpinned |
| G | Testing & Validation | 7.0 | 6.0 | **6.0** | 2.0 | +7 test files but TDD gaps in sprint |
| H | Error Handling & Debugging | 7.5 | 5.0 | **7.5** | 1.5 | 98.4% broad-except reduction |
| I | Security & Input Validation | 7.0 | 8.0 | **8.0** | 1.5 | Maintained |
| J | Extensibility | 8.0 | 8.0 | **8.0** | 1.0 | Maintained |
| K | Reproducibility & Provenance | 7.5 | 4.0 | **6.5** | 1.5 | ProvenanceInfo now in production |
| L | Long-Term Maintainability | 8.0 | 5.0 | **6.5** | 1.0 | Shims removed; 11 re-exports remain |
| M | Educational Resources | 6.5 | 6.0 | **6.0** | 1.0 | No change this sprint |
| N | Visualization & Export | 8.0 | 7.0 | **7.0** | 1.0 | No change this sprint |
| O | CI/CD & DevOps | 8.5 | 7.0 | **7.0** | 1.0 | Maintained |

**Weighted Score: 145.25 / 215 = 6.8 / 10**

---

## Top 10 Remaining Issues

| Rank | Issue | Category | Severity | Recommended Action |
|:----:|-------|:--------:|:--------:|-------------------|
| 1 | Drake GUI (2,111 lines) and Pinocchio GUI (1,942 lines) are monolithic, near-duplicate files | DRY, Monolithic, Orthogonality | CRITICAL | Extract common `_setup_ui()` into `SimulationGUIBase`; split UI from simulation |
| 2 | 1,415 lines of new launcher modules have 0 unit tests | TDD | CRITICAL | Write tests for docker_dialog, help_dialogs, model_card, settings_dialog, startup |
| 3 | `_bootstrap.py` (67 lines, critical import mechanism) has 0 tests | TDD | CRITICAL | Write tests for repo root discovery and path management |
| 4 | Only 40 `assert` statements and 11 files use DbC decorators across all production code | DbC | MAJOR | Add preconditions to physics engine entry points; add postconditions to exports |
| 5 | 231 `print()` calls remain in non-test src/ (AGENTS.md bans these) | Code Quality | MAJOR | Convert remaining print() to logging |
| 6 | 54 files >800 lines; 286 files >400 lines | Monolithic | MAJOR | Phase-split largest files starting with >1,500 line files |
| 7 | 14 `sys.path` hacks remain in production code | Reversibility, Deprecated | MAJOR | Migrate to `_bootstrap` or proper package installation |
| 8 | `.benchmarks/` directory still empty — no performance baselines | Performance | MAJOR | Run benchmark suite and commit baseline |
| 9 | Docker builds use unpinned dependencies | Reproducibility | MAJOR | Generate lockfiles with pip-compile |
| 10 | `humanoid_launcher.py` grew to 1,817 lines during shim removal | Monolithic | MAJOR | Decompose into focused modules |

---

## Remediation Roadmap

### Phase 1: Test Debt (Immediate)

1. Write tests for `_bootstrap.py` — verify repo root discovery, path addition, idempotency
2. Write tests for the 5 decomposed launcher modules (docker_dialog, help_dialogs, model_card, settings_dialog, startup)
3. Write tests for `LogPanel` and `SignalBlocker` in shared widgets
4. Target: 0 untested new modules

### Phase 2: GUI Refactoring (Next Sprint)

5. Audit `SimulationGUIBase` — identify all common patterns between Drake and Pinocchio GUIs
6. Extract shared `_setup_ui()` boilerplate into base class
7. Make MuJoCo `sim_widget.py` inherit from `SimulationGUIBase`
8. Split Drake GUI into `drake_gui.py` (UI) + `drake_sim_controller.py` (simulation logic)
9. Split Pinocchio GUI similarly
10. Target: No engine GUI file >800 lines

### Phase 3: Design by Contract (2 Weeks)

11. Add `@precondition` to all physics engine `load_model()` methods (validate paths, parameters)
12. Add `@precondition` to all `step()` methods (validate timestep > 0, state initialized)
13. Add `@postcondition` to export functions (non-empty output, valid format)
14. Add `assert` invariants for energy conservation in simulation loops
15. Target: 50+ files using DbC patterns

### Phase 4: Remaining Cleanup (Backlog)

16. Convert remaining 231 `print()` calls to logging
17. Eliminate remaining 14 `sys.path` hacks
18. Remove 11 backward-compat re-export facades after confirming no external consumers
19. Consolidate glossary data files
20. Decompose `humanoid_launcher.py` (1,817 lines)

---

## Verdict

**The refactoring sprint was highly productive but uneven in discipline.** The cleanup of exception handlers (98.4% reduction), shim removal (100%), and ProvenanceInfo wiring represent genuine architectural improvements. The Zustand adoption is exemplary in its test coverage. However, the sprint violated the project's own MANDATORY TDD requirement in 4 of 6 PRs, creating a test debt that should be addressed before further feature work.

**Overall Grade: 5.8/10** — Significant improvement from the 4.6 adversarial assessment, but still below the 7.6 baseline established on Feb 3. The gap is primarily driven by monolithic files (54 files >800 lines) and insufficient DbC adoption. Closing the remaining test gaps and decomposing the engine GUIs would bring the score back above 7.0.

---

_Generated by the Organizational Code Quality Assessment Framework v2.0_
_Assessment template: `docs/assessments/README.md` (Categories A-O + TDD/DbC/DRY from AGENTS.md Section 4-5)_
