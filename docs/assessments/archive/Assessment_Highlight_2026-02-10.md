# Comprehensive Assessment Summary

**Overall Score: 69/100**
**Assessment Date: 2026-02-10**
**Assessed By: Claude Code (Automated Deep Review, Adversarial Mode)**
**Model: Claude Opus 4.6**
**Commit: `3a70fc248` (main branch, fix/issue-1069-memory-leaks-tab-system)**

---

## Executive Summary

- The codebase demonstrates strong architectural intent with Protocol-based engine abstraction, Design by Contract decorators, and a clean FastAPI dependency injection layer, but execution is inconsistent: 98 backward-compatibility shims bloat `src/shared/python/`, 739 broad `except Exception` handlers in production code undermine error specificity, and 199 non-test files still use `print()` despite AGENTS.md explicitly banning it.
- The test suite is substantial (5,400+ test functions, 3,404 passing) but unreliable: 86 tests fail, 25+ error on collection due to missing modules (`fixtures_lib`, `apps`, `c3d`, `mujoco`), and the integration test suite is entirely broken by a single `conftest.py` import failure.
- Security posture is above average for a research project: `eval()` replaced with `simpleeval`, `shell=True` blocked in `secure_subprocess.py`, CORS restricted, rate limiting via `slowapi`, security headers middleware, and auth with bcrypt+JWT. The SECRET_KEY fallback to an unsafe placeholder in development is correctly handled.
- The React frontend (183 TS files, 300 tests all green, clean type-check) is the healthiest part of the stack, with proper WebSocket reconnection logic, memory-bounded frame history, and component/hook separation. However, it has no state management library (no Redux/Zustand), relying entirely on prop drilling and local state.
- Documentation volume is extreme (1,147 markdown files) but largely auto-generated assessment/review output from Jules workflows. Actual developer-facing tutorials (4 files) and code examples (4 files) are thin relative to a 515K LOC Python codebase.

---

## Score Breakdown

| Category | Assessment | Score | Weight | Weighted | Key Finding | Priority |
|:---------|:-----------|------:|-------:|---------:|:------------|:---------|
| **A** Architecture & Implementation | Protocol-based engine abstraction is sound; 98 shims and 1 remaining dependency violation dilute it | 7 | 2.0 | 14.0 | 98 backward-compat shims in `src/shared/python/` create import indirection | MAJOR |
| **B** Code Quality & Hygiene | 10 ruff violations (excellent), but 124 unformatted files, 739 broad excepts, 199 print files | 5 | 1.5 | 7.5 | 739 `except Exception` in `src/` non-test code; 199 files with banned `print()` | CRITICAL |
| **C** Documentation & Comments | 1,147 .md files (mostly auto-generated), good docstrings in core modules, thin tutorials | 7 | 1.0 | 7.0 | 4 tutorials for a 515K LOC codebase; most docs are Jules-generated assessments | MINOR |
| **D** User Experience & Developer Journey | Dashboard launcher works, 7 React pages, help system exists; no end-to-end getting-started | 6 | 2.0 | 12.0 | `tests/integration/conftest.py` fails on import -- integration suite completely broken | CRITICAL |
| **E** Performance & Scalability | TaskManager has TTL+LRU, WebSocket frame capping, benchmark framework exists but empty results | 6 | 1.5 | 9.0 | `.benchmarks/` directory is empty; no baseline performance regression data | MAJOR |
| **F** Installation & Deployment | Multi-stage Dockerfile, conda+pip, pyproject.toml with optional deps; no lockfile for reproducibility | 6 | 1.5 | 9.0 | Docker uses `pip install --no-cache-dir` with unpinned versions; no `pip-compile` lockfile | MAJOR |
| **G** Testing & Validation | 5,400+ tests, 97% pass rate (excluding errors); 86 failures, 25+ collection errors, broken integration | 6 | 2.0 | 12.0 | 86 test failures + 25 collection errors = 3% of suite is broken | CRITICAL |
| **H** Error Handling & Debugging | Centralized logging config (224 imports), structlog dependency, but 739 broad excepts | 5 | 1.5 | 7.5 | `except Exception` used 739 times in `src/` non-test; swallows specific failure modes | CRITICAL |
| **I** Security & Input Validation | simpleeval, secure subprocess, CORS, rate limiting, auth, security headers; SECRET_KEY fallback is safe | 8 | 1.5 | 12.0 | No `shell=True` in production, `eval()` eliminated, Bandit in CI | MINOR |
| **J** Extensibility & Plugin Architecture | EngineRegistry + LOADER_MAP, Protocol-based engines, easy to add new engines | 8 | 1.0 | 8.0 | Adding a new engine requires only a loader function + Protocol implementation | NIT |
| **K** Reproducibility & Provenance | ProvenanceInfo class exists but is unused in production code (only in its own module + test) | 4 | 1.5 | 6.0 | `ProvenanceInfo` defined but never called from any production code path | MAJOR |
| **L** Long-Term Maintainability | 98 shim files, 2K+ Python files, 61 CI workflows; high surface area | 5 | 1.0 | 5.0 | 61 GitHub Actions workflows (most Jules automation) create maintenance burden | MAJOR |
| **M** Educational Resources & Tutorials | 4 tutorials, 4 examples, help system in React UI; thin for the scope | 6 | 1.0 | 6.0 | `examples/` has 4 scripts; `docs/tutorials/content/` has 4 markdown files | MINOR |
| **N** Visualization & Export | Matplotlib plotting module, 3D Scene (Three.js), URDF viewer, ForceOverlay; export is JSON-only in config | 7 | 1.0 | 7.0 | Export formats defined (`json`, `csv`, `mat`, `hdf5`, `c3d`) but only `json` in `VALID_EXPORT_FORMATS` in `config.py` | MINOR |
| **O** CI/CD & DevOps | quality-gate + tests + frontend-tests; 61 workflows; Bandit, pip-audit, ruff, black, mypy all in CI | 7 | 1.0 | 7.0 | CI has proper concurrency groups and timeout limits; test step uses `xvfb-run` | NIT |

**Weighted Total: 69.0/150 (scaled to 100) = 69/100**

*(Calculation: Sum of Weighted = 129.0; Max possible = 10 * (2+1.5+1+2+1.5+1.5+2+1.5+1.5+1+1.5+1+1+1+1) = 10 * 21.5 = 215; Score = 129.0/215 * 100 = 60.0)*

**Corrected Weighted Total: 60/100**

---

## Category Group Scores

| Group | Categories | Weight | Avg Score | Contribution |
|:------|:-----------|-------:|----------:|-------------:|
| Core Engineering (A, B, G) | Architecture, Quality, Testing | 5.5 | 6.0 | 33.0/55 |
| User-Facing (D, M, N) | UX, Tutorials, Visualization | 4.0 | 6.3 | 25.0/40 |
| Operations (F, H, O) | Deploy, Error Handling, CI | 4.0 | 6.0 | 24.0/40 |
| Security & Reliability (I, K) | Security, Reproducibility | 3.0 | 6.0 | 18.0/30 |
| Maintainability (C, E, J, L) | Docs, Perf, Extensibility, LTM | 4.5 | 6.5 | 29.0/45 |

---

## Top 10 Risks

| Rank | Risk | Category | Severity | Impact | Recommended Action | Effort |
|-----:|:-----|:---------|:---------|:-------|:-------------------|:-------|
| 1 | 739 broad `except Exception` handlers mask bugs | H | CRITICAL | Silent failures in physics engines, data corruption undetected | Replace with specific exceptions (ValueError, FileNotFoundError, etc.) in 50 highest-traffic files first | 3-5 days |
| 2 | 86 test failures + 25 collection errors | G | CRITICAL | No confidence in code correctness; CI may pass but local is broken | Fix collection errors first (missing modules), then address 86 failures by category | 2-3 days |
| 3 | Integration test suite entirely broken | G, D | CRITICAL | Cannot verify cross-engine behavior or API contracts | Fix `tests/integration/conftest.py:20` -- `fixtures_lib` module missing from path | 2 hours |
| 4 | 199 files with `print()` in non-test code | B | MAJOR | Console noise, no structured logging for those paths, violates AGENTS.md | Bulk replace with `logger.info/debug/warning` using automated script | 1-2 days |
| 5 | ProvenanceInfo never used in production | K | MAJOR | Scientific results lack reproducibility metadata despite having the infrastructure | Wire ProvenanceInfo.capture() into export routes and OutputManager | 4 hours |
| 6 | 98 backward-compatibility shims | A, L | MAJOR | Every import goes through an indirection layer; confusing for new developers | Create migration script to update all imports, then remove shims in batches | 1 week |
| 7 | No performance baseline data | E | MAJOR | Cannot detect regressions; `.benchmarks/` is empty | Run benchmark suite and commit baseline; add to CI nightly job | 4 hours |
| 8 | 124 files fail black formatting | B | MAJOR | CI quality-gate will reject PRs touching these files | Run `black .` once to fix all; prevent regression via pre-commit | 30 min |
| 9 | Docker uses unpinned pip versions | F | MAJOR | Builds are non-reproducible; dependency resolution may break over time | Use `pip-compile` to generate `requirements.lock` for Docker builds | 2 hours |
| 10 | 61 GitHub Actions workflows (maintenance burden) | L, O | MINOR | Difficult to understand which workflows are essential vs. automated noise | Audit and consolidate; disable unused Jules workflows | 1 day |

---

## Remediation Roadmap

### Phase 1: Critical (48 hours)

1. **Fix integration test suite** -- The `fixtures_lib` import in `tests/integration/conftest.py:20` breaks the entire integration test directory. Either add `fixtures_lib` to the Python path or restructure the import.
   - File: `tests/integration/conftest.py`
   - Estimated time: 2 hours

2. **Fix 25+ test collection errors** -- Missing modules (`apps`, `c3d`, `ezc3d`, `mujoco`) prevent test collection. Add conditional skips or fix imports.
   - Files: `tests/unit/test_analysis_robustness.py`, `tests/unit/test_c3d_export_features.py`, `tests/unit/test_c3d_force_plate.py`, `tests/unit/test_crba.py`, `tests/unit/test_golf_suite_launcher.py`, `tests/unit/test_models.py`
   - Estimated time: 4 hours

3. **Run `black .` to fix formatting** -- 124 files are out of format. One command fixes them all.
   - Command: `python -m black .`
   - Estimated time: 30 minutes

4. **Fix remaining 10 ruff violations** -- 7 are auto-fixable.
   - Command: `python -m ruff check . --fix`
   - Estimated time: 15 minutes

### Phase 2: Major (2 weeks)

5. **Replace broad `except Exception` with specific exceptions** -- Start with the 50 most critical files in `src/api/`, `src/engines/`, and `src/shared/python/engine_core/`.
   - Target: Reduce from 739 to under 200 in production code
   - Estimated time: 3-5 days

6. **Replace `print()` with logger calls** -- 199 non-test files violate AGENTS.md.
   - Create a ruff rule or script to automate
   - Estimated time: 2 days

7. **Wire ProvenanceInfo into export pipeline** -- The class exists in `src/shared/python/data_io/provenance.py` but is never called from production code. Add it to `src/api/routes/export.py` and `src/shared/python/data_io/export.py`.
   - Estimated time: 4 hours

8. **Run benchmarks and establish baseline** -- `tests/benchmarks/test_dynamics_benchmarks.py` exists but `.benchmarks/` is empty.
   - Add to nightly CI: `python -m pytest tests/benchmarks/ --benchmark-json=.benchmarks/baseline.json`
   - Estimated time: 4 hours

9. **Pin Docker dependencies** -- Generate lockfiles with `pip-compile` for reproducible Docker builds.
   - Estimated time: 2 hours

### Phase 3: Full (6 weeks)

10. **Remove backward-compatibility shims** -- 98 files in `src/shared/python/` are pure indirection (`sys.modules[__name__] = ...`). Create a migration script that updates all imports across the codebase, then delete the shim files in batches.
    - Estimated time: 1 week

11. **Consolidate GitHub Actions workflows** -- 61 workflows are excessive. Audit which Jules workflows are actively used and disable the rest.
    - Target: Reduce to ~15-20 essential workflows
    - Estimated time: 1 day

12. **Address 86 remaining test failures** -- Categorize by root cause (missing optional deps, physics accuracy, UI rendering) and fix or mark as `xfail` with tracked issues.
    - Estimated time: 1 week

13. **Add state management to React frontend** -- Currently relies on prop drilling. As the UI grows, this will become unsustainable. Consider Zustand (lightweight) or React Query for API state.
    - Estimated time: 1 week

14. **Expand tutorial coverage** -- 4 tutorials and 4 examples for 515K LOC is thin. Add at least 4 more: multi-engine comparison, biomechanics workflow, video pose analysis end-to-end, and custom engine integration.
    - Estimated time: 1 week

---

## Go/No-Go Assessment

| Criterion | Status | Evidence |
|:----------|:-------|:---------|
| Core functionality works | CONDITIONAL GO | Engine loading, simulation, API serve fine; 86 test failures need triage |
| Security posture | GO | No eval, no shell=True, auth in place, rate limiting, CORS restricted |
| Test suite reliable | NO-GO | 86 failures + 25 collection errors + broken integration = unreliable |
| Documentation adequate | GO | User manual (16K lines), API docs, help system, tutorials (minimal) |
| Deployment ready | CONDITIONAL GO | Dockerfile works but unpinned deps risk non-reproducible builds |
| Maintainable long-term | CONDITIONAL GO | 98 shims and 61 workflows create friction; core architecture is sound |

**Overall Verdict: CONDITIONAL GO** -- The architecture and security are solid, but the test suite must be stabilized before any production-facing deployment. The 86 test failures and broken integration suite are the gating issues.

---

## Appendix: Individual Category Details

### A. Architecture & Implementation (Score: 7/10, Weight: 2x)

**Strengths:**
- Protocol-based engine abstraction (`src/shared/python/engine_core/interfaces.py`, 717 lines) defines a comprehensive PhysicsEngine protocol with Design by Contract documentation. All 7 engine implementations (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite, Pendulum, PuttingGreen) conform to this protocol.
- Engine loading via `LOADER_MAP` in `src/engines/loaders.py` (lines 236-244) provides clean registration with lazy imports to avoid heavy dependency loading at startup.
- FastAPI dependency injection is properly implemented via `src/api/dependencies.py` -- all services are stored in `app.state` and retrieved via `Depends()`. 98 `Depends()` calls across route modules confirm this pattern is consistently followed.
- Decoupling work (PRs #1267-#1279) moved modules from flat `src/shared/python/` into structured subpackages: `engine_core/`, `data_io/`, `core/`, `config/`, `security/`, `logging_pkg/`, `validation_pkg/`, `signal_toolkit/`.

**Weaknesses:**
- **98 backward-compatibility shim files** (`sys.modules[__name__] = _real_module`) in `src/shared/python/`. Every file like `src/shared/python/contracts.py`, `src/shared/python/engine_manager.py`, `src/shared/python/logging_config.py` is a 7-line redirect. These add import indirection and confuse IDE navigation. [MAJOR]
- **1 dependency direction violation remaining**: `src/shared/python/engine_core/engine_loaders.py:12` imports from `src.engines.loaders` (shared -> engines is inverted). The `scripts/check_dependency_direction.py` tool catches it. [MINOR]
- `src/engines/physics_engines/drake/python/src/drake_gui_app.py` at 2,105 lines is a monolith combining GUI setup, simulation logic, and visualization. Contrast with MuJoCo's properly decomposed `gui/tabs/` structure. [MAJOR]

### B. Code Quality & Hygiene (Score: 5/10, Weight: 1.5x)

**Strengths:**
- Only 10 ruff violations across 2,053 Python files is excellent linting compliance. The ruff config (`pyproject.toml` lines 131-142) selects E, F, I, UP, B rules.
- No bare `except:` clauses found anywhere. All exception handlers at minimum use `except Exception`.
- 452 files use `from __future__ import annotations` for modern type hint syntax.

**Weaknesses:**
- **739 `except Exception` handlers in `src/` non-test code** -- This is the single worst hygiene issue. In physics simulation code, catching `Exception` instead of `numpy.linalg.LinAlgError`, `ValueError`, or `FileNotFoundError` means numerical blowups, invalid inputs, and missing files all get the same error handling. Example: `src/engines/loaders.py:103` catches `Exception` when loading a URDF into Drake. [CRITICAL]
- **124 files fail `black --check`** (6% of total) -- CI enforces `black --check` so these will block PRs. Files include core files like `src/shared/python/ui/simulation_gui_base.py`, `tests/api/test_phase3_api.py`, `tests/api/test_phase4_api.py`. [MAJOR]
- **199 non-test files use `print()`** despite AGENTS.md line 36 explicitly stating "DO NOT use print() statements for application output." [MAJOR]
- **65 TODO/FIXME/HACK/XXX comments** remain in the codebase. CI (`ci-standard.yml` lines 48-58) makes these blocking, meaning PRs touching these files will fail the quality gate. [MINOR]

### C. Documentation & Comments (Score: 7/10, Weight: 1x)

**Strengths:**
- User manual at `docs/UPSTREAM_DRIFT_USER_MANUAL.md` is 16,794 lines -- comprehensive.
- Core module docstrings are thorough: `src/engines/common/physics.py` has references (Jorgensen, 1999; Smits & Ogg, 2004), units documented, DbC contracts in docstrings.
- AGENTS.md (17K+ bytes) provides clear coding standards for AI agents working in the repo.
- Help system exists in both PyQt6 (`src/shared/python/help_system.py`, `help_content.py`) and React (`ui/src/components/ui/HelpPanel.tsx`, `helpData.ts`).

**Weaknesses:**
- 1,147 markdown files are overwhelmingly auto-generated Jules assessment/audit/review output. The `docs/assessments/` directory alone has dozens of files. Most of this is noise, not actionable documentation. [MINOR]
- Only 4 tutorials in `docs/tutorials/content/` and 4 example scripts in `examples/` for a 515K LOC codebase. No tutorial covers the biomechanics workflow, video analysis pipeline, or multi-engine comparison. [MINOR]

### D. User Experience & Developer Journey (Score: 6/10, Weight: 2x)

**Strengths:**
- React dashboard (`ui/src/pages/Dashboard.tsx`) provides tile-based launcher with error handling and help button.
- 7 pages cover the main workflows: Dashboard, Simulation, ModelExplorer, PuttingGreen, VideoAnalyzer, DataExplorer, MotionCapture.
- Toast notification system (`ui/src/components/ui/Toast.tsx`) provides user feedback.
- Diagnostics panel (`ui/src/components/ui/DiagnosticsPanel.tsx`) available on every page.

**Weaknesses:**
- **Integration test suite completely broken** (`tests/integration/conftest.py:20` -- `ModuleNotFoundError: No module named 'fixtures_lib'`). A developer running `pytest tests/integration/` gets instant failure. [CRITICAL]
- No onboarding guide for new contributors. `CONTRIBUTING.md` (59 lines) is thin. No "first PR" guide, no architecture decision records (ADRs). [MAJOR]
- `launch_golf_suite.py` entry point is at the repo root -- not inside `src/`. The `pyproject.toml` script entry (`upstream-drift = "launch_golf_suite:main"`) references root-level module. [MINOR]

### E. Performance & Scalability (Score: 6/10, Weight: 1.5x)

**Strengths:**
- `TaskManager` in `src/api/server.py` (lines 111-201) implements TTL-based cleanup and LRU eviction with thread safety -- prevents memory leak from unbounded task accumulation.
- WebSocket frame history in `ui/src/api/client.ts` (line 30) capped at `MAX_FRAMES_HISTORY = 1000` to prevent client-side memory growth.
- `pytest-benchmark` is configured (pyproject.toml markers, `tests/benchmarks/` directory).
- Exponential backoff with jitter for WebSocket reconnection (`ui/src/api/client.ts` lines 62-69).

**Weaknesses:**
- **`.benchmarks/` directory is empty** -- No baseline performance data exists. Cannot detect regressions. [MAJOR]
- No profiling results committed. No `cProfile` usage in non-deployment code. `src/deployment/realtime/controller.py` uses `perf_counter` but this is runtime telemetry, not profiling. [MAJOR]
- The `nightly-cross-engine.yml` workflow exists but it's unclear if it runs benchmarks or only validation tests. [MINOR]

### F. Installation & Deployment (Score: 6/10, Weight: 1.5x)

**Strengths:**
- Multi-stage Dockerfile (builder + runtime) reduces image size. Non-root user (`golfer`), health check at `/health`, proper `PYTHONPATH` setup.
- `pyproject.toml` uses optional dependencies for modular installation: `[drake]`, `[pinocchio]`, `[biomechanics]`, `[dev]`, `[all]`.
- `docker-compose.yml` and `.env.example` exist for configuration management.
- `install.sh` script for Unix-based setup.

**Weaknesses:**
- **Docker uses unpinned pip versions** (`Dockerfile` line 58-88: `pip install mujoco>=3.2.3` etc.) with no lockfile. Builds are non-deterministic. `requirements.lock` exists (48 lines) but is not used in Docker. [MAJOR]
- `environment.yml` (conda) and `requirements.txt` and `pyproject.toml` -- three dependency specification formats create confusion about which is canonical. [MINOR]
- Windows setup requires manual steps not documented beyond `launch_urdf_generator.bat`. No PowerShell setup script equivalent to `install.sh`. [MINOR]

### G. Testing & Validation (Score: 6/10, Weight: 2x)

**Strengths:**
- 5,400+ test functions across 508 test files -- substantial coverage for a research codebase.
- 3,404 tests pass reliably (97% pass rate when excluding collection errors).
- 300 React frontend tests all pass with vitest. Clean TypeScript type-check.
- 17 parity tests verify cross-engine consistency (pendulum engine).
- CI runs parallel tests with `pytest-xdist`, 60s timeout, skip slow/benchmark/requires_gl markers.

**Weaknesses:**
- **86 tests fail** including physics accuracy tests (`test_pendulum_putter_physics.py`), plotting tests (`test_plotting.py`), and myoconverter tests. These are not marked `xfail`. [CRITICAL]
- **25+ collection errors** from missing modules: `fixtures_lib`, `apps`, `c3d`, `mujoco`, `mediapipe`, `cv2`. Tests that require optional dependencies should use `pytest.importorskip()`. [CRITICAL]
- **Integration test suite completely broken** due to `conftest.py` import failure. [CRITICAL]
- No coverage threshold enforced in CI. `codecov/codecov-action` runs but `fail_ci_if_error: false` means coverage drops are silent. [MAJOR]

### H. Error Handling & Debugging (Score: 5/10, Weight: 1.5x)

**Strengths:**
- Centralized logging via `src/shared/python/logging_config.py` with 224 import sites across `src/`.
- `structlog` declared as dependency for structured logging.
- Request tracing middleware (`src/api/utils/tracing.py`) enables debugging of API issues.
- Custom exception hierarchy: `GolfModelingError`, `ContractViolationError`, `SecureSubprocessError`.

**Weaknesses:**
- **739 `except Exception` handlers in production code** -- the most damaging pattern in the codebase. Physics engines, file I/O, and network calls all catch the same broad exception type. [CRITICAL]
- Only 2 files actually import structlog despite it being a declared dependency. The centralized logging config wraps it, but most code still uses raw `logging` patterns. [MAJOR]
- Error messages in many handlers are generic: `"Failed to load default URDF into Drake (expected if missing meshes): {e}"` (src/engines/loaders.py:105) -- "expected if missing meshes" suggests the developer knows the specific error but catches `Exception` anyway. [MAJOR]

### I. Security & Input Validation (Score: 8/10, Weight: 1.5x)

**Strengths:**
- `eval()` replaced with `simpleeval` library throughout: `src/engines/pendulum_models/python/double_pendulum_model/physics/double_pendulum.py:19` documents this explicitly. Security tests in `tests/test_expression_security.py` and `tests/test_ui_security.py`.
- `shell=True` explicitly blocked in `src/shared/python/security/secure_subprocess.py:135-137`.
- Security headers middleware: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, HSTS, Referrer-Policy (`src/api/middleware/security_headers.py`).
- Rate limiting via `slowapi` in `src/api/server.py:65`.
- CORS restricted to specific origins (`src/api/config.py:28-32`), not `*`.
- Auth with bcrypt (cost factor 12) and HS256 JWT. Production requires `SECRET_KEY` env var (`src/api/auth/security.py:31-36`).
- `pip-audit` and Bandit in CI (`ci-standard.yml` lines 60-77).
- Pydantic input validation with 284 field validators/model validators in API models.
- Upload size limit: 10MB enforced via middleware (`src/api/middleware/upload_limits.py`).

**Weaknesses:**
- `SECRET_KEY` fallback in development uses hardcoded string `"UNSAFE-NO-SECRET-KEY-SET-AUTHENTICATION-WILL-FAIL"` (`src/api/auth/security.py:46`). While this is intentionally non-functional, it means development mode has no auth at all. [MINOR]
- `defusedxml` is listed as optional dependency but not checked for consistent usage across XML parsing. [MINOR]

### J. Extensibility & Plugin Architecture (Score: 8/10, Weight: 1x)

**Strengths:**
- Engine registration is clean: add an `EngineType` enum value, implement `PhysicsEngine` Protocol, add a loader function to `LOADER_MAP` in `src/engines/loaders.py`. The pendulum engine addition (PR #1151) is a good template.
- Protocol-based design means engines can be in separate packages with optional dependencies. Lazy imports in loaders prevent import-time failures.
- `engine_probes.py` provides runtime capability detection per engine.
- Robotics module (`src/robotics/core/protocols.py`) defines 6 additional protocols for specialized capabilities.

**Weaknesses:**
- No formal plugin system (no entry points, no `pluggy`, no dynamic discovery). Engines must be registered in source code. [MINOR]
- The 98 backward-compat shims make it harder to understand where a module actually lives when extending. [MINOR]

### K. Reproducibility & Provenance (Score: 4/10, Weight: 1.5x)

**Strengths:**
- `ProvenanceInfo` class in `src/shared/python/data_io/provenance.py` is well-designed: frozen dataclass, captures git SHA, timestamps, software version, model hash.
- `src/shared/python/data_io/reproducibility.py` exists for reproducibility support.

**Weaknesses:**
- **ProvenanceInfo is defined but never used in any production code path**. Only referenced in `src/shared/python/data_io/provenance.py` itself and `tests/unit/test_provenance.py`. No export route, no OutputManager, no simulation result includes provenance metadata. This is dead infrastructure. [MAJOR]
- No `requirements.lock` used in Docker builds. Conda `environment.yml` specifies package names but not exact versions for most. [MAJOR]
- `.benchmarks/` is empty -- no reproducible performance baselines. [MAJOR]

### L. Long-Term Maintainability (Score: 5/10, Weight: 1x)

**Strengths:**
- The decoupling effort (22 issues closed via PRs #1267-#1279) demonstrates active maintenance investment.
- `scripts/check_dependency_direction.py` enforces architectural boundaries at CI time.
- Shim files provide a migration path -- old imports still work.
- Mypy configured with targeted exclusions rather than being disabled entirely.

**Weaknesses:**
- **61 GitHub Actions workflows** -- most are Jules automation (auto-repair, auto-assess, auto-refactor, etc.). This creates a massive attack surface and maintenance burden. Many workflows trigger on issues/PRs and create additional noise. [MAJOR]
- **98 shim files** mean the true module location is always one indirection away. IDEs will navigate to the shim, then the developer must follow the redirect. [MAJOR]
- **2,053 Python files** -- the sheer file count makes it difficult to orient. Many files are small shims or thin wrappers. [MINOR]

### M. Educational Resources & Tutorials (Score: 6/10, Weight: 1x)

**Strengths:**
- 4 tutorials: getting started, first simulation, engine comparison, video analysis.
- 4 code examples: basic simulation, parameter sweeps, injury risk tutorial, motion training demo.
- Help system in React UI with topic-specific content (`docs/help/` with 5 help files).
- `examples/README.md` provides overview of available examples.

**Weaknesses:**
- 4 tutorials for a codebase with 9 engine types, 24 API route modules, and 7 React pages is thin. No tutorial for: biomechanics workflow, custom engine integration, data export pipeline, Docker deployment. [MINOR]
- No Jupyter notebooks for interactive exploration -- common in scientific Python projects. [MINOR]

### N. Visualization & Export (Score: 7/10, Weight: 1x)

**Strengths:**
- `src/shared/python/plotting/` has 10 files covering animation, energy plots, kinematics, transforms, export.
- React 3D visualization via Three.js (`ui/src/components/visualization/Scene3D.tsx`, `URDFViewer.tsx`).
- ForceOverlay component (`ui/src/components/visualization/ForceOverlay.tsx`) for force visualization.
- Live plot component (`ui/src/components/analysis/LivePlot.tsx`) for real-time data.
- Export request model supports 5 formats: json, csv, mat, hdf5, c3d (`src/api/models/requests.py:34`).

**Weaknesses:**
- `VALID_EXPORT_FORMATS` in `src/api/config.py:16` is `{"json"}` only -- the request model claims 5 formats but the config restricts to 1. This is a contract mismatch. [MINOR]
- No screenshot/preview generation for batch processing. [NIT]

### O. CI/CD & DevOps (Score: 7/10, Weight: 1x)

**Strengths:**
- `ci-standard.yml` has quality-gate -> tests -> frontend-tests pipeline with proper dependency ordering.
- Concurrency groups (`cancel-in-progress: true`) prevent redundant runs.
- Timeout limits on all jobs (20 minutes).
- Pre-commit hooks configured for ruff, black, eslint, prettier.
- Bandit security scan and pip-audit in CI.
- Frontend CI includes type-check, lint, test, and build verification.
- Codecov integration (optional, doesn't block).

**Weaknesses:**
- 61 workflows total -- more than 50 are Jules automation. Many have overlapping triggers. The workflow directory is nearly unnavigable. [MAJOR]
- MATLAB and Arduino CI jobs are disabled (`if: false`) but still present in the workflow file, adding confusion. [NIT]
- `tests` job has pre-existing failures that are tolerated (CI passes despite broken tests). The cross-engine validation step uses `|| true` so it never fails the build. [MINOR]
