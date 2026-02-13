# Repository Code Quality & Architecture Assessment

**Repository:** UpstreamDrift (formerly Golf Modeling Suite)
**Assessment Date:** 2026-01-31
**Assessor:** Claude Code (Automated Analysis)

---

## A) Executive Summary

### Top 5 Risks

| #   | Risk                                          | Location                                                                                              | Failure Modes                                                                                 |
| --- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 1   | **Security: Unsafe `eval()` usage**           | `src/shared/python/signal_toolkit/fitting.py`                                                         | Code injection vulnerability if expression input is user-controlled; arbitrary code execution |
| 2   | **Security: Pickle deserialization**          | `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/motion_optimization.py`               | Arbitrary code execution from untrusted pickle files                                          |
| 3   | **God modules exceed maintainability limits** | `src/launchers/golf_launcher.py` (2,539 LOC), `src/shared/python/statistical_analysis.py` (2,221 LOC) | High coupling, difficult testing, merge conflicts, onboarding friction                        |
| 4   | **Incomplete dependency locking**             | `requirements.lock` (13 entries vs 100+ deps), loose constraints in `requirements.txt`                | Non-reproducible builds, supply chain vulnerabilities, silent breakage                        |
| 5   | **Frontend accessibility critical gaps**      | `ui/src/components/**`                                                                                | WCAG violations, legal liability, excluded users, failed accessibility audits                 |

### Top 5 Opportunities

| #   | Opportunity                                         | Payoff                                                                |
| --- | --------------------------------------------------- | --------------------------------------------------------------------- |
| 1   | **Expand mypy coverage** (10+ excluded directories) | Catch type errors early, reduce runtime bugs, improve IDE support     |
| 2   | **Refactor god modules into focused components**    | 50% faster code reviews, easier testing, lower merge conflict rate    |
| 3   | **Complete dependency lockfile**                    | Reproducible builds, faster CI, security audit compliance             |
| 4   | **Add frontend component tests**                    | Prevent UI regressions, enable safe refactoring                       |
| 5   | **Centralize error handling patterns**              | Consistent API responses, easier debugging, better client integration |

### Stop / Start / Keep

| Category  | Practices                                                                                                                                               |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **STOP**  | Using `eval()` on any user-influenced input; adding new modules to mypy exclusion list; ignoring CVEs without documented remediation plan               |
| **START** | Breaking up large modules proactively (>500 LOC trigger); adding ARIA attributes to new UI components; running Docker security scans in CI              |
| **KEEP**  | 90% test coverage threshold; structured error codes (GMS-XXX-NNN); Design by Contract patterns; TaskManager TTL cleanup; comprehensive CI quality gates |

---

## B) Architecture Map

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │  React UI    │  │  PyQt6 GUI   │  │  CLI Tools                   │  │
│  │  (ui/)       │  │  (launchers/)│  │  (src/tools/, scripts/)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Server (src/api/)                                        │  │
│  │  - REST endpoints (/api/engines, /api/simulate, /api/analysis)    │  │
│  │  - WebSocket streaming (/api/ws/simulate)                         │  │
│  │  - Authentication (JWT + API Keys)                                │  │
│  │  - Rate limiting, security headers                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           SERVICE LAYER                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐   │
│  │SimulationService │  │ AnalysisService  │  │  VideoService      │   │
│  └──────────────────┘  └──────────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENGINE ABSTRACTION LAYER                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  EngineManager + UnifiedEngineInterface (src/shared/python/)      │  │
│  │  - Protocol-based engine abstraction                              │  │
│  │  - Cross-engine validation                                        │  │
│  │  - State management                                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│      MuJoCo           │ │        Drake          │ │      Pinocchio        │
│  (Recommended)        │ │   (Model-Based)       │ │   (Fast Dynamics)     │
└───────────────────────┘ └───────────────────────┘ └───────────────────────┘
                    │               │               │
                    ▼               ▼               ▼
┌───────────────────────┐ ┌───────────────────────┐
│      OpenSim          │ │       MyoSuite        │
│   (Biomechanics)      │ │   (Muscle Models)     │
└───────────────────────┘ └───────────────────────┘
```

### Data Flow

1. **Request Path:** React UI → FastAPI → Service → EngineManager → Physics Engine → Response
2. **WebSocket Streaming:** Client ↔ FastAPI WebSocket ↔ Physics Engine (60fps frames)
3. **Background Tasks:** API → BackgroundTasks → TaskManager (TTL=1hr, max=1000)
4. **Analysis Pipeline:** Simulation Result → AnalysisService → Statistical/Biomechanics Analysis

### Dependency Direction Violations

| Violation                                 | Files                                            | Impact                            |
| ----------------------------------------- | ------------------------------------------------ | --------------------------------- |
| UI components import engine-specific code | `golf_launcher.py` imports from multiple engines | Breaks engine abstraction         |
| Shared utilities import from launchers    | Some utils reference launcher-specific code      | Circular dependency risk          |
| API routes contain business logic         | `simulation.py`, `analysis.py` with inline logic | Harder to test, violates layering |

### Coupling Hotspots ("God" Modules)

| File                                                            | LOC    | Responsibilities                                                | Coupling Score |
| --------------------------------------------------------------- | ------ | --------------------------------------------------------------- | -------------- |
| `src/launchers/golf_launcher.py`                                | 2,539  | UI, Docker, environment, model loading, splash screens          | CRITICAL       |
| `src/shared/python/statistical_analysis.py`                     | 2,221  | 7 mixins: Energy, Phase, GRF, Stability, Momentum, Swing, Stats | HIGH           |
| `src/engines/physics_engines/drake/python/src/drake_gui_app.py` | 2,050  | GUI + physics engine management                                 | HIGH           |
| `src/launchers/ui_components.py`                                | 2,600+ | 20+ UI component classes                                        | HIGH           |
| `src/shared/python/checkpoint.py`                               | 17,800 | Checkpoint management (likely auto-generated)                   | MEDIUM         |

---

## C) Code Quality Findings (Ranked)

### CRITICAL

#### C-1: Unsafe `eval()` Usage

- **Severity:** CRITICAL
- **Category:** Security
- **Evidence:** `src/shared/python/signal_toolkit/fitting.py` - `return eval(expression, {"__builtins__": {}}, local_dict)  # noqa: S307`
- **Why it matters:** Code injection if `expression` comes from user input; Bandit security check explicitly bypassed with `noqa`
- **Recommended fix:** Replace with `ast.literal_eval()` for simple expressions or `simpleeval` library (already in dependencies)
- **Migration:** 1) Audit all callers of this function, 2) Replace with safe evaluation, 3) Remove noqa comment
- **Effort:** S | **Risk:** HIGH

#### C-2: Pickle Deserialization Vulnerability

- **Severity:** CRITICAL
- **Category:** Security
- **Evidence:** `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/motion_optimization.py` - `data = np.load(filename, allow_pickle=True)`
- **Why it matters:** Arbitrary code execution from untrusted files; pickle is not safe for untrusted data
- **Recommended fix:** Use `.npz` format without pickle, or validate file source; consider `safetensors` for model weights
- **Migration:** 1) Audit all `allow_pickle=True` usages, 2) Convert data formats, 3) Add file provenance validation
- **Effort:** M | **Risk:** HIGH

### HIGH

#### H-1: God Module - golf_launcher.py

- **Severity:** HIGH
- **Category:** Maintainability
- **Evidence:** `src/launchers/golf_launcher.py` (2,539 LOC) handles UI, Docker checks, environment dialogs, splash screens, model cards
- **Why it matters:** Merge conflicts, difficult testing, onboarding friction, violates SRP
- **Recommended fix:** Extract into: `DockerManager`, `EnvironmentDialog`, `SplashScreen`, `ModelCardWidget`, `LauncherCore`
- **Migration:** Strangler Fig pattern - extract one component at a time, maintain backward compatibility
- **Effort:** L | **Risk:** MEDIUM

#### H-2: Incomplete Dependency Locking

- **Severity:** HIGH
- **Category:** Build/Deploy
- **Evidence:** `requirements.lock` has only 13 entries vs 100+ actual dependencies; `requirements.txt` uses `>=` without upper bounds
- **Why it matters:** Non-reproducible builds, silent breakage, security vulnerabilities from uncontrolled updates
- **Recommended fix:** Generate complete lockfile with `pip-compile` or `pip freeze`; add upper bounds to critical deps
- **Migration:** 1) Generate full lock, 2) Test against lock, 3) Update dependabot to use lock
- **Effort:** S | **Risk:** LOW

#### H-3: Frontend Accessibility Gaps

- **Severity:** HIGH
- **Category:** DX / Compliance
- **Evidence:** `ui/src/components/**` - No ARIA attributes, no keyboard navigation indicators, Canvas without fallback content
- **Why it matters:** WCAG violations, potential legal liability, excluded users
- **Recommended fix:** Add `aria-label` to all interactive elements, `role` attributes, keyboard focus styles
- **Migration:** 1) Add ESLint a11y plugin, 2) Fix existing violations, 3) Add accessibility tests
- **Effort:** M | **Risk:** LOW

#### H-4: Missing Frontend Component Tests

- **Severity:** HIGH
- **Category:** Testability
- **Evidence:** `ui/src/api/client.test.ts` is the only test file; no component render tests, no interaction tests
- **Why it matters:** UI regressions undetected, unsafe to refactor, poor confidence in changes
- **Recommended fix:** Add React Testing Library tests for all components; target 80% component coverage
- **Migration:** 1) Add testing-library deps, 2) Test critical paths first, 3) Add to CI
- **Effort:** M | **Risk:** LOW

#### H-5: Main Dockerfile Security Issues

- **Severity:** HIGH
- **Category:** Security / Build
- **Evidence:** `/Dockerfile` uses `:latest` base, no non-root user, no multi-stage build, no health check
- **Why it matters:** Non-reproducible builds, container escape risk, no health monitoring
- **Recommended fix:** Pin base image version, add non-root user, implement multi-stage build (use `Dockerfile.unified` as template)
- **Migration:** 1) Copy patterns from Dockerfile.unified, 2) Test builds, 3) Update CI
- **Effort:** M | **Risk:** LOW

### MEDIUM

#### M-1: MyPy Exclusion List Too Large

- **Severity:** MEDIUM
- **Category:** Maintainability
- **Evidence:** `pyproject.toml` excludes 10+ directories from type checking including `src/api/`, `src/shared/python/signal_toolkit/`
- **Why it matters:** Type errors slip through, IDE support degraded, runtime bugs
- **Recommended fix:** Gradually add type hints and remove directories from exclusion list
- **Migration:** One directory per sprint; start with most-used modules
- **Effort:** L (ongoing) | **Risk:** LOW

#### M-2: Generic Exception Catching

- **Severity:** MEDIUM
- **Category:** Reliability
- **Evidence:** 50+ instances of `except Exception:` in API routes; `src/api/routes/simulation.py:45`, `src/api/server.py:231-235`
- **Why it matters:** Masks specific errors, harder to debug, poor error messages to clients
- **Recommended fix:** Catch specific exceptions; use structured error codes from `error_codes.py`
- **Migration:** 1) Identify exception types in each handler, 2) Add specific catches, 3) Log original exceptions
- **Effort:** M | **Risk:** LOW

#### M-3: Unused Frontend Dependencies

- **Severity:** MEDIUM
- **Category:** Build/DX
- **Evidence:** `ui/package.json` includes Zustand, react-hook-form, zod - none are imported in codebase
- **Why it matters:** Larger bundle size, security surface, confusing for developers
- **Recommended fix:** Remove unused dependencies or implement planned features using them
- **Migration:** Run `depcheck`, remove unused packages, verify build
- **Effort:** S | **Risk:** LOW

#### M-4: CVE Exceptions Without Remediation Plan

- **Severity:** MEDIUM
- **Category:** Security
- **Evidence:** `.pre-commit-config.yaml` ignores CVE-2024-23342; `ci-standard.yml` ignores CVE-2026-0994
- **Why it matters:** Security vulnerabilities remain unaddressed; audit failures
- **Recommended fix:** Document each CVE exception with expected fix date and mitigation; track in issues
- **Migration:** Create tracking issues for each CVE; add to `SECURITY.md`
- **Effort:** S | **Risk:** MEDIUM

#### M-5: Missing WebSocket Reconnection Logic

- **Severity:** MEDIUM
- **Category:** Reliability
- **Evidence:** `ui/src/api/client.ts` - WebSocket closes without auto-reconnect; no exponential backoff
- **Why it matters:** Poor UX on network issues, lost simulation data
- **Recommended fix:** Add reconnection with exponential backoff; preserve state during reconnect
- **Migration:** 1) Add reconnect logic, 2) Add connection status indicator, 3) Handle in-flight messages
- **Effort:** M | **Risk:** LOW

### LOW

#### L-1: Pytest Configuration Duplication

- **Severity:** LOW
- **Category:** Maintainability
- **Evidence:** Identical `pytest.ini` files in 4+ engine subdirectories with same configuration
- **Why it matters:** DRY violation, maintenance burden, configuration drift
- **Recommended fix:** Use root `pyproject.toml` [tool.pytest] with path-specific overrides
- **Migration:** Consolidate configs, test each engine, remove duplicate files
- **Effort:** S | **Risk:** LOW

#### L-2: Inline CSS Class Duplication (Frontend)

- **Severity:** LOW
- **Category:** Maintainability
- **Evidence:** `ui/src/components/simulation/SimulationControls.tsx` - Button classes repeated 4+ times
- **Why it matters:** Inconsistent styling risk, harder to maintain
- **Recommended fix:** Extract to Tailwind component classes or utility functions
- **Migration:** Create `buttonStyles` utility object, replace inline classes
- **Effort:** S | **Risk:** LOW

#### L-3: Missing Docker Image Scanning in CI

- **Severity:** LOW
- **Category:** Security
- **Evidence:** No Trivy/Scout/Grype action in any CI workflow
- **Why it matters:** Container vulnerabilities undetected until production
- **Recommended fix:** Add `aquasecurity/trivy-action` to CI workflow after Docker builds
- **Migration:** Add workflow step, triage initial findings, set severity threshold
- **Effort:** S | **Risk:** LOW

---

## D) Contracts & API Review

### Boundary Contracts

| Boundary        | Type             | Status  | Location                                            |
| --------------- | ---------------- | ------- | --------------------------------------------------- |
| REST API        | Pydantic models  | GOOD    | `src/api/models/requests.py`, `responses.py`        |
| WebSocket       | JSON messages    | PARTIAL | No schema validation on incoming frames             |
| Database        | SQLAlchemy ORM   | GOOD    | `src/api/auth/models.py` with CHECK constraints     |
| File I/O        | Path validation  | GOOD    | `src/api/utils/path_validation.py` whitelist        |
| Error responses | Structured codes | GOOD    | `src/api/utils/error_codes.py` (GMS-XXX-NNN format) |

### Evaluation Results

| Criterion                          | Status  | Notes                                                 |
| ---------------------------------- | ------- | ----------------------------------------------------- |
| Inputs validated at boundary       | PARTIAL | Pydantic validates REST; WebSocket JSON not validated |
| Outputs well-defined and versioned | PARTIAL | Response models defined; no API versioning            |
| Errors structured and consistent   | GOOD    | `GMS-{CATEGORY}-{NUMBER}` format with request tracing |

### Recommendations

1. **Add JSON Schema validation to WebSocket messages** - Currently raw JSON.parse without type checking
2. **Implement API versioning** - Add `/api/v1/` prefix for future compatibility
3. **Generate OpenAPI spec** - FastAPI auto-generates; ensure it's exported and versioned
4. **Add request ID to all error responses** - Currently partial implementation

---

## E) Testing Strategy Review

### Current State

| Test Type         | Count     | Coverage        | Status       |
| ----------------- | --------- | --------------- | ------------ |
| Unit tests        | 117 files | 90% threshold   | GOOD         |
| Integration tests | 10+ files | Included in 90% | GOOD         |
| E2E tests         | Limited   | Not measured    | NEEDS WORK   |
| Frontend tests    | 1 file    | Minimal         | CRITICAL GAP |

### Test Pyramid Compliance

```
Target:                          Actual:
    /\                               /\
   /E2E\  (5%)                      /??\  (unmeasured)
  /------\                         /----\
 /Integr. \  (15%)                /Integr\  (~10%)
/----------\                     /--------\
/   Unit    \  (80%)            /  Unit    \  (~90%)
--------------                  --------------
```

### Flakiness Sources Identified

1. **Time-dependent tests** - Some tests don't mock `datetime.now()`
2. **Network-dependent** - Integration tests may hit real endpoints
3. **Random seed inconsistency** - Most but not all tests pin `np.random.seed(42)`
4. **Resource contention** - Physics engines may conflict when run in parallel

### Recommendations

1. **Add frontend component tests** - Priority: Simulation page, EngineSelector, Scene3D
2. **Mock all external calls in unit tests** - Ensure determinism
3. **Add E2E smoke test** - Single happy-path test for React UI → API → Engine
4. **Standardize seed management** - Create `@pytest.fixture(autouse=True)` for random seeds

---

## F) Tooling & Process

### Code Quality Tooling

| Tool                 | Status   | Configuration         |
| -------------------- | -------- | --------------------- |
| Formatting (Black)   | ENFORCED | 88 chars, CI blocking |
| Linting (Ruff)       | ENFORCED | CI blocking           |
| Type checking (MyPy) | PARTIAL  | Many exclusions       |
| Security (Bandit)    | ENFORCED | low+high impact       |
| Security (pip-audit) | ENFORCED | 2 CVE exceptions      |

### CI/CD

| Check            | Required | Notes                  |
| ---------------- | -------- | ---------------------- |
| Linting          | YES      | Ruff + Black           |
| Type checking    | YES      | MyPy (with exclusions) |
| Unit tests       | YES      | 90% coverage threshold |
| Security scan    | YES      | Bandit + pip-audit     |
| TODO/FIXME check | YES      | Blocks merge           |
| Frontend build   | YES      | Vite build             |
| Frontend tests   | NO       | Missing requirement    |

### Dependency Hygiene

| Aspect            | Status     | Notes                       |
| ----------------- | ---------- | --------------------------- |
| Python lockfile   | INCOMPLETE | Only 13 of 100+ deps locked |
| npm lockfile      | COMPLETE   | Full package-lock.json      |
| Conda env file    | PARTIAL    | Loose version constraints   |
| Dependabot        | ENABLED    | Weekly updates              |
| Supply chain risk | MEDIUM     | CVE exceptions documented   |

---

## G) Repository Structure Assessment

### Current Structure (Simplified)

```
UpstreamDrift/
├── src/
│   ├── api/                 # FastAPI server (5,400 LOC)
│   ├── engines/             # Physics engines (255 files)
│   ├── launchers/           # GUI launchers (9,600 LOC)
│   ├── shared/python/       # Shared utilities (87 files)
│   ├── tools/               # Analysis tools
│   └── config/              # Configuration
├── ui/                      # React frontend (12 files)
├── tests/                   # Test suite (120+ files)
├── docs/                    # Documentation (50+ files)
├── scripts/                 # Utility scripts (40+ files)
├── examples/                # Usage examples
└── data/                    # Datasets
```

### Assessment

| Criterion                     | Status  | Notes                                       |
| ----------------------------- | ------- | ------------------------------------------- |
| Clear separation of concerns  | PARTIAL | Launchers mix UI and logic                  |
| Consistent naming conventions | GOOD    | Python: snake_case, TS: camelCase           |
| Test co-location              | GOOD    | Separate tests/ directory                   |
| Documentation organization    | GOOD    | Well-structured docs/                       |
| Configuration centralization  | PARTIAL | Multiple config files with some duplication |

---

## Comparison with Existing GitHub Issues

### Currently Open Issues

| Issue # | Title                                               | Overlap with Assessment      |
| ------- | --------------------------------------------------- | ---------------------------- |
| #980    | Implement SMPL-X integration for mesh generation    | Feature request - No overlap |
| #979    | Implement MakeHuman integration for mesh generation | Feature request - No overlap |

### Assessment Findings NOT in Existing Issues

All findings in this assessment are NEW and should be filed as GitHub issues.

---

## Recommended GitHub Issues to Create

### Critical Priority (Create Immediately)

1. **[Security] Replace unsafe `eval()` in signal_toolkit/fitting.py**

   - Labels: `security`, `critical`, `python`
   - Assignee: Security review required

2. **[Security] Remove pickle deserialization in motion_optimization.py**
   - Labels: `security`, `critical`, `python`

### High Priority

3. **[Refactor] Break up golf_launcher.py god module**

   - Labels: `refactoring`, `maintainability`, `python`, `size/L`

4. **[DevOps] Complete Python dependency lockfile**

   - Labels: `dependencies`, `ci/cd`, `size/M`

5. **[A11y] Add accessibility attributes to React components**

   - Labels: `accessibility`, `frontend`, `size/M`

6. **[Testing] Add React component tests**

   - Labels: `tests`, `frontend`, `size/M`

7. **[Security] Harden main Dockerfile (non-root user, pinned base)**
   - Labels: `security`, `docker`, `size/M`

### Medium Priority

8. **[TypeScript] Reduce mypy exclusion list**

   - Labels: `typing`, `python`, `size/L`

9. **[Reliability] Replace generic exception catches with specific handlers**

   - Labels: `error-handling`, `python`, `size/M`

10. **[Cleanup] Remove unused frontend dependencies (Zustand, react-hook-form, zod)**

    - Labels: `dependencies`, `frontend`, `size/S`

11. **[Security] Document CVE exception remediation plans**

    - Labels: `security`, `documentation`, `size/S`

12. **[Reliability] Add WebSocket reconnection with exponential backoff**
    - Labels: `reliability`, `frontend`, `size/M`

### Low Priority

13. **[DRY] Consolidate duplicate pytest.ini configurations**

    - Labels: `maintainability`, `tests`, `size/S`

14. **[Cleanup] Extract repeated Tailwind button classes**

    - Labels: `frontend`, `maintainability`, `size/S`

15. **[Security] Add Docker image scanning to CI (Trivy)**
    - Labels: `security`, `ci/cd`, `size/S`

---

## Summary Metrics

| Category                  | Score | Notes                                                       |
| ------------------------- | ----- | ----------------------------------------------------------- |
| Security                  | 6/10  | Critical eval/pickle issues; good auth/validation otherwise |
| Maintainability           | 7/10  | God modules and type gaps offset by good testing            |
| Testability               | 8/10  | Strong Python testing; frontend tests lacking               |
| Build/Deploy              | 7/10  | Good CI gates; Docker and locking need work                 |
| Documentation             | 9/10  | Comprehensive docs, good structure                          |
| DX (Developer Experience) | 7/10  | Good tooling; onboarding could be smoother                  |

**Overall Assessment: GOOD with specific critical fixes needed**

The repository demonstrates professional engineering practices with comprehensive testing, structured error handling, and good documentation. The critical security issues (eval/pickle) should be addressed immediately. The god module refactoring and accessibility fixes are high-value medium-term investments.
