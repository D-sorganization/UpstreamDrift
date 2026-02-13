# Dual GUI Parity Architecture Assessment and Multi-Phase Implementation Plan

Date: 2026-02-10  
Author: Codex (mainline assessment)

## 1. Scope and Method

This assessment covers three repositories and evaluates the ability to maintain a **React + PyQt dual GUI** model with a **shared backend**, while enabling parallel development with low drift risk.

Repositories assessed on remote main snapshots:

- `Tools` at commit `f43af9a`
- `Gasification_Model` at commit `7c9a6052`
- `UpstreamDrift` at commit `31a6cbde`

To avoid disturbing active local work, analysis was run on clean `origin/main` worktrees:

- `/home/dieterolson/Linux_Repositories/_mainline_worktrees/Tools-main`
- `/home/dieterolson/Linux_Repositories/_mainline_worktrees/Gasification-main`
- `/home/dieterolson/Linux_Repositories/_mainline_worktrees/UpstreamDrift-main`

## 2. Executive Summary

Current architecture is directionally good, but not yet operationally safe for long-term parity:

- There is a real shared-backend foundation (`vendor/ud-tools` from `Tools`) consumed by both `Gasification_Model` and `UpstreamDrift`.
- `Tools` has strong modular structure for many dual-surface calculators (both PyQt and React).
- `UpstreamDrift` has a strong single-manifest design for launcher parity, but runtime pathing has inconsistencies between API entry modes.
- `Gasification_Model` still relies on runtime `sys.path` bootstrapping for shared code imports, creating packaging and reproducibility risk.
- Cross-repo drift in shared backend version is already significant and unmanaged by automation.

Overall: **feasible to maintain dual GUI parity**, but only after standardizing dependency/version flow, endpoint composition, and parity gates in CI.

## 3. Observed Architecture State (Mainline)

### 3.1 Tools repository (`f43af9a`)

Strengths:

- Modular tool layout with clear `python` and `web` subdirectories for many tools.
- Measured on mainline:
  - dual-surface modules (`python` + `web`): `16`
  - PyQt-only modules: `8`
  - `launch_web.py` files: `13`
  - `gui_registration.py` files: `25`
- Existing DRY tests enforce launcher consistency patterns.

Risks:

- Launcher metadata has **two parallel sources**:
  - `tools.json` consumed by unified launcher UI (`src/tools/gui/windows/unified_launcher_window.py`)
  - `gui_registration.py` auto-discovery path (`launch.py`, `gui_launcher.registry`)
- Without generation/sync enforcement, this can drift and create parity failures.

### 3.2 Gasification_Model repository (`7c9a6052`)

Strengths:

- API service layer exists and React clients call typed API endpoints.
- Contains API/PyQt parity-oriented tests and fixtures.

Risks:

- Shared code imports still depend on runtime path mutation:
  - `src/integrated_process_simulator/api/services/tools_service.py` inserts vendor path into `sys.path`.
  - `src/integrated_process_simulator/__main__.py` does similar bootstrap.
- This is fragile across environments and weak for clean dependency versioning.
- Submodule pointer for `vendor/ud-tools` is old: commit `d4326b6` (31 commits behind Tools mainline `f43af9a`).

### 3.3 UpstreamDrift repository (`31a6cbde`)

Strengths:

- Launcher manifest model is well designed:
  - single source at `src/config/launcher_manifest.json`
  - API exposure via `src/api/routes/launcher.py`
  - React hook consumes `/api/launcher/manifest`
- Main API server (`src/api/server.py`) includes launcher router.

Risks:

- Local runtime server (`src/api/local_server.py`) does **not** include launcher router, but frontend expects launcher endpoint in local mode.
- Endpoint style is inconsistent in `src/api/routes/engines.py`:
  - both `/engines/...` and `/api/engines/...` are declared in same router.
- Submodule pointer for `vendor/ud-tools` is old: commit `44b6afa` (45 commits behind Tools mainline).

## 4. Root Causes of Parity Drift

1. Shared backend is consumed as submodule snapshots, but version movement is manual and unscheduled.
2. Runtime import bootstrapping (`sys.path`) bypasses package dependency contracts.
3. Multiple metadata sources for launchers and tool registration.
4. API route composition differs between runtime entrypoints (`server.py` vs `local_server.py`).
5. Parity tests are not yet a full end-to-end contract gate across PyQt, API, and React.

## 5. Safety-First Multi-Phase Implementation Plan

The sequence below is ordered from easiest/safest to highest-impact structural changes.

---

## Phase 0: Baseline and Guardrails (Low Risk, Immediate)

Goal: Make current state observable and prevent accidental regressions before refactors.

Tasks:

1. Add a repo-level `PARITY_BASELINE.md` in each repo documenting:
   - active GUI surfaces per module
   - current shared backend source and version
   - API endpoint map used by each frontend
2. Add CI checks that fail when:
   - branch upstream is gone (for protected branches)
   - submodule pointer changed without parity tests
3. Add scripted inventory command output artifact in CI:
   - number of `gui_registration.py`
   - number of `launch_web.py`
   - dual-surface count

Validation:

- CI emits baseline artifacts on every PR.
- No functional code path changes.

Rollback:

- Remove workflow and baseline docs only.

---

## Phase 1: Runtime Endpoint Consistency Fixes (Easy, Safe)

Goal: Remove obvious parity breaks without touching core business logic.

Tasks:

1. UpstreamDrift: include launcher router in `src/api/local_server.py`.
2. UpstreamDrift: ensure `/api/launcher/manifest` works in both:
   - `src/api/server.py`
   - `src/api/local_server.py`
3. Add test set that runs against both app factories and asserts same route coverage for critical endpoints (`/api/launcher/*`, engine probe/load, health).

Validation:

- Local and full server pass common route parity tests.
- React dashboard manifest load works in both runtime modes.

Rollback:

- Revert local_server router inclusion; no data migrations involved.

---

## Phase 2: Metadata Unification for Launchers (Safe Refactor)

Goal: Establish one source of truth for tool/launcher definitions.

Tasks:

1. Declare canonical metadata source (recommended: `gui_registration.py` + typed schema).
2. Generate `tools.json` from canonical source (do not hand-edit `tools.json`).
3. Add CI check:
   - regeneration is deterministic
   - generated file diff must be clean
4. Optionally generate TypeScript launcher manifest types from the same schema.

Validation:

- `tools.json` and discovered registrations always match.
- Launcher UI renders same tool set after regeneration.

Rollback:

- Keep generator but allow temporary manual mode via feature flag.

---

## Phase 3: Shared Backend Version Flow (Moderate Risk, Controlled)

Goal: Make shared backend updates predictable and reviewable.

Tasks:

1. Introduce explicit shared-backend version policy:
   - pin target Tools commit/tag in each consumer repo
   - add `SHARED_BACKEND_VERSION.md`
2. Add drift CI job in Gasification/Upstream that reports:
   - current `vendor/ud-tools` pointer
   - commits behind target Tools baseline
3. Add automation to propose submodule bump PRs on Tools changes to shared packages.

Validation:

- Each consumer repo has visible drift metric.
- Submodule bumps become regular and test-gated.

Rollback:

- Disable auto-bump bot; keep drift reporting.

---

## Phase 4: Remove Runtime `sys.path` Bootstrapping (Moderate Risk)

Goal: Replace implicit imports with explicit package dependencies.

Tasks:

1. Package shared Python backend from Tools as installable artifact (wheel/sdist).
2. In Gasification and Upstream:
   - replace `sys.path.insert(...)` patterns with standard dependency imports
   - use environment-specific dependency files (`requirements`/`pyproject` extras)
3. Keep temporary compatibility shim with deprecation warning for one release cycle.

Validation:

- App startup succeeds without vendor path mutation.
- Tests run in clean envs with package install only.

Rollback:

- Re-enable shim imports for one cycle if packaging issue is discovered.

---

## Phase 5: Contract-First Parity Gates (Higher Value, Higher Effort)

Goal: Prevent future drift by enforcing parity across surfaces.

Tasks:

1. Define canonical API contracts (OpenAPI/Pydantic) as source of truth.
2. Generate frontend clients/types from contract.
3. Build parity fixture runner that executes same vectors through:
   - backend API endpoint
   - PyQt adapter path
   - React client path
4. Set numeric tolerance and failure budgets by calculator/engine.

Validation:

- PR fails if parity deviation exceeds tolerance.
- Parity report artifact includes passed/failed modules and deltas.

Rollback:

- Keep tests non-blocking initially; move to blocking after burn-in.

---

## Phase 6: Parallel Development Workflow (Process + Tooling)

Goal: Make dual-GUI parallel work straightforward for agents and contributors.

Tasks:

1. Add `feature template` requiring four deliverables:
   - backend contract change
   - PyQt integration
   - React integration
   - parity tests
2. Add PR template checklist with links to generated parity artifacts.
3. Add CODEOWNERS mappings so shared backend and both frontends are reviewed.

Validation:

- Every feature PR has explicit parity evidence.
- Review bottlenecks reduced via clear ownership.

Rollback:

- Relax template requirements temporarily if throughput drops.

## 6. Agent-Executable Structured Plan

Use this as an implementation playbook for an autonomous agent.

### Milestone A (Days 1-3): Safe Runtime Corrections

- [ ] Add launcher router to Upstream local server.
- [ ] Add route parity tests for server/local_server.
- [ ] Add baseline docs and inventory CI artifacts.

Acceptance:

- [ ] `/api/launcher/manifest` available in both runtime modes.
- [ ] No behavior changes to solver outputs.

### Milestone B (Days 4-7): Metadata Synchronization

- [ ] Implement metadata generator (`gui_registration` -> `tools.json`).
- [ ] Add deterministic generation CI gate.
- [ ] Remove manual edit paths from docs.

Acceptance:

- [ ] Generator idempotent.
- [ ] Launcher card counts unchanged after migration.

### Milestone C (Week 2): Shared Backend Drift Control

- [ ] Add drift report job in Gasification and Upstream.
- [ ] Add automated submodule bump proposal workflow.
- [ ] Define shared backend update policy and cadence.

Acceptance:

- [ ] Drift metric visible on every PR.
- [ ] At least one successful automated bump cycle.

### Milestone D (Weeks 3-4): Dependency Hardening

- [ ] Package shared backend from Tools.
- [ ] Remove runtime path hacks in Gasification and Upstream.
- [ ] Keep temporary shim for compatibility.

Acceptance:

- [ ] Fresh environment install works without vendor path insertion.
- [ ] Existing parity/integration tests still pass.

### Milestone E (Weeks 4-6): Full Parity Enforcement

- [ ] Generate frontend client/types from API contract.
- [ ] Implement tri-surface fixture runner (API/PyQt/React).
- [ ] Promote parity gate from informational to blocking.

Acceptance:

- [ ] CI blocks parity regressions.
- [ ] Drift and parity dashboards both green.

## 7. Immediate Next Actions (Recommended)

1. Implement Phase 1 first in UpstreamDrift (`local_server` launcher router + tests).
2. Implement Phase 2 generator in Tools to prevent metadata drift now.
3. Add Phase 3 drift-report CI jobs before any larger packaging refactor.
4. Start Phase 4 packaging pilot with one calculator family before full rollout.

## 8. Reference Evidence Files

- `Tools`: `src/tools/gui/windows/unified_launcher_window.py`, `launch.py`, `tests/test_dry_compliance.py`
- `Gasification_Model`: `src/integrated_process_simulator/api/services/tools_service.py`, `src/integrated_process_simulator/__main__.py`
- `UpstreamDrift`: `src/api/routes/launcher.py`, `src/api/local_server.py`, `src/api/server.py`, `src/api/routes/engines.py`, `ui/src/api/useLauncherManifest.ts`, `src/config/launcher_manifest.json`
