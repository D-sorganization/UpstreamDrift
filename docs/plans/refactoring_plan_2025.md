# Golf Modeling Suite - Comprehensive Refactoring Plan 2025

Based on reviews from Claude and Codex, this plan outlines a 4-phase approach to modernizing the codebase, eliminating technical debt, and professionalizing the engineering standards.

## Phase 1: Critical Stabilization & CI Reliability (Week 1)
**Goal:** Ensure the build is stable, dependencies are unified, and duplicate code is halted.

- [x] **1.1 Fix Pytest Coverage Configuration**
  - **Issue:** `pyproject.toml` currently includes `--cov=engines` in `addopts` but blindly omits `engines/` in the `[tool.coverage.run]` section, creating a configuration contradiction.
  - **Action:** align coverage targets to include shared components and launchers first, then selectively add engines.
- [x] **1.2 Consolidate Dependency Management**
  - **Issue:** 16+ `requirements.txt` files exist, causing version drift.
  - **Action:** Migrate all dependencies to the root `pyproject.toml` using `optional-dependencies` groups (e.g., `[project.optional-dependencies] mujoco = [...]`). Delete all `requirements.txt` files.
- [x] **1.3 Implement Duplicate File Prevention**
  - **Issue:** 7+ copies of `matlab_quality_check.py`.
  - **Action:** canonicalize the script in `tools/` and add a CI check (`scripts/check_duplicates.py`) to fail PRs if duplicates are detected.
- [x] **1.4 Fix Python Version Metadata**
  - **Issue:** README says 3.10+, `pyproject.toml` requires 3.11+.
  - **Action:** Update metadata to reflect the true requirement (3.11+) and align CI matrices.

## Phase 2: Technical Debt Reduction (Weeks 2-4)
**Goal:** Improve maintainability by breaking down monoliths and removing dead code.

- [ ] **2.1 GUI Refactoring (Single Responsibility Principle)**
  - **Issue:** `advanced_gui.py` is 3,933 lines long.
  - **Action:** Split into sub-modules:
    - [x] Extract `VisualizationTab` (`gui/tabs/visualization_tab.py`)
    - [ ] Extract `PhysicsTab`
    - [ ] Extract `ControlsTab`
    - [ ] Extract `AnalysisTab`
    - [ ] Refactor `AdvancedGolfAnalysisWindow` to use composed tabs
- [x] **2.2 Archive & Legacy Cleanup**
  - **Issue:** 13+ archive directories bloat the repo.
  - **Action:** Move valid historical code to a `legacy/` branch and delete these folders from `main`.
- [x] **2.3 Constants Normalization**
  - **Issue:** Multiple `constants.py` files with conflicting values (e.g., `GRAVITY`).
  - **Action:** Create `shared/python/physics_constants.py` as the source of truth and update all engines to import from it.

## Phase 3: Testing & Documentation (Month 2)
**Goal:** Increase confidence in cross-engine validity.

- [ ] **3.1 Cross-Engine Integration Tests**
  - **Issue:** No automated way to verify that MuJoCo, Drake, and Pinocchio results match.
  - **Action:** Create `tests/integration/test_physics_consistency.py` that runs simple pendulums on all 3 engines and asserts result proximity.
- [ ] **3.2 Architecture Documentation**
  - **Issue:** Missing high-level diagrams.
  - **Action:** Add Mermaid diagrams for "Engine Loading Flow" and "Data Pipeline" to `docs/architecture/`.
- [ ] **3.3 Launcher Configuration Abstraction**
  - **Issue:** `golf_launcher.py` contains hardcoded paths and mixed concerns.
  - **Action:** Move model definitions to `config/models.yaml` and create a `ModelRegistry` class.

## Phase 4: Performance Optimization (Month 3)
**Goal:** smooth user experience and faster feedback loops.

- [ ] **4.1 Async Engine Loading**
  - **Issue:** App freezes while MATLAB/MuJoCo loads.
  - **Action:** Move engine initialization to background threads with a splash screen/progress bar.
- [ ] **4.2 Lazy Import implementation**
  - **Issue:** `golf_launcher.py` imports heavy PyQt6 modules at the top level.
  - **Action:** Move imports inside functions where possible to speed up CLI response time.
