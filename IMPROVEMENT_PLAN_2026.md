# Golf Modeling Suite Improvement Plan (Jan 2026)

Based on the assessment in `Golf_Modeling_Suite_Assessment.md` and current code verification, the following plan aims to elevate the `3D_Golf_Model` to a production-ready state.

## 1. Critical Remediation (Security & Correctness)

**Priority**: Impact High | Effort Low
**Target**: `logger_utils.py`, `c3d_reader.py`

- [ ] **Fix Seed Validation**: Modify `logger_utils.set_seeds` to reject negative inputs or values exceeding `uint32`.
- [ ] **Enforce CSV Sanitization**: In `c3d_reader.py`, remove the `sanitize=True` optional flag from `_export_dataframe`. Enforce sanitization by default for all CSV exports to prevent Excel Injection vulnerabilities.
- [ ] **Harden Unit Logic**: Improve `_unit_scale` to handle unknown units gracefully (e.g., fallback to 1.0 with a warning) instead of crashing with `ValueError`.

## 2. Architectural Refactoring (The Monolith)

**Priority**: Impact High | Effort Medium
**Target**: `python/src/apps/c3d_viewer.py`

The viewer is currently a 670-line monolith. We will decompose it into a clean MVC structure:

- [ ] **Extract UI Components**:
    -   `ui/widgets/mpl_canvas.py`: The `MplCanvas` class.
    -   `ui/tabs/`: Separate classes for `OverviewTab`, `MarkerPlotTab`, `Viewer3DTab`, etc.
- [ ] **Refactor Main Window**: `C3DViewerMainWindow` should only handle layout and signal connections, delegating logic to controllers or the extracted tabs.

## 3. Testing & Validation

**Priority**: Impact Medium | Effort Medium
**Target**: `python/tests/`

- [ ] **Headless GUI Smoke Test**: Create `tests/ui/test_c3d_viewer_headless.py` using `pytest-qt` to verify that the main window launches and tears down without error in a headless environment (using `xvfb` patterns).
- [ ] **Unit Tests**: Add specific tests for the new Seed Validation and Sanitization logic.

## 4. Packaging & Hygiene

**Priority**: Impact Low | Effort Low

- [ ] **Verify Dependencies**: Ensure `pyproject.toml` correctly targets the new structure.
- [ ] **Linting**: Run `ruff` and `mypy` on the refactored code to ensure zero regressions.

## Execution Timeline

- **Phase 1**: Critical Remediations (Step 1).
- **Phase 2**: Refactoring (Step 2).
- **Phase 3**: Verification (Step 3).
