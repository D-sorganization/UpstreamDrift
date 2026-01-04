# Remediation Strategy & Implementation Plan

**Source:** Based on `Golf_Modeling_Suite_Assessment.md` (Root Assessment)
**Target:** `Golf_Modeling_Suite` (Simscape Multibody Models / 3D Golf Model)
**Date:** Jan 3, 2026

## Phase 1: Critical Stabilization (Next 48 Hours)
**Goal:** Stop the bleeding. Address correctness and security criticals.

### 1.1 Correctness & Determinism (F2)
- **Objective:** Fix undefined RNG behavior.
- **Action:** Modify `logger_utils.set_seeds` to validate input seeds (0 through uint32 max). Raise `ValueError` for invalid inputs.
- **Target File:** `engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/logger_utils.py`

### 1.2 Security Hardening (F5)
- **Objective:** Prevent CSV Injection attacks (Excel Formula Injection).
- **Action:** Modify `c3d_reader.export_points` and `export_analog` to **enforce** sanitization. Remove the `sanitize=True` optional flag (make it mandatory/internal).
- **Target File:** `engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/c3d_reader.py`

### 1.3 Error Handling & UX (F1, F8)
- **Objective:** Prevent GUI freezes on invalid files.
- **Action:** Wrap `_load_c3d` in `apps/c3d_viewer.py` with specific `try/except` blocks to catch `KeyError` (missing metadata) and `ValueError` (shape mismatch), surfacing them as actionable `QMessageBox` errors instead of crashing.
- **Target File:** `engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py`

## Phase 2: Architectural De-Risking (Next 2 Weeks)
**Goal:** Decouple the Monolith and establish Test Safety.

### 2.1 UI Decomposition (F3)
- **Objective:** Break the `500+ LOC` monolith `c3d_viewer.py`.
- **Action:** Extract components into `src/ui/`:
    - `src/ui/widgets/mpl_canvas.py`: Matplotlib canvas.
    - `src/ui/tabs/overview.py`: Metadata table.
    - `src/ui/tabs/plotting.py`: 2D/3D Plotters.
    - `src/ui/controllers/main_window.py`: Layout and orchestration only.

### 2.2 Schema Validation (F4, F11)
- **Objective:** Enforce data integrity.
- **Action:** Introduce `pydantic` or `dataclasses` for Metadata/Point/Analog structures to ensure valid shapes before DataFrame construction.

### 2.3 Headless Testing (F9, F12)
- **Objective:** Enable CI for GUI.
- **Action:** Create `tests/ui/test_headless_smoke.py` using `pytest-qt` and `xvfb` patterns to verify app launch/shutdown in CI logic.

## Phase 3: Observability & Production (Next 6 Weeks)
**Goal:** Deep observability and packaging.

### 3.1 Structured Observability (F6)
- **Objective:** Debuggability.
- **Action:** Add `structlog` or equivalent structured logging to `c3d_viewer.py` loading paths.

### 3.2 Packaging (F10)
- **Objective:** Reproducibility.
- **Action:** Finalize `pyproject.toml` with strict dependencies and `requirements.lock`.

## Execution Status
- **Phase 1**: [Completed]
- **Phase 2**: [Pending]
- **Phase 3**: [Pending]
