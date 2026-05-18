# test(c3d-viewer): bring `apps/` (C3D Viewer) subtree to ≥85% coverage

## Goal

Raise unit + integration coverage of `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/` to **≥85% line, ≥75% branch**. Test-only — no production code changes.

## Current state

17 production `.py` files (services, ui/tabs, ui/widgets, core/models). Recent additions from #4640 / #4661 added 16+ tests under `tests/unit/engines/simscape/three_d_gui/`. Likely gaps:

- `core/models.py` — `C3DDataModel`, `Marker`, raw_parameters round-trip
- `services/c3d_loader.py` — error paths, malformed C3D, missing groups
- `services/loader_thread.py` — Qt thread signals, cancellation
- `services/segment_set_io.py` — JSON schema validation, version migration
- `services/marker_export.py` — CSV / JSON / NPZ format details, frame-range edge cases
- `services/analysis.py` — kinematic stats edge cases (single-frame, all-NaN, etc.)
- `ui/tabs/overview_tab.py` — metadata tree population
- `ui/tabs/marker_plot_tab.py` — component switching, missing markers
- `ui/tabs/analog_plot_tab.py` — empty-analog branch
- `ui/tabs/analysis_tab.py` — recompute button
- `ui/tabs/force_plot_tab.py` — empty-force-plate branch
- `ui/tabs/segments_tab.py` — table editing, dialog flows
- `ui/widgets/mpl_canvas.py`

## Process

Same as the tracking issue (run coverage, identify gaps, add unit + Qt-offscreen tests). The per-directory `conftest.py` (already on `main`) sets up `sys.modules['src']` so the engine package and the repo's `src.shared.*` both resolve.

For Qt-driven tests:

- `QT_QPA_PLATFORM=offscreen`
- matplotlib `Agg` backend
- `QApplication.instance() or QApplication([])`
- `QTimer.singleShot(0, lambda: ...)` + `QEventLoop` for async-flow tests

## Acceptance

- [ ] `pytest tests/unit/engines/simscape/three_d_gui/ --cov=src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps --cov-report=term-missing --cov-branch` reports **≥85% line, ≥75% branch**.
- [ ] PR body lists per-file delta.
- [ ] No production changes.
- [ ] mypy + ruff + file-size budget clean.

## Files touched

- New / extended: `tests/unit/engines/simscape/three_d_gui/test_*.py`
