# Extract analysis/plotting logic from PyQt6 widgets into shared services (parity prerequisite)

## Problem

The single largest structural obstacle to PyQt6↔web parity is that the rich analysis features live _inside_ PyQt6 widgets, so they cannot be exposed over the API without duplication. `docs/architecture/dual-gui-architecture-review.md` §3.2 Step 1 already identifies the pattern: `UnifiedDashboardWindow.refresh_static_plot()` dispatches 20+ plot types directly to `GolfSwingPlotter` (`src/shared/python/analysis/plot_engine.py`) rendering into matplotlib canvases. The same coupling applies to counterfactual analysis (ZTCF/ZVCF, induced acceleration) and export logic in the dashboard windows.

Until this logic is extracted, every web-parity feature would have to be reimplemented twice — exactly the divergence this epic is meant to stop.

## Proposed fix

1. Create `AnalysisOrchestrator` (e.g. `src/shared/python/analysis/orchestrator.py`) whose methods return **structured data** (typed dataclasses / numpy-backed `PlotData`), never matplotlib figures:
   - `get_plot_data(plot_type, recorder) -> PlotData` for every static plot type currently in the dashboard dispatch;
   - `compute_counterfactual(kind, ...) -> CounterfactualResult` for ZTCF/ZVCF/induced-acceleration;
   - export-format enumeration and serialization hooks.
2. Refactor the PyQt6 dashboards to consume the orchestrator and render `PlotData` locally (thin `render_to_canvas(data, canvas)` adapter). Behavior must be unchanged — this is a pure extraction.
3. The orchestrator becomes the implementation behind the new API endpoints (tracked separately: static plots, counterfactuals, export parity issues in this epic). API routes should be thin wrappers in `src/api/services/analysis_service.py`.
4. Tests: golden-data tests on the orchestrator outputs (engine-agnostic, headless-safe) so both frontends are validated by one suite.

## Acceptance criteria

- [ ] All static-plot dispatch branches in the unified dashboard route through `AnalysisOrchestrator.get_plot_data`
- [ ] ZTCF/ZVCF/induced-acceleration computation callable headlessly with no Qt import
- [ ] PyQt6 dashboards visually unchanged (existing GUI tests pass)
- [ ] Orchestrator outputs are JSON-serializable (Pydantic-model compatible) so the API issues in this epic can build on them without further refactoring
- [ ] No logic duplicated between widget and orchestrator (DRY gate)

## References

- `docs/architecture/dual-gui-architecture-review.md` §3.1–3.2 (recommended shared service layer)
- `src/shared/python/analysis/plot_engine.py` (GolfSwingPlotter)
- `src/api/services/analysis_service.py` (existing thin service to extend)
