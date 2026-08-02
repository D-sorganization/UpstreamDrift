# Static analysis plots (20+ types) absent from web app — expose plot-data API + web Analysis parity

## Gap (PyQt6 = model)

The PyQt6 unified dashboard offers 20+ static analysis plot types after a simulation (joint angles, joint velocities/torques, muscle forces, power, energy, club/hand kinematics, EMG-style activations, etc.) via `GolfSwingPlotter` (`src/shared/python/analysis/plot_engine.py`) dispatched from the dashboard's `refresh_static_plot()`. The web app has only the live-streaming `LivePlot` (`ui/src/components/analysis/LivePlot.tsx`) and a stub `AnalysisTools` page. **Static post-run analysis coverage on the web is 0%** — this is the single largest functional parity gap.

## Proposed fix

Builds on the service-layer extraction issue in this epic (`AnalysisOrchestrator` returning `PlotData`):

1. API: `GET /api/v1/analysis/plot-data/{plot_type}?session=...` returning structured series data (time vectors, named channels, units, axis labels) — never rendered images. Add `GET /api/v1/analysis/plot-types` enumerating available types per engine/recording so the web UI is data-driven and new plot types added on the PyQt6 side appear on the web automatically.
2. Web: extend the Analysis page to render any `PlotData` generically with Recharts (one generic multi-series renderer + per-type config map). Channel names/units come from the payload (pairs with the `ProvenanceValue`/`HelpfulField` UX primitives).
3. Recording source: plot data must be computable from a completed simulation session (recorder retained server-side or persisted recording id) — coordinate with the export-parity issue for recording persistence.
4. Parity test: a pytest that asserts the set of plot types exposed by the API equals the set dispatched by the PyQt6 dashboard (single source: the orchestrator's registry), so the two UIs can never drift silently.

## Acceptance criteria

- [ ] Every plot type available in the PyQt6 dashboard is enumerable and fetchable via the API
- [ ] Web Analysis page renders all types from live API data (no canned data)
- [ ] Adding a new plot type to the orchestrator automatically surfaces it in both UIs (manifest/enumeration-driven, no per-type web PR required)
- [ ] Parity test pinning PyQt6 dispatch set == API set

## References

- `docs/architecture/dual-gui-architecture-review.md` §3.2 Step 2 (`GET /api/analysis/plot-data/{type}` proposal)
- Depends on: service-layer extraction issue (this epic)
