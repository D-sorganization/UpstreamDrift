# Cross-engine robustness dashboard (perturbation comparison, CV/robustness charts) has no web equivalent

## Gap (PyQt6 = model)

Desktop `src/launchers/cross_engine_dashboard.py` runs perturbation-based comparisons across engines (MuJoCo/Drake/Pinocchio/pendulum), computing robustness scores (1 − CV) and CV charts per metric, with configurable perturbation parameters — and it already has a headless CLI mode (`--no-gui`), meaning the computation is _already_ UI-independent. The web `EngineComparisonPanel` (`ui/src/components/simulation/EngineComparisonPanel.tsx`) only selects engines for side-by-side rollouts; there is no perturbation study, no robustness metrics, no comparison charts.

## Proposed fix

1. Extract the dashboard's compute path (the same code its `--no-gui` mode uses) into a service callable from the API: `POST /api/v1/analysis/cross-engine` `{engines[], model, perturbation config}` → async task → structured results (per-engine per-metric mean/std/CV/robustness).
2. Web: a Cross-Engine Comparison view (extend the comparison panel or a dedicated route) with engine checkboxes, perturbation parameter form, run-with-progress, and bar charts for robustness score and CV (Recharts; reuse generic plot renderer from the static-plots issue).
3. PyQt6 dashboard refactors to consume the same service (no duplicated compute logic).

## Acceptance criteria

- [ ] Same perturbation study produces identical numbers via desktop dashboard, CLI, and API (golden test)
- [ ] Web view renders robustness + CV comparisons for any subset of available engines
- [ ] Compute logic exists exactly once

## References

- `src/launchers/cross_engine_dashboard.py` (note: 38KB — extraction also helps the file-size budget)
- Depends on: service-layer pattern from this epic
