# ZTCF/ZVCF and induced-acceleration analyses are desktop-only — expose via API + web UI

## Gap (PyQt6 = model)

Counterfactual analyses — ZTCF (zero-torque counterfactual), ZVCF (zero-velocity counterfactual), and induced-acceleration analysis — are available in the PyQt6 unified dashboard (`compute_analysis()` path) but have no API endpoint and no web UI. These are headline scientific features of the suite; the web app cannot run them at all.

Note: the ZTCF/ZVCF definitions were part of the 2026-06-11 scientific accuracy audit — implementers should verify the canonical definitions against the corrected articles/issues before mirroring them to the API, so we don't propagate a wrong identity into a second frontend.

## Proposed fix

1. After the service-layer extraction (this epic), expose:
   - `POST /api/v1/analysis/counterfactual` with `{kind: "ztcf"|"zvcf"|"induced_acceleration", session/recording id, params}` — async-task pattern (these are expensive; reuse the existing `/simulate/async` TaskManager + `GET /simulate/status/{task_id}` machinery).
   - Results returned as the same `PlotData`/structured series used by the static-plot endpoint so the web renderer is reused.
2. Web: add a Counterfactual section to the Analysis page — kind selector, parameter form (`HelpfulField`), run button with task progress, results rendered with the generic plot renderer.
3. Capability-gate by engine: not all engines support these; surface via the existing engine capabilities endpoint (`/launcher/engines/{id}/capabilities`) rather than hardcoding in the web UI.

## Acceptance criteria

- [ ] ZTCF, ZVCF, induced-acceleration runnable from the web against a completed session, with results matching the PyQt6 dashboard for the same recording (golden-data test)
- [ ] Engine capability gating data-driven from the API
- [ ] Feature-parity registry entries flip from `gap` to `parity`

## References

- `docs/architecture/dual-gui-architecture-review.md` §3.2 Step 2 (`POST /api/analysis/counterfactual`)
- Depends on: service-layer extraction issue (this epic)
