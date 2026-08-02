# Epic: PyQt6 ↔ Tauri/React functional parity program (2026-06-12 review)

## Scope and principle

**The PyQt6 desktop app is the model.** The Tauri/React app must match it in functional parity, and both must remain developable in parallel without re-diverging. This epic comes from a full review of `src/launchers/` + `src/tools/` (desktop), `ui/` (web), and `src/api/` (shared backend) at `origin/main` (88fdf2dc2).

Current state in one line: launcher tiles, engine load/probe, the basic simulation loop (WS streaming), and chat transport have parity; **post-run analysis (20+ plot types), counterfactual analyses, export formats, recordings, motion-capture breadth, diagnostics depth, settings, and several whole tools exist only on the desktop**. The dual-GUI review doc estimates web coverage at ~30% of the desktop feature set.

This epic deliberately does NOT cover visual/UX quality (that is epic #7444, filed 2026-06-11) or scientific correctness of the physics (epic #7414). It covers _which features exist where_, and the machinery to keep that true.

## Why it diverged (root causes, so the fix sticks)

1. Analysis/plot/export logic lives inside PyQt6 widgets, so it cannot be served over the API without extraction (`docs/architecture/dual-gui-architecture-review.md` §3.2).
2. No machine-checked parity ledger: only launcher tiles have parity tests; features drift silently.
3. Stub API endpoints let web pages _look_ implemented when they aren't.
4. Hand-written TS payload types drift from the Pydantic contract.

## Foundation (do these first — they make parallel development maintainable)

- [ ] #7445 **F1** Feature-parity registry (`feature_parity.json`) + CI gate + generated matrix doc — the standing mechanism replacing one-off audits
- [ ] #7446 **F2** Extract analysis/plot/counterfactual logic from PyQt6 widgets into shared services returning data (prereq for G1/G2/G7)
- [ ] #7447 **F3** Generate TypeScript API types from Pydantic models in CI
- [ ] #7448 **F4** Complete or remove stub/partial API endpoints (analysis_tools, character_builder, data_explorer, model_explorer, CSV export)

## Feature gaps (PyQt6 has it, web doesn't)

- [ ] #7449 **G1** Static analysis plots (20+ types) — plot-data API + generic web renderer _(largest gap; depends F2)_
- [ ] #7450 **G2** ZTCF/ZVCF + induced-acceleration via API + web UI _(depends F2)_
- [ ] #7451 **G3** Export/recording parity: HDF5/MAT/C3D/CSV/video downloads + persisted recordings API + web panel
- [ ] #7452 **G4** Wire dead web SimulationControls (camera presets, recording toggle, trajectory export) to existing endpoints
- [ ] #7453 **G5** Live app/engine context in web chat (desktop Sidekick has it)
- [ ] #7454 **G6** Motion-capture breadth: C3D upload/playback + OpenPose source on web
- [ ] #7455 **G7** Cross-engine robustness dashboard (perturbation/CV) on web _(depends F2 pattern)_
- [ ] #7456 **G8** Shot Tracer / ball-flight page on web (API exists)
- [ ] #7457 **G9** Web settings/preferences surface + server-side persistence
- [ ] #7458 **G10** Diagnostics + integrations-health parity (browser mode included)
- [ ] #7459 **G11** About/version info + onboarding on web
- [ ] #7460 **G12** Decide + document desktop-only exemptions (pose editing, document library, Docker, MCP config, terminal/REPL/Jupyter, MATLAB suite, dashboards) in the registry
- [ ] #7461 **G13** Manifest tile web-reachability contract (route / native-window / unavailable) + parity test

## Suggested execution order

1. **F1 + F4** (registry + honest endpoints) — establishes ground truth and stops the bleeding.
2. **F2** (service extraction) — unblocks the big three (G1, G2, G7).
3. **G13 + G4** — cheap, high-visibility honesty fixes in the web shell; **F3** alongside.
4. **G1 → G3 → G2** — the analysis/export core that makes the web app scientifically useful.
5. **G5–G11** in any order; **G12** whenever F1 lands.

## Standing policy (the "maintainable in the future" part)

- Every PR that adds a user-facing feature to the PyQt6 app must add/update a `feature_parity.json` entry (`parity`, `gap`+issue, or `exempt`+reason). CI enforces (F1).
- New backend features land in a shared service + API endpoint first; both UIs are thin consumers (per `docs/architecture/dual-gui-architecture-review.md` §3.1, ADR-0028).
- No web UI affordance ships ahead of its backend (F4 rule); no Pydantic payload gets a hand-written TS twin (F3 rule).

## Prior art consulted (not duplicated)

- ADR-0028 multi-window decision; Feb-2026 `launcher_parity_assessment.md` (Phase 1 manifest unification — done); closed #1162–#1178, #6896 (reconnect-restart accepted as explicit-status), #5470 (desktop chat context); UI/UX epic #7444 (visual quality, a11y, theming tokens — adjacent issues cross-referenced, not re-filed).
