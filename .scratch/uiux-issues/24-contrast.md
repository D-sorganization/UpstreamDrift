## Problem

Multiple text/control colors fail WCAG AA contrast (4.5:1 for text, 3:1 for UI components) on the app's dark backgrounds (`gray-900`/`gray-950`):

- `text-gray-500` (#6B7280) on gray-900 ≈ 2.8:1 — used for helper text, empty states, e.g. `ui/src/components/model-explorer/InspectorPanel.tsx` ~10, `ui/src/components/simulation/ParameterPanel.tsx` help text (~line 109), many others (`grep -rn "text-gray-500" ui/src`).
- Disabled buttons `bg-gray-600 text-gray-400` ≈ 2.1:1 — `ui/src/components/simulation/SimulationControls.tsx` ~54; `text-gray-500 bg-gray-700` in `AnalysisPanel.tsx` (~60); `disabled:text-gray-500` in `ModelExplorer.tsx` ~212.
- Range slider track `bg-gray-600` on dark panel ≈ 2.5:1 — `ui/src/components/simulation/ActuatorSlider.tsx` ~84.

Reference on bg #111827: gray-300 ≈ 4.2:1 (AA ✓), gray-400 ≈ 3.2:1 (large text only), gray-500 ≈ 2.8:1 (fail), gray-600 ≈ 2.0:1 (fail).

## Fix

1. Global sweep: replace foreground `text-gray-500` with `text-gray-400` minimum (or `text-gray-300` for body-size helper text). Keep gray-500 only for decorative/disabled-adjacent uses that are not the sole information carrier.
2. Disabled controls: standardize on `disabled:bg-gray-700 disabled:text-gray-400 disabled:cursor-not-allowed` (≥3:1) — encode this in the shared `Button` primitive (see UI-primitives issue) rather than per-call-site.
3. Slider tracks: `bg-gray-500` plus explicit `accentColor: '#3B82F6'` so the filled portion and thumb are clearly visible.
4. Add an automated check: a vitest + `vitest-axe` (or jest-axe) pass over representative rendered pages asserting no `color-contrast` violations, so regressions are caught.

## Acceptance criteria

- `grep -rn "text-gray-500" ui/src` hits reviewed: none remain as primary text on dark backgrounds.
- Disabled buttons readable (≥3:1); axe contrast test green on Simulation, DataExplorer, ModelExplorer.

Part of the UI/UX overhaul epic (see tracking issue).
