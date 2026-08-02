## Problem

Three color-consistency defects make the app look stitched together:

1. **Slate vs gray family mix.** `ui/src/pages/CanonicalCoreShell.tsx` (~lines 27–49) uses `bg-slate-950 / bg-slate-900 / border-slate-800 / text-slate-100` while the entire rest of the app uses `gray-*` (e.g. `Simulation.tsx` ~238–362, `DataExplorer.tsx` ~381–624). The slate page visibly differs in hue.
2. **Fragmented primary accent.** Primary actions are green in `SimulationControls.tsx` (`bg-green-600 hover:bg-green-700`), purple in `AnalysisTools.tsx` ~188 (`bg-purple-600 hover:bg-purple-500`), and blue in `EngineSelector.tsx` ~80–86 / `ParameterPanel.tsx` ~162. There is no recognizable primary color.
3. **Inconsistent semantic status colors.** `Toast.tsx` ~30–42 uses `yellow-*` for warnings while `DataExplorer.tsx` ~493 uses `amber-*`; `ErrorBoundary.tsx` ~64 uses a gray card with faint red border for errors. Icon status colors are hardcoded per-component (`ConnectionStatus.tsx` ~14–56, `EngineSelector.tsx` ~16–24).

## Fix

1. **Canonical neutrals: `gray-*` only.** Replace all `slate-*` classes in `CanonicalCoreShell.tsx` with the `gray-*` equivalents. Verify zero remaining: `grep -rn "slate-" ui/src`.
2. **Canonical primary: blue.** `blue-600` base / `blue-700` hover for primary action buttons. Keep green strictly for "run/start/success" semantics, red for destructive. Change `AnalysisTools.tsx` purple buttons to primary blue. Encode these as the `variant` colors of the shared `Button` component (see UI-primitives issue).
3. **Semantic palette** — standardize and apply in `Toast.tsx`, `DataExplorer.tsx`, `ErrorBoundary.tsx`, `ConnectionStatus.tsx`:
   - success: `green-*`, error: `red-*`, warning: `amber-*` (not yellow), info: `blue-*`
   - surface pattern: `bg-{color}-900/80 border-{color}-600 text-{color}-100`
4. Add the semantics to `tailwind.config.js` `theme.extend.colors` (e.g. `primary`, `success`, `warning`, `danger` aliases) so future code references tokens, not raw palette names.
5. Add a stylelint/ESLint guard or a simple CI grep test that fails on new `slate-`, `zinc-`, `neutral-`, or `purple-` utility classes in `ui/src` (allowlist file for intentional exceptions).

## Acceptance criteria

- `grep -rn "slate-\|purple-" ui/src` → no hits (or only allowlisted).
- CanonicalCore pages match the rest of the app's neutrals.
- Warning UI uses amber consistently in Toast and DataExplorer.
- Guard test in place.

Part of the UI/UX overhaul epic (see tracking issue).
