## Problem

Batch of small layout/formatting execution bugs found in the UI/UX review. Each is small; together they read as unpolished. Fix all in one PR.

1. **LivePlot strip height hardcoded** — `ui/src/pages/Simulation.tsx` (~line 357): `<div className="h-64 bg-gray-800 border-t border-gray-700 p-2">`. On short windows 256px is half the screen. Change to `h-40 sm:h-48 lg:h-64`.
2. **ModelExplorer tree area clips instead of scrolling** — `ui/src/pages/ModelExplorer.tsx` (~line 230): the tree container uses `overflow-hidden`; a model with many nodes is cut off with no scrollbar. Change the inner tree column to `overflow-y-auto` (keep `min-w-0` on it).
3. **DiagnosticsPanel can render under sidebars** — `ui/src/components/ui/DiagnosticsPanel.tsx` (~line 123): `absolute bottom-10 right-0 w-80 ...` has no z-index while page asides use `z-10`. Add `z-50`.
4. **LauncherDashboard grid density on small windows** — `ui/src/components/simulation/LauncherDashboard.tsx` (~line 385): `px-6 py-6` + `gap-4` waste space on narrow widths. Change to `px-3 sm:px-4 md:px-6 py-3 md:py-6` and `gap-2 sm:gap-3 md:gap-4`.
5. **CanonicalCoreShell padding** — `ui/src/pages/CanonicalCoreShell.tsx` (~line 28): `px-6 py-8` → `px-4 sm:px-6 py-4 sm:py-8`.
6. **Chat page wrapper duplication** — `ui/src/pages/Chat.tsx` (~lines 12–15): redundant nested flex wrappers around `ChatPanel`; simplify to a single `w-full h-screen flex justify-center p-2 sm:p-4` wrapper and let ChatPanel own its `max-w-3xl`.
7. **Dead Vite-template CSS** — `ui/src/App.css` (lines 1–46): unused logo/card styles from the scaffold. Delete the dead rules (keep the file only if anything imports needed styles; check `grep -rn "App.css" ui/src`).

## Acceptance criteria

- Each item above addressed as described; visual smoke check of Simulation, ModelExplorer, Launcher, CanonicalCore, Chat at 1024px and 1920px.
- No `overflow-hidden` clipping of the ModelExplorer tree with a 100+ node model.

Part of the UI/UX overhaul epic (see tracking issue).
