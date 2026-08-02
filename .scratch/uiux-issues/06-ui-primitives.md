## Problem

There is no shared component layer for basic controls, so every page hand-rolls buttons, inputs, selects, and badges with divergent Tailwind strings. Concrete divergence found in review:

**Buttons** — paddings `py-2 px-4` / `py-1 px-2` / `py-1.5 px-2.5` / `py-1.5 px-3`; radii `rounded` / `rounded-md` / `rounded-lg` / `rounded-full`:

- `ui/src/components/simulation/SimulationControls.tsx` ~49–59
- `ui/src/components/simulation/EngineSelector.tsx` ~154–182
- `ui/src/components/ui/HelpPanel.tsx` ~339–346
- `ui/src/components/model-explorer/JointManipulator.tsx` ~40

**Inputs/selects** — some with `border border-gray-600`, some `border-none`; focus `ring-2` vs `ring-1` vs `focus-visible:outline`; paddings `py-2` / `py-1.5` / `py-1`:

- `ui/src/components/simulation/ParameterPanel.tsx` ~104–106
- `ui/src/components/simulation/ActuatorPanel.tsx` ~273
- `ui/src/pages/DatasetGenerator.tsx` ~265
- `ui/src/pages/DataExplorer.tsx` ~450

**Focus rings** — `focus:ring-2 focus:ring-offset-2` (SimulationControls ~50) vs `focus:ring-1` (ActuatorPanel ~273) vs `focus-visible:outline-2` (DataExplorer ~89).

**Panels** — `p-2` / `p-3` / `p-4` mixed across SimulationControls ~229, ActuatorSlider ~59, ForceOverlayPanel ~107, DataExplorer ~443.

## Fix

Create `ui/src/components/ui/` primitives and migrate callers:

1. `Button.tsx`:
   ```tsx
   variant: "primary" | "secondary" | "danger" | "success" | "ghost";
   size: "sm" | "md" | "lg";
   // sm: px-2 py-1 text-xs rounded
   // md: px-3 py-1.5 text-sm rounded-md (default)
   // lg: px-4 py-2 text-base rounded-md
   // all: focus:outline-none focus:ring-2 focus:ring-blue-400,
   //      disabled:bg-gray-700 disabled:text-gray-400 disabled:cursor-not-allowed
   ```
2. `Input.tsx` / `Select.tsx`: `px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50 disabled:cursor-not-allowed`, with a `size="sm"` variant (`px-2 py-1 text-xs`) for dense panels.
3. `Badge.tsx` (pill, `px-2 py-0.5 rounded-full text-xs`), `Card.tsx` (panel wrapper, `bg-gray-800 rounded-md p-4`; `p-2` only for dense inline controls).
4. Add a `.focus-ring` component class in `index.css` (`@layer components`) for one-off focusable elements.
5. Migrate the files listed above (and other ad-hoc instances found via `grep -rn "px-3 py-1.5\|py-2 px-4" ui/src`) to the primitives. Mechanical migration; do not redesign behavior.
6. Document the primitives in `ui/README.md` ("all new buttons/inputs must use components from `components/ui/`").

## Acceptance criteria

- Primitives exist with unit tests (render variants, disabled state, focus class).
- SimulationControls, EngineSelector, ParameterPanel, ActuatorPanel, DataExplorer, DatasetGenerator, HelpPanel use them.
- Visual smoke check: consistent radii and focus rings across Simulation and DataExplorer pages.

Part of the UI/UX overhaul epic (see tracking issue). Coordinates with the color-system issue (primitives should consume its tokens).
