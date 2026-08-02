## Problem

Affordance gaps that make the app feel opaque to non-expert users:

1. **Disabled buttons don't explain themselves.** E.g. `ModelExplorer.tsx` ~212 (compare disabled until both trees load), `EngineSelector.tsx` unload disabled when the engine is selected (has `title` but nothing for AT). Users see dead buttons with no reason.
2. **Keyboard shortcuts are undiscoverable.** `SimulationControls` binds Space/Escape/Period; HelpPanel binds F1 — none of these are hinted in the UI except F1.
3. **Frankenstein mode has no undo.** Tree copy/swap operations in ModelExplorer are irreversible until page reload.

## Fix

1. Standardize the "disabled with reason" pattern: when a button is disabled for a knowable reason, set both `title` and `aria-describedby` to a short reason ("Load both models to compare", "Deselect engine before unloading"). If the shared `Button` primitive exists (UI-primitives issue), add a `disabledReason?: string` prop that wires both automatically.
2. Add shortcut hints: append the key to the control's `title`/`aria-label` (e.g. "Start simulation (Space)") and add a "Keyboard shortcuts" section to `helpData.ts` so HelpPanel documents Space/Escape/Period/F1 in one place.
3. Frankenstein undo: keep a bounded history stack (e.g. last 20 tree states) in ModelExplorer state; add Undo button + Ctrl+Z handler scoped to the page. Each mutating operation pushes the previous `targetTree` snapshot.
4. Tests: disabled compare button exposes its reason; Ctrl+Z reverts a copy operation.

## Acceptance criteria

- All knowingly-disabled buttons in ModelExplorer/EngineSelector/SimulationControls expose a reason via tooltip + aria.
- HelpPanel lists all global/page shortcuts; Frankenstein operations undoable.

Part of the UI/UX overhaul epic (see tracking issue).
