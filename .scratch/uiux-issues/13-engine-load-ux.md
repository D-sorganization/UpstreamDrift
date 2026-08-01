## Problem

Engine load/unload flow has three robustness/UX gaps:

1. **Double-click sends duplicate requests.** `ui/src/components/simulation/EngineSelector.tsx` ~133–152: the Load/Unload buttons call `onLoad`/`onUnload` with no in-flight state; rapid clicks fire multiple POSTs and the UI can flicker if responses land out of order.
2. **Auto-select can select an engine that is no longer loaded.** `ui/src/stores/useEngineStore.ts` ~230–233: after `loadEngineApi` resolves, the store auto-selects the engine without re-checking its current `loadState` — with two windows (or a backend-side unload racing the load) the UI ends up with a selected engine the backend doesn't have, and Start fails confusingly.
3. **No confirmation for unload.** Unloading is disruptive (kills the engine mid-session) and is a single small icon click with no confirm step.

## Fix

1. Track in-flight operations: `loadState` already exists — set it to `'loading'`/`'unloading'` synchronously when the request starts, disable the corresponding button while in flight, and render a spinner (`Loader2`) in place of the Power icon. The store-level guard should also early-return if an operation for that engine is already pending (idempotence even if a button misses the disabled state).
2. In `requestLoad`'s auto-select, re-check the just-updated state before selecting:
   ```ts
   const justLoaded = get().engines.find((e) => e.name === engineName);
   if (!get().selectedEngine && justLoaded?.loadState === "loaded") {
     set({ selectedEngine: engineName });
   }
   ```
3. Add a lightweight inline confirm for unload (two-step button: click → "Confirm unload? ✓ / ✕", auto-cancel after 3s), not a modal. Skip confirm if no simulation has been run with that engine this session, if that's easy to detect; otherwise always confirm.
4. Tests: double-click load issues one request; unload requires confirm; auto-select skipped when loadState is not 'loaded'.

## Acceptance criteria

- Buttons visibly disabled with spinner during load/unload; duplicate requests impossible from the UI.
- Unload requires a second click to confirm.
- Tests above pass in `EngineSelector.test.tsx` / store tests.

Part of the UI/UX overhaul epic (see tracking issue).
