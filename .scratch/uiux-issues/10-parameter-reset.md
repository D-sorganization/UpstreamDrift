## Problem (verified in code)

User-set simulation parameters are silently reset to engine defaults whenever `ParameterPanel` remounts (navigate away from /simulation and back) or the selected engine changes.

`ui/src/components/simulation/ParameterPanel.tsx`:

- Lines 53–59: local `useState` seeded from hardcoded `ENGINE_DEFAULTS`, never from the store.
- Lines 62–65: `useEffect` resets duration/timestep to defaults on engine change.
- Lines 68–79: `notifyChange` effect fires on mount, pushing those defaults up via `onChange`.

`ui/src/pages/Simulation.tsx` lines 118–123: `handleParameterChange` → `replaceParameters(params)` — so the mount-time defaults **overwrite the user's values in `useSimulationStore`**.

Reproduction: set Duration to 10s on Simulation → navigate to Dashboard → back to Simulation → panel shows 3.0s and the store now holds 3.0s; the next run uses 3s.

Related defects to fix in the same PR:

- `ParameterPanel.tsx` line 102: `onChange={(e) => setDuration(parseFloat(e.target.value) || 3.0)}` — clearing the field mid-edit snaps the value to 3.0 while typing. Same pattern on the timestep input.
- The same defaults are duplicated in three places: `ENGINE_DEFAULTS` (ParameterPanel ~17–43), `DEFAULT_PARAMETERS` (`ui/src/stores/useSimulationStore.ts` ~45), and hardcoded `duration: 3.0, timestep: 0.002` inside the WebSocket start payload (`ui/src/api/client.ts` ~122–130). The `SimulationParameters` interface is also defined in both ParameterPanel and the store.

## Fix

1. Make `useSimulationStore` the single source of truth. `ParameterPanel` should read `parameters` from the store (or receive them as a controlled `value` prop) and write changes through `setParameters`; delete the local mirror state and the mount-time `notifyChange` effect entirely.
2. On engine change, apply engine defaults **only** if the user hasn't customized values (track a `dirty` flag in the store), or better: prompt nothing and keep user values, exposing a "Reset to engine defaults" button.
3. Move `ENGINE_DEFAULTS` into `useSimulationStore.ts` (exported), import it in ParameterPanel; remove the hardcoded `duration/timestep` literals in `client.ts` — the caller already passes the real config (Simulation.tsx ~131–135), so `client.ts` should only fill `live_analysis` fallback or throw if duration/timestep are absent.
4. Export `SimulationParameters` from the store only; import the type in ParameterPanel.
5. Fix numeric input handling: keep the raw string in local state while editing, parse/clamp on blur or on valid parse; never substitute a default mid-edit. Disallow NaN reaching the store.
6. Add tests: (a) remounting ParameterPanel does not change store values; (b) engine switch preserves user-modified values; (c) clearing the duration field then typing "12" yields 12, not 3.0.

## Acceptance criteria

- Repro above no longer resets values; tests in (6) pass.
- `grep -rn "3.0" ui/src/api/client.ts` shows no hardcoded simulation defaults.
- One definition of `SimulationParameters` and one defaults table.

Part of the UI/UX overhaul epic (see tracking issue).
