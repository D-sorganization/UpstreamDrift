## Problem

Two slider→API paths misbehave under rapid input:

1. **Speed slider sends one API call per drag tick.** `ui/src/pages/Simulation.tsx` ~69–76: `handleSpeedChange` awaits `setSpeed(value)` directly from the range input's `onChange`. A single drag fires dozens of POSTs to the speed endpoint; responses can land out of order, settling on a stale speed.
2. **ActuatorSlider debounce reports success regardless of outcome.** `ui/src/components/simulation/ActuatorSlider.tsx` ~26–41: the debounced callback fires `onValueChange(...)` fire-and-forget and unconditionally `setIsDragging(false)`. If the send fails, the slider keeps showing the user's value while the backend holds the old one — silent divergence.

## Fix

1. Extract a shared hook `ui/src/utils/useDebouncedCommand.ts`:
   ```ts
   function useDebouncedCommand<T>(
     send: (v: T) => Promise<{ success: boolean; error?: string }>,
     delayMs: number,
     onError: (msg: string) => void,
   );
   ```
   - updates local/UI state immediately, debounces the network call (~150–300ms), clears pending timeout on unmount;
   - tracks a monotonically increasing request id and ignores/resolves only the latest, so out-of-order responses can't apply stale values;
   - on failure invokes `onError` AND reverts the displayed value to the last confirmed backend value.
2. Use it in `Simulation.tsx` for `handleSpeedChange` (toast on failure already exists via `showError`).
3. Refactor `ActuatorSlider` onto the same hook: on send failure, surface the error (prop callback up to ActuatorPanel's toast) and snap `dragValue` back to `actuator.value`.
4. Tests: fake timers — rapid changes produce a single send with the final value; failed send reverts the displayed value and reports the error; unmount during pending debounce does not call send.

## Acceptance criteria

- Dragging the speed slider end-to-end issues ≤2 requests (verify in devtools/network or unit test).
- A failed actuator/speed send shows an error and the UI value reverts to the confirmed value.

Part of the UI/UX overhaul epic (see tracking issue).
