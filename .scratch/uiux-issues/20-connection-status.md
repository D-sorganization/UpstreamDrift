## Problem

The WebSocket connection status `'lost'` (set when the socket drops mid-simulation) persists in the UI after it is no longer meaningful. Navigating away from /simulation and back still shows the "connection lost" treatment (`ui/src/components/ui/ConnectionStatus.tsx`, ~lines 49–55 config; state lives in the simulation client hook `ui/src/api/client.ts`), even though a fresh connection attempt would succeed. Users see a stale scary banner and may not realize they can just press Start.

## Fix

1. Reset connection status to `'disconnected'`/`'idle'` when the simulation client hook unmounts (page navigation) — add it to the existing unmount cleanup in `client.ts`.
2. When the user presses Start (which opens a fresh socket), immediately transition `lost → connecting` so the stale banner clears at the moment of action.
3. In `ConnectionStatus`, pair the `lost` state with an inline "Reconnect" affordance (or hint text "press Start to reconnect") instead of a dead-end message.
4. Test: simulate close → status `lost`; unmount/remount hook → status not `lost`; `start()` clears `lost`.

## Acceptance criteria

- Stale `lost` status never survives navigation or a new start; tests pass.

Part of the UI/UX overhaul epic (see tracking issue).
