## Problem

Two major versions of zustand are bundled:

- `zustand@^5.0.13` — direct dependency (`ui/package.json` line ~36)
- `zustand@4.5.7` — transitive via `tunnel-rat@0.1.2` (`ui/package-lock.json`, `node_modules/tunnel-rat/node_modules/zustand`)

This double-bundles the library and risks subtle store-instance mismatches if any code path resolves the v4 copy.

## Fix

1. Check whether a newer `tunnel-rat` (or its parent, likely `@react-three/drei`) declares zustand v5 — `npm ls zustand` to see the chain, then upgrade the parent if available.
2. Otherwise add an override in `ui/package.json`:
   ```json
   "overrides": { "tunnel-rat": { "zustand": "$zustand" } }
   ```
3. `npm install` and verify `npm ls zustand` shows a single deduped v5 entry; run the UI test suite and a manual smoke of the Simulation page (drei's `Html`/tunnel features are the consumer).

## Acceptance criteria

- `npm ls zustand` shows one version; build output contains a single zustand module; tests green.

Part of the UI/UX overhaul epic (see tracking issue).
