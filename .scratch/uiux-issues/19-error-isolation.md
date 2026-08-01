## Problem

1. **One crash takes down the whole app.** `ui/src/main.tsx` (~lines 13–14) wraps the entire app in a single `ErrorBoundary`. A render error in any page replaces everything with the generic fallback; the user cannot navigate to a working page because the router is inside the dead tree.
2. **Console noise in production.** `ui/src/api/client.ts` (~lines 159, 164, 194) and `ui/src/components/ui/ErrorBoundary.tsx` (~line 39) call `console.error/warn` unconditionally.

## Fix

1. Keep the top-level boundary as last resort, but add a route-level boundary so navigation survives page crashes. Create a small wrapper used in `App.tsx`:
   ```tsx
   const page = (el: ReactNode) => (
     <ErrorBoundary resetKeys={[location.pathname]}>{el}</ErrorBoundary>
   );
   ```
   Simplest robust approach: one `ErrorBoundary` around `<Routes>` that resets when the pathname changes (add a `resetKey` prop to ErrorBoundary that, when changed, clears the error state) — the user can then use browser/sidebar navigation to recover. The fallback UI should include a "Back to Dashboard" link (plain `<a href="/">` works even when the router context is unavailable).
2. Include the failing route path in the fallback message to aid bug reports.
3. Add `ui/src/utils/logger.ts` with `error/warn/info` that no-op (or forward to a diagnostics buffer) when `import.meta.env.PROD`; replace the raw console calls in `client.ts` and `ErrorBoundary.tsx`. Note: use `import.meta.env`, not `process.env`, in Vite client code.
4. Tests: throwing page component shows fallback; navigating to another route clears it.

## Acceptance criteria

- A crash on one page no longer bricks navigation; fallback offers a working way home.
- No unconditional console.error/warn in `ui/src` production paths (`grep -rn "console\." ui/src` reviewed; test files exempt).

Part of the UI/UX overhaul epic (see tracking issue).
