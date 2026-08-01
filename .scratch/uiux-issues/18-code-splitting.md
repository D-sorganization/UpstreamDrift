## Problem (verified in code)

`ui/src/App.tsx` lines 1–14 eagerly imports all 13 page components, so the initial bundle includes every page — including three.js / @react-three/fiber / @react-three/drei (~hundreds of KB) pulled in by `Scene3D`, `URDFViewer`, and `ModelPreviewViewport` — even when the user opens only the Dashboard or Chat. No `React.lazy`, no `Suspense`, no manual chunks.

## Fix

1. Convert all page imports in `App.tsx` to `React.lazy`:
   ```tsx
   const SimulationPage = lazy(() =>
     import("./pages/Simulation").then((m) => ({ default: m.SimulationPage })),
   );
   ```
   (pages use named exports — either map them as above or add default exports).
2. Wrap `<Routes>` in `<Suspense fallback={<PageLoadingFallback />}>` with a small centered spinner component consistent with the app theme.
3. Add a vendor chunk for 3D libs in `vite.config.ts`:
   ```ts
   build: {
     rollupOptions: {
       output: {
         manualChunks: {
           three: ["three", "@react-three/fiber", "@react-three/drei"];
         }
       }
     }
   }
   ```
4. Verify `npm run build` output: the entry chunk should shrink substantially and `three-*.js` should load only on pages that render a 3D view (check the network panel from the Dashboard route).
5. Make sure the ErrorBoundary in `main.tsx` still catches lazy-load failures (chunk load errors should show the boundary fallback, not a blank screen).

## Acceptance criteria

- Entry JS chunk no longer contains three.js (inspect `dist/assets` or use `rollup-plugin-visualizer`).
- Navigating to each route works with a brief themed loading fallback; tests pass.

Part of the UI/UX overhaul epic (see tracking issue).
