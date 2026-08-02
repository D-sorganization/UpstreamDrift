## Problem

3D canvas containers sit inside nested flex layouts with `flex-1` but without `min-h-0`/`min-w-0`. Flexbox's default `min-height: auto` prevents the flex child from shrinking below its content size, so the canvas overflows its container or squashes adjacent panels (e.g. the analysis plot strip) when space is tight.

**Affected:**

- `ui/src/pages/Simulation.tsx` (~lines 316–324) — wrapper around `<Scene3D>`:
  ```tsx
  <main className="flex-1 flex flex-col relative min-w-0">
    <div className="flex-1 relative bg-gray-950">   {/* missing min-h-0 */}
      <Scene3D engine={...} />
    </div>
  ```
- `ui/src/pages/ModelExplorer.tsx` (~lines 355–360) — wrapper around `ModelPreviewViewport`
- `ui/src/pages/MotionCapture.tsx` (~line 483)
- `ui/src/pages/PuttingGreen.tsx` (~line 552)
- `ui/src/pages/CharacterBuilder.tsx` (~line 173+)

## Fix

1. Add `min-h-0` (and `min-w-0` where the container is in a row-direction flex) to every `flex-1` wrapper that directly contains a WebGL canvas component (`Scene3D`, `URDFViewer`, `ModelPreviewViewport`, `SkeletonRenderer`, R3F `<Canvas>`).
2. Audit with: `grep -rn "flex-1" ui/src/pages ui/src/components/visualization ui/src/components/model-explorer` and check each hit whose children include a canvas component.
3. Canonical pattern to adopt everywhere:
   ```tsx
   <div className="flex-1 relative min-h-0 min-w-0 bg-gray-950">
     <Canvas ... />
   </div>
   ```
4. Verify each canvas component handles container resize (R3F does automatically; for any manual three.js renderer confirm a ResizeObserver is attached).

## Acceptance criteria

- Resizing the window on Simulation with the bottom LivePlot strip visible never pushes the plot off-screen or lets the canvas overflow its bounds.
- Same check on ModelExplorer, MotionCapture, PuttingGreen, CharacterBuilder.
- No layout regression at full-screen desktop size.

Part of the UI/UX overhaul epic (see tracking issue).
