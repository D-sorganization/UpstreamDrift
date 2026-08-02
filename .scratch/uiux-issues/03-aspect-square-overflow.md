## Problem

Square viewports are constrained only by a fixed max width, so on narrow windows the `aspect-square` box (forced height = width) overflows the viewport.

**Affected:**

- `ui/src/pages/PuttingGreen.tsx` (~line 553):
  ```tsx
  <div className="w-full max-w-[600px] aspect-square p-4">
  ```
- `ui/src/pages/MotionCapture.tsx` (~line 484): same pattern with `max-w-2xl`.

At ~375–700px widths (or a narrow Tauri window), the square is wider than the area remaining next to the fixed `w-80` sidebar, causing horizontal overflow and a clipped canvas.

## Fix

1. Constrain by both dimensions so the square always fits:
   ```tsx
   <div className="w-full max-w-[min(600px,90vw,calc(100vh-8rem))] aspect-square p-2 sm:p-4">
   ```
   (the `calc(100vh-…)` term keeps the square from exceeding available height once toolbars are accounted for — adjust the offset to the page's actual header/footer height), OR use responsive steps: `max-w-[90vw] sm:max-w-[400px] md:max-w-[600px]`.
2. Reduce padding on small widths: `p-2 sm:p-4`.
3. Apply the identical fix to both pages and extract the wrapper if a shared `SquareViewport` helper is cleaner.

## Acceptance criteria

- At any window width ≥360px, the square viewport on PuttingGreen and MotionCapture fits fully inside the visible main area with no horizontal scrollbar.
- Desktop appearance (≥1280px) unchanged.

Part of the UI/UX overhaul epic (see tracking issue).
