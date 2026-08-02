## Problem (verified in code)

`ui/index.html`:

- Line 7: `<title>ui</title>` — browser tab / Tauri window shows "ui".
- Line 5: `<link rel="icon" type="image/svg+xml" href="/vite.svg" />` — the favicon is the Vite scaffold logo.

There is also no per-route `document.title`, so every page is indistinguishable in the tab bar, window switcher, and to screen readers (WCAG 2.4.2 Page Titled).

## Fix

1. `index.html`: set `<title>Golf Modeling Suite</title>` (match the product name used in `ui/src-tauri/tauri.conf.json`) and add `<meta name="description" content="Golf swing analysis and physics simulation platform" />`.
2. Replace the favicon: add a proper app icon to `ui/public/` (reuse the Tauri icon source in `ui/src-tauri/icons/` exported as SVG/PNG) and point the `<link rel="icon">` at it. Delete `public/vite.svg` if unused elsewhere.
3. Add a tiny hook `ui/src/utils/usePageTitle.ts`:
   ```ts
   export function usePageTitle(title: string) {
     useEffect(() => {
       document.title = `${title} — Golf Modeling Suite`;
     }, [title]);
   }
   ```
   Call it in every page component (Dashboard, Simulation, Model Explorer, Putting Green, Video Analyzer, Data Explorer, Motion Capture, Terrain, Dataset Generator, Analysis Tools, Character Builder, Canonical Core (per mode), Chat, NotFound).
4. Test: rendering a page sets the expected `document.title` (one representative test is fine).

## Acceptance criteria

- Tab/window title reflects the active page; favicon shows the app icon, not the Vite logo.
- `grep -rn "vite.svg" ui/` → no hits.

Part of the UI/UX overhaul epic (see tracking issue).
