## Problem

Every multi-panel page hardcodes fixed-width sidebars with no responsive breakpoints. The standard layout is `w-80` (320px) left sidebar + `flex-1` main + `w-72` (288px) right sidebar = 608px of fixed chrome. Below ~1024px the main content area collapses to unusably narrow; at tablet widths the layout is effectively broken.

**Affected pages (all in `ui/src/pages/`):**

- `Simulation.tsx` (~lines 238–240, 363): `w-80` left aside + `w-72` right aside
- `ModelExplorer.tsx` (~lines 189, 362)
- `DataExplorer.tsx`
- `AnalysisTools.tsx` (~line 50)
- `MotionCapture.tsx`
- `PuttingGreen.tsx`
- `VideoAnalyzer.tsx`
- `Terrain.tsx`
- `DatasetGenerator.tsx`
- `CharacterBuilder.tsx` (~line 187)

Representative code (Simulation.tsx):

```tsx
<div className="flex h-screen bg-gray-900 overflow-hidden">
  <aside className="w-80 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto flex-shrink-0 z-10">
  ...
  <aside className="w-72 bg-gray-800 border-l border-gray-700 p-4 flex-shrink-0 z-10 overflow-y-auto">
```

## Fix

Create ONE canonical responsive shell component and migrate all pages to it, rather than patching each page ad hoc.

1. Create `ui/src/components/layout/WorkspaceShell.tsx` that accepts `leftPanel`, `rightPanel`, `children` (main), and optional `bottomPanel` props and implements:
   ```tsx
   <div className="flex flex-col lg:flex-row h-screen bg-gray-900 overflow-hidden">
     <aside className="hidden lg:flex lg:w-72 xl:w-80 flex-col flex-shrink-0 bg-gray-800 border-r border-gray-700 overflow-y-auto">
       {leftPanel}
     </aside>
     <main className="flex-1 flex flex-col min-w-0 min-h-0">{children}</main>
     <aside className="hidden xl:flex xl:w-72 flex-col flex-shrink-0 bg-gray-800 border-l border-gray-700 overflow-y-auto">
       {rightPanel}
     </aside>
   </div>
   ```
2. Below `lg`, render the side panels as toggleable overlay drawers (slide-in `fixed inset-y-0` panels with a backdrop and a toolbar toggle button), so all controls remain reachable on narrow windows. A simple `useState` + two buttons in a top bar is sufficient; no new dependency needed.
3. Migrate each of the 10 pages listed above to `WorkspaceShell`, deleting their bespoke aside markup.
4. Keep page-specific content (panel contents) untouched — this issue is only about the frame.

## Acceptance criteria

- At 768px and 1024px window widths, every listed page shows a usable main content area (>50% of viewport width) with side panels reachable via drawers.
- At ≥1280px the current three-column appearance is preserved.
- All 10 pages render through the shared `WorkspaceShell`; `grep -r "w-80 bg-gray-800" ui/src/pages` returns no hits.
- Existing page tests still pass; add a test for `WorkspaceShell` rendering panels and toggling drawers.

Part of the UI/UX overhaul epic (see tracking issue).
