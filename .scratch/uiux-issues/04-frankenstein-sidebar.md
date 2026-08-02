## Problem

`ui/src/pages/ModelExplorer.tsx` (~line 189): enabling Frankenstein mode widens the left sidebar to a hardcoded `w-[42rem]` (672px):

```tsx
<aside className={`${frankensteinMode ? 'w-[42rem]' : 'w-80'} ... transition-all duration-300`}>
```

With the `w-72` right sidebar, that consumes ~960px of fixed chrome. At 1024–1280px window widths the 3D viewport is squeezed to a sliver; below ~1024px it is unusable.

## Fix

1. Cap the expanded width relative to the viewport:
   ```tsx
   frankensteinMode ? "w-[min(42rem,55vw)]" : "w-80";
   ```
2. Below `lg`, instead of widening in place, present the dual-tree Frankenstein view as a full-width overlay panel (or tabbed single-tree view) so the viewport isn't crushed:
   - Desktop (`lg:`+): side-by-side trees as today, with the `min()` cap.
   - Smaller: render the two trees stacked or tabbed inside a drawer that overlays the viewport.
3. Also auto-collapse the right `w-72` sidebar while Frankenstein mode is active below `xl` (it competes for the same space).

## Acceptance criteria

- At 1280px width with Frankenstein mode on, the 3D viewport retains ≥40% of the window width.
- At 1024px and below, Frankenstein mode remains fully usable (both trees reachable) without reducing the viewport to <300px.
- Transition animation preserved.

Part of the UI/UX overhaul epic (see tracking issue).
