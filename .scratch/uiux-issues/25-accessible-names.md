## Problem

Batch of missing accessible names/announcements (WCAG 1.1.1, 4.1.2, 4.1.3) found in review:

1. **Icon-only buttons without `aria-label`** — e.g. expand/collapse glyph buttons in `ui/src/components/analysis/AnalysisPanel.tsx` (~50) and tree expanders in `ModelTree.tsx` (`▼`/`▶` text glyphs). Some buttons (EngineSelector power buttons, HelpPanel close) already have labels — bring the rest up to that standard. Inventory with: `grep -rn "<button" ui/src | grep -v aria-label` and review each icon-only hit.
2. **Loading spinners not announced** — `Loader2` spinners carry `aria-hidden="true"` with no sibling status text, e.g. `EngineSelector.tsx` ~16, `HelpPanel.tsx` ~185. Pattern to apply:
   ```tsx
   <span role="status">
     <Loader2 className="animate-spin" aria-hidden="true" />
     <span className="sr-only">Loading…</span>
   </span>
   ```
3. **Decorative emoji unlabeled** — e.g. `📊` in `TreeDiffModal.tsx` ~34 → `aria-hidden="true"` (sweep: `grep -rnP "[\x{1F300}-\x{1FAFF}]" ui/src`).
4. **Tree items without accessible names/states** — `ModelTree.tsx` `role="treeitem"` nodes should carry `aria-label` (name + type) and `aria-selected`/`aria-expanded`.
5. **Connection status for AT** — `ConnectionStatus.tsx` (~70): colored pulse dot + text exists; add `role="status"` + `aria-live="polite"` on the container so status _changes_ are announced.

## Fix

Work through the five categories above; keep `aria-hidden="true"` on the icons themselves and put the name on the interactive element. Add a vitest-axe smoke test (`button-name`, `aria-*` rules) over the main pages to lock it in.

## Acceptance criteria

- No icon-only button without an accessible name (axe `button-name` rule green on Simulation, ModelExplorer, DataExplorer, Chat).
- Spinners announce; tree items expose name/selected/expanded.

Part of the UI/UX overhaul epic (see tracking issue).
