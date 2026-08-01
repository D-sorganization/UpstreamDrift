## Problem

The UI is hard-coded dark with no theming infrastructure:

- `ui/src/index.css` (~lines 92–127) defines `--sidekick-color-*` CSS variables with dark values only; there is no `prefers-color-scheme` handling and no light counterpart.
- `tailwind.config.js` does not set `darkMode`, and `grep -rn "dark:" ui/src` returns zero hits — components style raw dark colors (`bg-gray-900` etc.) directly.

A dark-first app is fine, but with no token layer every future theme/contrast adjustment means editing hundreds of class strings.

## Fix (infrastructure, not a visual redesign)

1. Set `darkMode: 'class'` in `tailwind.config.js` and add `class="dark"` at the root (`index.html` or `App.tsx`) so the existing appearance is preserved.
2. Extend the `--sidekick-color-*` variable set into a complete token sheet (canvas, surface, surface-raised, border, text, text-muted, primary, success, warning, danger) and map them in `tailwind.config.js` `theme.extend.colors` so utilities like `bg-surface` / `text-muted` become available.
3. Migrate the shared primitives (`Button`, `Input`, `Card` — see UI-primitives issue) and the page shells to the token utilities first; deeper migration can follow incrementally.
4. Add a light-mode variable block (`:root:not(.dark)` or `@media (prefers-color-scheme: light)`) with reasonable values, even if no toggle is exposed yet — it validates that the token layer actually works.
5. Document the tokens in `ui/README.md`.

## Acceptance criteria

- `darkMode: 'class'` configured; app renders identically to today by default.
- Token utilities exist and are used by the shared primitives and page shells.
- Flipping the root class to light produces a legible (not necessarily polished) light theme — proof the plumbing works.

Part of the UI/UX overhaul epic (see tracking issue). Depends on the UI-primitives and color-system issues.
