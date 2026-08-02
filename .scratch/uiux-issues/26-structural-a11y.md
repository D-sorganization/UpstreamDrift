## Problem

Structural accessibility gaps (WCAG 2.4.1, 1.3.1, 2.3.3):

1. **No skip link** — keyboard users must Tab through the entire sidebar on every page to reach the main content.
2. **Heading hierarchy skips ranks** — e.g. `Simulation.tsx` sidebar starts at `h2` with no page `h1`; several pages jump h1→h3. (Coordinates with the typography-scale issue, which defines the classes.)
3. **Animations not gated on `prefers-reduced-motion`** — only the dead Vite template CSS checks it. Ungated: `animate-in fade-in`/`zoom-in` (TreeDiffModal), `slide-in-from-right-5` (Toast), `animate-ping` (ConnectionStatus), `animate-pulse` (Scene3D overlay), various `transition-all`.

## Fix

1. Add as the first element inside the app shell:
   ```tsx
   <a
     href="#main-content"
     className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:bg-blue-600 focus:text-white focus:px-3 focus:py-2 focus:rounded"
   >
     Skip to main content
   </a>
   ```
   and give each page's `<main>` the id `main-content` (`WorkspaceShell` from the responsive-shell issue is the natural single place).
2. Heading sweep: each page gets one `h1` (visually styled or `sr-only`), sections h2, subsections h3 — no skips. Verify with axe `heading-order` rule.
3. Add a global reduced-motion rule to `ui/src/index.css`:
   ```css
   @media (prefers-reduced-motion: reduce) {
     *,
     *::before,
     *::after {
       animation-duration: 0.01ms !important;
       animation-iteration-count: 1 !important;
       transition-duration: 0.01ms !important;
     }
   }
   ```
   (Tailwind's `motion-reduce:` variants can refine individual cases later; the global rule is the safety net.)

## Acceptance criteria

- First Tab press on any page reveals the skip link; activating it focuses main content.
- axe `heading-order` green on all pages; reduced-motion rule present and effective (toasts/modals appear without animation when the OS setting is on).

Part of the UI/UX overhaul epic (see tracking issue).
