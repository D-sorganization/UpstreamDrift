Tracking issue for the 2026-06-11 comprehensive UI/UX review of the React/Vite/Tailwind frontend (`ui/`). Five parallel review passes (layout/responsiveness, design-system consistency, functional bugs, accessibility, app shell/architecture) produced 28 actionable issues. Goal: a modern, responsive, professional-grade application.

Line numbers in the child issues reference the working tree near `main` as of 2026-06-11; treat them as approximate anchors.

## Foundation first (other issues build on these)

- [ ] #7420 Shared UI primitives (Button/Input/Select/Badge/Card)
- [ ] #7421 Color system standardization (gray-only neutrals, blue primary, semantic statuses)
- [ ] #7415 Responsive WorkspaceShell for all multi-panel pages

## P0/P1 — broken or clearly wrong

- [ ] #7430 No 404 catch-all route; no scroll reset
- [ ] #7432 Branding: "ui" title, Vite favicon, no per-route titles
- [ ] #7424 Simulation parameters silently reset on navigation (verified)
- [ ] #7416 Missing min-h-0/min-w-0 on 3D viewport flex containers
- [ ] #7433 No route-level code splitting (three.js in initial bundle)
- [ ] #7438 Modal keyboard accessibility (TreeDiffModal, HelpPanel)
- [ ] #7439 WCAG contrast failures (gray-500 text, disabled states)

## P2 — bugs and fragility

- [ ] #7417 aspect-square viewport overflow
- [ ] #7418 Frankenstein-mode sidebar width
- [ ] #7419 Layout polish batch
- [ ] #7425 Slider debounce / failure feedback
- [ ] #7426 Chat retry attachments + stuck streaming indicator
- [ ] #7427 Engine load/unload hardening
- [ ] #7428 Toast stacking/dedup/announcement
- [ ] #7429 Polynomial panel empty-coefficient submission
- [ ] #7434 Per-route error isolation + production logging
- [ ] #7435 Stale "connection lost" status
- [ ] #7436 Tauri backend orphaned on window close
- [ ] #7437 Duplicate zustand v4+v5
- [ ] #7442 HelpfulField violations invisible to user

## P2 — consistency, a11y, polish

- [ ] #7422 Typography scale
- [ ] #7423 Theming tokens / darkMode infrastructure
- [ ] #7440 Accessible-name batch (icon buttons, spinners, tree items)
- [ ] #7441 Skip link, heading order, prefers-reduced-motion
- [ ] #7443 Affordances: disabled reasons, shortcut hints, undo

## Suggested sequencing for implementing agents

1. #7420 + #7421 (primitives + colors) — everything else consumes them.
2. #7415 + #7416 (shell + canvas sizing) — structural layout.
3. #7430, #7432, #7433 (routing/branding/splitting) — independent quick wins, can run in parallel with 1–2.
4. Functional bug cluster (#7424–#7429, #7434–#7437, #7442) — independent of each other; parallelizable.
5. A11y and polish (#7438–#7441, #7417–#7419, #7422, #7423, #7443).

Per repo policy: claim a lease before starting any child issue, one focused PR per issue (`Closes #N`), tests in the same PR.
