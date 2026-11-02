# #14255: Launch Footer Clear of the Diagnostics Badge

Captured with Playwright/Chromium against local Vite dev servers: `before` is
main at the branch base, `after` is this branch. Desktop widths run with a
mouse at 1440×900, 1280×800 and 1024×768; `390` runs at 390×844 with
`isMobile` and `hasTouch`, so `(pointer: coarse)` matches. The launcher
manifest was stubbed; other endpoints were left unanswered, so the badge reads
`? Backend`.

"Overlap" lists every visible button or link whose rectangle, clipped to its
scrolling ancestors and the viewport, intersects the fixed diagnostics badge.

| Width | Badge (before → after) | Overlap before            | Overlap after |
| ----- | ---------------------- | ------------------------- | ------------- |
| 1440  | 91×30 → 91×30          | Launch Simulation (83×30) | none          |
| 1280  | 91×30 → 91×30          | Launch Simulation (83×30) | none          |
| 1024  | 91×30 → 91×30          | Launch Simulation (83×30) | none          |
| 390   | 91×44 → 44×44          | Launch Simulation (83×44) | none          |

After the change the Launch button ends 21px left of the badge at every
desktop width and 20px left of it at 390. Below `md` the badge shows only its
status glyph; the `Backend` word stays in its accessible name.
