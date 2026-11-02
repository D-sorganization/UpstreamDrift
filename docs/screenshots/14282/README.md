# #14282: Wide Content Scrolls Instead of Clipping

Captured with Playwright/Chromium against local Vite dev servers: `before` is
main at the branch base, `after` is this branch. API responses were stubbed.
Lift Baseline uses the real `pack_parity_baseline.json` receipt rendered
through `baseline_view.py`.

"Overflowing" counts visible elements whose right edge passes the viewport
and that are not inside a narrower horizontal scroll container. "Page h-scroll"
is the sideways scroll of the page-level content area.

| Page           | Width | Overflowing (before → after) | Page h-scroll (before → after) |
| -------------- | ----- | ---------------------------- | ------------------------------ |
| Lift Baseline  | 390   | 549 → 0                      | 397px → 0                      |
| Lift Baseline  | 1280  | 0 → 0                        | 0 → 0                          |
| Analysis Tools | 390   | 0 → 0                        | 0 → 0                          |
| Analysis Tools | 1280  | 0 → 0                        | 0 → 0                          |
| Golf Simulator | 390   | 17 → 0                       | 0 → 0                          |
| Golf Simulator | 1280  | 0 → 0                        | 0 → 0                          |

With stubbed values, the Analysis Tools grids did not overflow before the
change; the live-data clipping reported in the issue is covered by
`responsiveLayout.test.ts`. That test also covers Terrain and Data Explorer,
which need a live terrain or dataset to render their grids.

At 1280 the Lift Baseline tables are still wider than their card. Before the
change the card clipped them silently, so the counts above read 0; now each
table scrolls horizontally inside its card. Analysis Tools and Golf Simulator
look the same at 1280.
