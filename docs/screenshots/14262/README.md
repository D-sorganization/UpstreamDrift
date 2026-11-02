# #14262: Planned Tools Kept Off Home and Primary Workspaces

Captured with Playwright/Chromium against local Vite dev servers, with the
onboarding panel dismissed and the launcher manifest stubbed from a fixture:
`before` is main with the canonical-core tiles at `experimental`; `after` is
this branch with the same fixture marking them `planned`, which is what the
backend now serves from `src/config/models.yaml`. `1280` runs at 1280×800
with a mouse; `390` runs at 390×844 with `isMobile` and `hasTouch`.

| Surface                   | Before                                                              | After                                       |
| ------------------------- | ------------------------------------------------------------------- | ------------------------------------------- |
| Home summary              | 58 tiles · 5 engines · 53 tools                                     | 56 tiles · 5 engines · 51 tools             |
| Home canonical-core tiles | 2 clickable tiles (`tile-canonical_core_*`)                         | None                                        |
| Results & Compare         | Canonical Core Comparison and Estimation offered with **Open Tool** | Neither listed                              |
| Developer & Research      | Neither listed                                                      | Both listed, still reachable for developers |

Neither width scrolls horizontally before or after. The home captures show
the summary count; the canonical-core tiles sat below the fold on main, so the
Results & Compare captures are the visible difference.
