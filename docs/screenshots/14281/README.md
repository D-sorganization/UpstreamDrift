# #14281: Touch Targets and a Minimum Text Size

Captured with Playwright/Chromium against local Vite dev servers: `before` is
main at the branch base, `after` is this branch. `390` runs at 390×844 with
`isMobile`, `hasTouch` and DPR 2, so `(pointer: coarse)` matches. `1280` runs
at 1280×900 with a mouse. The launcher manifest and the Lift Baseline receipt
were stubbed; other endpoints were left unanswered.

"Small" counts visible buttons, links, selects, inputs, tabs and summaries
under 44×44px. A checkbox or radio is measured by the label that wraps it. The
1×1 sr-only skip link is excluded. "Tiny" counts text nodes outside SVG/canvas
whose computed font size is under 12px.

| Page          | 390 small (before → after) | 390 tiny (before → after) |
| ------------- | -------------------------- | ------------------------- |
| Dashboard     | 12/71 → 0/71               | 271 → 0                   |
| Workspace     | 10/11 → 0/11               | 0 → 0                     |
| Dataset       | 6/7 → 0/7                  | 0 → 0                     |
| Data Explorer | 3/4 → 0/4                  | 0 → 0                     |
| Putting Green | 1/3 → 0/1                  | 0 → 0                     |
| Lift Baseline | 1/3 → 0/3                  | 288 → 0                   |
| Settings      | 11/11 → 0/11               | 0 → 0                     |

At 1280 with a mouse the touch-target rules do not apply, so small-target
counts are unchanged (for example Dashboard 13/71, Settings 11/11). The only
1280 difference is the text floor: Dashboard tiny text 271 → 0 and Lift
Baseline 303 → 0. Putting Green's control count varies between runs because
its panels depend on unanswered API calls.
