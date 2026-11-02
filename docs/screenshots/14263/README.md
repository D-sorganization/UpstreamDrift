# #14263: Non-Blocking First-Run Onboarding

Captured with Playwright/Chromium against local Vite dev servers, with the
launcher manifest and `/api/v1/about/onboarding` stubbed from fixtures and no
stored dismissal: `before` is main at the branch base, `after` is this branch.
`1280` runs at 1280×800 with a mouse; `390` runs at 390×844 with `isMobile`
and `hasTouch`.

| State         | Before                                               | After                                                                  |
| ------------- | ---------------------------------------------------- | ---------------------------------------------------------------------- |
| Onboarding    | Full-screen `aria-modal` dialog over a dark backdrop | Docked `region` panel above the footer; no backdrop, no modal          |
| Home content  | Hidden and unclickable until the dialog is closed    | Visible immediately; tiles and Launch Simulation take clicks           |
| Panel actions | Inside the dialog                                    | "Get Started" and the opt-out checkbox stay visible; only cards scroll |

Measured after: at 1280 the panel sits at left 16, 448 wide, ending at 688,
above the footer at 735. At 390 it spans 358 wide and ends at 732, above the
footer at 759. A hit test at the centre of Launch Simulation reaches the
button at both widths (it did not before). The dismissal key
`upstreamdrift.onboarding.dismissed` is unchanged, so users who already
dismissed the dialog do not see the panel again.
