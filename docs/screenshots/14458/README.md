# #14458: Metric and Imperial Units on the React Putting Page

Captured with Playwright/Chromium against local Vite dev servers: `before` is
main at the branch base, `after` is this branch. Desktop runs at 1280×900 and
phone at 390×844. `/api/v1/*` and `/health` were stubbed, and the 3D simulate
endpoint returned a fixed SI response, so both unit systems show the same
physics converted only for display.

| File                                                 | Shows                                              |
| ---------------------------------------------------- | -------------------------------------------------- |
| `before-{desktop-1280,phone-390x844}.png`            | Main: fixed m/s, yards and mm, no unit selector    |
| `after-{metric,imperial}-desktop-1280.png`           | Units selector: m/s, m, mm or mph, ft, in readouts |
| `after-{metric,imperial}-phone-390x844-controls.png` | Phone controls drawer with the Units selector      |
| `after-{metric,imperial}-phone-390x844-readout.png`  | Phone readout drawer after a simulated putt        |

Stimp stays in feet in both systems, by the stimpmeter convention.
