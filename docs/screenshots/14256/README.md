# #14256: Plain-Language API Errors on the Golf Simulator

Captured with Playwright/Chromium against local Vite dev servers with no API
running: `before` is main at the branch base, `after` is this branch. `1280`
runs at 1280×800 with a mouse; `390` runs at 390×844 with `isMobile` and
`hasTouch`. With no backend, `/tools/golf-simulator/destinations` falls
through to the SPA and returns `index.html`, which reproduces the reported
banner exactly.

| State                         | Before                                                    | After                                                                                                    |
| ----------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Error banner                  | `Unexpected token '<', "<!doctype "... is not valid JSON` | `Simulator API not reachable` with a Retry button                                                        |
| Lifecycle and replay controls | Disabled, no explanation                                  | Disabled, with "Connect to a simulator to enable shot and replay controls." linked by `aria-describedby` |

Six controls were checked at both widths: Prepare Shot, Arm for Impact,
Trigger Impact Submit, Play, Pause and Stop. All are disabled before and after;
after, each one is described by the reason note. Neither width scrolls
horizontally.
