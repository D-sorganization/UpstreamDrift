# #14460: Noise Channels and Dispersion Batches on the React Putting Page

Captured with Playwright/Chromium against local Vite dev servers: `before` is
main before #14532 (which made no UI change), `after` is this branch. Desktop runs at 1280×900 and
phone at 390×844, in imperial units. The putting-analytics routes were served
by the real shared core (`/dispersion`, `/make-spread`), so every plotted value
comes from `service_dispersion`; the 3D simulate route was not called.

| File                                      | Shows                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| `before-{desktop-1280,phone-390x844}.png` | Main: no noise channels, batches or make-% spread on the simulator     |
| `after-desktop-1280.png`                  | Seven noise channels, 40/100/1000 batch buttons and a 100-putt scatter |
| `after-desktop-1280-spread.png`           | The 0–45 ft make-% spread from `/make-spread` on the 2% green          |
| `after-phone-390x844-readout.png`         | Phone readout drawer with the channels, batch result and rest scatter  |
