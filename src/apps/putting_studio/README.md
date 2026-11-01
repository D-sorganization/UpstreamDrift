# Standalone Putting Studio

An offline, shareable HTML studio for the putting model (epic #14141). Open
`index.html` directly in a browser: it loads nothing from a network, so it works
from `file://`, a USB stick or an email attachment.

## What It Shows

- **3D Vector Studio:** the dual-hand grip force couples, midpoint torques and
  the putter on its inclined stroke plane.
- **Ball Impact and Green Roll:** a seeded Monte Carlo batch of putts on an
  undulating green, with lip-outs from the chord capture bound.
- **Strokes-Gained Capture Chart:** the model's expected strokes against entry
  speed.
- **Equations and Turnover Guide:** the model's frames and equations.

Every number on the page is model output, not a measurement. The page says so in
its header, and it makes no calibration or strokes-gained claim.

## Layout

| Path                    | Role                                                    |
| ----------------------- | ------------------------------------------------------- |
| `index.html`            | Markup only; classic `<script src>` tags (no modules)   |
| `css/studio.css`        | Studio styles beside the vendored Tailwind build        |
| `js/putting_physics.js` | Temporary JS copy of the shared physics (`SUNSET.md`)   |
| `js/*.js`               | Scene, charts, controls, state and the JSON / agent API |
| `vendor/`               | three.js r128 and Tailwind, with hashes and licences    |

The single source of truth for the physics is
`src/shared/python/putting_dynamics`. The JS copy stays in step through the
shared fixtures and parity tests in `tests/apps/putting_studio/`.

## Budgets

`tests/apps/putting_studio/test_studio_budget.py` holds the studio to a
1,000,000-byte offline bundle (835 KB measured) and a 2 s headless script load
(163 ms measured), and checks that the studio's own code makes no network
access. A replacement for the JS physics must meet the same budgets.

## Agent API

`window.PuttingSimulatorAPI` exposes the state, scenario export and load, the
Stimpmeter roll and a JSON-RPC `tools/list` / `tools/call` entry point inside
the page.

## Open Work

Physics bridge and PyQt6 parity: #14200 (`tools.putting_studio` in
`src/config/feature_parity.json`).
