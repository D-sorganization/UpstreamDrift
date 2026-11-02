# Mobile 3D Page Drawers (#14279)

Before (main) and after (this branch) screenshots of Simulation, Model Explorer
and Character Builder. They were captured with headless Chromium against the
Vite dev server with no backend, so the pages show their idle states and the
WebGL canvas renders blank.

## Measurements

| Page              | Viewport | Before canvas | After canvas | After toggles (px) |
| ----------------- | -------- | ------------- | ------------ | ------------------ |
| Simulation        | 390×844  | 300×150       | 390×623      | 128×44, 92×44      |
| Simulation        | 768×1024 | 160×832       | 768×771      | 128×44, 92×44      |
| Simulation        | 1280×800 | 672×544       | 672×544      | docked             |
| Model Explorer    | 390×844  | 300×150       | 390×783      | 103×44, 93×44      |
| Model Explorer    | 768×1024 | 160×1024      | 768×963      | 103×44, 93×44      |
| Model Explorer    | 1280×800 | 672×800       | 672×800      | docked             |
| Character Builder | 390×844  | 300×844       | 390×783      | 144×44             |
| Character Builder | 768×1024 | 384×1024      | 768×963      | 144×44             |
| Character Builder | 1280×800 | 896×800       | 896×800      | docked             |

The desktop layout at 1280 wide is unchanged: the docked asides are 320 and
288 px, as before.

## Files

- `before-*-desktop.png` / `after-*-desktop.png`: 1280×800.
- `before-*-phone.png` / `after-*-phone.png`: 390×844.
- `after-simulation-phone-drawer-open.png`: the Engine Controls drawer open on
  a phone.
