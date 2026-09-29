# ADR 0030: C3D Viewer Renderer Backend

## Status

Accepted

## Context

The C3D 3D viewer currently renders marker playback through matplotlib in
`viewer_3d_tab.py`. That path already supports scrubbing, speed control, looped
playback, marker groups, view presets, and skeleton overlays, but it is
CPU-bound during interactive playback. Issue #7214 requires a top-of-the-line
interactive path targeting 60 fps on CMU C3D clips around 45 markers and 1000
frames while preserving the existing matplotlib fallback until feature parity is
verified.

Two candidate families were considered:

- `pyqtgraph.opengl` in-process with Qt. This matches the existing desktop
  launcher model, reuses the optional PyQtGL marker renderer in
  `src/shared/python/plot_style/renderers/pyqtgl.py`, and avoids introducing a
  realtime transport boundary for the first production slice.
- WebGL/Three.js in the Tauri/React UI. This remains attractive for long-term
  web parity, but it requires a marker-frame streaming protocol and a second UI
  implementation before it can replace the desktop tab.

## Decision

Use `pyqtgraph.opengl` as the first GPU backend for the desktop C3D viewer and
keep matplotlib as the fallback backend. Backend selection is explicit and
contract-tested in `viewer_3d_backend.py`: prefer PyQtGL when requested and
available, otherwise use matplotlib. The PyQtGL path must satisfy the parity
checklist before it can become the default replacement for the existing tab:
scrubbing, speed control, loop playback, marker groups, view presets, and
skeleton overlay.

The first implementation slice intentionally adds only the backend decision
contract. The follow-up slice should wire a PyQtGL scene adapter into
`Viewer3DTab` without growing that file further; new renderer orchestration
belongs in sibling modules.

## Consequences

- The desktop launcher can get a GPU-rate path without waiting for Tauri/WebGL
  transport work.
- Matplotlib remains available for headless tests, environments without
  `pyqtgraph.opengl`, and parity fallback.
- The backend contract gives future benchmark tests a stable place to assert the
  60 fps target and CMU acceptance dimensions.
- A later WebGL ADR or extension can reuse the same parity checklist and should
  define the marker-frame streaming boundary before adding React rendering code.

## Validation

- `tests/unit/apps/test_viewer_3d_backend.py` covers PyQtGL selection,
  matplotlib fallback, explicit fallback preference, dataset-shape DbC, and the
  required parity checklist.
