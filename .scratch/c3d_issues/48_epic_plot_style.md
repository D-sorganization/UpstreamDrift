# [EPIC] `plot_style` — fleet-wide marker / color / colormap library + data-driven coloring

## Vision

Today every plotting surface in the repo (C3D Viewer 2D plots, the Motion-Match Preview 3D matcher, per-engine viz modules, the cross-engine dashboard, the trajectory overlays) hard-codes its own marker shapes, sizes, colors, and palettes. Users can't:

1. **Pick a marker shape** (sphere / cube / cross / star / diamond / custom STL) consistently across plots.
2. **Pick a marker size + edge** in a way that persists.
3. **Pick a color** from a UI palette + custom hex.
4. **Apply a colormap** that drives marker color from a *data channel* — e.g. clubhead-velocity-magnitude colors the marker red at peak speed, blue at rest.

The user has asked for a single shared toolkit that does all four, in a way that's reusable across the C3D Viewer, the matcher, the URDF generator's preview, the cross-engine dashboard, and any future plotting surface.

This epic creates a new shared package `src/shared/python/plot_style/` that **complements** (not replaces):

- The existing `src/shared/python/plot_theme/` (matplotlib stylesheet themes)
- The existing `src/shared/python/theme/` (Qt/CSS application themes)
- The existing `src/shared/python/body_part_viz/` (segment / mesh rendering — separate concern)

`plot_style` owns: **per-marker style + colormap + data-driven coloring**. The other three packages own their respective concerns. Clean separation.

## Architecture

```
                ┌─────────────────────────────────────────────────┐
                │ src/shared/python/plot_style/                   │
                │                                                 │
                │  Contracts (Protocols + dataclasses):           │
                │   • MarkerStyle, MarkerShape (StrEnum)          │
                │   • ColorScale, ColorMapId (StrEnum)            │
                │   • DataChannel (per-frame scalar source)       │
                │   • PlotStyleSpec, PlotStyleSet                 │
                │                                                 │
                │  Marker shapes (matplotlib + pyqtgraph adapters)│
                │   • SphereMarker, CubeMarker, CrossMarker,      │
                │     StarMarker, DiamondMarker, CustomMeshMarker │
                │                                                 │
                │  Color resolution:                              │
                │   • StaticColor (single hex)                    │
                │   • PaletteColor (from named palette)           │
                │   • DataDrivenColor (channel → colormap → rgba) │
                │                                                 │
                │  Colormap registry:                             │
                │   • viridis, plasma, magma, inferno (built-in)  │
                │   • velocity, force, acceleration (semantic)    │
                │   • custom palettes loadable from JSON          │
                │                                                 │
                │  UI widgets:                                    │
                │   • MarkerStylePicker (Qt widget)               │
                │   • ColorPicker (Qt widget; hex + palette)      │
                │   • ColormapPicker (Qt widget; preview swatch)  │
                │   • DataChannelEditor (channel selector +       │
                │     min/max bounds + colormap)                  │
                │                                                 │
                │  Renderers (backend-agnostic adapters):         │
                │   • MatplotlibMarkerRenderer                    │
                │   • PyQtGLMarkerRenderer                        │
                │                                                 │
                │  Persistence: JSON v1 schema                    │
                └────────────┬────────────────────────────────────┘
                             │
   ┌────────────┬────────────┼────────────────┬────────────────────┐
   ▼            ▼            ▼                ▼                    ▼
 C3D Viewer  Matcher     URDF Generator  Cross-engine        Per-engine
 (3D + 2D)   live view   preview         dashboard           viz modules
                         (link visuals)
```

## Children (14 issues)

| # | Title | Type | Priority |
|---|---|---|---|
| 1 | feat(plot-style): core contracts + dataclasses (MarkerStyle / ColorScale / DataChannel / PlotStyleSpec) | architecture | high |
| 2 | feat(plot-style): marker shape primitives — sphere, cube, cross, star, diamond, custom-mesh | feature | high |
| 3 | feat(plot-style): color resolution layer — Static / Palette / DataDriven | feature | high |
| 4 | feat(plot-style): colormap registry + named semantic colormaps (velocity / force / acceleration) | feature | high |
| 5 | feat(plot-style): DataChannel abstraction (per-frame scalar source from any np.ndarray of shape (T,) or (T, M)) | feature | high |
| 6 | feat(plot-style): MarkerStylePicker / ColorPicker / ColormapPicker / DataChannelEditor Qt widgets | feature | high |
| 7 | feat(plot-style): MatplotlibMarkerRenderer (3D scatter + 2D scatter backends) | feature | high |
| 8 | feat(plot-style): PyQtGLMarkerRenderer (GLScatterPlotItem backend) | feature | medium |
| 9 | feat(plot-style): PlotStyleSet JSON v1 persistence with theme presets | feature | high |
| 10 | feat(c3d-viewer): integrate plot_style into 2D + 3D plot tabs (Markers / Analog / 3D Viewer) | integration | high |
| 11 | feat(matcher): integrate plot_style into live view controller marker rendering | integration | high |
| 12 | feat(cross-engine-dashboard): integrate plot_style into trajectory overlays | integration | medium |
| 13 | test(plot-style): comprehensive TDD coverage + golden image snapshots | testing | high |
| 14 | docs(plot-style): ADR + user guide + colormap-author guide | docs | medium |

## Cross-cutting principles (binding for every child PR)

### TDD

- Every child PR: failing tests first, then implementation.
- ≥85% line coverage on new code; ≥75% branch.

### DbC

- Use `src/shared/python/core/contracts/decorators.py`.
- Dataclasses validate in `__post_init__`; raise `ValueError`/`TypeError` with descriptive messages.
- Every public function documents postconditions.

### LOD

- No method chains > 2 levels.
- Public API returns data, not widgets — UI integration code is the only place that touches Qt or matplotlib.

### DRY

- Shared utilities live in one place. No duplication across engine viz modules.
- Existing `plot_theme` colors stay where they are; `plot_style` references them via theme tokens.

### Orthogonality

- Adding a new marker shape doesn't touch colormap / data-channel / renderer code.
- Adding a new colormap doesn't touch marker shapes.
- Adding a new renderer doesn't touch shapes / colormaps / data channels.

### Generic naming

No vendor / lab / person / study names anywhere in code, error messages, log messages, docstrings, or test names.

### Performance

- Per-frame marker style update at ≥ 60 fps for 38 markers × 654 frames.
- DataDriven color update at ≥ 60 fps too: pre-compute the lookup-table at load time, slice per frame.

## Sequencing

```
1. Contracts ──┬─► 2. Marker shapes ────┐
               ├─► 3. Color resolution ─┤
               ├─► 4. Colormaps ────────┼──► 7. Mpl renderer ──┐
               ├─► 5. DataChannel ──────┤                       │
               └─► 6. Qt widgets ───────┴──► 8. PyQtGL renderer ┤
                                                                 │
                                                                 ▼
                                                         9. Persistence
                                                                 │
                       ┌───────────────────────────┬─────────────┴───────────────┐
                       ▼                           ▼                              ▼
              10. C3D Viewer integ        11. Matcher integ           12. Cross-engine integ
                       │                           │                              │
                       └───────────────────────────┴───────┬──────────────────────┘
                                                           ▼
                                                   13. Tests + snapshots
                                                           │
                                                           ▼
                                                    14. Docs + ADR
```

## Data-driven coloring — the headline capability

```python
from src.shared.python.plot_style import (
    DataDrivenColor, DataChannel, ColormapId,
    MarkerStyle, MarkerShape,
)

# Build a data channel from clubhead speed (T,) magnitude:
speed_channel = DataChannel.from_array(
    name="clubhead_speed",
    values=clubhead_speed_per_frame,   # shape (T,)
    unit="m/s",
)

# A color that maps clubhead speed via the "velocity" semantic colormap:
color = DataDrivenColor(
    channel=speed_channel,
    colormap=ColormapId.VELOCITY,
    vmin=0.0,                          # auto if None
    vmax=55.0,                         # auto if None
    nan_color="#888888",               # sub-NaN color
)

# Compose marker style:
style = MarkerStyle(
    shape=MarkerShape.SPHERE,
    size_px=12.0,
    edge_color="#000000",
    edge_width=0.5,
    fill_color=color,                  # ← data-driven!
)

# Apply via the renderer:
renderer.update_marker_style(handle, style, frame_idx=t)
```

The same pattern works for:
- Per-marker scalar (shape `(T, M)`) → per-marker color per frame.
- Joint torque magnitude → bone color.
- Force-plate force magnitude → ground reaction marker color.

This pattern lays the groundwork for the future "scale colors on custom color scales based on magnitude of selected parameters" the user explicitly asked for.

## Out of scope for v1

- Editable colormaps (drag stops in a dialog) — v2.
- Multi-channel composite coloring (color from channel A, alpha from channel B) — v2.
- WebGL / Three.js renderer for the React/Tauri UI — separate effort.
- Animated colormaps (Jet → Viridis transition) — pure aesthetic.
- Marker rotation per frame — current scope is fixed-orientation markers.

## Definition of done (epic)

- All 14 children closed.
- C3D Viewer 2D + 3D plot tabs use `plot_style` for marker rendering.
- Motion-Match Preview matcher uses `plot_style` for marker + segment rendering.
- Cross-engine dashboard's trajectory overlays use `plot_style`.
- A user can: open the C3D Viewer → select a marker → click "Style…" → see the MarkerStylePicker, ColorPicker, ColormapPicker, DataChannelEditor in one dialog → preview live → save.
- ≥ 85% line / ≥ 75% branch coverage on `plot_style`.
- Performance: 60 fps with data-driven coloring on the canonical 38-marker × 654-frame driver test.
- ADR + user guide + colormap-author guide on `main`.
