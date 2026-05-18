# plot_style child issues — full bodies for all 14 children

## Child 02 — Marker shape primitives

### feat(plot-style): marker shape primitives — sphere, cube, cross, star, diamond, custom-mesh

Depends on contracts (#1).

Implement six classes in `src/shared/python/plot_style/shapes/`:

- `_sphere_marker.py` — UV-sphere triangulation (configurable lat/lon).
- `_cube_marker.py` — 12-triangle cube.
- `_cross_marker.py` — 3 crossed quads (Greek cross).
- `_star_marker.py` — 5-pointed star extrusion.
- `_diamond_marker.py` — bipyramid (octahedron stretched along Z).
- `_custom_mesh_marker.py` — wraps a `body_part_viz.shapes.MeshShape` for arbitrary STL/OBJ.

Each class implements `MarkerShapeRenderer` (Protocol from #1) returning vertices + faces in unit-radius local frame; size scales via `MarkerStyle.size_px`.

Tests under `tests/unit/plot_style/shapes/`:

- Vertex / face counts at default params match reference.
- Unit-bounding-box check: every shape inscribed in unit cube before scale.
- DbC: invalid params raise.

Constraints + commit + PR + cleanup pattern matches body_part_viz children.

---

## Child 03 — Color resolution layer

### feat(plot-style): color resolution layer — Static / Palette / DataDriven

Depends on contracts (#1).

Implement the `ColorResolver` Protocol with three strategies in `src/shared/python/plot_style/resolvers/`:

- `static_resolver.py` — passes through hex.
- `palette_resolver.py` — looks up index in named palettes (matplotlib + custom).
- `data_driven_resolver.py` — applies channel → vmin/vmax normalisation → colormap sampling.

Bulk path `resolve_array(scale, n_frames, n_markers)` is the perf-critical one: pre-computes a lookup-table once at load time, slices per frame.

Tests verify:

- Static: hex round-trips through resolver.
- Palette: index in range; index out of bounds raises.
- DataDriven: per-frame, per-(frame, marker), and bulk paths produce identical results.
- DataDriven NaN handling: NaN values yield `nan_color`.

≥90% line coverage.

---

## Child 04 — Colormap registry

### feat(plot-style): colormap registry + named semantic colormaps

Depends on contracts (#1).

`src/shared/python/plot_style/registry.py`:

```python
def get_colormap(cmap_id: ColormapId | str) -> matplotlib.colors.Colormap:
    """Resolve a ColormapId or custom name to a matplotlib Colormap."""

def register_custom_colormap(cmap: CustomColormap) -> None:
    """Register a CustomColormap so future get_colormap('custom_name') resolves."""

def list_colormaps() -> tuple[str, ...]:
    """Return all available colormap names (built-in + registered custom)."""
```

Semantic alias map (immutable at module load):

- VELOCITY → plasma
- FORCE → inferno
- ACCELERATION → turbo
- HEIGHT → viridis
- GENERIC_DIVERGING → coolwarm

Tests:

- All ColormapId values resolve.
- Custom colormap registration round-trips.
- Idempotent registration → no-op.
- Different colormap with same name → ValueError.

≥95% line coverage.

---

## Child 05 — DataChannel abstraction

### feat(plot-style): DataChannel abstraction (per-frame scalar source)

Depends on contracts (#1).

Already partially specified in #1's `channels.py`. This child fully implements:

- `DataChannel` dataclass with `from_array` constructor.
- `value_at(frame_idx, marker_idx=None)` — NaN for OOB; supports both 1-D and 2-D.
- `auto_range()` — returns `(min, max)` over finite values; raises `ValueError` if all NaN.
- `slice(frame_range, marker_subset=None)` — derived channel.
- Vector channels: a helper `magnitude_channel(vector_per_frame: np.ndarray, ...)` that takes shape `(T, 3)` velocity → returns `DataChannel` of magnitudes.

Tests:

- All API exercised; NaN paths covered.
- Magnitude helper round-trips against synthetic constant-velocity input.

≥95% line coverage.

---

## Child 06 — Qt widgets

### feat(plot-style): MarkerStylePicker / ColorPicker / ColormapPicker / DataChannelEditor

Depends on contracts (#1).

Widgets in `src/shared/python/plot_style/widgets/`:

- `marker_style_picker.py` — `QGroupBox` with: shape combobox, size spin, edge-width spin, edge-color picker button (opens `QColorDialog`), opacity slider. Emits `style_changed: pyqtSignal[MarkerStyle]`.

- `color_picker.py` — `QToolButton` showing the current color as a swatch; click opens `QColorDialog` + tab for palette swatches. Returns hex on selection. Emits `color_chosen: pyqtSignal[str]`.

- `colormap_picker.py` — combobox showing colormap names with mini-swatch icons (16-step gradient as the icon). Selection emits `colormap_chosen: pyqtSignal[ColormapId|str]`.

- `data_channel_editor.py` — for the data-driven coloring case: shows channel name + current vmin/vmax + colormap picker; "auto" toggles auto-detection of vmin/vmax. Emits `style_updated: pyqtSignal[DataDrivenColor]`.

Tests:

- Headless (`QT_QPA_PLATFORM=offscreen`).
- Each widget instantiated; programmatic state-change emits the right signal.

≥85% line coverage.

---

## Child 07 — Matplotlib renderer

### feat(plot-style): MatplotlibMarkerRenderer (3D scatter + 2D scatter backends)

Depends on contracts (#1), shapes (#2), resolvers (#3), colormaps (#4), channels (#5).

Implements `MarkerRenderer` Protocol for matplotlib.

Two backends:

- 3D: `Axes3D.scatter` for shape == POINT/SPHERE; `Poly3DCollection` for the rest.
- 2D: `Axes.scatter` (always — even non-sphere shapes degrade to filled patches).

Per-frame update via `set_offsets3d` (3D) or `set_offsets` (2D); never `ax.clear()`.

For `DataDrivenColor`: pre-compute the per-frame color array once at `add_markers()`, slice in `update_frame`. ≥60 fps target.

Tests:

- Headless. Each shape rendered. Each ColorScale variant rendered. Per-frame update timing ≤16 ms.

≥90% line coverage.

---

## Child 08 — pyqtgraph renderer

### feat(plot-style): PyQtGLMarkerRenderer (GLScatterPlotItem backend)

Depends on contracts (#1), shapes (#2), resolvers (#3).

Implements `MarkerRenderer` for `pyqtgraph.opengl.GLViewWidget`. Optional dependency (extra `plot-style-gl` in pyproject.toml).

Lazy-imports `pyqtgraph` — package's `__init__` doesn't load it.

For sphere markers: `GLScatterPlotItem` with per-point colors. For non-sphere shapes: `GLMeshItem` per marker (slower; fall back to sphere if shape == POINT).

Tests gated on `pytest.importorskip("pyqtgraph")`. ≥80% coverage.

---

## Child 09 — Persistence

### feat(plot-style): PlotStyleSet JSON v1 persistence with theme presets

Depends on contracts (#1) + all subsequent shape/color/colormap/channel/resolver children for round-trip coverage.

`src/shared/python/plot_style/persistence.py`:

- `PlotStyleSpec` and `PlotStyleSet` dataclasses (already in #1).
- `PlotStyleSet.load(path) / save(path)` round-trips JSON.

Theme presets shipped in `src/shared/python/plot_style/presets/`:

- `default.json` — neutral palette.
- `scientific_violet.json` — matches existing `plot_theme.themes`.
- `monochrome.json` — for print.
- `high_contrast.json` — for accessibility.

`PresetLibrary.default()` resolves preset names to `PlotStyleSet` instances.

Tests: round-trip every shape/color/colormap variant. ≥95% line coverage.

---

## Child 10 — C3D Viewer integration

### feat(c3d-viewer): integrate plot_style into 2D + 3D plot tabs

Depends on all prior children.

Three tabs touched in `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/ui/tabs/`:

- `marker_plot_tab.py` (2D markers): "Style…" button per selected marker → opens `MarkerStylePicker`.
- `analog_plot_tab.py` (2D analog): "Style…" per channel.
- `viewer_3d_tab.py` (3D): "Style…" per marker group; data-channel editor for "color by speed/force/...".

Uses `MatplotlibMarkerRenderer`. Persists to `~/.golf_modeling_suite/c3d_viewer_plot_styles.json`.

Tests: headless smoke for each tab + style-pick + persistence round-trip.

---

## Child 11 — Matcher integration

### feat(matcher): integrate plot_style into live view controller marker rendering

Depends on all prior children.

`src/tools/starting_pose_matcher/live_view_controller.py` swaps its `BodyMarkerLayer` for `MatplotlibMarkerRenderer`. Style for body markers vs club markers configurable separately. Session-schema bumps to persist.

≥85% coverage.

---

## Child 12 — Cross-engine dashboard integration

### feat(cross-engine-dashboard): integrate plot_style into trajectory overlays

`src/launchers/cross_engine_dashboard.py` uses `MatplotlibMarkerRenderer` for engine-output trajectory overlays. One color-per-engine via `PaletteColor`.

≥85% coverage.

---

## Child 13 — Tests + golden snapshots

### test(plot-style): comprehensive TDD coverage + golden image snapshots

`tests/integration/plot_style/test_renderer_snapshots.py` — committed PNG snapshots at fixed DPI for each shape × each ColorScale variant. ~0.5% RMS pixel-diff tolerance.

Performance regression: 38 markers × 654 frames data-driven coloring update timing.

Cross-tool integration: same `PlotStyleSet` JSON consumed by C3D Viewer + matcher + dashboard renderers; results consistent.

---

## Child 14 — Docs

### docs(plot-style): ADR + user guide + colormap-author guide

ADR `docs/adr/00<next>-plot-style-toolkit.md`. User guide pages:

- `quickstart.md` — picking a marker shape and color.
- `data_driven_coloring.md` — coloring markers by clubhead speed.
- `colormap_author_guide.md` — how to register a custom colormap.

AGENTS.md updated.
