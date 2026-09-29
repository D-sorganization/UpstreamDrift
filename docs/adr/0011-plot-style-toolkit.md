# ADR 0011: Plot-Style Toolkit

- Status: Accepted
- Date: 2026-05-08
- Decision Makers: UpstreamDrift core maintainers
- Related Issues/PRs: EPIC [#4796](https://github.com/D-sorganization/UpstreamDrift/issues/4796),
  child issues [#4803](https://github.com/D-sorganization/UpstreamDrift/issues/4803)–
  [#4813](https://github.com/D-sorganization/UpstreamDrift/issues/4813),
  this ADR closes [#4812](https://github.com/D-sorganization/UpstreamDrift/issues/4812)

## Context

Marker styling — the visual recipe that says "this marker is a 6-px violet
sphere with a black 0.5-px outline" — was scattered across three desktop
tools and three render backends:

- The **C3D Viewer**'s 2D and 3D plot tabs hard-coded scatter `marker=`,
  `s=`, `c=` per call site, with magic-number sizes and palette colors
  inlined alongside the data path.
- The **starting-pose matcher**'s live-view controller built its own
  matplotlib styling helpers, sibling-implementations of the viewer's,
  not a shared call.
- The **cross-engine dashboard** had its own per-trajectory color cycler
  for engine overlays, with no relationship to the viewer or the matcher.

That meant three places to touch every time a styling concept (e.g.
"add a `star` shape", "switch the default palette") crossed tool
boundaries, and it foreclosed a feature we actually needed:
**data-driven coloring** — coloring markers by clubhead speed, by
ground-reaction-force magnitude, by per-frame error — was impossible
without per-tool ad-hoc code, because no tool had a notion of "channel"
or "colormap" outside the matplotlib symbol space.

The body-part visualisation toolkit ([ADR 0008](0008-body-part-viz-toolkit.md))
solved the _segment-shape_ drift problem with a shared package; this ADR
solves the parallel _marker-style_ drift problem with the same pattern.

Constraints:

- Backend-agnostic: matplotlib for the desktop tools, PyQtGL for tools
  that need GPU-rate redraws. No matplotlib types in the public API.
- Frozen public dataclasses, Design-by-Contract validation in
  `__post_init__`, total `resolve` methods (NaN inputs map to a
  configured fallback color, never raise).
- Pre-computed 256-entry colormap LUTs so per-frame resolution during
  animation is allocation-free beyond the output array.
- Persistence: every user customisation round-trips through a versioned
  JSON schema and survives a v1 → v2 migration.
- Optional GUI surface: `widgets/` requires PyQt6 but the rest of the
  package must import in a headless environment.

## Decision

Ship `src/shared/python/plot_style/` as the canonical marker-styling
stack for every tool that draws markers. The package defines four
runtime-checkable Protocols, three frozen `ColorScale` variants, a
single `MarkerStyle` dataclass, and the registries that wire them
together.

**Canonical record.** `markers.py` defines a frozen `MarkerStyle`
dataclass:

```python
from src.shared.python.plot_style import MarkerStyle, MarkerShape, StaticColor

style = MarkerStyle(
    shape=MarkerShape.SPHERE,
    size_px=6.0,
    edge_color="#000000",
    edge_width=0.5,
    fill_color=StaticColor("#1f77b4"),
    opacity=1.0,
)
# Validated on construction:
#   - size_px and edge_width are finite and non-negative,
#   - opacity in [0, 1],
#   - edge_color and fill_color are parseable,
#   - CUSTOM_MESH shape pairs with a CustomMeshSpec.
```

**Color model.** `colors.py` defines three `ColorScale` variants — the
union covers every styling case currently in use:

| Variant           | Use-case                                                            |
| ----------------- | ------------------------------------------------------------------- |
| `StaticColor`     | One constant color (most marker groups).                            |
| `PaletteColor`    | Categorical pick from a named matplotlib palette (engine overlays). |
| `DataDrivenColor` | Channel value → vmin/vmax normalisation → colormap LUT sample.      |

`MarkerStyle.fill_color` accepts any of the three; `__post_init__`
rejects anything else.

**Protocols** (`contracts.py`):

| Protocol              | Purpose                                                 |
| --------------------- | ------------------------------------------------------- |
| `MarkerRenderer`      | `add_markers / update_frame / update_style / remove`    |
| `MarkerShapeRenderer` | `style -> (vertices, faces)` mesh in marker-local frame |
| `ColorResolver`       | `scale -> RGBA` via `resolve_one` and `resolve_array`   |

All three are `@runtime_checkable` so call sites and tests can fail
fast.

**Registries.** Three module-level dispatch tables let callers go from
"a thing the user picked" to "a thing the renderer can use" without
isinstance ladders:

- `COLORMAP_REGISTRY` (in `registry.py`, surfaced via
  `get_colormap` / `list_colormaps` /
  `register_custom_colormap`) — `ColormapId` enum and registered
  `CustomColormap` names → `matplotlib.colors.Colormap`. Semantic
  aliases (`VELOCITY → PLASMA`, `FORCE → INFERNO`,
  `ACCELERATION → TURBO`, `HEIGHT → VIRIDIS`,
  `GENERIC_DIVERGING → COOLWARM`) keep call sites readable when the
  underlying matplotlib pick changes.
- `SHAPE_REGISTRY` (in `shapes/__init__.py`) — `MarkerShape` enum →
  default `MarkerShapeRenderer` factory. Adding a new built-in shape
  means dropping a module under `shapes/` and adding one entry here.
- `RESOLVER_REGISTRY` (in `resolvers/__init__.py`) — `ColorScale`
  dataclass → matching `ColorResolver` class.

**Backends.** Two renderer implementations ship in-tree:

- `MatplotlibMarkerRenderer` is the canonical 2D / 3D backend; it
  satisfies the `MarkerRenderer` Protocol and additionally exposes a
  stateless `draw(ax, positions, style, colors)` helper for callers
  that already have a resolved RGBA array in hand.
- `PyQtGLMarkerRenderer` ships alongside for tools that need GPU-rate
  redraws (the C3D Viewer's 3D tab during animation playback). Both
  implement the same Protocol so the same `MarkerStyle` targets either
  backend with no call-site change.

**Persistence.** `persistence.py` defines a `PlotStyleSet` JSON schema
versioned at `SCHEMA_VERSION = 1`. The set is a name-keyed dict of
`PlotStyleSpec` entries, each of which serialises one `MarkerStyle`.
Unknown fields round-trip; a v0-style payload (a bare list of
`MarkerStyle` dicts without a `version` key) is auto-migrated on load.

**Preset library.** `preset_library.py` ships four curated themes
(`default`, `scientific_violet`, `monochrome`, `high_contrast`) as v1
JSON documents under `plot_style/presets/`. `PresetLibrary.default()`
loads every built-in; `BUILTIN_PRESET_NAMES` is the public listing for
UI dropdowns.

**Widgets.** `widgets/` ships four PyQt6 editor widgets that compose
into a complete styling UI: `ColorPicker`, `ColormapPicker`,
`MarkerStylePicker`, `DataChannelEditor`. They are imported lazily so
headless consumers (CI, tests, the API server) can still
`import plot_style` without PyQt6 installed.

## Alternatives Considered

### A. Extend matplotlib's defaults directly

Configure `matplotlib.rcParams` and a per-tool `mplstyle` file.

Rejected. matplotlib's styling sits inside the matplotlib type system —
`Line2D`, `PathCollection`, `Axes3D`. The PyQtGL backend has none of
those types; styling that lives in `rcParams` cannot reach the GL
renderer at all. We would still have written `MarkerStyle` for the GL
path and would then have two parallel styling systems to keep in sync —
the very drift this ADR exists to fix.

### B. Per-tool styling configs

Let each tool keep its own styling helpers behind a shared YAML / JSON
spec listing the conventional shapes and colors so the three
implementations can converge by hand-coordination.

Rejected. The three implementations had already drifted before the
epic — that is the situation the epic exists to fix. A spec without an
enforcement mechanism is the same trap repeated. DRY violations across
tools are exactly the case `AGENTS.md` section A is meant to catch.

### C. Adopt an external library (plotly, vispy, k3d)

Reuse one of the existing scientific-plot libraries in place of writing
our own.

Rejected. None of the candidates ship the combination we need:
matplotlib + PyQt-GL backends behind the same Protocol, frozen
data-driven `ColorScale` with a 256-entry LUT cache, named semantic
colormap aliases tied to kinematic / kinetic data families, and
JSON-roundtrippable presets. The renderers themselves are thin
matplotlib / pyqtgraph wrappers; the value of the toolkit is the
contract, not the rendering code.

## Consequences

### Positive

- **Cross-tool consistency.** The C3D Viewer (2D + 3D tabs), the
  matcher's live-view controller, and the cross-engine dashboard share
  one source of truth for what a marker looks like.
- **Data-driven coloring is now a first-class feature.** Any tool that
  plots markers can color them by clubhead speed, ground-reaction
  force, per-frame error, or any other channel without writing per-tool
  styling code. The bulk path (`resolve_array`) is allocation-free
  beyond the output and clears 60 fps for 1000 frames × 32 markers.
- **Portability.** A user customisation saved by one tool re-opens
  identically in any other. The four built-in presets are good
  defaults for publication, screen-share, and accessibility-focused
  setups.
- **Extensibility.** Adding a new shape kind means dropping one module
  under `shapes/` plus an entry in `SHAPE_REGISTRY`. Adding a colormap
  means a single `register_custom_colormap` call. No tool has to
  change.

### Negative

- One extra package between the tools and matplotlib / PyQtGL. Small
  in practice — the package is pure Python with numpy + matplotlib
  imports — but it is still a layer.
- `widgets/` introduces an optional PyQt6 dependency. Consumers that
  only need the data layer must avoid a top-level
  `from plot_style.widgets import ...` and instead defer to the lazy
  re-export from the package root.

### Follow-ups

- Issue #4805 — `MatplotlibMarkerRenderer` (landed).
- Issue #4806 — `PyQtGLMarkerRenderer` (landed).
- Issue #4807 — Qt widgets (landed).
- Issue #4808 — matcher integration (landed).
- Issue #4810 — cross-engine dashboard integration (landed).
- Issue #4811 — C3D Viewer integration (landed).
- Issue #4813 — preset library + theme JSONs (landed).
- Future: web-frontend bridge so the same `MarkerStyle` JSON drives the
  WASM viewer.

## Validation

- Every public dataclass (`MarkerStyle`, `CustomMeshSpec`, `StaticColor`,
  `PaletteColor`, `DataDrivenColor`, `DataChannel`, `CustomColormap`,
  `PlotStyleSpec`, `PlotStyleSet`) runs `__post_init__` validation.
  Frozen + validated is the design contract; renderers and resolvers
  can trust the shape of what they receive.
- The Protocols are `@runtime_checkable` so tests assert that each new
  resolver / renderer / shape satisfies the contract before the type
  checker sees it.
- Cross-tool integration tests pin the contract: a fixture builds a
  `PlotStyleSet`, round-trips it through JSON, drives a
  `DataDrivenColor` resolver from a `DataChannel`, renders via
  `MatplotlibMarkerRenderer`, and asserts pixel parity with the
  PyQtGL backend. Any divergence between tools surfaces as a test
  failure, not as a runtime UI bug.
- CI gates: `ruff check`, `ruff format --check`, file-size budget,
  pytest with the coverage `fail_under` from `pyproject.toml`.

## See also

- User guide: [`docs/user_guide/plot_style/quickstart.md`](../user_guide/plot_style/quickstart.md)
- Data-driven coloring: [`docs/user_guide/plot_style/data_driven_coloring.md`](../user_guide/plot_style/data_driven_coloring.md)
- Colormap-author guide: [`docs/user_guide/plot_style/colormap_author_guide.md`](../user_guide/plot_style/colormap_author_guide.md)
- [ADR 0008 — Body-Part Visualisation Toolkit](0008-body-part-viz-toolkit.md)
  — predecessor decision that introduced the same Protocol-driven
  pattern for body-segment shapes.
