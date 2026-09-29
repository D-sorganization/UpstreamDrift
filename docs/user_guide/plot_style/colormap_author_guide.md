# Plot-Style Toolkit — Colormap & Palette Author Guide

This guide walks through the workflow for _adding_ colormaps and
palettes to the plot-style toolkit — both ad-hoc registrations made at
runtime and contributions intended to ship in-tree.

For background and the design contract, see
[ADR 0011 — Plot-Style Toolkit](../../adr/0011-plot-style-toolkit.md).

## How custom colormaps are registered

A `CustomColormap` is a frozen dataclass that lists a unique name and a
sequence of `(position, hex)` stops. `__post_init__` validates that
positions lie in `[0, 1]`, are strictly increasing, and that every hex
parses through matplotlib's `is_color_like`.

```python
from src.shared.python.plot_style import (
    CustomColormap,
    register_custom_colormap,
)

cmap = CustomColormap(
    name="ud_violet_amber",
    stops=(
        (0.0, "#1a0a3a"),
        (0.25, "#5b21b6"),
        (0.5, "#a855f7"),
        (0.75, "#fbbf24"),
        (1.0, "#fef3c7"),
    ),
)
register_custom_colormap(cmap)
```

`register_custom_colormap` is _idempotent_ for the same logical
colormap — re-registering an identical `(name, stops)` is a no-op.
Re-registering a _different_ colormap under an existing name raises
`ValueError`. Names that clash with a built-in `ColormapId` value
(`viridis`, `plasma`, `velocity`, ...) are also rejected.

Once registered, the name is usable anywhere a `ColormapId` is — the
registry's `get_colormap(cmap_id)` accepts both enum values and string
custom names:

```python
from src.shared.python.plot_style import get_colormap, list_colormaps

colormap = get_colormap("ud_violet_amber")
print(list_colormaps())
# ('viridis', 'plasma', ..., 'velocity', ..., 'ud_violet_amber')
```

`unregister_custom_colormap(name)` removes a colormap from the
registry — primarily intended for tests that need to reset the global
registry between cases.

## Sampling semantics

The `DataDrivenColor` resolver pre-computes a **256-entry LUT** from
the colormap once at construction time:

```python
sample_points = np.linspace(0.0, 1.0, 256, dtype=np.float64)
self._lut = np.asarray(cmap(sample_points), dtype=np.float64)
```

Per-frame resolution is then a normalisation, a clip into `[0, 1]`, a
round to the nearest integer index, and a gather into the LUT. The
LUT size of 256 matches matplotlib's default colormap resolution and
is the inflection point for 8-bit display banding — going higher costs
memory without a perceptible quality gain, going lower introduces
visible quantisation steps on smooth gradients.

Two practical implications:

- A `CustomColormap` with thousands of stops is wasted work — the LUT
  bake will sample only 256 of them. Five to ten well-chosen stops in
  perceptually-uniform spacing are enough.
- The LUT is built with the resolved colormap (after semantic alias
  resolution), so a semantic alias like `ColormapId.VELOCITY` is
  indistinguishable in performance from its underlying built-in
  (`PLASMA`).

## Shipping a custom palette via `PaletteColor`

`PaletteColor` is the categorical sibling of `DataDrivenColor` — it
picks a single color from a named matplotlib qualitative palette by
integer index. It validates at construction that the named palette is
registered with matplotlib:

```python
from src.shared.python.plot_style import PaletteColor

drake_color = PaletteColor(palette_name="tab10", palette_index=0)
pinocchio_color = PaletteColor(palette_name="tab10", palette_index=1)
```

The palette index wraps modulo the palette size at resolve time, so
`palette_index=12` in `tab10` is deterministic (resolves to index 2).
This keeps the dataclass frozen-and-validated without forcing callers
to track palette sizes.

To ship a custom palette, register it as a `CustomColormap` with one
stop per swatch and refer to its name from a `PaletteColor`. Any
matplotlib-registered colormap (built-in or custom) is a valid
`palette_name`.

## Conventions

### Naming

`ColormapId` enum values follow matplotlib's lowercase convention
(`viridis`, `plasma`, `coolwarm`). Custom colormap names should
follow the same convention:

- All lowercase, ASCII letters / digits / underscores.
- Prefix with `ud_` for repository-shipped colormaps to avoid
  collisions with matplotlib built-ins added in future releases.
- Avoid embedding vendor / lab / person names in the identifier
  (citations in docstrings are fine — see ADR 0006).

### Perceptually-uniform recommendations

When the data is _quantitative_ (continuous numeric channel) the
colormap should be perceptually uniform — equal differences in
underlying value should look like equal differences in color. The
matplotlib built-ins that satisfy this for sequential data are
`viridis`, `plasma`, `magma`, `inferno`, and `cividis`; for diverging
data, `coolwarm` is the codebase default.

The semantic aliases ship the perceptually-uniform pick for each data
family — prefer them over the underlying matplotlib name at call
sites:

| Semantic alias      | Underlying built-in | Use-case                      |
| ------------------- | ------------------- | ----------------------------- |
| `VELOCITY`          | `PLASMA`            | Speed magnitudes (0 → max)    |
| `FORCE`             | `INFERNO`           | Force / pressure magnitudes   |
| `ACCELERATION`      | `TURBO`             | Higher-frequency derivatives  |
| `HEIGHT`            | `VIRIDIS`           | Vertical position / elevation |
| `GENERIC_DIVERGING` | `COOLWARM`          | Signed deviations around zero |

Using the alias keeps call sites readable — and if a future review
changes the underlying matplotlib pick, every existing caller updates
for free.

For **categorical** data (engine names, marker groups, target ids)
prefer a qualitative palette via `PaletteColor`: `tab10`, `Set2`,
`Set3`, or `Dark2`. Avoid using a perceptually-uniform sequential
colormap for categories — equal-spaced category indices look like
ordered values, which they aren't.

### Accessibility

The `cividis` built-in is the recommended sequential colormap for
deuteranopia / protanopia accessibility — it remains monotonic in
luminance for the most common forms of color-vision deficiency. The
`monochrome` and `high_contrast` presets shipped in
`PresetLibrary.default()` are designed for screen-share and
print-friendly contexts and avoid relying on hue alone.

## Where to next

- Quickstart and the static / palette paths:
  [`quickstart.md`](quickstart.md).
- Wire a colormap to a numeric channel:
  [`data_driven_coloring.md`](data_driven_coloring.md).
- ADR with the full surface-area discussion:
  [ADR 0011](../../adr/0011-plot-style-toolkit.md).
