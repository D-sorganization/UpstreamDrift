# Plot-Style Toolkit — Data-Driven Coloring

A _data-driven_ color is one that varies with a numeric channel:
"colour the clubhead marker by clubhead speed", "colour each foot
marker by ground-reaction-force magnitude", "colour each frame by
per-frame fit error". This is a first-class feature of the plot-style
toolkit — no per-tool styling code needed.

For background and the design contract, see
[ADR 0011 — Plot-Style Toolkit](../../adr/0011-plot-style-toolkit.md).

## End-to-end recipe — colour by clubhead speed

The recipe has three steps:

1. Build a `DataChannel` from the numeric values you want to colour by.
2. Wrap it in a `DataDrivenColor` scale that names a colormap.
3. Assemble a `MarkerStyle` with that scale as its `fill_color`, then
   hand the style to any renderer.

### 1. Configure a `DataChannel`

A `DataChannel` is a frozen wrapper around a numeric `numpy.ndarray`.
A 1-D array of shape `(T,)` denotes one scalar per frame; a 2-D array
of shape `(T, M)` denotes one scalar per (frame, marker).

```python
import numpy as np

from src.shared.python.plot_style import DataChannel

# 600 frames, one scalar per frame, in m/s.
speed_values = np.linspace(0.0, 50.0, 600, dtype=np.float64)
speed = DataChannel(name="clubhead_speed", values=speed_values, unit="m/s")
```

`DataChannel.value_at(frame, marker)` returns `NaN` for out-of-bounds
indices instead of raising — callers may safely vectorise without
bounds-guarding. `DataChannel.auto_range()` returns `(min, max)` over
finite values, used as the default normalisation bounds.

For vector-valued sources (e.g. a `(T, 3)` velocity vector), use the
`magnitude_channel` helper to collapse to a scalar:

```python
from src.shared.python.plot_style import magnitude_channel

velocity_xyz = np.zeros((600, 3))  # (T, 3) — one 3-vector per frame
speed = magnitude_channel(name="clubhead_speed", vector_per_frame=velocity_xyz, unit="m/s")
```

### 2. Wire to a `DataDrivenColor` resolver

`DataDrivenColor` binds a channel to a colormap and sets the
normalisation bounds. The `colormap` argument is a `ColormapId` enum
member — built-in matplotlib names (`VIRIDIS`, `PLASMA`, ...) and
semantic aliases (`VELOCITY`, `FORCE`, `ACCELERATION`, `HEIGHT`,
`GENERIC_DIVERGING`) both work.

```python
from src.shared.python.plot_style import ColormapId, DataDrivenColor

# `speed` is the DataChannel built in step 1.
fill = DataDrivenColor(
    channel=speed,
    colormap=ColormapId.VIRIDIS,
    vmin=0.0,
    vmax=50.0,
    nan_color="#888888",
)
```

`vmin` / `vmax` may be left `None` to auto-detect via the channel's
`auto_range()`. Non-finite values and a degenerate range (`vmax <=
vmin`) both map to `nan_color`.

### 3. Assemble a `MarkerStyle`

`MarkerStyle.fill_color` accepts any `ColorScale` variant —
`StaticColor`, `PaletteColor`, or `DataDrivenColor`:

```python
from src.shared.python.plot_style import MarkerShape, MarkerStyle

style = MarkerStyle(
    shape=MarkerShape.SPHERE,
    size_px=8.0,
    edge_color="#000000",
    edge_width=0.5,
    fill_color=fill,
    opacity=1.0,
)
```

That's the complete object. Pass it to `MatplotlibMarkerRenderer` or
`PyQtGLMarkerRenderer` exactly as in the
[quickstart](quickstart.md#3-apply-a-markerstyle-to-a-renderer); the
renderer asks the resolver registry for the right `ColorResolver` and
samples per frame.

## Hooking into the C3D Viewer's 3D tab

The C3D Viewer's 3D tab (`viewer_3d_tab.py`) installs a
`DataChannelEditor` widget that drives the data-driven path live. Call
`install_data_channel_editor` with the channels available for the
loaded capture:

```python
from src.shared.python.plot_style import DataChannel, magnitude_channel

# Build the channels you want the viewer to expose.
channels = (
    magnitude_channel(name="clubhead_speed", vector_per_frame=v_xyz, unit="m/s"),
    DataChannel(name="rms_error", values=err_per_frame, unit="m"),
)

viewer_3d_tab.install_data_channel_editor(channels)
```

The editor exposes the channel combobox, `vmin` / `vmax` spinboxes, and
a "fit to data" button (which calls `DataChannel.auto_range()` by
default; pass a custom `fit_fn` for windowed or robust quantile
ranges). Selecting a channel re-resolves the active marker group's
style with a fresh `DataDrivenColor` and pushes it through the
renderer — no scene rebuild.

## Bulk path for animation playback

Per-frame `resolve_one` is fine for one-shot lookups, but the
animation playback loop wants to pre-compute the entire frame ×
marker × RGBA cube once and gather per frame. Use
`ColorResolver.resolve_array`:

```python
from src.shared.python.plot_style import DataDrivenColor as DataDrivenColorScale
from src.shared.python.plot_style.resolvers import DataDrivenColor as DataDrivenResolver

scale = DataDrivenColorScale(
    channel=speed,
    colormap=ColormapId.VELOCITY,  # semantic alias -> PLASMA
    vmin=0.0,
    vmax=50.0,
)

resolver = DataDrivenResolver(scale)

# (n_frames, n_markers, 4) RGBA cube — pre-compute once.
n_frames, n_markers = 600, 8
rgba_cube = resolver.resolve_array(scale, n_frames=n_frames, n_markers=n_markers)

# In the playback loop:
for frame_idx in range(n_frames):
    frame_rgba = rgba_cube[frame_idx]   # (n_markers, 4) — a no-copy view
    renderer.draw(ax, positions[frame_idx], style, frame_rgba)
```

Internally the resolver pre-computes a 256-entry colormap LUT once at
construction time, normalises the channel values into LUT indices in a
single vectorised step, and gathers. For per-marker-mean and frame-only
channels the slab building is also vectorised. NaN values yield the
configured `nan_color`. The path is allocation-free beyond the output
array and clears 60 fps for 1000 frames × 32 markers on a developer
laptop.

`resolve_array` accepts `n_markers=None` for 1-D channels (one scalar
per frame) and tolerates callers asking for more frames than the
channel actually has — the trailing rows are NaN-padded and resolve to
`nan_color`.

## Where to next

- Register your own colormap or palette: see
  [`colormap_author_guide.md`](colormap_author_guide.md).
- Quickstart for the static / palette paths:
  [`quickstart.md`](quickstart.md).
- Public API reference: the package's
  [`__init__.py`](../../../src/shared/python/plot_style/__init__.py)
  is the curated entry-point — every public name is in `__all__`.
