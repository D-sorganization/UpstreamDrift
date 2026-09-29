# Plot-Style Toolkit — Quickstart

This guide walks through the most common workflow: build a `MarkerStyle`
either interactively (via the `MarkerStylePicker` widget) or
programmatically from a built-in preset, then apply it to the canonical
matplotlib renderer.

The toolkit lives in `src/shared/python/plot_style/` and is shared
across the C3D Viewer, the starting-pose matcher, and the cross-engine
dashboard. Anything you save here is portable across all three.

For background and the design contract, see
[ADR 0011 — Plot-Style Toolkit](../../adr/0011-plot-style-toolkit.md).

## 1. Pick a marker shape and color in the GUI

The simplest way to build a `MarkerStyle` is the `MarkerStylePicker`
widget. It composes a shape combobox, a size and edge-width spinbox,
and an embedded `ColorPicker` for the edge color. Whenever any
sub-control changes, the widget rebuilds an immutable `MarkerStyle` and
emits `styleChanged(MarkerStyle)`:

```python
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout

from src.shared.python.plot_style import MarkerShape, MarkerStyle
from src.shared.python.plot_style.widgets import MarkerStylePicker


def edit_style(initial: MarkerStyle) -> MarkerStyle:
    app = QApplication.instance() or QApplication([])
    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    picker = MarkerStylePicker(initial=initial, parent=dialog)
    layout.addWidget(picker)

    edited: list[MarkerStyle] = [initial]
    picker.styleChanged.connect(lambda style: edited.__setitem__(0, style))

    dialog.exec()
    return edited[0]


initial = MarkerStyle(shape=MarkerShape.SPHERE, size_px=8.0)
style = edit_style(initial)
```

The fill color is intentionally out of scope for this widget — it is
governed by `DataChannelEditor` (data-driven) or by the parent dialog's
own `ColorPicker` (static / palette). See
[`data_driven_coloring.md`](data_driven_coloring.md) for the wiring.

## 2. Load a built-in preset

If you don't need a per-marker custom build, the preset library has
four curated themes ready to use:

```python
from src.shared.python.plot_style import BUILTIN_PRESET_NAMES, PresetLibrary

print(BUILTIN_PRESET_NAMES)
# ('default', 'scientific_violet', 'monochrome', 'high_contrast')

library = PresetLibrary.default()
style_set = library["scientific_violet"]
```

A `PlotStyleSet` is a name-keyed mapping of `PlotStyleSpec` entries.
Pull the `MarkerStyle` for a specific marker group:

```python
spec = style_set.specs["wrists"]   # or any other group key
marker_style = spec.style
```

The four built-in presets all ship with the same group keys so you can
swap themes without touching call sites.

## 3. Apply a `MarkerStyle` to a renderer

The canonical 2D / 3D backend is `MatplotlibMarkerRenderer`. It
satisfies the stateful `MarkerRenderer` Protocol
(`add_markers` / `update_frame` / `update_style` / `remove`) and also
exposes a stateless `draw(ax, positions, style, colors)` helper for
callers that already have a resolved RGBA array in hand.

For a stateful workflow (animation, frame scrubbing) you give the
renderer a default axes and call `add_markers` per group:

```python
import matplotlib.pyplot as plt
import numpy as np

from src.shared.python.plot_style import MarkerStyle, MatplotlibMarkerRenderer

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

renderer = MatplotlibMarkerRenderer(default_ax=ax)

# Per-frame positions: (T, M, 3) ndarray.
positions = np.random.default_rng(0).uniform(-1.0, 1.0, size=(120, 8, 3))

handle = renderer.add_markers(
    positions=positions,
    style=MarkerStyle(),
    label="left_hand",
)

renderer.update_frame(handle, frame_idx=42)
plt.show()
```

`update_frame` re-positions the existing artist; `update_style` swaps
the style without rebuilding the artist; `remove` tears it down.

## 4. Worked example — preset + matplotlib renderer

End-to-end: load the `scientific_violet` preset, pick the marker style
for the `wrists` group, and render a 32-frame trajectory in 3D.

```python
import matplotlib.pyplot as plt
import numpy as np

from src.shared.python.plot_style import (
    MatplotlibMarkerRenderer,
    PresetLibrary,
)

# 1. Load a preset and pull a MarkerStyle.
library = PresetLibrary.default()
style = library["scientific_violet"].specs["wrists"].style

# 2. Build a 3D scene.
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)

# 3. Add a synthetic 32-frame trajectory: (T=32, M=2, 3).
rng = np.random.default_rng(seed=42)
positions = rng.uniform(-0.8, 0.8, size=(32, 2, 3))

# 4. Hand it to the renderer.
renderer = MatplotlibMarkerRenderer(default_ax=ax)
handle = renderer.add_markers(positions=positions, style=style, label="wrists")

# 5. Scrub a frame.
renderer.update_frame(handle, frame_idx=15)
plt.show()
```

The same code targets the GPU-rate `PyQtGLMarkerRenderer` by swapping
the constructor — both implement the same `MarkerRenderer` Protocol.

## Where to next

- Color markers by a numeric channel (clubhead speed, force, error):
  see [`data_driven_coloring.md`](data_driven_coloring.md).
- Ship a custom colormap or palette: see
  [`colormap_author_guide.md`](colormap_author_guide.md).
- Public API reference: the package's
  [`__init__.py`](../../../src/shared/python/plot_style/__init__.py)
  is the curated entry-point — every public name is in `__all__`.
