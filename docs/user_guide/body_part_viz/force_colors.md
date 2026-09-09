# Segment Force Colors

The optional display colors segments blue in tension and red in compression.
It changes existing surface or line colors and preserves geometry and opacity.
It is disabled by default. Settings include separate saturation limits in newtons,
a neutral band around zero, and custom tension/compression/neutral RGB colors.
The range stays fixed during playback so the same color retains the same meaning.

Missing, stale or nonfinite forces retain the original model color. They are not
zero-force measurements. The legend names the sign convention: positive tension,
negative compression. Axial force at a declared segment section is not a stress
field, joint torque, or a tissue injury threshold.

## Python Hosts

Both `MatplotlibRenderer` and `PyQtGLRenderer` expose the optional
`ColorOverrideRenderer` capability. A `ForceColorDisplay` connects arbitrary
segment identifiers to renderer handles without accessing native artist internals:

```python
from src.shared.python.body_part_viz import (
    ForceColorDisplay, ForceColorScale, SegmentLoadSeries,
)

# renderer and handle come from the host's existing shape registration.
display = ForceColorDisplay(renderer, {"link_a": handle})
display.set_loads(SegmentLoadSeries(
    time_s=(0.0, 0.01, 0.02),
    values_n={"link_a": (100.0, -100.0, None)},
    source="Declared analytical axial load at the distal section of link_a",
))
display.configure(ForceColorScale(enabled=True, tension_limit_n=100.0))
# Call alongside the geometry update when playing or seeking.
display.update_frame(1)
display.configure(ForceColorScale())  # Immediately restore original colors.
```

Hosts must align the load series with the geometry's sample times before binding
it. The controller validates sample counts and index bounds but cannot infer the
geometry clock from a renderer handle. Replacing loads resets to frame zero;
`set_loads(None)` clears stale coloring. Recreate the controller when replacing
the model's handles. Clear it before removing the underlying shapes.

The optional PyQt6 widget is deliberately imported separately from the headless API:

```python
from src.shared.python.body_part_viz.force_color_controls import ForceColorControls

controls = ForceColorControls()
controls.scale_changed.connect(display.configure)
```

Store emitted settings using `scale.to_dict()` and restore them with
`ForceColorScale.from_dict()`. Invalid edits preserve the last valid scale, and
the toggle can always turn off the display even when an unapplied edit is invalid.

## Web Hosts

`Scene3D` has a collapsible Segment Force Colors panel. Supply the optional
`segmentLoads` prop with `time_s` matching `frame.time`, `units: "N"`,
`sign_convention: "tension-positive"`, a nonempty `source`, and `values_n` keyed
by model link names. Absent or mismatched frames display an unavailable message.
Existing solver streams do not yet populate this field automatically.

Other Three.js interfaces can reuse `ForceColorControls`, `forceColor`, and
`SegmentMaterialColors`. Bind their own stable segment IDs to mesh lists. The
adapter clones materials while coloring to avoid altering shared glTF assets,
reuses clones during playback, and restores/disposes only owned clones on cleanup.
Python and JavaScript share conformance examples in
`schemas/force-color-examples.json`.

## Qualification Status

The shared policy, native renderer object updates, controls and web scene wiring
have local automated coverage. Offscreen OpenGL object tests do not qualify GPU
raster output. Native solver adapters, automatic force stream population, and
remaining desktop host integrations are pending in the
[epic draft](../../development/segment_force_color_epic.md).
