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
The WebSocket stream populates `frame.segment_loads` when its engine implements
`AxialLoadProvider`; Scene3D consumes this field automatically. Missing capability
is explicitly unavailable. The timestamp is checked before wire rounding.

Other Three.js interfaces can reuse `ForceColorControls`, `forceColor`, and
`SegmentMaterialColors`. Bind their own stable segment IDs to mesh lists. The
adapter clones materials while coloring to avoid altering shared glTF assets,
reuses clones during playback, and restores/disposes only owned clones on cleanup.
Python and JavaScript share conformance examples in
`schemas/force-color-examples.json`.

## Qualification Status

The Pendulums View menu provides Segment Force Colors (Ctrl+Shift+F). Double and
triple pendulums use the existing transmitted joint reactions, projected at the
proximal section. Both flat and tapered segment views share the color policy.
The golfer's point-mass net forces are not transmitted section reactions and are
not mislabeled as axial segment loads.

The MuJoCo view has a Segment Force Colors button. The native source uses scratch
simulation data and supports named bodies with one capsule/cylinder whose endpoint
coincides with the proximal joint. Ambiguous axes and free bodies remain unchanged.
Tendons, plugins and global dynamics callbacks are unavailable in this adapter.
MuJoCo documents a spatial-tendon limitation in `cfrc_int`; see the
[native API reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-rnepostconstraint).
The same engine provider feeds the web stream and MuJoCo's MeshCat view.
`MeshcatForceColors` accepts a public property setter and explicit segment-to-leaf
object bindings with original RGBA values. For meshcat-python pass
`lambda path, prop, value: viewer[path].set_property(prop, value)`; for Drake pass
`meshcat.SetProperty`. This adapter supports multiple geometries per segment and
restores base RGBA on disable or missing data. Recreate it when base materials
change. Its native meshcat-python command test qualifies transport, not browser
raster output. Drake, Pinocchio, OpenSim and other host/source integrations remain
tracked under #9833.

The C3D/Simscape viewer exposes the same controls for user-defined shapes. Its
`set_segment_axial_loads(loads, segment_indices)` method accepts a qualified
`SegmentLoadSeries` and an explicit load-ID to segment-index mapping. Sample times
must exactly equal the model's point times. Replacing the model or segment set
clears loads to prevent stale bindings. Motion capture alone supplies no axial loads.

The shared policy, native renderer updates, controls and web scene wiring have
local automated coverage. Native MuJoCo raster verification confirms blue tension,
red compression and pixel-exact off restoration. PyQtGraph OpenGL object tests
do not qualify GPU raster output. Remaining desktop host integrations are tracked
in [epic #9833](https://github.com/D-sorganization/UpstreamDrift/issues/9833) and
[PR #9840](https://github.com/D-sorganization/UpstreamDrift/pull/9840).
