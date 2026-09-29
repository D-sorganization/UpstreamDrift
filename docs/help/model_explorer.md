---
title: Model Explorer
tile_id: model_explorer
status: complete
---

# Model Explorer

## Purpose

Model Explorer is the interactive editor for the rigid-body models
UpstreamDrift simulates. You browse the bundled model library, load an
existing model file, add or edit links and joints segment by segment,
watch a 3D preview update as you go, and save the result as URDF for one
of the physics engines. Internally it is still the "Interactive URDF
Generator", and URDF is the format it writes.

## Inputs

| Input | Format / units |
| --- | --- |
| Model file (File > Open) | `*.urdf`, `*.xml`, `*.mjcf`, `*.osim` (the file dialog filter in `gui.py`) |
| OpenSim model | `.osim`, converted to URDF text by `osim_loader.OsimLoader.to_urdf` before display |
| Library model | A category plus model key chosen in the library panel; golf-club models are generated on demand, human models are resolved to a file on disk |
| Segment definition | A dict of link/joint properties supplied by the segment panel: geometry, mass in kg, inertia, and joint limits in rad (revolute/continuous) or m (prismatic) |
| Attachment manifest | JSON validated against `src/tools/model_explorer/attachment_manifest.schema.json` |

Joint types the segment editor offers are fixed, revolute, prismatic,
continuous, floating, and planar.

## Outputs

| Output | Format / units |
| --- | --- |
| Saved model | URDF XML written as `*.urdf` or `*.xml`, UTF-8 |
| Engine export | URDF XML via `export_for_engine(engine, default_filename)` |
| `urdf_generated` signal | The current URDF document as a `str`, emitted to the launcher host |
| `segment_added` / `segment_removed` signals | Segment dict / segment name |
| Web tree response | `ModelExplorerResponse` JSON from the REST API, a collapsible node tree parsed from the model file |

## Method

The PyQt6 surface is `MainWidget` in
[`gui.py`](../../src/tools/model_explorer/gui.py); `main_window.py` is
only a `QMainWindow` shell that owns the menu bar, window icon, and the
unsaved-changes prompt, and delegates everything else to that widget.
`_embed_adapter.py` exposes the same widget to the launcher as a tab or
dock through the `EmbeddableTool` protocol.

URDF XML is assembled by `URDFBuilder` (`urdf_builder.py`) from the
segment hierarchy held by `SegmentManager`. `visualization_widget.py`
re-renders the preview from the builder's URDF text on every segment
add / remove / modify. `.osim` input is translated to URDF by
`osim_loader.py`; `sdf_loader.py` handles SDF.

The web surface at `/tools/model-explorer` is a different, read-only
implementation: `src/api/routes/model_explorer.py` parses a discovered
URDF/MJCF file into a node tree
(`GET /tools/model-explorer/{model_name}`,
`POST /tools/model-explorer/inspect`) and can diff two models
(`POST /tools/model-explorer/compare`, "Frankenstein mode").

## Limitations

- It writes URDF only. `export_for_engine` is a hook, not a converter:
  its own docstring says it "Currently exports a generic URDF (Drake /
  Pinocchio / MuJoCo all accept it)", so the MuJoCo, Drake, and
  Pinocchio menu entries all produce the same bytes. No MJCF or SDF is
  ever emitted.
- Loading is wider than saving. `.mjcf` and `.osim` can be opened, but
  there is no round-trip: the model is converted to URDF and saved as
  URDF.
- The web page cannot edit. It parses and compares models; all authoring
  is PyQt6-only.
- `src/tools/model_explorer/README.md` is stale. It documents a
  `tools/urdf_generator/` layout and a `launch_urdf_generator.py`
  entry point that no longer exist, and lists 3D visualization as "in
  progress" although `visualization_widget.py` and `mujoco_viewer.py`
  are present. Treat this page, not that README, as current.

## See Also
- [Model Explorer package README](../../src/tools/model_explorer/README.md)
  (stale in the ways noted above)
- [Attachment manifests](../model_explorer/attachment-manifests.md)
- [Engine selection](engine_selection.md)
- [Visualization](visualization.md)
