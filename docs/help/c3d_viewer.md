---
title: C3D Viewer
tile_id: c3d_viewer
status: complete
---

# C3D Viewer

## Purpose

Open a `.c3d` file from an optical motion-capture lab and inspect it:
what markers and analog channels it contains, how each channel behaves
over time, where the markers sit in 3D, and simple per-marker kinematic
summaries. Use it to sanity-check a capture before feeding it to the
motion pipeline, and to export a chosen subset of markers.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| `.c3d` file | binary C3D | Loaded on a background thread. Requires `ezc3d`. |
| Marker selection | list of marker names | Chosen in the export dialog. |
| Component selection | `x`, `y`, `z` or `all` | Which position components to export. |
| Animation export path | filesystem path ending in `.mp4` | A non-`.mp4` suffix is rejected. |
| Animation frame rate | Hz | Defaults to the file's own point rate; falls back to 30 Hz when the file reports 0. Must be positive. |
| Segment definitions | marker pairs plus a shape | Optional; defined interactively in the Segments tab. |

The file's own `POINT:UNITS` parameter is honoured. When a writer (for
example `ezc3d`) omits it, the loader assumes **millimetres** before
parsing metadata.

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| Marker trajectories in memory | metres | Positions are converted to metres on load (`target_units="m"`), regardless of the unit declared in the file. Array shape is `(N, 3)` per marker. |
| Point rate / analog rate | Hz | Shown in the Overview tab. |
| Time vectors | seconds | `point_time = arange(frame_count) / frame_rate`; `analog_time = arange(n_samples) / analog_rate`. |
| Events | seconds | Named C3D events with their timestamps. |
| Analog channels | file-declared unit per channel | The unit string is read from `ANALOG:UNITS` and carried with each channel; the viewer does not rescale analog data. |
| Per-marker statistics | path length in metres, max/mean speed in metres per second | Derived from the metre-converted positions. The panel labels these generically as "units" and "units/s". |
| Marker export | `.csv` (long format, one row per frame-marker pair), `.json` (with a `metadata` block), or `.npz` (one array per marker plus a JSON `_meta` string) | CSV cells beginning `=`, `+`, `-` or `@` are prefixed with `'` to defang spreadsheet formula injection. |
| Animation | `.mp4` | Rendered off the GUI thread, with progress reporting and cancellation. |

## Method

The viewer is an MVC Qt application:
[`c3d_viewer.py`](../../src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py)
composes the tabs, `core/models.py` holds the `C3DDataModel` dataclass,
and `services/c3d_loader.py` does the parsing by delegating to
`C3DDataReader` (which wraps `ezc3d`) and calling
`points_dataframe(target_units="m")`. Loading runs on a
`C3DLoaderThread` so the window stays responsive.

Tabs: Overview (metadata, marker and channel lists), Marker Plot,
Analog Plot, Force Plot (ground-reaction-force time series and
centre-of-pressure trajectory), Analysis (speed, path length, extrema),
Segments (user-defined marker-pair segments rendered as lines,
cylinders, ellipsoids, capsules, library shapes or imported meshes) and
3D Viewer.

C3D's place in the wider ingestion story is described in the
[format matrix](../motion_pipeline/formats.md).

## Limitations

- **It is a viewer and an exporter, not an editor.** There is no gap
  filling, no filtering, no marker-swap repair and no re-labelling. A
  capture with occlusions comes out of this tool exactly as occluded as
  it went in.
- **No inverse kinematics, no retargeting, no scaling.** It does not
  drive a physics model; it shows you the markers. Solving belongs to
  the [motion pipeline](../motion_pipeline/README.md).
- **No coordinate-frame conversion.** Axis convention is whatever the
  source system wrote. Only the length unit is normalised (to metres);
  a Y-up capture stays Y-up.
- **Analog channels are not calibrated or scaled** beyond what the file
  already contains, and the unit string is taken on trust from the file.
- The Analysis panel's labels read "units" and "units/s" rather than
  naming metres explicitly.
- Requires `ezc3d`. Without it, loading fails.

## See Also
- [Motion pipeline format matrix](../motion_pipeline/formats.md)
- [Motion pipeline workflow guide](../motion_pipeline/README.md)
- [Motion pipeline troubleshooting](../motion_pipeline/troubleshooting.md) - start here for millimetre-versus-metre and occlusion symptoms
- [Motion Capture (FreeMoCap sidecar)](motion_capture.md)
- [Motion-Match Preview](motion_target_preview.md) - consumes C3D as a club or body-marker target
- [Visualization](visualization.md), [Analysis Tools](analysis_tools.md)
