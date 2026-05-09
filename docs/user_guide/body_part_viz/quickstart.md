# Body-Part Viz — Quickstart

This guide walks through the most common workflow: load a C3D capture
in the C3D Viewer, open the **Segments** tab, swap the default
between-marker line for a richer library shape, and save the result so
the next session re-opens with the same look.

The toolkit lives in `src/shared/python/body_part_viz/` and is shared
across the C3D Viewer, the starting-pose matcher, and the URDF
generator. Anything you save here is portable across all three.

## 1. Open a C3D file

From the launcher (the **C3D Viewer** tile registered in
`src/config/models.yaml`) or directly:

```bash
python3 src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py \
    path/to/capture.c3d
```

The viewer opens with the marker cloud rendered in 3D and a row of
tabs along the bottom. Click **Segments**.

## 2. Read the default segment list

The first time you open a capture, the Segments tab populates the
table with the default body-segment set returned by
`default_body_segments` (pelvis, spine, torso, shoulders, elbows,
wrists, hands, feet — the same set the motion-matching cost terms
dispatch on; see [ADR 0006](../../adr/0006-multi-source-motion-targets.md)).

Each row shows:

| Column | What it is |
| --- | --- |
| **Segment** | The canonical segment label (e.g. `LeftForearm`). |
| **Binding** | Which markers the segment is attached to. |
| **Shape** | The current shape kind — `line`, `cylinder`, `ellipsoid`, `capsule`, `mesh_file`, `library_shape`, or `composite`. |
| **Fitter** | How the shape is positioned each frame — `between_two`, `cluster_kabsch`, or `procrustes_anisotropic`. |
| **Visible** | Whether the segment is drawn. |

The defaults are minimal — every row starts as a `line` between two
markers — to keep the first render fast. The next step swaps in
something more visual.

## 3. Swap a line for a library shape

Pick a row, e.g. `LeftForearm`. Click **Shape → library_shape**, then
choose **forearm** from the library picker. The Segments tab now:

1. Loads the bundled forearm mesh from
   `assets/body_part_shapes/default/forearm.stl`.
2. Looks up the binding template in
   `assets/body_part_shapes/default/manifest.json` — for the forearm
   that is `between_two` on `LELB` / `LWRA`.
3. Re-fits the mesh against the live marker positions for every frame.

The 3D view immediately re-draws the segment with the mesh in place
of the line. The frame slider, time scrubber, and export-frame
controls all keep working.

The same swap works for any of the bundled shapes:

- `head` (between two head markers, ellipsoid mesh)
- `torso` (cluster fit, box mesh)
- `upper_arm`, `forearm` (between two markers, cylinder mesh)
- `hand` (anchored on a single marker, ellipsoid mesh)
- `thigh`, `shin` (between two markers, cylinder mesh)
- `foot` (between heel + toe markers, box mesh)

If you want a shape that isn't in the library yet, see
[`mesh_import.md`](mesh_import.md) for importing a custom mesh and
[`asset_author_guide.md`](asset_author_guide.md) for adding it to the
default library.

## 4. Recolour by group

The **Theme** column on each row controls the colour group. The
matplotlib renderer applies the shared dark-theme palette
(`src/shared/python/theme/matplotlib_style.py`); changing a group
re-runs the palette without re-fitting the geometry, so it is cheap.

Typical groupings:

- `arms` — both upper arms + both forearms.
- `legs` — both thighs + both shins.
- `axial` — pelvis + spine + torso + head.
- `extremities` — hands + feet.

The colours match the matcher's pose-overlay plots, so a segment that
looks blue in the C3D Viewer also looks blue in the matcher's
clubhead-trace overlay.

## 5. Save the segment-viz set

**File → Save Segment Viz Set** writes a JSON v2 file (the
`SegmentVizSet` schema; see
[`docs/api/body_part_viz.md`](../../api/body_part_viz.md)). The default
location is alongside the C3D capture, with a `.viz.json` extension.

Re-opening the same capture auto-detects the sidecar file and applies
it. Legacy `.segments.json` files written by the pre-toolkit Segments
tab are still loadable — they auto-migrate from v1 to v2 on read,
and are saved back as v2 the next time you save.

The file is portable: copy it next to the same capture on another
machine and the segment configuration travels with it.

## 6. Hand it to another tool

The starting-pose matcher and the humanoid character builder both read
the same `SegmentVizSet` format. To use the same segment configuration
in the matcher:

```bash
python3 -m src.tools.starting_pose_matcher \
    --target path/to/capture.c3d \
    --segment-viz path/to/capture.viz.json
```

The matcher's preview pane now shows the same shapes you configured
in the C3D Viewer.

To export the same shapes as URDF visuals, see the URDF bridge
section of [`docs/api/body_part_viz.md`](../../api/body_part_viz.md).

## Where to next

- [`mesh_import.md`](mesh_import.md) — bring in a custom mesh.
- [`asset_author_guide.md`](asset_author_guide.md) — add a shape to the
  default library so every capture starts with it.
- [ADR 0008](../../adr/0008-body-part-viz-toolkit.md) — design rationale
  for the toolkit.
