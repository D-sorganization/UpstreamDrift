# feat(c3d-viewer): integrate body_part_viz into Segments tab — shape picker, mesh-import dialog

Depends on the `body_part_viz` package (#1–#8).

## Why

Replace the C3D Viewer's hand-rolled cylinder rendering (PR #4664) with the canonical `body_part_viz` toolkit. This gives the user shape choice + mesh import inside the existing Segments tab.

## What

### Segments-tab UI changes

```
Segments                                    [+ Add segment]  [Reset to default]
+────────────────────────────────────────────────────────────────────────────+
| ☑ | A           | B          | Shape                | Fitter         | × |
+────────────────────────────────────────────────────────────────────────────+
| ☑ | WaistLeft   | WaistRight | [Line          ▾]   | between_two   | × |
| ☑ | LShoulderTop| LElbowOut  | [Library: arm  ▾]   | between_two   | × |
| ☑ | Marker_2:2:1| Marker_2:2:2| [Mesh: club.stl…  ] | cluster_kabsch| × |
+────────────────────────────────────────────────────────────────────────────+

[Save segment set…] [Load…] [Import shape file…] [Library…]
```

`Shape` column is a combobox: `Line / Cylinder / Ellipsoid / Capsule / Library shape… / Mesh file…`.

When the user picks "Mesh file…", a `QFileDialog` opens; selected file is loaded via `MeshShape.load()` and stored. The shape file path is remembered relative to the JSON.

When the user picks "Library shape…", a chooser shows the entries in `ShapeLibrary.default()`.

### Code changes

- Replace `precompute_segments_from_pairs` (in `gui_playback.py` / equivalent) with `body_part_viz.fitters.*.fit`.
- Replace the cylinder `Poly3DCollection` rebuild with `MatplotlibRenderer`.
- Move `SegmentSet` / `SegmentSpec` (engine subtree) → adapter that calls into `SegmentVizSet`.

The OLD `SegmentSpec` struct stays as a **deprecated** shim that calls into `SegmentVizSpec` for one release; remove in a follow-up.

### Backward compat

- v1 segment-set JSON files load (auto-migrate via #8).
- The on-disk default location remains `~/.golf_modeling_suite/c3d_viewer_segments.json`.
- All existing tests in `tests/unit/engines/simscape/three_d_gui/` continue to pass.

## Tests

`tests/unit/engines/simscape/three_d_gui/test_segments_tab_v2.py`:

- Open the Segments tab; assert Shape column is a combobox with 6 options.
- Select Cylinder; assert the rendered artist is a `Poly3DCollection` from the renderer.
- Import a synthetic STL; assert the new segment uses `MeshShape`.
- Library-shape chooser shows ≥ 5 default entries.
- Save → load round-trip preserves the shape choices.
- Old v1 JSON loads + auto-migrates.

## Acceptance criteria

- [ ] Segments tab Shape column has 6 options.
- [ ] Mesh import works end-to-end via `QFileDialog`.
- [ ] Library shape chooser populated from `ShapeLibrary.default()`.
- [ ] OLD cylinder code removed from `gui_playback.py`.
- [ ] All existing C3D Viewer tests still pass.
- [ ] Performance unchanged or better (≥ 60 fps scrub on the canonical 26-segment driver test).

## Files touched

- Edit: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/ui/tabs/segments_tab.py`
- Edit: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/ui/tabs/viewer_3d_tab.py`
- Edit: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/services/segment_set_io.py` → thin adapter to `body_part_viz.persistence`
- Edit: `src/tools/starting_pose_matcher/gui_playback.py` → use `MatplotlibRenderer`
- New: `tests/unit/engines/simscape/three_d_gui/test_segments_tab_v2.py`
