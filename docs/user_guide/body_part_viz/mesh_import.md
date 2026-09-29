# Body-Part Viz — Custom Mesh Import

This guide covers bringing your own triangle mesh into the toolkit:
which file formats are supported, how the mesh is sized and oriented,
how the rest-pose fit works, and what to watch out for at high
vertex counts.

If you only want to use one of the bundled shapes, see
[`quickstart.md`](quickstart.md). If you want to add a shape to the
default library so every capture starts with it, see
[`asset_author_guide.md`](asset_author_guide.md).

## Supported formats

`MeshShape` loads any of the formats `trimesh` can parse:

| Extension | Notes                                                   |
| --------- | ------------------------------------------------------- |
| `.stl`    | Binary or ASCII STL. The bundled defaults are STL.      |
| `.obj`    | Triangulated; multi-material OBJs are flattened.        |
| `.ply`    | Binary or ASCII; vertex colours are dropped.            |
| `.glb`    | Embedded binary glTF; the first mesh primitive is used. |

Loading goes through `body_part_viz.shapes._mesh_io.load_mesh`, which
returns a `(vertices, faces)` pair and the mesh's oriented-bounding-
box (OBB) extents. Anything `trimesh` can read should work; if your
file fails to load, it almost certainly is failing inside `trimesh`
and the error message will say so.

## Importing in the C3D Viewer

In the Segments tab:

1. Pick the row whose segment you want to skin.
2. Click **Shape → mesh_file**.
3. In the file picker, choose your `.stl` / `.obj` / `.ply` / `.glb`.
4. The viewer immediately re-fits the mesh against the live markers
   and redraws.

The picker remembers the last directory; for repeated imports it is
fastest to keep your custom meshes under a known directory and reuse
that.

## Sizing — what happens to the mesh

Three things happen to your file before it becomes a `MeshShape`:

1. **Re-centring on the OBB centroid.** The mesh is translated so its
   oriented-bounding-box centre sits at the origin. This makes the
   subsequent fit independent of where the modeller put their origin.
2. **OBB rest-dimensions.** `rest_dimensions` is set to the OBB
   extents (not the axis-aligned bounding box). For a forearm
   modelled along the +Z axis this ends up as roughly
   `(0.07, 0.07, 0.27)` metres.
3. **Decimation.** If the input has more than the per-shape vertex
   budget (default 5000 triangles), `_mesh_decimation.decimate` runs
   quadric decimation to bring the count down. The default budget is
   conservative; it preserves silhouette but discards fine surface
   detail.

The final shape is in metres. If your mesh ships in millimetres or
inches, scale it before import — there is no UI scale slider, and the
fitter assumes metres throughout.

## Fitting at rest pose

Once the mesh is in place, the binding determines how it is fit each
frame:

- **`between_two`**. Pick two markers (e.g. elbow and wrist for a
  forearm). The fitter aligns the mesh's longest OBB axis to the
  marker-to-marker direction and stretches the mesh isotropically
  along that axis to span the live distance. Cross-axis dimensions
  stay at their rest values. Best for limbs.
- **`cluster_kabsch`**. Pick three or more markers (e.g. four torso
  markers). The fitter runs a rigid Kabsch fit between the rest-pose
  marker positions and the live positions, applying only rotation +
  translation. Best for rigid body segments where you trust the
  marker cluster.
- **`procrustes_anisotropic`**. Same as cluster_kabsch but allows
  per-axis scale. Best when the rest-pose marker spread is a poor
  match for the subject (e.g. you used a generic torso template
  against a notably wider or narrower subject).

Pick the binding from the Segments-tab row's **Fitter** column; the
toolkit re-runs the fit on the active capture without reloading the
mesh.

## Performance considerations

The matplotlib backend renders the full vertex list every frame.
`MatplotlibRenderer` is sufficient for ≲ 5 000 triangles per segment
across ≲ 25 segments. Beyond that, scrubbing the time slider gets
sluggish. Two levers:

1. **Lower the vertex budget.** Re-import with a lower
   `max_vertices` (in code: `MeshShape(..., max_vertices=2000)`).
   Re-decimating typically drops triangle count by 4-10× with no
   perceptible silhouette change.
2. **Switch to the Qt-GL backend.** `PyQtGLRenderer` shares the same
   `ShapeRenderer` Protocol and pushes geometry to the GPU once, then
   updates a small per-frame transform. The C3D Viewer's **View →
   Backend → Qt-GL** menu item flips between the two; the matcher
   exposes the same toggle in its preferences.

If neither lever is enough — for instance you are visualising a high-
poly clinical mesh — consider building a `CompositeShape` of cheaper
primitives (an `EllipsoidShape` for the bulk plus a `CylinderShape`
where the joint sits) instead of importing the high-poly geometry
directly.

## Handedness, axes, and units

The toolkit assumes:

- **Right-handed Z-up** coordinate frame.
- **Metres** throughout.
- The mesh's local +Z axis is the segment's "long" axis (matters
  only for `between_two` fitters; the OBB pre-pass usually picks the
  right axis automatically, but if your forearm comes out crosswise,
  rotate the source file 90° before exporting).

Source files exported from MeshLab, Blender, MakeHuman, OpenSim's
GUI, and Blender-exported FBX-via-conversion all import cleanly at
the correct scale and orientation in our test fixtures. If yours
does not, the most common causes are unit mismatch (mm vs. m) and
Y-up vs. Z-up convention.

## Troubleshooting

**Mesh loads but is invisible.** It almost certainly loaded with
zero-area faces or got rejected by the OBB pre-pass. Check
`MeshShape.rest_dimensions`; if any axis is `~0`, the source mesh has
a degenerate dimension and needs cleanup in the source modeller.

**Mesh loads but is the wrong scale.** Unit mismatch. Verify by
inspecting `rest_dimensions` — if it reads in centimetres or
millimetres, you have a unit-conversion issue at export time, not
inside the toolkit.

**Mesh loads but tracks the wrong markers.** The binding is wrong;
swap markers in the Segments tab row and re-fit.

**Frame scrubbing is sluggish.** See "Performance considerations"
above. Lower the vertex budget or switch backends.

## Where to next

- [`asset_author_guide.md`](asset_author_guide.md) — turn a custom
  mesh into a permanent library entry.
- [`docs/api/body_part_viz.md`](../../api/body_part_viz.md) — public
  API including the URDF bridge for re-using imported meshes as URDF
  visuals.
