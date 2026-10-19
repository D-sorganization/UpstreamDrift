# R9a MeshCat Versus MuJoCo Appearance Stills (#12688)

These stills compare the shared `model_appearance` document as drawn by Drake
MeshCat, Pinocchio MeshCat, and the MuJoCo native viewer. They are evidence
only; appearance never feeds dynamics.

## Expected Differences Other Than Lighting

- **Shading model:** MeshCat uses Phong-style materials on triangle meshes;
  MuJoCo uses its own rasteriser. Specular highlights and floor reflections
  differ even when RGBA matches the library.
- **Head hair:** The CC0 head mesh is identical; MeshCat may show more
  contrast on fine hair triangles than MuJoCo at 960×720.
- **Garment normals:** Shirt and trouser meshes are the same OBJs; MeshCat and
  MuJoCo may disagree slightly on back-face culling at grazing angles.
- **Club shaft:** Both use the shared club meshes; MeshCat draws the shaft as a
  smooth mesh while MuJoCo may tessellate differently near the grip.

## What Must Match

- Library material RGBA on every dressed body (checked in
  `test_meshcat_appearance.py` and the MuJoCo cross-engine signature test).
- Trunk ML extent versus the spec hint semi-axis (5% tolerance).
- Pelvis garment material (pants), torso garment (shirt), and foot material
  (shoe) from `library.CLOTHING`.

## Reproduction

From a Linux host with pydrake, pinocchio, coal, meshcat, playwright, and
Chromium:

```bash
PYTHONPATH=.:src python3 \
  docs/development/full_body_models/evidence/appearance/engines/drake_pinocchio/render_drake_pinocchio_appearance.py \
  --datastore /path/to/swing-datastore \
  --out docs/development/full_body_models/evidence/appearance/engines/drake_pinocchio
```

On Windows without Drake or Pinocchio, pass `--engines mujoco` to capture the
MuJoCo reference row only; commit the MeshCat panels after a brick render.
