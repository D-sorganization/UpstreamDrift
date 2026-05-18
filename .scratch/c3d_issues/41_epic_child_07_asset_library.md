# feat(body-part-viz): asset library — default body-part meshes + manifest

Depends on `MeshShape` (#3).

## Why

Curated default meshes for the most-used body parts so users don't have to find their own. Generic, low-poly, public-domain.

## What

`assets/body_part_shapes/default/`:

- `head.stl` — anonymized human head, ~2000 triangles.
- `torso.stl`
- `upper_arm.stl`
- `forearm.stl`
- `hand.stl`
- `thigh.stl`
- `shin.stl`
- `foot.stl`
- `manifest.json` — metadata for each mesh.

### `manifest.json` schema

```json
{
  "schema_version": 1,
  "shapes": {
    "head": {
      "file": "head.stl",
      "rest_dimensions": [0.18, 0.22, 0.20],
      "binding_template": {
        "kind": "between_two",
        "marker_names": ["HeadFront", "HeadTop"],
        "rest_orientation_quat": [1.0, 0.0, 0.0, 0.0]
      },
      "license": "CC0-1.0",
      "source": "procedural-low-poly"
    },
    ...
  }
}
```

## Asset generation strategy

To keep the assets generic and re-distributable:

- Use `trimesh.creation.icosphere` / `cylinder` / `capsule` to generate basic shapes procedurally at build time, OR
- Vendor low-poly CC0 meshes from public sources (e.g. MakeHuman base mesh, downsampled to ~2k tris).

The procedural path is preferred because it's reproducible from a script in the repo; vendor path requires a `LICENSES.md` entry per mesh.

## Loader

`src/shared/python/body_part_viz/asset_library.py`:

```python
class ShapeLibrary:
    """Resolve named body-part shapes to MeshShape instances."""

    def __init__(self, asset_root: Path | None = None) -> None: ...

    def names(self) -> tuple[str, ...]: ...
    def get(self, name: str) -> MeshShape: ...
    def binding_template(self, name: str) -> MarkerBinding: ...

    @classmethod
    def default(cls) -> "ShapeLibrary":
        """Load the bundled default library."""
```

## Tests

`tests/unit/body_part_viz/test_asset_library.py`:

- `ShapeLibrary.default()` loads without raising.
- Every name in the manifest resolves to a `MeshShape` whose `rest_dimensions` matches the manifest within 1e-3.
- Unknown name → `KeyError` with available names listed.
- `binding_template("head")` returns a valid `MarkerBinding`.

## Acceptance criteria

- [ ] 8 mesh files + manifest committed to `assets/body_part_shapes/default/`.
- [ ] `ShapeLibrary.default()` loads them all.
- [ ] Each mesh ≤ 5000 triangles (decimation budget).
- [ ] All licenses documented in `assets/body_part_shapes/default/LICENSES.md`.
- [ ] No vendor / lab / person identifiers anywhere in mesh metadata.
- [ ] ≥ 90% line coverage on `asset_library.py`.

## Files touched

- New: `assets/body_part_shapes/default/*.stl` (8 files)
- New: `assets/body_part_shapes/default/manifest.json`
- New: `assets/body_part_shapes/default/LICENSES.md`
- New: `assets/body_part_shapes/default/_generate.py` (procedural-generation script)
- New: `src/shared/python/body_part_viz/asset_library.py`
- New: `tests/unit/body_part_viz/test_asset_library.py`
