# feat(body-part-viz): MeshShape with STL/OBJ/PLY/GLB loaders via trimesh

Depends on contracts (#1).

## Why

User wants to import custom body-part shape files (STL / OBJ / PLY / GLB) and bind them to mocap markers. This issue ships the loader + decimator + the `MeshShape` class.

## What

### `MeshShape` (in `shapes/mesh_shape.py`)

```python
class MeshShape:
    """Triangle-mesh body-part shape loaded from a file."""

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        max_vertices: int = 5000,
        decimation_strategy: Literal["quadric", "uniform"] = "quadric",
    ) -> "MeshShape":
        """Load a mesh, decimate to ``max_vertices`` if needed.

        Supported formats: STL (ascii + binary), OBJ, PLY, GLB.
        Raises FileNotFoundError, ValueError on malformed file.
        """

    def __init__(
        self,
        vertices: np.ndarray,    # (V, 3)
        faces: np.ndarray,       # (F, 3) int
        rest_dimensions: tuple[float, float, float],   # bounding box extents
        source_path: Path | None = None,
    ) -> None: ...
```

### Loader internals (`shapes/_mesh_io.py`)

- Use `trimesh` for the file IO + mesh ops (already a transitive dependency via `humanoid_character_builder/mesh/`).
- Convert to canonical NumPy arrays before storing — never expose the trimesh object publicly.
- Rest dimensions = oriented-bounding-box extents (use `trimesh.bounds.oriented_bounds`).
- Re-centre the mesh on its bounding-box centroid, then store.

### Decimator (`shapes/_mesh_decimation.py`)

- Quadric edge-collapse decimation when `len(vertices) > max_vertices`.
- Fallback to uniform face removal if quadric decimation fails (e.g. non-manifold).
- Reuse helpers from `src/shared/python/humanoid_character_builder/mesh/_cg_decimation.py` if their contract fits — DRY rule applies. If contract doesn't fit, the new helper documents WHY in its docstring.

### Format support matrix

| Extension | Read | Notes                            |
| --------- | ---- | -------------------------------- |
| `.stl`    | ✓    | binary or ascii                  |
| `.obj`    | ✓    | vertex-only; mtl ignored in v1   |
| `.ply`    | ✓    | binary or ascii                  |
| `.glb`    | ✓    | first mesh node only             |
| `.gltf`   | ✗    | rejected with "use .glb" message |

## Tests

`tests/unit/body_part_viz/shapes/test_mesh_shape.py`:

- Synthesize a known triangle-mesh on disk (use `trimesh.creation.box(extents=(1,2,3))` as fixture; write to tmp_path as STL/OBJ/PLY).
- Load each format; assert vertex / face count + bounding-box extents.
- Bad path → `FileNotFoundError`.
- `.gltf` → `ValueError("use .glb")`.
- 100k-vertex synthetic mesh → decimated to ≤ 5000 vertices.
- Loaded mesh's `rest_dimensions` matches bbox of input within 1e-6.

`tests/unit/body_part_viz/shapes/test_mesh_decimation.py`:

- Quadric decimation reduces a high-poly icosphere to target.
- Uniform fallback runs when quadric fails on non-manifold.

## Performance

Decimating a 100k-vertex mesh to 5k should run in < 1 s on a typical office laptop. If trimesh's quadric decimation is too slow, fall back to fast uniform decimation and surface a warning.

## Acceptance criteria

- [ ] `MeshShape` implements `BodyPartShape`.
- [ ] Loader supports STL, OBJ, PLY, GLB; rejects gltf with helpful message.
- [ ] Decimation honours `max_vertices`; quadric → uniform fallback.
- [ ] Bounding-box extents stored as `rest_dimensions`.
- [ ] ≥ 90% line coverage.
- [ ] mypy + ruff + file-size budget clean.

## Files touched

- New: `src/shared/python/body_part_viz/shapes/mesh_shape.py`
- New: `src/shared/python/body_part_viz/shapes/_mesh_io.py`
- New: `src/shared/python/body_part_viz/shapes/_mesh_decimation.py`
- New: `tests/unit/body_part_viz/shapes/test_mesh_shape.py`
- New: `tests/unit/body_part_viz/shapes/test_mesh_decimation.py`
