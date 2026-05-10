# feat(body-part-viz): primitive shapes — Line, Cylinder, Ellipsoid, Capsule, Composite

Depends on contracts (#1).

## Why

Five concrete `BodyPartShape` implementations covering the most-used cases without needing a mesh file. These cover ~80% of the user's segment-rendering needs.

## Shapes to ship

### `LineShape` (in `shapes/line_shape.py`)

- `rest_dimensions = (length,)`
- `vertices_at_rest()` returns `np.array([[0,0,0],[length,0,0]])`.
- `faces()` returns empty `(0, 3)` array.
- `transform(fitted)` honours fitted.centroid, rotation, scale[0] (length).

### `CylinderShape` (in `shapes/cylinder_shape.py`)

- `rest_dimensions = (length, radius)`
- N-facet cylinder (default 16 facets, configurable).
- `vertices_at_rest()` returns 2 caps + side ring vertices.
- `faces()` returns triangle indices for the two caps + side quads as triangle pairs.
- `transform(fitted)` applies length scale to local x-axis, radius scale to local y-z.
- Constructor: `CylinderShape(length=1.0, radius=0.05, n_facets=16)`.

### `EllipsoidShape` (in `shapes/ellipsoid_shape.py`)

- `rest_dimensions = (a, b, c)` (three semi-axes).
- UV-sphere triangulation (default 16 longitudes × 8 latitudes).
- `transform(fitted)` applies anisotropic scale per axis.

### `CapsuleShape` (in `shapes/capsule_shape.py`)

- Cylinder + two hemisphere caps (a "stadium" in 3D).
- `rest_dimensions = (length, radius)`.
- Reuses `CylinderShape` + half-`EllipsoidShape` instances internally; demonstrates the composite pattern but ships as a single shape.

### `CompositeShape` (in `shapes/composite_shape.py`)

- `rest_dimensions` = concatenated child dims.
- Holds a list of `(child_shape, local_transform)` pairs.
- `vertices_at_rest()` concatenates child vertices, transformed by their local_transform.
- `faces()` re-indexes per child.
- Used to build "head + neck + torso" in a single shape with a single fitter.

## Implementation notes

- Use NumPy only — no matplotlib or trimesh imports in this module.
- Constructors validate via DbC: `length > 0`, `radius > 0`, `n_facets >= 3`.
- For triangulation, reuse a small helper module `shapes/_mesh_primitives.py` with pure-NumPy `make_uv_sphere(n_longitudes, n_latitudes)`, `make_cylinder(length, radius, n_facets)`, `make_ellipsoid(a, b, c, n_lon, n_lat)` — these are testable in isolation.

## Tests

`tests/unit/body_part_viz/shapes/test_line_shape.py`:
- Vertex count == 2; face count == 0.
- `transform` with identity fitted shape returns input unchanged.
- `transform` with rotation rotates correctly (Procrustes-shaped check).

`tests/unit/body_part_viz/shapes/test_cylinder_shape.py`:
- Default 16-facet cylinder has `2 * (n_facets + 1) = 34` vertices, `4 * n_facets = 64` triangles.
- DbC: `n_facets < 3` raises ValueError; negative `radius` raises ValueError.
- Anisotropic scale applies length to x and radius to yz independently.

`tests/unit/body_part_viz/shapes/test_ellipsoid_shape.py`:
- Vertex count matches `n_lon * (n_lat + 1)`.
- All vertices satisfy `(x/a)^2 + (y/b)^2 + (z/c)^2 == 1` to 1e-9.

`tests/unit/body_part_viz/shapes/test_capsule_shape.py`:
- Vertex count = cylinder + 2 hemispheres.
- Capsule with `radius == length / 2` reduces to a sphere shape.

`tests/unit/body_part_viz/shapes/test_composite_shape.py`:
- Composite of 2 cylinders → vertex count is sum + each cylinder's local transform applied.
- DbC: empty children list raises ValueError.

`tests/unit/body_part_viz/shapes/test_mesh_primitives.py`:
- `make_uv_sphere(16, 8)`: 16*9 = 144 vertices.
- `make_cylinder(1, 0.5, 16)` produces watertight mesh.

## Acceptance criteria

- [ ] All 5 shapes implement the `BodyPartShape` Protocol (verified by `isinstance` Protocol runtime check).
- [ ] All shapes' constructors DbC-validate dimensions + facet counts.
- [ ] `_mesh_primitives.py` is pure-NumPy and reusable.
- [ ] ≥ 90% line coverage on the new files.
- [ ] mypy + ruff + file-size budget clean.
- [ ] No new production-code dependency outside what `body_part_viz` already needs (numpy only).

## Files touched

- New: `src/shared/python/body_part_viz/shapes/{line,cylinder,ellipsoid,capsule,composite}_shape.py`
- New: `src/shared/python/body_part_viz/shapes/_mesh_primitives.py`
- Edit: `src/shared/python/body_part_viz/shapes/__init__.py` (re-exports)
- New: `tests/unit/body_part_viz/shapes/test_*.py`
