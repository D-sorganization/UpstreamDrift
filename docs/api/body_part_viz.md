# `body_part_viz` — API Reference

Public API surface of the `body_part_viz` toolkit. Hand-curated index;
the docstrings on each symbol in `src/shared/python/body_part_viz/` are
the authoritative reference.

The toolkit has no Sphinx / pdoc auto-build configured in
`pyproject.toml`; if a doc-build pipeline is added later, this file
should be regenerated from the docstrings rather than maintained by
hand.

For design rationale see
[ADR 0008](../adr/0008-body-part-viz-toolkit.md). For end-user
workflows see [`docs/user_guide/body_part_viz/`](../user_guide/body_part_viz/quickstart.md).

## Top-level package

`src/shared/python/body_part_viz/__init__.py` re-exports the public
surface:

| Symbol               | Kind                                               | Source module |
| -------------------- | -------------------------------------------------- | ------------- |
| `SCHEMA_VERSION`     | `int` constant (currently `2`)                     | `persistence` |
| `VALID_SHAPE_KINDS`  | tuple of valid `shape_kind` strings                | `persistence` |
| `VALID_FITTER_KINDS` | tuple of valid `fitter_kind` strings               | `persistence` |
| `BindingKind`        | `str`-enum (`BETWEEN_TWO`, `CLUSTER`, `ON_MARKER`) | `bindings`    |
| `BodyPartShape`      | runtime-checkable Protocol                         | `contracts`   |
| `FittedShape`        | frozen dataclass                                   | `_types`      |
| `MarkerBinding`      | frozen dataclass                                   | `bindings`    |
| `SegmentVizSet`      | frozen dataclass                                   | `persistence` |
| `SegmentVizSpec`     | frozen dataclass                                   | `persistence` |
| `ShapeFitter`        | runtime-checkable Protocol                         | `contracts`   |
| `ShapeRenderer`      | runtime-checkable Protocol                         | `contracts`   |
| `ShapeTheme`         | frozen dataclass                                   | `theme`       |
| `migrate_v1_to_v2`   | function                                           | `persistence` |

## Contracts (`body_part_viz.contracts`)

Three runtime-checkable Protocols. Implementations live under the
`shapes/`, `fitters/`, and `renderers/` sub-packages.

### `BodyPartShape`

```python
class BodyPartShape(Protocol):
    shape_id: str
    rest_dimensions: tuple[float, ...]

    def vertices_at_rest(self) -> np.ndarray: ...
    def faces(self) -> np.ndarray: ...
    def transform(self, fitted: FittedShape) -> np.ndarray: ...
```

A geometric body-part visualisation. Implementations expose rest-pose
vertices / faces and apply a `FittedShape` transform per frame.

### `ShapeFitter`

```python
class ShapeFitter(Protocol):
    def fit(
        self,
        shape: BodyPartShape,
        binding: MarkerBinding,
        markers_xyz: dict[str, np.ndarray],
    ) -> FittedShape: ...
```

Computes a per-frame `FittedShape` from a marker dictionary.

### `ShapeRenderer`

```python
class ShapeRenderer(Protocol):
    def add_shape(
        self,
        shape: BodyPartShape,
        fitted: FittedShape,
        theme: ShapeTheme,
    ) -> str: ...
    def update_frame(self, handle: str, frame_idx: int) -> None: ...
    def set_visible(self, handle: str, visible: bool) -> None: ...
    def remove(self, handle: str) -> None: ...
```

Backend-specific renderer. The contract is intentionally Qt- and
matplotlib-free so the same shape / fitter pair targets any backend.

## Dataclasses

### `FittedShape` (`body_part_viz._types`)

Frozen dataclass; per-frame placement output of any `ShapeFitter`.

| Field             | Type                                  | Notes                                                 |
| ----------------- | ------------------------------------- | ----------------------------------------------------- |
| `shape_id`        | `str`                                 | Stable, non-empty identifier.                         |
| `binding`         | `MarkerBinding`                       | The binding used for this fit.                        |
| `centroid`        | `np.ndarray` shape `(T, 3)`           | World-frame centroid per frame.                       |
| `rotation_matrix` | `np.ndarray` shape `(T, 3, 3)`        | Per-frame rotation.                                   |
| `scale`           | `np.ndarray` shape `(T, 3)`           | Anisotropic scale, strictly positive on valid frames. |
| `valid_mask`      | `np.ndarray` shape `(T,)`, dtype bool | Per-frame validity.                                   |

`__post_init__` validates shapes, dtypes, and per-frame finiteness on
valid frames.

### `MarkerBinding` (`body_part_viz.bindings`)

Frozen dataclass; rest-pose binding of a shape to one or more markers.

| Field                   | Type                 | Notes                                     |
| ----------------------- | -------------------- | ----------------------------------------- |
| `kind`                  | `BindingKind`        | `BETWEEN_TWO`, `CLUSTER`, or `ON_MARKER`. |
| `marker_names`          | `tuple[str, ...]`    | Length depends on `kind` (2, ≥3, or 1).   |
| `rest_dimensions`       | `tuple[float, ...]`  | Optional; entries strictly positive.      |
| `rest_orientation_quat` | `(w, x, y, z)` tuple | Unit quaternion within `1e-6`.            |

`__post_init__` enforces the per-kind marker count, finiteness +
positivity of `rest_dimensions`, and unit-norm of the quaternion.

### `ShapeTheme` (`body_part_viz.theme`)

Frozen dataclass; visual styling for a shape. All colour strings
validated via matplotlib's `is_color_like`.

| Field         | Type    | Default     | Notes                             |
| ------------- | ------- | ----------- | --------------------------------- |
| `color`       | `str`   | `"#1f77b4"` | Any matplotlib-recognised colour. |
| `opacity`     | `float` | `0.8`       | In `[0, 1]`.                      |
| `edge_color`  | `str`   | `"#000000"` | Any matplotlib-recognised colour. |
| `edge_width`  | `float` | `0.5`       | Non-negative, in points.          |
| `flat_shaded` | `bool`  | `True`      | Per-face vs. smooth shading.      |
| `group`       | `str`   | `"default"` | Logical palette group.            |

### `BindingKind` (`body_part_viz.bindings`)

`str`-valued enum:

| Member        | Value           | Marker count |
| ------------- | --------------- | ------------ |
| `BETWEEN_TWO` | `"between_two"` | exactly 2    |
| `CLUSTER`     | `"cluster"`     | ≥ 3          |
| `ON_MARKER`   | `"on_marker"`   | exactly 1    |

## Shapes (`body_part_viz.shapes`)

Concrete `BodyPartShape` implementations, each in its own module.

| Class            | Module            | Description                           |
| ---------------- | ----------------- | ------------------------------------- |
| `LineShape`      | `line_shape`      | Zero-volume line between two markers. |
| `CylinderShape`  | `cylinder_shape`  | Radius + length along local Z.        |
| `EllipsoidShape` | `ellipsoid_shape` | Three semi-axes `(a, b, c)`.          |
| `CapsuleShape`   | `capsule_shape`   | Cylinder + hemispherical caps.        |
| `MeshShape`      | `mesh_shape`      | Triangle mesh from STL/OBJ/PLY/GLB.   |
| `CompositeShape` | `composite_shape` | Composition of child shapes.          |

`MeshShape` exposes a `MeshShape.load(path, max_vertices=5000)`
classmethod that reads a file via `trimesh`, re-centres on OBB
centroid, and decimates to the vertex budget.

## Fitters (`body_part_viz.fitters`)

Concrete `ShapeFitter` strategies.

| Class                         | Module                   | Best for                                           |
| ----------------------------- | ------------------------ | -------------------------------------------------- |
| `BetweenTwoMarkersFitter`     | `between_two`            | Limbs (e.g. forearm between elbow + wrist).        |
| `ClusterKabschFitter`         | `cluster_kabsch`         | Rigid segments with a marker cluster (e.g. torso). |
| `ProcrustesAnisotropicFitter` | `procrustes_anisotropic` | Cluster fit with per-axis scale.                   |

## Renderers (`body_part_viz.renderers`)

Concrete `ShapeRenderer` backends.

| Class                | Module                | Notes                                                             |
| -------------------- | --------------------- | ----------------------------------------------------------------- |
| `MatplotlibRenderer` | `matplotlib_renderer` | Canonical 3D backend; suitable for ≲ 5 000 triangles per segment. |
| `PyQtGLRenderer`     | `pyqtgl_renderer`     | GPU-accelerated alternative for high-poly scenes; same Protocol.  |

`MatplotlibRenderer` is the canonical 3D rendering helper for any new
tool that needs marker / mesh rendering — see AGENTS.md section B.

## Persistence (`body_part_viz.persistence`)

JSON v2 schema for saving / loading segment-viz configurations.

### `SegmentVizSpec`

Frozen dataclass; one segment's full spec.

| Field          | Type             | Notes                              |
| -------------- | ---------------- | ---------------------------------- |
| `binding`      | `MarkerBinding`  | How the shape attaches to markers. |
| `shape_kind`   | `str`            | One of `VALID_SHAPE_KINDS`.        |
| `shape_params` | `dict[str, Any]` | Validated per-`shape_kind`.        |
| `fitter_kind`  | `str`            | One of `VALID_FITTER_KINDS`.       |
| `theme`        | `ShapeTheme`     | Visual styling.                    |
| `visible`      | `bool`           | Whether the segment is rendered.   |

Constructors: `SegmentVizSpec.from_dict(data)`. Serialisation:
`spec.to_dict()`. Numerics are rounded to `1e-9` for stable JSON
round-trip.

### `SegmentVizSet`

Frozen dataclass; schema-versioned collection.

| Field            | Type                         | Notes                                    |
| ---------------- | ---------------------------- | ---------------------------------------- |
| `schema_version` | `int`                        | Always `SCHEMA_VERSION` (currently `2`). |
| `segments`       | `tuple[SegmentVizSpec, ...]` | Ordered.                                 |

Methods:

- `SegmentVizSet.from_dict(payload)` — parse a JSON-loaded dict;
  auto-migrates v1.
- `SegmentVizSet.load(path)` — read JSON file; auto-migrates v1.
- `viz_set.to_dict()` — serialise to JSON-ready dict (always v2).
- `viz_set.save(path)` — write JSON v2 file.

### `migrate_v1_to_v2(v1_dict) -> dict`

Lift a legacy v1 `SegmentSet` payload (the pre-toolkit C3D Viewer
format) into a v2 dict. Maps `(a, b)` markers to a `between_two`
binding, `geometry` to `shape_kind`, and carries forward the legacy
`group` / `visible` / `radius` fields.

## Asset library (`body_part_viz.asset_library`)

### `ShapeLibrary`

Resolves named body-part shapes to `MeshShape` instances backed by a
manifest-described asset directory.

```python
from src.shared.python.body_part_viz.asset_library import ShapeLibrary

lib = ShapeLibrary.default()           # bundled assets/body_part_shapes/default
head_mesh = lib.get("head")            # cached MeshShape
binding = lib.binding_template("head") # MarkerBinding from manifest
names = lib.names()                    # tuple of available shape names
```

Constructors:

- `ShapeLibrary()` — load the bundled default library.
- `ShapeLibrary(asset_root)` — load any directory that follows the
  `manifest.json` schema documented in
  [`asset_author_guide.md`](../user_guide/body_part_viz/asset_author_guide.md).
- `ShapeLibrary.default()` — class method, equivalent to `ShapeLibrary()`.

Methods:

- `names() -> tuple[str, ...]` — manifest shape names in insertion order.
- `get(name) -> MeshShape` — load (and cache) the named mesh.
- `binding_template(name) -> MarkerBinding` — return the manifest's
  binding template.

## URDF bridge (`body_part_viz.urdf_bridge`)

Forward and inverse mapping between `BodyPartShape` and URDF
`<visual>` elements.

| Symbol                                                                                 | Kind           | Notes                                                                                           |
| -------------------------------------------------------------------------------------- | -------------- | ----------------------------------------------------------------------------------------------- |
| `DEFAULT_PACKAGE`                                                                      | `str` constant | Default `package://` prefix (`"body_part_viz"`).                                                |
| `shape_to_urdf_visual(shape, *, rest_origin_xyz=…, rest_origin_rpy=…, package_name=…)` | function       | Returns an `xml.etree.ElementTree.Element` (`<visual>`) or a list of them for `CompositeShape`. |
| `urdf_to_shape(visual)`                                                                | function       | Inverse mapping; recovers the original shape within `1e-9` for supported kinds.                 |

Mapping (forward):

| Shape            | URDF                                                                         |
| ---------------- | ---------------------------------------------------------------------------- |
| `LineShape`      | `ValueError` (URDF cannot render lines)                                      |
| `CylinderShape`  | `<cylinder length=L radius=R>`                                               |
| `EllipsoidShape` | `<mesh filename="package://body_part_viz/__bpv_ellipsoid__a_b_c.obj">`       |
| `CapsuleShape`   | `<mesh filename="package://body_part_viz/__bpv_capsule__length_radius.obj">` |
| `MeshShape`      | `<mesh filename="package://body_part_viz/<stem>.<ext>">`                     |
| `CompositeShape` | `list[Element]` (caller wraps in `<link>`)                                   |

The `__bpv_ellipsoid__` / `__bpv_capsule__` filename conventions
encode the geometry in the filename because URDF has no native
capsule and only an isotropic-scale sphere; `urdf_to_shape` decodes
them.

## Tests

Reference tests live under `tests/unit/body_part_viz/` mirroring the
source layout. Useful entry points for adding new tests:

- `test_imports.py` — smoke test that the public surface re-exports
  cleanly.
- `test_contracts.py` — Protocol satisfaction tests; copy the pattern
  when adding a new shape / fitter / renderer.
- `test_persistence.py` — JSON round-trip and v1→v2 migration tests.
- `test_asset_library.py` — manifest validation; mirror this when
  adding library entries.
- `test_urdf_bridge.py` — forward / inverse URDF mapping tests.
