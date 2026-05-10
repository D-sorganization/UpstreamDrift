# feat(body-part-viz): core contracts + dataclasses (Shape / Binding / Fitter / Theme)

Child of the [EPIC] body_part_viz tracking issue. **First** issue — every other child depends on these contracts.

## Why

Define the abstract surface that every shape, fitter, and renderer talks across. No primitives, no meshes, no rendering — pure contracts and dataclasses. Once this lands, the rest of the epic is N parallel implementations.

## Files to create

```
src/shared/python/body_part_viz/
├── __init__.py
├── contracts.py        # Protocols: BodyPartShape, ShapeFitter, ShapeRenderer
├── shapes/__init__.py  # placeholder; primitives + meshes land in #2/#3
├── fitters/__init__.py # placeholder; fitters land in #4
├── renderers/__init__.py # placeholder; backends land in #5/#6
├── theme.py            # ShapeTheme dataclass
├── bindings.py         # MarkerBinding dataclass + binding-type enum
└── _types.py           # FittedShape dataclass + helper aliases
```

## Public API

### `MarkerBinding` (in `bindings.py`)

```python
class BindingKind(StrEnum):
    """How a shape attaches to mocap markers."""
    BETWEEN_TWO = "between_two"   # length = ‖b - a‖, axis = b - a
    CLUSTER = "cluster"           # >=3 markers; rigid Kabsch + scale
    ON_MARKER = "on_marker"       # static at one marker (e.g. head sphere)


@dataclass(frozen=True)
class MarkerBinding:
    kind: BindingKind
    marker_names: tuple[str, ...]            # length depends on kind
    rest_dimensions: tuple[float, ...] = ()  # rest-pose lengths/diameters
    rest_orientation_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # invariants in __post_init__:
    #   BETWEEN_TWO -> len(marker_names) == 2
    #   CLUSTER     -> len(marker_names) >= 3
    #   ON_MARKER   -> len(marker_names) == 1
    #   rest_dimensions all positive
    #   quaternion unit-norm
```

### `BodyPartShape` (in `contracts.py`)

```python
@runtime_checkable
class BodyPartShape(Protocol):
    """A geometric body-part visualisation.

    Implementations: LineShape, CylinderShape, EllipsoidShape, CapsuleShape,
    MeshShape, CompositeShape (added in #2 / #3).
    """

    shape_id: str
    """Stable, human-readable identifier (e.g. 'cylinder', 'mesh:head_v1')."""

    rest_dimensions: tuple[float, ...]
    """Rest-pose dimension tuple. Semantics vary per shape:
    - line: (length,)
    - cylinder: (length, radius)
    - ellipsoid: (a, b, c) semi-axes
    - capsule: (length, radius)
    - mesh: (extent_x, extent_y, extent_z) of bounding box
    - composite: tuple of child rest_dimensions concatenated
    """

    def vertices_at_rest(self) -> np.ndarray:
        """Return ``(V, 3)`` vertex array in the shape's local frame."""
        ...

    def faces(self) -> np.ndarray:
        """Return ``(F, 3)`` triangle index array, or empty array for line shapes."""
        ...

    def transform(self, fitted: FittedShape) -> np.ndarray:
        """Return ``(V, 3)`` vertices after fitting transformation."""
        ...
```

### `ShapeFitter` (in `contracts.py`)

```python
@runtime_checkable
class ShapeFitter(Protocol):
    """Compute a per-frame transform from markers to a fitted shape."""

    def fit(
        self,
        shape: BodyPartShape,
        binding: MarkerBinding,
        markers_xyz: dict[str, np.ndarray],   # marker_name -> (T, 3)
    ) -> FittedShape:
        ...
```

### `FittedShape` (in `_types.py`)

```python
@dataclass(frozen=True)
class FittedShape:
    """Per-frame placement of a shape."""

    shape_id: str
    binding: MarkerBinding
    centroid: np.ndarray             # (T, 3) world-frame centroid per frame
    rotation_matrix: np.ndarray      # (T, 3, 3) rotation per frame
    scale: np.ndarray                # (T, 3) anisotropic scale per frame
    valid_mask: np.ndarray           # (T,) bool — frame valid iff all
                                     # source markers finite
    # __post_init__: shapes consistent; valid_mask correctly populated
```

### `ShapeRenderer` (in `contracts.py`)

```python
@runtime_checkable
class ShapeRenderer(Protocol):
    """Backend-specific renderer (matplotlib, pyqtgraph, etc.).

    Implementations live in ``renderers/`` and are added by #5 (matplotlib)
    and #6 (pyqtgraph). This contract intentionally exposes no Qt or
    matplotlib types; subclasses pick up the backend as needed.
    """

    def add_shape(
        self,
        shape: BodyPartShape,
        fitted: FittedShape,
        theme: ShapeTheme,
    ) -> str:
        """Return a handle that can later be used to update / remove."""
        ...

    def update_frame(self, handle: str, frame_idx: int) -> None:
        ...

    def set_visible(self, handle: str, visible: bool) -> None:
        ...

    def remove(self, handle: str) -> None:
        ...
```

### `ShapeTheme` (in `theme.py`)

```python
@dataclass(frozen=True)
class ShapeTheme:
    color: str = "#1f77b4"           # any matplotlib-recognised
    opacity: float = 0.8             # 0.0..1.0
    edge_color: str = "#000000"
    edge_width: float = 0.5
    flat_shaded: bool = True
    group: str = "default"           # used by themed colour palettes
    # Validates color strings, opacity range, edge_width >= 0.
```

## Validation rules (DbC)

Every dataclass validates in `__post_init__`:
- arrays are NumPy ndarrays of the documented shape;
- floats finite, in their documented range;
- string identifiers non-empty;
- enum values are members of the enum.

Public functions on the Protocols document postconditions in their docstring.

## Tests

`tests/unit/body_part_viz/test_contracts.py`:

- `BindingKind` round-trips through string serialisation.
- `MarkerBinding(BETWEEN_TWO, ("a", "b"))` succeeds; `("a",)` raises ValueError.
- `MarkerBinding` rejects negative `rest_dimensions`, non-unit quaternion.
- `ShapeTheme` rejects opacity > 1.0, negative edge_width, empty color string.
- A stub `BodyPartShape` instance satisfies `isinstance(stub, BodyPartShape)` (Protocol runtime check).
- Same for `ShapeFitter`, `ShapeRenderer`.
- `FittedShape` rejects mismatched array dimensions.

`tests/unit/body_part_viz/test_imports.py`:

- Public API exports load without raising.
- `from src.shared.python.body_part_viz import BodyPartShape, ShapeFitter, ShapeRenderer, MarkerBinding, ShapeTheme, FittedShape, BindingKind` succeeds.

## Acceptance criteria

- [ ] Five files created: `contracts.py`, `bindings.py`, `theme.py`, `_types.py`, `__init__.py`.
- [ ] Three Protocols + four dataclasses + one StrEnum.
- [ ] `__init__.py` re-exports the public API.
- [ ] All dataclasses are frozen.
- [ ] All public callables decorated with appropriate `precondition` / `postcondition`.
- [ ] ≥ 95% line coverage on the new files.
- [ ] mypy strict on the new module.
- [ ] ruff check + ruff format clean.
- [ ] file-size budget clean.
- [ ] No production code outside `src/shared/python/body_part_viz/` is touched.

## Out of scope

- Implementations of the shapes / fitters / renderers (separate child issues).
- Persistence schema (issue #8).
- Asset library (issue #7).

## References

- Existing `GeometryType` / `GeometrySpec` in `src/shared/python/humanoid_character_builder/core/segment_definitions.py` — DO NOT modify; this new package is intentionally separate (the humanoid_character_builder ships URDF segments; we ship visualisation overlays). Issue #11 wires them together.
- Existing `SegmentSpec` in `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/services/segment_set_io.py` — also stays untouched in this issue; superseded by `SegmentVizSpec` in issue #8.
