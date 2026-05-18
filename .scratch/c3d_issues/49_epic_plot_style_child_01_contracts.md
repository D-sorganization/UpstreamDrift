# feat(plot-style): core contracts + dataclasses (MarkerStyle / ColorScale / DataChannel / PlotStyleSpec)

Child #1 of plot_style EPIC. **First** issue — every other child depends on these contracts. No primitives, no rendering — pure data model.

## Why

Define the abstract surface. Once this lands, the rest of the epic is N parallel implementations.

## Files to create

```
src/shared/python/plot_style/
├── __init__.py            # public re-exports
├── contracts.py           # Protocols: MarkerRenderer, ColorResolver
├── markers.py             # MarkerStyle, MarkerShape (StrEnum)
├── colors.py              # ColorScale + StaticColor / PaletteColor / DataDrivenColor dataclasses
├── colormaps.py           # ColormapId (StrEnum), CustomColormap dataclass
├── channels.py            # DataChannel dataclass + helpers
├── persistence.py         # PlotStyleSpec, PlotStyleSet
└── _types.py              # internal helper aliases (rgba tuple, etc.)
```

Plus stub directories: `widgets/__init__.py`, `renderers/__init__.py` for later children.

## Public API spec

### `MarkerShape` (in `markers.py`)

```python
class MarkerShape(str, Enum):
    """Built-in marker shape identifiers."""
    SPHERE = "sphere"
    CUBE = "cube"
    CROSS = "cross"
    STAR = "star"
    DIAMOND = "diamond"
    PLUS = "plus"
    POINT = "point"           # 1-px point, fastest path
    CUSTOM_MESH = "custom_mesh"  # paired with a CustomMeshSpec
```

### `MarkerStyle` (in `markers.py`)

```python
@dataclass(frozen=True)
class MarkerStyle:
    """All visual properties of a marker except its position."""
    shape: MarkerShape = MarkerShape.SPHERE
    size_px: float = 6.0           # > 0
    edge_color: str = "#000000"    # any matplotlib-recognised
    edge_width: float = 0.5        # >= 0
    fill_color: ColorScale = StaticColor("#1f77b4")
    opacity: float = 1.0           # 0..1
    custom_mesh: CustomMeshSpec | None = None  # required if shape == CUSTOM_MESH

    # __post_init__ validates:
    # - size_px > 0
    # - edge_width >= 0
    # - opacity in [0, 1]
    # - edge_color is a parseable matplotlib color
    # - fill_color is a ColorScale instance
    # - shape == CUSTOM_MESH iff custom_mesh is not None
```

### `ColorScale` (in `colors.py`)

```python
ColorScale = Union[StaticColor, PaletteColor, DataDrivenColor]


@dataclass(frozen=True)
class StaticColor:
    """A constant color across all markers / frames."""
    hex_value: str
    # __post_init__ validates hex_value is a parseable matplotlib color.

    def resolve(self, frame_idx: int, marker_idx: int | None = None) -> tuple[float, ...]:
        """Return (r, g, b, a) tuple in [0, 1]."""


@dataclass(frozen=True)
class PaletteColor:
    """A color picked from a named palette by index."""
    palette_name: str          # e.g. "tab10", "set2", "scientific_violet"
    palette_index: int         # 0..palette_size-1

    def resolve(self, frame_idx, marker_idx=None) -> tuple[float, ...]:
        ...


@dataclass(frozen=True)
class DataDrivenColor:
    """Color derived from a DataChannel + colormap.

    For each frame (or each (frame, marker) pair), look up the channel
    value, normalise via vmin/vmax, and sample the colormap.

    Channel may be:
    - shape (T,)     -> one color per frame; same color for every marker
    - shape (T, M)   -> per-marker color per frame
    """
    channel: DataChannel
    colormap: ColormapId
    vmin: float | None = None      # auto-detect from channel if None
    vmax: float | None = None      # auto-detect from channel if None
    nan_color: str = "#888888"     # color for non-finite values

    def resolve(self, frame_idx, marker_idx=None) -> tuple[float, ...]:
        ...
```

### `ColormapId` (in `colormaps.py`)

```python
class ColormapId(str, Enum):
    """Built-in + semantic colormaps.

    Matplotlib built-ins (passed through):
    """
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    MAGMA = "magma"
    INFERNO = "inferno"
    CIVIDIS = "cividis"
    TURBO = "turbo"
    COOLWARM = "coolwarm"
    SPECTRAL = "spectral"

    # Semantic — registered by us, alias to a built-in:
    VELOCITY = "velocity"           # alias for plasma
    FORCE = "force"                 # alias for inferno
    ACCELERATION = "acceleration"   # alias for turbo
    HEIGHT = "height"               # alias for viridis
    GENERIC_DIVERGING = "generic_diverging"   # alias for coolwarm


@dataclass(frozen=True)
class CustomColormap:
    """User-defined colormap from a list of stops."""
    name: str                       # unique, non-empty
    stops: tuple[tuple[float, str], ...]   # (position 0..1, hex)
    # __post_init__ validates:
    # - len(stops) >= 2
    # - positions strictly increasing in [0, 1]
    # - hex strings parseable
```

### `DataChannel` (in `channels.py`)

```python
@dataclass(frozen=True)
class DataChannel:
    """A per-frame (or per-(frame, marker)) scalar source.

    Examples:
    - clubhead speed (T,) m/s
    - per-marker residuals (T, M)
    - joint torque (T, n_joints)
    """
    name: str                       # unique, non-empty
    values: np.ndarray              # shape (T,) or (T, M)
    unit: str = ""                  # display unit string
    # __post_init__ validates:
    # - name non-empty
    # - values is np.ndarray with ndim in {1, 2}
    # - values dtype is numeric

    @classmethod
    def from_array(cls, name: str, values: np.ndarray, unit: str = "") -> "DataChannel":
        """Convenience constructor."""

    def value_at(self, frame_idx: int, marker_idx: int | None = None) -> float:
        """Get the scalar value at (frame, marker). Returns NaN if out of bounds."""

    def auto_range(self) -> tuple[float, float]:
        """Return (min, max) over all finite values for auto vmin/vmax."""
```

### `MarkerRenderer` Protocol (in `contracts.py`)

```python
@runtime_checkable
class MarkerRenderer(Protocol):
    """Backend-specific marker renderer (matplotlib, pyqtgraph, etc.)."""

    def add_markers(
        self,
        positions: np.ndarray,       # (T, M, 3) or (T, 3)
        style: MarkerStyle,
        label: str = "",
    ) -> str:
        """Return a handle for later updates."""

    def update_frame(self, handle: str, frame_idx: int) -> None: ...
    def update_style(self, handle: str, style: MarkerStyle) -> None: ...
    def set_visible(self, handle: str, visible: bool) -> None: ...
    def remove(self, handle: str) -> None: ...
```

### `ColorResolver` Protocol (in `contracts.py`)

```python
@runtime_checkable
class ColorResolver(Protocol):
    """Resolves a ColorScale to an (r, g, b, a) tuple per frame / marker."""

    def resolve_one(
        self, scale: ColorScale, frame_idx: int, marker_idx: int | None = None
    ) -> tuple[float, float, float, float]:
        ...

    def resolve_array(
        self, scale: ColorScale, n_frames: int, n_markers: int | None = None
    ) -> np.ndarray:                  # shape (n_frames, [n_markers,] 4)
        """Bulk-resolve for performance — useful for pre-computed LUTs."""
```

### `PlotStyleSpec` and `PlotStyleSet` (in `persistence.py`)

```python
@dataclass(frozen=True)
class PlotStyleSpec:
    """A single named style entry — usually one per marker group."""
    name: str                      # unique within a set
    target: str                    # e.g. "marker_group:body" / "trace:clubhead"
    style: MarkerStyle


@dataclass(frozen=True)
class PlotStyleSet:
    """A collection of named styles, JSON-persistable."""
    schema_version: int = 1
    entries: tuple[PlotStyleSpec, ...] = ()

    @classmethod
    def load(cls, path: Path) -> "PlotStyleSet": ...
    def save(self, path: Path) -> None: ...
```

## Validation rules (DbC)

Every dataclass validates in `__post_init__`:

- Strings non-empty.
- Numerical fields in their documented range.
- ColorScale fields: hex strings parseable, palette index in range, channel array dtype numeric.
- Enum values are members of their enum.

Public functions on the Protocols document postconditions in their docstring.

## Tests

`tests/unit/plot_style/test_contracts.py`:

- Each Protocol satisfied by a stub — `isinstance(stub, MarkerRenderer)` etc.

`tests/unit/plot_style/test_markers.py`:

- `MarkerStyle` validates every constraint (one happy + one fail per rule).
- `shape == CUSTOM_MESH` requires `custom_mesh` set.

`tests/unit/plot_style/test_colors.py`:

- `StaticColor` validates hex.
- `PaletteColor` validates palette name + index.
- `DataDrivenColor` validates channel + colormap.
- All three `resolve()` methods return a 4-tuple in [0, 1].

`tests/unit/plot_style/test_colormaps.py`:

- `ColormapId` round-trips through string.
- `CustomColormap` validates stops monotonic + parseable.

`tests/unit/plot_style/test_channels.py`:

- `DataChannel.from_array` happy path.
- 1-D and 2-D shapes both supported.
- `value_at` returns NaN for OOB index, not raises.
- `auto_range` ignores NaN.

`tests/unit/plot_style/test_persistence.py`:

- Round-trip of every ColorScale variant.
- v1 schema with extra unknown keys is tolerated (forward-compat).

`tests/unit/plot_style/test_imports.py`:

- All public names import cleanly.

## Acceptance criteria

- [ ] Files created per the layout above.
- [ ] All Protocols are `@runtime_checkable`.
- [ ] All dataclasses are frozen.
- [ ] All public callables decorated with appropriate `precondition`/`postcondition`.
- [ ] ≥ 95% line coverage on the new files.
- [ ] mypy strict on the new module.
- [ ] ruff check + format clean.
- [ ] file-size budget clean.
- [ ] No production code outside `src/shared/python/plot_style/` is touched.

## Out of scope

- Implementations (separate child issues).
- Qt widgets (issue #6).
- Renderers (#7 / #8).
- Asset library / preset themes (issue #9).
