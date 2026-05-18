# feat(body-part-viz): SegmentVizSpec JSON v2 persistence (extends current SegmentSpec)

Depends on contracts (#1), primitives (#2), meshes (#3), fitters (#4), asset library (#7).

## Why

The current `SegmentSpec` (in `src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/services/segment_set_io.py`) only knows `geometry: line | cylinder`. The new `SegmentVizSpec` knows about every shape kind, every fitter, theming, and library shape ids — and round-trips to JSON v2.

## What

`src/shared/python/body_part_viz/persistence.py`:

```python
@dataclass(frozen=True)
class SegmentVizSpec:
    """A single segment's full visualisation spec."""

    binding: MarkerBinding
    shape_kind: Literal[
        "line", "cylinder", "ellipsoid", "capsule",
        "mesh_file", "library_shape", "composite"
    ]
    shape_params: dict[str, Any]   # shape-specific params, validated below
    fitter_kind: Literal["between_two", "cluster_kabsch", "procrustes_anisotropic"]
    theme: ShapeTheme
    visible: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "SegmentVizSpec": ...
    def to_dict(self) -> dict: ...

@dataclass(frozen=True)
class SegmentVizSet:
    schema_version: int = 2
    segments: tuple[SegmentVizSpec, ...] = ()

    @classmethod
    def load(cls, path: Path) -> "SegmentVizSet": ...
    def save(self, path: Path) -> None: ...
```

## v1 → v2 migration

When loading a v1 file (`SegmentSpec` shape):

```python
def migrate_v1_to_v2(v1_dict: dict) -> dict:
    """v1 had geometry='line'|'cylinder' and a, b marker names.
    Map to v2 with binding kind=BETWEEN_TWO, fitter=between_two,
    shape_kind=line|cylinder, theme=ShapeTheme(group=v1['group']).
    """
```

Migration is automatic on load; round-trip ALWAYS writes v2.

## v2 sample

```json
{
  "schema_version": 2,
  "segments": [
    {
      "binding": {
        "kind": "between_two",
        "marker_names": ["WaistLeft", "WaistRight"],
        "rest_dimensions": [0.32]
      },
      "shape_kind": "cylinder",
      "shape_params": { "length": 0.32, "radius": 0.04, "n_facets": 16 },
      "fitter_kind": "between_two",
      "theme": { "color": "#1f77b4", "opacity": 0.8, "group": "pelvis" },
      "visible": true
    },
    {
      "binding": {
        "kind": "between_two",
        "marker_names": ["LShoulderTop", "LElbowOut"],
        "rest_dimensions": [0.3]
      },
      "shape_kind": "library_shape",
      "shape_params": { "library_name": "default", "shape_id": "upper_arm" },
      "fitter_kind": "between_two",
      "theme": { "color": "#ff7f0e", "opacity": 1.0, "group": "left_arm" },
      "visible": true
    }
  ]
}
```

## Tests

`tests/unit/body_part_viz/test_persistence.py`:

- v1 sample loads; round-trip writes v2.
- All shape kinds round-trip.
- All fitter kinds round-trip.
- DbC: unknown shape_kind on load → `ValueError` listing valid kinds.
- DbC: unknown fitter_kind → `ValueError`.
- DbC: shape_params missing required keys → `ValueError`.

## Acceptance criteria

- [ ] `SegmentVizSpec` + `SegmentVizSet` dataclasses, frozen, validated.
- [ ] v1 → v2 migration on load.
- [ ] Round-trip preserves every field bit-exact (rounded to 1e-9 in JSON).
- [ ] ≥ 95% line coverage.

## Files touched

- New: `src/shared/python/body_part_viz/persistence.py`
- New: `tests/unit/body_part_viz/test_persistence.py`
- Edit: `src/shared/python/body_part_viz/__init__.py`
