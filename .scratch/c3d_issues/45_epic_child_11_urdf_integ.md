# feat(urdf-generator): bind URDF link visuals to body_part_viz shapes (cross-tool reuse)

Depends on the `body_part_viz` package (#1–#8) and asset library (#7).

## Why

The URDF generator at `src/shared/python/humanoid_character_builder/generators/urdf_generator.py` ships URDF visual elements as `<box>` / `<cylinder>` / `<sphere>` / `<mesh>`. These should map onto the same shape contracts the C3D Viewer + matcher use, so a user who imports a custom mesh into the C3D Viewer can use it as the URDF visual link as well.

## What

### Bridge module

`src/shared/python/body_part_viz/urdf_bridge.py`:

```python
def shape_to_urdf_visual(
    shape: BodyPartShape,
    *,
    rest_origin_xyz: tuple[float, float, float] = (0, 0, 0),
    rest_origin_rpy: tuple[float, float, float] = (0, 0, 0),
) -> ET.Element:
    """Translate a body_part_viz shape into a URDF <visual> element.

    - LineShape -> not supported (raises ValueError).
    - CylinderShape -> <cylinder length=... radius=...>
    - EllipsoidShape -> <mesh> referencing a generated icosphere with anisotropic scale
    - CapsuleShape -> <mesh> (no native URDF capsule)
    - MeshShape -> <mesh filename=package://...>
    - CompositeShape -> a list of <visual> elements (caller wraps in a link).
    """


def urdf_to_shape(
    visual_element: ET.Element,
    asset_resolver: Callable[[str], Path],
) -> BodyPartShape:
    """Inverse mapping for round-trip."""
```

### URDF generator wiring

- `urdf_generator.py` accepts a `BodyPartShape` per link instead of a hardcoded `<cylinder>`.
- `urdf_geometry.py:create_geometry_dict` becomes a thin call into `shape_to_urdf_visual`.
- A new `--shape-library default` CLI flag picks the default ShapeLibrary as link visuals.

### Round-trip guarantee

- `MeshShape("foo.stl")` → URDF `<mesh filename="package://body_part_viz/foo.stl">` → loader recovers the same MeshShape.
- `CylinderShape(L, R)` → `<cylinder length=L radius=R>` → recovers within 1e-9.

## Tests

`tests/unit/body_part_viz/test_urdf_bridge.py`:

- Round-trip every supported shape kind.
- LineShape → ValueError ("URDF cannot render line visuals; use cylinder").
- CompositeShape → multi-visual list.

`tests/unit/humanoid_character_builder/test_urdf_generator_with_shapes.py`:

- Generate a URDF using the default ShapeLibrary; assert each link has the expected `<visual>` payload.

## Acceptance criteria

- [ ] Bridge module covers Cylinder / Ellipsoid / Capsule / Mesh / Composite.
- [ ] URDF generator accepts `BodyPartShape` per link.
- [ ] Round-trip preserves shape identity within numerical tolerance.
- [ ] ≥ 90% line coverage on the bridge.
- [ ] Existing humanoid_character_builder tests still pass.

## Files touched

- New: `src/shared/python/body_part_viz/urdf_bridge.py`
- Edit: `src/shared/python/humanoid_character_builder/generators/urdf_generator.py`
- Edit: `src/shared/python/humanoid_character_builder/generators/urdf_geometry.py`
- New: `tests/unit/body_part_viz/test_urdf_bridge.py`
- New: `tests/unit/humanoid_character_builder/test_urdf_generator_with_shapes.py`
