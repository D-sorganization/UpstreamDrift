# feat(motion-matching): body-skeleton segments — connect anatomical markers into a stick figure

## Why

A `BodyTarget` carries 28 disjoint marker positions per frame. To visualise the body in the matcher view as a **moving stick figure** (rather than a point cloud) we need a canonical set of **segments** (pairs of marker names) that describe the skeleton.

The connectivity is data-driven — the C3D files use a Plug-in-Gait subset, and the segment list is fully determined by which markers are present.

## What to build

`src/shared/python/motion_matching/body_skeleton.py`:

```python
@dataclass(frozen=True)
class BodySegment:
    a: str
    b: str
    group: Literal["torso", "head", "left_arm", "right_arm",
                   "left_leg", "right_leg", "pelvis"]


def default_body_segments(marker_names: Sequence[str]) -> tuple[BodySegment, ...]:
    """Return the segment list for the given marker set.

    Only segments whose BOTH endpoints are present in marker_names are returned.
    Always safe to call: if some markers are missing the figure is partial,
    not broken.
    """
```

### Canonical segment table (28-marker anatomical subset)

| Group     | a              | b                          |
| --------- | -------------- | -------------------------- |
| pelvis    | `WaistLeft`    | `WaistRight`               |
| pelvis    | `WaistLeft`    | `WaistLBack`               |
| pelvis    | `WaistRight`   | `WaistRBack`               |
| pelvis    | `WaistLBack`   | `WaistRBack`               |
| torso     | `BackTop`      | `BackLeft`                 |
| torso     | `BackTop`      | `BackRight`                |
| torso     | `BackTop`      | `WaistLBack` (proxy spine) |
| torso     | `BackTop`      | `WaistRBack`               |
| head      | `HeadTop`      | `HeadFront`                |
| head      | `HeadTop`      | `HeadSide`                 |
| left_arm  | `LShoulderTop` | `LShoulderBack`            |
| left_arm  | `LShoulderTop` | `LUArmHigh`                |
| left_arm  | `LUArmHigh`    | `LElbowOut`                |
| left_arm  | `LElbowOut`    | `LWristTop`                |
| right_arm | `RShoulderTop` | `RShoulderBack`            |
| right_arm | `RShoulderTop` | `RUArmHigh`                |
| right_arm | `RUArmHigh`    | `RElbowOut`                |
| right_arm | `RElbowOut`    | `RWristTop`                |
| left_leg  | `WaistLeft`    | `LKneeOut`                 |
| left_leg  | `LKneeOut`     | `LAnkleOut`                |
| left_leg  | `LAnkleOut`    | `LToeIn`                   |
| left_leg  | `LToeIn`       | `LToeOut`                  |
| right_leg | `WaistRight`   | `RKneeOut`                 |
| right_leg | `RKneeOut`     | `RAnkleOut`                |
| right_leg | `RAnkleOut`    | `RToeIn`                   |
| right_leg | `RToeIn`       | `RToeOut`                  |

### Renderer integration

`src/shared/python/motion_matching/diagnostics/_skeleton_render.py` — extend with:

```python
def draw_body_target_frame(ax, target: BodyTarget, frame_idx: int, *,
                           segment_groups: Sequence[str] | None = None,
                           color_map: dict[str, str] | None = None,
                           linewidth: float = 1.5) -> None: ...
```

with a sensible default colour map (one colour per group). NaN-marker segments are silently skipped (any endpoint NaN → segment hidden for that frame).

## Generic naming

`BodySegment`, `default_body_segments`, `draw_body_target_frame`. No source-specific names.

## Acceptance criteria

- [ ] `BodySegment` and `default_body_segments` added.
- [ ] Returns only segments whose markers are present in input.
- [ ] `RShoulderTop` known-occluded → those segments NaN-skip cleanly each frame.
- [ ] `draw_body_target_frame` integrated and called by the matcher's animated preview.
- [ ] Unit tests cover segment-presence pruning, NaN-skip, group filtering.
- [ ] Mypy + ruff clean.

## Files touched

- New: `src/shared/python/motion_matching/body_skeleton.py`
- Edit: `src/shared/python/motion_matching/diagnostics/_skeleton_render.py`
- Edit: `src/shared/python/motion_matching/__init__.py`
- New: `tests/unit/motion_matching/test_body_skeleton.py`

## References

- Marker name list verified by ezc3d probe of the four reference C3D files.
- Existing 3D matplotlib helpers: `src/shared/python/motion_matching/diagnostics/_skeleton_render.py`.
