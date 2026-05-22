# Loading Motion Targets

This guide walks through loading club, ball-aware, and full-body motion
targets and assembling them into a `MultiSourceTarget` for the
motion-matching pipeline. See
[ADR 0018](../../adr/0018-multi-source-motion-targets.md) for the
design rationale.

## Source formats at a glance

| Source                     | Loader             | Returns      |
| -------------------------- | ------------------ | ------------ |
| Wiffle-style xlsx workbook | `load_club_target` | `ClubTarget` |
| C3D club capture           | `load_club_target` | `ClubTarget` |
| MAT club capture           | `load_club_target` | `ClubTarget` |
| C3D full-body marker set   | `load_body_target` | `BodyTarget` |

Both dispatchers route on file extension, so call sites do not need to
special-case the source format.

## Loading a club track

```python
from pathlib import Path

from src.shared.python.motion_matching import load_club_target
from src.shared.python.motion_matching.target import AlignOptions

# xlsx workbook -- a sheet name is required.
club = load_club_target(
    Path("data/Club_Data.xlsx"),
    sheet="Driver",
    opts=AlignOptions(),
)

# C3D capture -- no sheet argument; impact alignment is read from the
# event channel.
club = load_club_target(Path("data/C3D_TA_Driver.c3d"))

# MAT capture -- routed the same way.
club = load_club_target(Path("data/some_capture.mat"))
```

The returned `ClubTarget` carries a unit-normalised quaternion track,
position samples on the simulation timegrid, and impact-pinned
indices. Validation runs in `__post_init__`; an invalid file raises
`ValueError`.

## Loading a body track

```python
from src.shared.python.motion_matching import load_body_target

body = load_body_target(Path("data/C3D_TA_Driver.c3d"))
```

`load_body_target` returns a `BodyTarget` containing a labelled
mapping of segment trajectories. The default segment set is exposed
by the `default_body_segments` helper for callers that want to
restrict cost terms to a known subset:

```python
from src.shared.python.motion_matching import default_body_segments

segments = default_body_segments()
# ('pelvis', 'spine', 'torso', 'left_shoulder', 'right_shoulder',
#  'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
#  'left_hand', 'right_hand', ...)
```

Unsupported extensions raise `ValueError` with the supported set listed
in the message, matching the `load_club_target` behaviour.

## Sharing a clock between body and club

When the body capture and the club capture come from the same session
and share an impact event, pass `impact_source` so the aggregator
re-times both onto the same grid:

```python
from src.shared.python.motion_matching import MultiSourceTarget

target = MultiSourceTarget(
    club=club,
    body=body,
    impact_source="club",   # club track defines t=0 at impact
)
```

`impact_source` accepts `"club"`, `"body"`, or `"explicit"`. If the
two tracks already agree on time, `MultiSourceTarget` validates that
their time vectors are equal within `TIME_EPS`. If not, the body
track is re-aligned to the chosen impact source and re-validated.

## Worked example: a representative full-body mocap capture

The repository ships with a small representative full-body mocap
capture, `data/C3D_TA_Driver.c3d`, that contains both a club track and
a body marker set on the same clock. This example walks through
loading both, assembling a `MultiSourceTarget`, and querying the
aggregator from cost-function code.

```python
from pathlib import Path

from src.shared.python.motion_matching import (
    MultiSourceTarget,
    load_body_target,
    load_club_target,
)

capture_path = Path("data/C3D_TA_Driver.c3d")

# 1.  Load the club track.  AlignOptions defaults to a 1 kHz sim grid
#     pinned at impact.
club = load_club_target(capture_path)

# 2.  Load the body markers from the same file.  The body loader
#     reads the same impact event, so both tracks land on the same
#     clock.
body = load_body_target(capture_path)

# 3.  Assemble a MultiSourceTarget.  impact_source="club" pins the
#     shared clock to the club's impact frame.
target = MultiSourceTarget(
    club=club,
    body=body,
    impact_source="club",
)

assert target.has_club()
assert target.has_body()
assert not target.has_ball()  # this capture has no launch ball state
```

A cost-function call site then dispatches on the `has_*()` accessors:

```python
def total_cost(target: MultiSourceTarget, prediction) -> float:
    cost = 0.0
    if target.has_club():
        cost += club_pose_cost(target.club, prediction.club)
    if target.has_body():
        cost += body_segment_cost(target.body, prediction.body)
    if target.has_ball():
        cost += launch_state_cost(target.ball, prediction.launch)
    return cost
```

Because the dispatcher / accessor pattern is uniform, adding a new
track type later (for example, a force-plate channel) is a matter of
adding one accessor and one dataclass — call sites that do not need
the new track do not change.

## Naming conventions

File-on-disk names follow whatever the capture system publishes; this
guide uses the literal filenames present in `data/`. Everything
above the loader boundary uses source-agnostic names: `BodyTarget`
not `MarkerSetTarget`, `load_body_target` not the per-vendor variant.
The motivation is documented in
[ADR 0018](../../adr/0018-multi-source-motion-targets.md).
