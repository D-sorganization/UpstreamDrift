# Loading Motion Targets

This guide walks through loading club, ball-aware, and full-body motion
targets and assembling them into a `MultiSourceTarget` for the
motion-matching pipeline. See
[ADR 0006](../../adr/0006-multi-source-motion-targets.md) for the
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
[ADR 0006](../../adr/0006-multi-source-motion-targets.md).

## Generic Capture Contract and Validation (#10361)

While the canonical tour captures (`TOUR_CAPTURE` and `TOUR_CAPTURE_IRON`) are pinned
by cryptographic hash, external users can load generic C3D files through a
`CaptureContract` without modifying library code:

```python
from pathlib import Path
from src.shared.python.motion_matching.tour_capture_contract import (
    CaptureContract,
    load_capture,
    validate_capture_contract,
)

# 1. Define expectations for an external capture
contract = CaptureContract(
    units="m",
    vertical_axis="y",
    required_segments=("trunk", "pelvis", "club"),
    label_map={
        "Subject1:LFHD": "HeadFront",
        "Subject1:RFHD": "HeadSide",
        "Subject1:LBHD": "HeadTop",
    },
    max_gap_fraction=0.5,
)

# 2. Audit the file before loading
report = validate_capture_contract(Path("data/my_capture.c3d"), contract=contract)
if not report.is_valid:
    print(f"Capture rejected: {report.reasons}")
    # e.g., 'invalid_units: expected m, found mm', 'missing_required_segment: club'
    report.raise_for_status()

# 3. Load with automatic label remapping and unit conversion (mm -> m)
capture = load_capture(Path("data/my_capture.c3d"), contract=contract)
```

Non-conforming files fail closed with actionable named diagnostic strings
(`invalid_units`, `missing_required_segment: <name>`, `missing_required_labels`,
`excessive_gap_fraction`, `missing_static_calibration`).

## Model Identifiability Probe (44-DOF Full-Body)

Before fitting target swings, the kinematic chain's observability is evaluated
via `probe_spec_identifiability`:

```python
from pathlib import Path
import json
from src.shared.python.motion_matching.identifiability import probe_spec_identifiability

spec = json.loads(Path("docs/development/full_body_models/full_body_spec_anthro_driver.json").read_text())

# 1. Uncalibrated baseline: lower limbs lack marker offsets (offset_m: null),
# reporting all 14 leg DOFs as strictly unobservable.
raw_result = probe_spec_identifiability(spec)
assert not raw_result.is_full_rank  # rank 28 of 44

# 2. With calibrated lower limb marker offsets and anthropometric prior regularization
# (prior_weight > 0.0), longitudinal spine trade-offs (#9769) are resolved to full rank.
calibrated_result = probe_spec_identifiability(
    spec,
    marker_offsets=calibrated_offsets,
    prior_weight=0.05,
)
assert calibrated_result.is_full_rank  # rank 44 of 44
```

