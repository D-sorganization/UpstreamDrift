# Pose Studio — Save formats

Every save path Pose Studio exposes routes through
`src.shared.python.pose_interchange.pose_io`. This page lists every
on-disk shape, plus a minimal downstream-reader example for each
format so you can integrate Pose Studio output into any consumer
(MATLAB pipeline, fit-swing solver, notebook, dashboard).

> The same engine identifiers everywhere: `drake`, `mujoco`,
> `pinocchio`, `opensim`, `simscape`. The
> `SUPPORTED_ENGINES` constant in `pose_io.py` is the source of truth.

> Round-trip parity is held to **1e-9** across every engine; the
> tests live in
> `tests/unit/pose_interchange/pose_io/test_starting_state_roundtrip.py`.

---

## A. Drake — pickle of `{q, v, model_metadata}`

`save_initial_state(pose, "drake", path)` writes a Python pickle
containing a dict with three keys:

```python
{
    "q": np.ndarray,           # [x, y, z, roll, pitch, yaw, *joints_rad]
    "v": np.ndarray,           # zeros, same shape as q
    "model_metadata": {
        "engine": "drake",
        "convention_tag": "canonical-v1",
        "pelvis_prefix": 6,
        "pelvis_layout": "xyz_rpy_rad",
        "joint_names": (...,),  # tuple of REFERENCE_GOLFER_FIELDS
    },
}
```

**Read it back** in any consumer:

```python
import pickle
from pathlib import Path

with Path("starting_state.pkl").open("rb") as fh:
    payload = pickle.load(fh)

q = payload["q"]
v = payload["v"]
print(f"pelvis xyz (m) = {q[:3]}, rpy (rad) = {q[3:6]}")
```

Or via the canonical loader:

```python
from src.shared.python.pose_interchange.pose_io import load_initial_state

pose = load_initial_state("drake", "starting_state.pkl")
```

---

## B. MuJoCo — standalone JSON `{qpos, qvel}`

`save_initial_state(pose, "mujoco", path)` writes UTF-8 JSON:

```json
{
  "qpos": [x, y, z, qw, qx, qy, qz, ...joints_rad],
  "qvel": [0.0, 0.0, ...],
  "convention_tag": "canonical-v1",
  "engine": "mujoco"
}
```

`qvel` has **one less DOF than `qpos`** (MuJoCo's convention: the
free-joint quaternion is 4 floats in `qpos` but 3 angular velocity
floats in `qvel`).

**Read it back** without importing this repo:

```python
import json
from pathlib import Path

payload = json.loads(Path("starting_state.json").read_text())
qpos = payload["qpos"]
qvel = payload["qvel"]
# qpos[3:7] is [qw, qx, qy, qz] — wxyz, NOT xyzw.
```

---

## C. Pinocchio — `np.savez(q=..., v=...)`

`save_initial_state(pose, "pinocchio", path)` writes a `.npz` archive
via `np.savez`. NumPy will append `.npz` if the path doesn't already
end in it.

```python
# Inside the archive:
#   q : np.ndarray  shape (7 + n_joints,)  layout [x, y, z, qx, qy, qz, qw, ...joints_rad]
#   v : np.ndarray  shape (6 + n_joints,)  zeros
```

**Read it back**:

```python
import numpy as np

with np.load("starting_state.npz") as bundle:
    q = bundle["q"]
    v = bundle["v"]
# Pinocchio quaternion is xyzw — q[3:7] = [qx, qy, qz, qw].
```

---

## D. OpenSim — `.sto` row format

`save_initial_state(pose, "opensim", path)` writes a one-row `.sto`
file with the following layout:

```text
name=initial_state
version=1
datacolumns=<N+1>
endheader
time	pelvis_tx	pelvis_ty	pelvis_tz	pelvis_rx	pelvis_ry	pelvis_rz	<joint_name_1>	<joint_name_2>	...
0.0	<x>	<y>	<z>	<rotX_deg>	<rotY_deg>	<rotZ_deg>	<angle_1_deg>	<angle_2_deg>	...
```

Floats are written with `%.17g` — enough to recover an IEEE-754
double on reload.

**Read it back** with plain Python (no OpenSim wheel required):

```python
from pathlib import Path

text = Path("initial_state.sto").read_text()
lines = text.splitlines()
end = lines.index("endheader")
columns = lines[end + 1].split("\t")
data = lines[end + 2].split("\t")
row = dict(zip(columns, data, strict=True))
print(f"pelvis_tx = {row['pelvis_tx']}, pelvis_rx_deg = {row['pelvis_rx']}")
```

Or via the canonical loader (`load_initial_state("opensim", path)`).

---

## E. Simscape — extends `starting_pose_offsets.json`

`save_initial_state(pose, "simscape", path)` writes UTF-8 JSON
matching the existing starting-pose-matcher schema:

```json
{
  "Tx": 0.0,
  "Ty": 0.0,
  "Tz": 0.0,
  "Rx": 0.0,
  "Ry": 0.0,
  "Rz": 0.0,
  "Scale": 1.0,
  "jointAngles": {
    "HipStartPositionX": 0.0,
    "HipStartPositionY": 0.0,
    "...": "every REFERENCE_GOLFER_FIELDS entry, in degrees"
  }
}
```

This is a **drop-in** for the existing
`starting_pose_offsets.json` consumed by `fit_swing_full_pipeline.m`,
so a Pose Studio save can be fed straight into the MATLAB pipeline:

```matlab
fit_swing_full_pipeline( ...
    'starting_pose_offsets', 'C:/path/to/starting_pose_offsets.json', ...
    'mocap_target',          'data/Wiffle_TA_Driver.xlsx');
```

**Read it back** in Python:

```python
import json
from pathlib import Path

payload = json.loads(Path("starting_pose_offsets.json").read_text())
print(f"hip x deg = {payload['jointAngles']['HipStartPositionX']}")
```

---

## F. Motion-matching target JSON

`save_motion_match_target(pose, path)` writes a `BodyTarget`-compatible
JSON. **Implementation note:** `BodyTarget` requires `N >= 2` frames,
so the file emits **two identical frames** at `t = 0` and
`t = 1 ms` (`_MOTION_MATCH_DT_S = 1.0e-3`). `impact_idx` is set to
`0` so the loader treats frame 0 as impact-aligned.

The marker set is the 13 named landmarks emitted by the shared FK:
`pelvis`, `spine_top`, `torso_top`, `l_shoulder`, `r_shoulder`,
`l_elbow`, `r_elbow`, `l_wrist`, `r_wrist`, `l_hand`, `r_hand`,
`butt`, `clubhead`. Order is fixed so downstream consumers see
deterministic columns.

**Read it back** via the canonical dispatcher:

```python
from src.shared.python.motion_matching.load_body_target import load_body_target

target = load_body_target("address_pose_target.json")
print(f"frames = {len(target.time_s)}, markers = {target.marker_names}")
```

---

## G. Reference-pose library

`save_reference_pose(pose, name)` and `list_saved_reference_poses()`
read/write canonical poses (not engine-native files) into the on-disk
library that lives alongside the FK helpers:

```text
src/shared/python/motion_matching/diagnostics/reference_pose_library/
    address_neutral.json
    top_of_backswing.json
    impact.json
    ...
```

Each file is the JSON form of `CanonicalPose` (pelvis SE(3) + flat
joint-angle dict, `convention_tag = "canonical-v1"`).

**List and reload** in Python:

```python
from src.shared.python.pose_interchange import CanonicalPose
from src.shared.python.pose_interchange.pose_io import (
    list_saved_reference_poses,
    save_reference_pose,
)

names = list_saved_reference_poses()  # sorted list of stems, no extension
print(names)

# Reload one by name:
from pathlib import Path
library_dir = Path(
    "src/shared/python/motion_matching/diagnostics/reference_pose_library"
)
pose = CanonicalPose.from_path(library_dir / "address_neutral.json")
```

The directory is created on first save. `save_reference_pose` rejects
empty names, names containing path separators, and `.` / `..` (basic
path-traversal guard).

---

## H. Where to go next

- [Cross-engine conventions](cross_engine_conventions.md) — the
  unit / sign / quaternion gotchas behind these formats.
- [Quickstart](quickstart.md) — the GUI walkthrough.
- [ADR-0012](../../adr/0012-canonical-pose-interchange.md) — design
  rationale for the canonical convention.
