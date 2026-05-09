# Pose Studio — Cross-engine conventions

The whole reason Pose Studio exists is that **every supported engine
disagrees about how to spell a pose**. This page is a one-stop
reference for the per-engine convention table that the canonical pose
adapters in `src/shared/python/pose_interchange/adapters/` insulate you
from.

> **Background.** The canonical convention itself is documented in
> [ADR-0012](../../adr/0012-canonical-pose-interchange.md). The
> `CanonicalPose` dataclass holds an SE(3) pelvis transform
> (`translation_m` + intrinsic XYZ Euler in **degrees**) plus a flat
> dict of joint angles in **degrees**, with field names matching
> `reference_golfer_setup`.

> Use this page to _understand_ what each engine wants on disk. Use
> [save_formats.md](save_formats.md) to _write_ a file in that
> convention.

---

## A. Convention matrix

| Engine    | Pelvis prefix in `q`                | Joint units | Quat order | Sign-flip gotchas                                                  |
| --------- | ----------------------------------- | ----------- | ---------- | ------------------------------------------------------------------ |
| Drake     | `[x, y, z, roll, pitch, yaw]` (rad) | rad         | n/a (RPY)  | URDF axes occasionally mirror canonical Y; tracked per joint slot. |
| MuJoCo    | `[x, y, z, qw, qx, qy, qz]`         | rad         | **wxyz**   | Free-joint quaternion is **w-first**.                              |
| Pinocchio | `[x, y, z, qx, qy, qz, qw]`         | rad         | **xyzw**   | Free-flyer quaternion is **w-last** — the opposite of MuJoCo.      |
| OpenSim   | `[x, y, z, rotX, rotY, rotZ]` (deg) | deg         | n/a (XYZ)  | Several `.osim` models invert Y for shoulder external rotation.    |
| Simscape  | `[x, y, z, rot_x, rot_y, rot_z]`    | deg         | n/a (XYZ)  | Torso-twist joint between spine and hub (see AGENTS.md §G).        |

The `JointSlot` dataclass in
`src/shared/python/pose_interchange/protocol.py` captures these per
joint: `units` (`"rad"` or `"deg"`), `sign` (`+1` or `-1`),
`start_index`, `lower_limit`, `upper_limit`. Adapter authors flip
the sign once at the slot level rather than scattering sign-flips
across function bodies.

---

## B. Joint angle units (deg vs rad)

| Engine    | Native joint units | Canonical conversion       |
| --------- | ------------------ | -------------------------- |
| Drake     | rad                | `np.radians()` at boundary |
| MuJoCo    | rad                | `np.radians()` at boundary |
| Pinocchio | rad                | `np.radians()` at boundary |
| OpenSim   | deg                | identity                   |
| Simscape  | deg                | identity                   |

Pose Studio always displays degrees because the canonical convention
_is_ degrees. The units badge in the bottom-right of the GUI tells
you what the engine's native side is so you don't get tripped up
when reading raw `.osim` / `.sto` files outside the tool.

---

## C. Pelvis representation (free-flyer `q` layout)

Three flavours:

1. **6-DOF Euler** (Drake, OpenSim, Simscape):
   `[x, y, z, rot_x, rot_y, rot_z]` — six floats. Drake is in
   radians, OpenSim and Simscape in degrees.
2. **7-DOF, w-first quaternion** (MuJoCo):
   `[x, y, z, qw, qx, qy, qz]`.
3. **7-DOF, w-last quaternion** (Pinocchio):
   `[x, y, z, qx, qy, qz, qw]`.

The `CanonicalPose` always uses (1) in degrees. Adapters synthesise
the engine's native quaternion via
`euler_xyz_deg_to_quat_wxyz` (and `quat_wxyz_to_xyzw` for Pinocchio)
in `pose_interchange.adapters._base`.

---

## D. Sign-flip gotchas

The `JointSlot.sign` field is the canonical place to capture a
per-joint sign flip. Adapters scan the layout once on
`from_canonical` / `to_canonical` and apply the sign at the slot
boundary. Known cases:

- **OpenSim shoulder external rotation.** Several `.osim` golfer
  models (notably the `Hamner_full_body` derivative) flip Y for
  glenohumeral external rotation. The adapter handles this via
  per-slot `sign=-1`; the canonical pose stays right-hand-rule.
- **Drake URDFs from third-party humanoids.** Some humanoid URDFs
  (e.g. derivatives of `talos`, `valkyrie`) declare hip joints with
  `axis="0 0 -1"` rather than `axis="0 0 1"`. Same fix: the
  adapter's `JointSlot.sign` carries the flip.

If you author a new adapter, do not scatter `q[i] *= -1` through the
function body. Put the sign in the slot. AGENTS.md §G has the
historical record of every sign-flip bug we've shipped — every one of
them was a scattered sign-flip.

---

## E. Quaternion order: MuJoCo `wxyz` vs Pinocchio `xyzw`

This is the single most-broken thing when you hand-translate between
the two engines. **MuJoCo puts the scalar first; Pinocchio puts it
last.** The canonical adapters convert through Euler degrees, so
they never have to write a pairwise `wxyz_to_xyzw` outside of
`_base.py`.

If you ever find yourself writing
`q[3], q[6] = q[6], q[3]` in a fit-swing script, stop and use the
adapter:

```python
from src.shared.python.pose_interchange import CanonicalPose
from src.shared.python.pose_interchange.adapters import ADAPTER_REGISTRY

pose = ADAPTER_REGISTRY["mujoco"]().to_canonical(mj_qpos)
pin_q = ADAPTER_REGISTRY["pinocchio"]().from_canonical(pose)
```

---

## F. OpenSim XYZ Euler vs canonical XYZ

Good news: **OpenSim's BodyKinematics already reports the pelvis as
XYZ Euler in degrees.** Same convention as the canonical pose. The
OpenSim adapter's pelvis path is therefore a direct copy — no
deg/rad conversion, no axis swap. Joint angles are also reported in
degrees by OpenSim (per `CoordinateSet`), so per-slot units are
`"deg"` with `sign=+1` by default.

The single place this can go wrong is when the `.osim` model itself
declares a non-XYZ rotation order on the pelvis joint set. The
adapter does not currently introspect the model XML to detect this;
if you load a custom `.osim` model and the pelvis rotation looks
mirrored, file an issue and we'll add a guard.

---

## G. Simscape torso-twist joint

Simscape's body chain has a `torso` joint **between spine and hub**
that hosts the revolute Z (twist) joint. Don't collapse to
`hip → spine → hub` directly — you'll lose the visible torso coil at
top-of-backswing. See AGENTS.md §G and the corresponding model block
in
`src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/mdl_reference/GolfSwing3D_Kinetic.mdl`,
block "Torso Kinetically Driven" (SID 8331).

The Simscape adapter (`adapters/simscape.py`) keeps the torso joint
as its own slot and exposes it as `TorsoStartPositionZ` in the
`jointAngles` dict — Pose Studio's joint panel will show it under the
_spine_ group.

---

## H. Wiffle xlsx CM-vs-inches

If your downstream pipeline includes Wiffle xlsx targets (the
canonical mocap source for the demo dataset), remember:

- **Wiffle xlsx values are in CENTIMETRES** despite the workbook's
  "Definitions" tab claiming inches.
- The MATLAB loader is the source of truth:
  `load_club_target_excel.m` line 53 sets `CM_TO_METRES = 0.01`.
- The Python loader in
  `src/shared/python/club_data/targets.py` matches the MATLAB
  convention.

This is unrelated to the canonical pose itself — Pose Studio always
emits SI units (metres, degrees) — but if you load a `BodyTarget`
that was authored from a Wiffle xlsx and the skeleton looks 100×
too large, the xlsx loader is the place to check first. See
AGENTS.md §G.

---

## I. Where to go next

- [Save formats](save_formats.md) — the on-disk shape per engine,
  with a downstream-reader code example for each.
- [Quickstart](quickstart.md) — the GUI walkthrough.
- [ADR-0012](../../adr/0012-canonical-pose-interchange.md) — the
  design reasoning behind the canonical convention.
