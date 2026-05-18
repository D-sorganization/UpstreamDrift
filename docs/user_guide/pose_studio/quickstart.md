# Pose Studio — Quickstart

Hand-edit a canonical skeleton pose, see it rendered through any of the
five supported engines (Drake, MuJoCo, Pinocchio, OpenSim, Simscape),
save it as an engine-native starting state, and feed that back into
`fit_swing_full_pipeline.m` (or any other downstream consumer) without
ever leaving the desktop tool.

> **Background.** Design rationale lives in
> [ADR-0012](../../adr/0012-canonical-pose-interchange.md). The
> per-engine convention table is in
> [cross_engine_conventions.md](cross_engine_conventions.md); the
> on-disk file shape per engine is in
> [save_formats.md](save_formats.md).

> **Screenshots.** Image references in this doc point at
> `docs/assets/pose_studio/`. Screenshots will be added in a follow-up
> PR — the placeholder paths below are the contract for that follow-up
> to satisfy.

---

## A. Launch

Two equivalent paths:

```bash
# 1. Direct module run from the repo root
python -m src.tools.pose_studio
```

```text
2. From the UpstreamDriftLauncher main grid: click the "Pose Studio" tile
   (registered as `pose_studio` in src/config/models.yaml).
```

Pose Studio requires the `gui-tools` extra (PyQt6 + matplotlib QtAgg
backend). Install once with:

```bash
pip install upstream-drift[gui-tools]
```

The tool falls back to a `MockKinematicsService` when an engine wheel
is absent — the engine pill turns yellow rather than refusing to
launch. See [cross_engine_conventions.md](cross_engine_conventions.md)
for which engines are real on your machine.

---

## B. Pick an engine

<!-- TODO: screenshot capture pending; placeholder OK -->

![Engine picker](../../assets/pose_studio/01_engine_picker.png)

The top-left **Engine picker** combo lists every engine in
`KINEMATICS_SERVICE_REGISTRY` (`drake`, `mujoco`, `pinocchio`,
`opensim`, `simscape`). The status pill next to it indicates:

| Pill colour | Meaning                                            |
| ----------- | -------------------------------------------------- |
| green       | real engine wheel is loaded; live FK is authentic. |
| yellow      | mock service in use; FK is the shared Python FK.   |
| red         | the engine factory itself failed to construct.     |

Switching engines does **not** clobber your edits — the
`CanonicalPose` is held by the controller and re-pushed through the
new engine's adapter.

---

## C. Edit the pose

<!-- TODO: screenshot capture pending; placeholder OK -->

![Joint sliders + 3D view](../../assets/pose_studio/02_joint_panel.png)

The left dock is an accordion of per-joint sliders grouped by body
region (pelvis, spine, shoulders, elbows, wrists). The right dock is
the live 3D skeleton view rendered by matplotlib's QtAgg backend.

- Drag a slider — the 3D view repaints in place. Numeric values are
  always displayed in the canonical convention (degrees, intrinsic
  XYZ Euler) regardless of the selected engine. The **units badge**
  in the bottom-right shows the engine's _native_ convention.
- Click a 3D landmark to highlight the corresponding slider group.
- `Ctrl+Z` / `Ctrl+Shift+Z` walk the undo stack of `CanonicalPose`
  snapshots maintained by the history controller.
- _Pose Library_ (top menu) loads `canonical_zero_pose` (T-pose at
  origin) or `canonical_from_reference_setup` (the canonical address
  pose). Saved poses from the on-disk reference-pose library are
  listed underneath.

---

## D. Save as a starting state

<!-- TODO: screenshot capture pending; placeholder OK -->

![Save dialog](../../assets/pose_studio/03_save_dialog.png)

`File → Save initial state…` writes an engine-native file via
`pose_io.save_initial_state`. The format is selected by the engine
combo, **not** by the file extension — pick the engine first, then
the path. See [save_formats.md](save_formats.md) for the on-disk
shape per engine.

For Simscape specifically, the dialog defaults to the
`starting_pose_offsets.json` filename used by the existing matcher;
the saved file is a drop-in for the MATLAB pipeline:

```matlab
% From MATLAB, after saving from Pose Studio:
fit_swing_full_pipeline( ...
    'starting_pose_offsets', 'C:/path/to/starting_pose_offsets.json', ...
    'mocap_target',          'data/Wiffle_TA_Driver.xlsx', ...
    'optimizer',             'fmincon');
```

The same file also loads back into Pose Studio via
`File → Load initial state…` so you can iterate without retyping
joint angles.

---

## E. Save as a motion-matching target (optional)

`File → Save motion-match target…` writes a `BodyTarget`-compatible
JSON via `pose_io.save_motion_match_target`. The target carries
Cartesian landmark positions (FK-evaluated through the canonical pose)
in two identical frames at `t = 0` and `t = 1 ms` so it satisfies
`BodyTarget`'s `N >= 2` invariant. Load it back via:

```python
from src.shared.python.motion_matching.load_body_target import load_body_target

target = load_body_target("address_pose_target.json")
```

---

## F. Where to go next

- [Cross-engine conventions](cross_engine_conventions.md) — the
  per-engine sign / unit / quaternion gotchas this canonical pose
  insulates you from.
- [Save formats](save_formats.md) — the on-disk shape of every
  starting-state file plus a downstream-reader code example for each.
- [ADR-0012](../../adr/0012-canonical-pose-interchange.md) — design
  rationale for the canonical convention.
- step-forward integration — see issue
  [#4901](https://github.com/D-sorganization/UpstreamDrift/issues/4901)
  (Subtask 7 of EPIC #4895; documentation will land alongside that PR).
