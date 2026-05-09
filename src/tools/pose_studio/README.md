# Pose Studio

Interactive cross-engine pose editor. Subtask 5 of EPIC #4895.

## Why

Hand-edit a `CanonicalPose` and see the result rendered through any of
the supported physics engines (Drake, MuJoCo, Pinocchio, OpenSim,
Simscape) without restarting the process. The same canonical pose
becomes the seed for the starting-pose matcher (#4899), the motion
matcher target, and any future engine-comparison tooling.

## Run

```bash
python -m src.tools.pose_studio
```

Requires the `gui-tools` extra (`pip install upstream-drift[gui-tools]`).

## Architecture

```
src/tools/pose_studio/
  core.py                — engine-agnostic state machine, no Qt
  controllers/
    engine_controller.py — owns active LiveKinematicsService + Adapter
    history_controller.py— undo/redo stack of CanonicalPose snapshots
  widgets/
    engine_picker.py     — combo + status pill
    joint_panel.py       — accordion of per-joint sliders
    view_3d.py           — matplotlib 3D skeleton view
    units_badge.py       — engine native-convention indicator
  gui.py                 — QMainWindow composing the above
```

The pure-data layer (`core.py` + `controllers/*`) imports no Qt and is
covered by `tests/unit/tools/pose_studio/`. The widget + GUI layer is
covered by smoke tests in `tests/ui/tools/pose_studio/`.

## Scope (v1)

- Engine swap with mock fallback (yellow pill) when wheel absent.
- Per-joint sliders grouped by body region.
- Click a 3D landmark to highlight it.
- Undo/redo (Ctrl+Z / Ctrl+Shift+Z).
- Pose Library: load `canonical_zero_pose` or
  `canonical_from_reference_setup`.

## Out of scope (deferred follow-ups)

- IK drag-handles — drag a clubhead, solve back through the chain.
- Save/load to JSON — lives in Subtask 6 / #4900.
- Mocap scrubbing — lives in Subtask 7 / #4901.
