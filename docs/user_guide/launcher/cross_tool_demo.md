# Cross-tool live-pose demo

This walkthrough shows the launcher's embedded-host plus realtime IPC
in action: edit a pose in **Pose Studio** and watch the **Pose
Subscriber (demo)** mirror it live, side-by-side, inside a single
launcher window.

> **Status:** experimental. Part of EPIC #4993 (cross-tool live data).
> The realtime publish path is gated by the
> `POSE_STUDIO_PUBLISH_REALTIME` env var; it is off by default so unit
> tests and CI never accidentally touch the IPC layer.

## What you'll see

- Pose Studio open as a tab on the left.
- The Pose Subscriber demo open as a dock on the right.
- Each slider drag in Pose Studio updates the demo's skeleton render
  within ~33 ms, and the demo's "Last update" timestamp ticks forward.

## Prerequisites

- Launcher built from this branch (the demo is registered in
  `src/config/models.yaml` under the `motion_matching` category).
- `POSE_STUDIO_PUBLISH_REALTIME=1` exported in your shell. Without it
  Pose Studio runs as before and the demo simply waits.

## Walkthrough

### 1. Open Pose Studio in a tab

In the launcher, find the **Pose Studio** tile. Right-click it and
pick **Launch in Tab**. The studio appears in the central tab area.

![Pose Studio open in a tab](../../images/launcher/pose_studio_tab.png)

> _Screenshot placeholder — populate after first successful build._

### 2. Open the Pose Subscriber demo in a dock

Find the **Pose Subscriber (demo)** tile (under the
**Motion Matching** category — the `experimental` chip identifies it).
Right-click and pick **Launch in Dock**. A dock panel appears on the
side; the demo widget says "Last update: (waiting…)" until the first
pose arrives.

![Pose Subscriber demo docked beside Pose Studio](../../images/launcher/pose_subscriber_dock.png)

> _Screenshot placeholder — populate after first successful build._

### 3. Drag a slider

In the joint panel on the right side of Pose Studio, grab any joint
slider (for example `HipStartPositionX`) and drag it. As you drag:

- Pose Studio's 3D view updates in real time.
- The demo's skeleton plot updates within one or two frames.
- The demo's "Last update" label ticks forward to the current
  wall-clock time.

The publish path is debounced to 30 Hz inside `EngineController`, so
even a frantic slider drag never floods the IPC layer.

### 4. (Optional) Verify the env var gate

Close everything, unset `POSE_STUDIO_PUBLISH_REALTIME`, and relaunch.
Pose Studio still runs normally; the Pose Subscriber demo still opens
but stays at "(waiting…)" forever, because no payloads are being
published. This is the default state — explicit opt-in is required to
turn realtime IPC on.

## Architecture (one diagram)

```
+-----------------+  publish    +---------------------+   tail (30 Hz)   +----------------------+
| Pose Studio     | ----------> | realtime.publish()  | ---------------> | MainWidget           |
| EngineController|             | -> FileTransport    |                  | (Pose Subscriber)    |
| _maybe_publish  |             |   ~/<tmp>/upstream- |                  | _on_pose_received    |
| _pose()         |             |   drift-realtime/   |                  | -> forward_kinematics|
+-----------------+             +---------------------+                  +----------------------+
```

Both ends use the same
[`src.shared.python.realtime`](../../../src/shared/python/realtime/__init__.py)
facade. Swapping the on-disk transport for a websocket or in-memory
queue is a single-file change.

## Related docs

- TODO: link to the embedded-view user guide once Subtask 7 lands
  (`docs/user_guide/launcher/embedded_view.md`).
- [`src/tools/pose_subscriber_demo/README.md`](../../../src/tools/pose_subscriber_demo/README.md)
  — tool-specific notes, including standalone launch instructions.
- EPIC #4993 — full design context.
- Issue #4999 — Subtask 6 spec for this demo.

## Troubleshooting

| Symptom                            | Likely cause                                                                                                                           |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Demo shows "(waiting…)" forever    | `POSE_STUDIO_PUBLISH_REALTIME` not set, or Pose Studio not yet open.                                                                   |
| Updates lag noticeably             | Slow filesystem; check that `REALTIME_FILE_ROOT` (if set) lives on a local disk.                                                       |
| Demo never appears in the launcher | The motion_matching category may be collapsed; expand it, or check that `pose_subscriber_demo` is present in `src/config/models.yaml`. |
