# Pose Subscriber (demo)

Live mirror of Pose Studio's canonical pose. Subscribes to the
`pose/canonical` channel exposed by the realtime IPC layer (Subtask 4
of EPIC #4993) and renders the most recent pose with a coarse forward-
kinematics skeleton.

## What it is

This tool exists to prove out the realtime IPC layer end-to-end with a
real GUI consumer. It is **read-only**: it never publishes anything,
never edits the pose, and never persists state.

## Running

### From the launcher

The tool is registered as embeddable. Right-click its tile in the
launcher and pick "Launch in Dock" — the demo prefers a dock layout
because it is supplementary to whatever editor (e.g. Pose Studio) the
user has in the main tab area.

### Standalone

```bash
python -m src.tools.pose_subscriber_demo
```

### Receiving live updates from Pose Studio

Pose Studio's realtime publisher is gated by an env var so unit tests
and CI never accidentally touch the IPC layer. To wire the two
together, set the env var before launching the launcher (or Pose
Studio directly):

```bash
POSE_STUDIO_PUBLISH_REALTIME=1 python -m src.launchers.unified_launcher
```

Open Pose Studio in a tab and the Pose Subscriber demo in a dock; drag
a slider in Pose Studio and watch the demo widget update.

## Architecture

```
Pose Studio (publisher)                    Pose Subscriber (subscriber)
└── EngineController.set_pose              └── MainWidget._on_realtime_payload
    └── _maybe_publish_pose                    └── forward_kinematics + matplotlib
        └── realtime.publish("pose/canonical", pose.to_dict())
              └── FileTransport (~/<temp>/upstreamdrift-realtime/...)
                  └── tail polling at 30 Hz
```

Both sides go through the same `src.shared.python.realtime` facade,
so substituting a websocket transport is a single-file change.

## Limitations

- The on-disk file transport polls at 30 Hz; expect ~33 ms median end-
  to-end latency on the same machine. The integration test asserts
  arrival within 200 ms to leave headroom for slow filesystems.
- The demo renders a coarse skeleton, not the same body-part viz that
  Pose Studio uses — that's intentional, it keeps the dependency
  surface small.
