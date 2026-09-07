# Camera Rig Capture

Version: 1.0.0

Issues: #9590 (child of #9422); Tools #4706

`motion_capture.rig` is UpstreamDrift's orchestration layer for a multi-camera
USB rig: it declares an experimental condition as a _rig plan_, checks that
plan against the live USB topology, captures every planned camera together,
and reports what each camera actually delivered. It owns no camera or capture
contract; those belong to Tools `sidekick.lab.mocap` under
[ADR-0041](../adr/0041-markerless-mocap-consumer-authority.md) and are
consumed through `tools_bridge` when the pinned Tools release ships them.

The constraints it encodes were measured in the
[USB camera rig bring-up](usb_camera_rig_bringup.md).

## Rig Plan

A plan binds named views to camera _identities_ and capture settings. Identity
is the USB serial, or the port-path fallback for units that expose none; it is
never an OpenCV index, which reshuffles on replug and would silently swap views.

```json
{
  "schema_version": "rig-plan/1.0.0",
  "name": "three-view-driver",
  "cameras": [
    { "view": "face_on", "serial": "2605160001" },
    { "view": "down_line", "serial": "2601240001", "mode": { "fps": 60 } },
    {
      "view": "overhead",
      "port_path": "path_D-D35A8F7-0-0000",
      "controls": { "exposure": -6, "auto_exposure": false }
    }
  ],
  "notes": "Sonnet root ports 4/5/6; TS4 free."
}
```

Changing an experimental condition means saving a new plan file, not editing
code: resolution, frame rate, exposure and gain are per camera, and the plan
travels with the session it produced. `RigPlan.load` rejects other schema
versions rather than guessing.

## Plan Check

```bash
python3 -m motion_capture.rig plan-check --plan plans/three-view-driver.json
```

Walks every camera's hub chain through Windows PnP, matches the plan by
identity, and reports missing cameras, cameras that share a USB 2.0 root port
(only one of them can stream), and enumerated cameras the plan does not claim.
Exit 0 means the plan is realizable on this host as wired.

## Capture Session

```bash
python3 -m motion_capture.rig capture --plan plans/three-view-driver.json \
  --duration 8 --out sessions/2026-09-06T14
```

Opens each camera in plan order (pausing between opens, because Media
Foundation tears down asynchronously), starts them behind one barrier so their
isochronous reservations compete for real, and measures per camera: achieved
frames per second, failed reads, worst inter-frame gap, and reopens. Every
frame is stamped in the `host_monotonic_ns` clock domain; arrival time is not
exposure time, and the manifest's `timing` block is where the sync stage will
record how they relate.

The manifest names one outcome using the acceptance-program vocabulary:

| Outcome       | Meaning                                                             | Exit |
| ------------- | ------------------------------------------------------------------- | ---- |
| `supported`   | every camera reached at least 90 % of its requested rate            | 0    |
| `degraded`    | every camera streamed, at least one below 90 %                      | 1    |
| `blocked`     | at least one camera opened but delivered nothing, or failed to open | 1    |
| `unavailable` | no camera delivered frames                                          | 2    |

A camera that stops delivering is reopened once (a lost reservation is
permanent on the old handle); if it still delivers nothing, the reason is
recorded and the outcome is `blocked`, never a silently shorter session.

## Recording

Decoding MJPEG to BGR costs about two cores per 1920x1200 at 60 fps stream;
copying the compressed stream to disk costs almost nothing. `recorder.py`
wraps `ffmpeg -f dshow ... -c:v copy` through `core.process_safety.managed_popen`
and addresses each camera by its DirectShow device path
(`recorder.dshow_device_ref`), which is what keeps three identically named
units distinct. Windows grants one process exclusive access to a camera, so a
session either observes through frame sources or records through recorders,
not both on the same camera.

## Extending the Rig

- **A new camera type** implements the `FrameSource` protocol in `sources.py`:
  `open(mode, controls)` negotiates and must prove frames arrive, `read()`
  never blocks forever, `close()` is idempotent. `SyntheticFrameSource` shows
  the minimum, including fault injection for tests.
- **A new recording path** implements `Recorder` (`start` / `stop`).
- **A new condition** is a plan file. Nothing in the package hard-codes the
  camera count, the resolution, or the views.

## Tools Schema Bridge

`tools_bridge.probe_tools_schema()` reports `unavailable` while the pinned
vendor tree lacks `sidekick.lab.mocap`, `incompatible` when it is present but
missing expected submodules, and `ready` otherwise. The result is written into
every manifest under `tools_schema`. No mapping to Tools records is attempted
until the pinned release documents its builders; that export is #9422's
responsibility, and inventing it here would be exactly the duplicate authority
ADR-0041 forbids.

## Time Sync

Three cameras on three USB root ports stamp frames in the host's monotonic
clock at _arrival_. ADR-0041 forbids promoting arrival time to exposure time,
so `sync.py` never rewrites a frame's timestamp. With `capture --timing` (issue
#9591) the session records each frame's mean brightness, finds the first frame
in which a shared strobe becomes visible per camera, and writes a `timing`
block to the manifest: per view, the offset of its arrival clock from the
reference view's, the uncertainty (both cameras' frame intervals combined in
quadrature, because a flash that lands anywhere inside one interval is first
seen in the next frame), the measured frame interval, and its deviation from
the nominal rate in parts per million. A view whose strobe is not found, or a
session whose reference view has none, is reported `unavailable` with the
reason; nothing is interpolated. The record is evidence for the reconstruction
stage to apply or reject, never a correction applied to frames.

## Diagnostic Script

`scripts/diagnose_mocap_camera_rig.py` is a thin CLI over this package for
bring-up: it builds a one-view-per-camera plan from the enumerated topology,
runs a solo session per camera and one concurrent session, and compares the
measured streaming count against the topology prediction.
