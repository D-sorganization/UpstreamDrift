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
      "unserialized": true,
      "controls": { "exposure": -6, "auto_exposure": false }
    }
  ],
  "notes": "Sonnet root ports 4/5/6; TS4 free."
}
```

### Naming the Cameras

The durable name of a camera is the **view** it is bound to; the binding is
what follows the unit around. Two of the three ELP units expose a USB serial
(`2605160001`, `2601240001`): Windows keys their device instance on it, so a
serial binding recognises the unit on any jack of any dock. The third unit
reports no serial. Bind it with `"unserialized": true`: `plan-check` resolves
it by elimination (the one enumerated ELP without a serial), so it also keeps
its view when moved. If a second serial-less unit ever appears the binding is
reported ambiguous rather than guessed, and a `port_path` binding is the
fallback for that case. Label the camera bodies with their serial (or "no
serial") and the view name from the plan, and the label and the software agree.

Changing an experimental condition means saving a new plan file, not editing
code: resolution, frame rate, exposure and gain are per camera, and the plan
travels with the session it produced. `RigPlan.load` rejects other schema
versions rather than guessing.

## Plan Check

```bash
python3 -m motion_capture.rig plan-check --plan plans/three-view-driver.json
```

Walks every camera's hub chain through Windows PnP (one enumeration plus one
bulk property query per hub tier, about 20 s for three cameras), matches the
plan by
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

## Layout Model

Multiview pictures (live preview, playback and composite export) are all
rendered through one pure model, `tools/capture_rig/layout_model.py` (#9810).
A `LayoutSpec` (JSON schema `rig-layout/1.0.0`) is a named `rows`x`cols` grid
(1..4 each, so up to 16 tiles) on a canvas; every `Tile` names a `source`
(`live`/`recorded`/`overlay` of a view plus optional variants, or `empty`), a
`cell` (row, col, rowspan, colspan; tiles may not overlap), `rotation`
(0/90/180/270), `flip_h`/`flip_v`, a normalised `crop`, `fit`
(`fit` letterboxes, `fill` centre-crops, `stretch`) and an optional label.
`to_dict()`/`from_dict()` round-trip it and name the offending field on a
`ValueError`; `with_tile`/`without_tile`/`move_tile` return new specs.
`compose(frames, spec, size, palette)` draws a mapping of BGR frames (keyed by
`SourceRef.key`, e.g. `live:face_on`) into one image with theme-derived
`Palette` colours; missing sources show a placeholder. Built-in presets:
`single`, `side_by_side`, `three_across`, `two_by_two`, `three_by_three`,
`four_by_four`, `primary_plus_strip`.

`layout_presets.LayoutStore` (#9811) saves named layouts with a provenance
stamp in two scopes: user (`<AppConfigLocation>/UpstreamDrift/capture_rig/layouts/`)
and session (`<session>/layouts/`, so a layout travels with a take); built-in
presets appear read-only in `list()`.

### Layout Editor

`tools/capture_rig/layout_editor.LayoutEditor` (#9812) is the interactive
front end of the layout model: grid spinners (up to 4x4) and a preset menu
(built-ins plus the layouts in the user and session scopes of `LayoutStore`),
a composed thumbnail of the whole layout drawn through the same `compose()`
the preview and export use, and per-tile controls: source, rotate, flip,
crop (switch _Crop_ on and drag a rectangle over the tile), fit, label and
span (+/- row and column). Click a tile to select it; drag it onto another
cell to move it (the two swap). _Save as..._, _Load_ and _Delete_ go through
the store; _Undo_/_Redo_ keep the last 20 edits. The widget emits
`layout_changed(LayoutSpec)` once per edit; frames for the thumbnails come
from an injected `frame_provider(SourceRef)`; every control's tooltip says
what it does and, when grey, why.

## Live Preview

_Preview cameras_ opens every planned view through the same camera binding
the recorder uses (`src/motion_capture/rig/binding.py`) and composites the
latest frame of every view into one canvas above the player (#9813), one
worker thread per camera, refreshed at up to 15 Hz. The plan path is prefilled with the lab plan and the session folder
with a fresh `sessions/<timestamp>-take`, so _Record_ works out of the box:
pressing it releases the cameras (ffmpeg needs the devices), runs the
recorder, loads the take into the player and resumes the preview. A plan that
cannot be realised on this machine is reported on the preview's status line
rather than raised.

### Multiview Live and Playback

Both viewing panes are drawn through a `LayoutSpec` (#9813, #9814). The
picker above each canvas lists the built-in presets and every saved layout;
_Edit..._ opens the layout editor beside it, with live thumbnails, and each
edit applies as you make it. A view that the layout does not show is still
captured, and the same view may appear twice (full plus a cropped detail).
The chosen layout name is saved with the pane arrangement, so both come back
on the next start.

Playback composites several of the session's sources at once: `recorded`
tiles show the raw recording (or its proxy) and `overlay` tiles the same
footage with the detector's pose and the ticked variants' models drawn on it,
so raw and overlay of one view can sit side by side. Frame *k* is the same
instant in every tile — when the manifest carries a strobe-alignment block
each reader is shifted by its whole-frame offset. Scrubbing, play/pause,
speed, single-frame stepping and _Export PNG..._ (the canvas exactly as shown)
sit under the canvas.

### Panes, Layouts and Recording Controls

The viewing panes (**Live preview**, **Playback**, **Results**) are dock
widgets: drag a title bar to move a pane to another edge, tear it off to float
it (onto a second monitor if you like), tab two panes together, or close one.
Every pane scrolls when its content is larger than the space it has, so the
window never grows past the screen. **Layout** (top right) saves the current
arrangement under a name, loads or deletes a saved one, and **Reset layout**
returns to the default with every pane shown. The last arrangement is restored
on the next start.

### Appearance

The tile follows the application theme (#9816). Its header is one toolbar:
the session line on the left, then a status strip of three chips (cameras
bound, recorder state with the live REC readout, and the outcome of the last
take) and the **Layout** bar on the right. The action buttons are grouped by
workflow step, each row labelled with the step it belongs to, so the grid
reads in the same order as the Workflow panel. Every colour and style comes
from `src/tools/capture_rig/styling.py`, which composes them from the active
palette and the fleet `Styles` constants: nothing in the tile names a colour,
spacing comes from `LayoutMetrics`, and switching theme (standalone window or
embedded in the launcher alike) restyles the header, chips, preview tiles and
the recording badge immediately. `tests/tools/capture_rig/test_theme_compliance.py`
fails the build on a literal colour or an ad-hoc stylesheet string in the package.

The live view opens with the tile. Under the tiles sits a transport strip:

- **Record / Stop**: one button; during the countdown it reads _Cancel_.
- **Take length**: 5 / 10 / 15 / 30 s presets or a custom spinner.
- **Countdown**: none, 3, 5 or 10 s between pressing Record and the recorder
  starting, so you can walk to address.
- **REC readout**: a blinking red indicator with elapsed / total time and a
  progress bar; the same `● REC 00:04 / 00:10` badge is stamped on every tile.

A camera cannot be opened twice, so during a take the recorder itself keeps the
view alive: with `--live-preview DIR` each ffmpeg process also decodes its stream at quarter
resolution and rewrites `DIR/<view>.jpg` eight times a second (atomically), and the
tile shows those snapshots until the take is written, then returns to the
direct preview. **Stop** ends a take early through `--stop-file PATH`: the
recorder polls for the file, stops every camera together and removes it.

## Recording a Session

```bash
python3 -m motion_capture.rig record --plan plans/three-view-driver.json   --duration 10 --out sessions/2026-09-06T14-record
```

Enumerates the cameras, checks the plan, resolves each planned camera's
DirectShow device path, and stream-copies its compressed MJPEG to
`<out>/<view>_<identity>.mkv` for the requested duration (issue #9600). The
result is a **session bundle**: `plan.json` (the plan as recorded),
`recordings.json` (per view: file, bytes, recorder exit code, requested mode)
and `session_manifest.json`, whose outcome reuses the capture vocabulary — a
view whose recorder failed or wrote nothing makes the session `blocked`, never
a quietly shorter dataset. `--dry-run` writes the bundle without touching a
camera and is what the tests exercise.

`--mode WxH@FPS[:FOURCC]` applies one capture mode to every selected view and
`--views a,b` restricts the run to a subset of the plan (plan order); both
derive a new plan whose name records the overrides, and both are accepted by
`plan-check`, `capture` and `record`. `recordings.json` also carries
`recorder_note` (ffmpeg's last stderr lines, kept when a recorder failed or
delivered nothing) and `recorder_wall_s` (host seconds the recorder ran). A
recorder that exits 0 but decodes zero frames makes the session `blocked`.

## Session Check

```bash
python3 -m motion_capture.rig session-check --session sessions/2026-09-06T14-record
```

Validates a bundle without opening any video: the three JSON files parse
against their schemas, every plan view has a recording entry, every successful
entry's file exists with the indexed size, and the manifest outcome is the one
the recordings imply. Exit 0 when sound. Later stages (ingest, alignment,
export) read bundles, so this is the gate between "the cameras ran" and "this
session can be trusted".

## Proxies

```bash
python3 -m motion_capture.rig proxy --session sessions/<date>-record [--encoder libx264|h264_nvenc|h264_mf] [--crf 18]
```

Writes a browser-playable H.264/yuv420p `.mp4` beside every usable recording
and `proxies.json` (encoder, exit code, bytes, reason). Proxies exist for the
web players; ingest and session-check never read them.

## Reconstruct

```bash
python3 -m motion_capture.rig reconstruct --session S --cameras cameras.json --anchor neck=0.53
```

Cleans the ingested views with the dynamics prior and jointly fits camera
placement, 3-D joints and bone lengths, starting from the given camera records
(a previous `reconstruction.json` works) or, with `--intrinsics` on a first
take, from a placement initialised from the golfer's joints. See
[Self-Calibrating Markerless Pipeline](self_calibrating_pipeline.md).

## Ingest

```bash
python3 -m motion_capture.rig ingest --session sessions/<date>-record
```

Runs the registered pose estimator (`mediapipe` by default — the Tasks-API
`MediaPipeEstimator`, issue #9602) over every successful recording in a bundle
and writes `observations/<view>.json`: one `KeypointObservation` per frame in
UpstreamDrift's existing observation records (pixel coordinates, per-keypoint
confidence, `time_s` from the recording's frame index and rate), the
`DetectorLayout` naming the keypoint order, provenance (estimator, model path
and variant, mediapipe version, camera identity, requested mode) and the
session's `timing` block copied verbatim. `observations.json` indexes the
views; a view whose recording failed is `unavailable` with the reason rather
than absent. Single-camera depth stays model-conditioned and is not written;
`CanonicalObservations` (which needs camera calibrations) is assembled by the
calibration stage, not here.

When the session was captured with `--timing` (issue #9603), each row also
carries `time_ref_s`, `time_ref_uncertainty_s` and `time_ref_source`: the same
instant expressed in the reference view's arrival clock through the strobe
offset, with the quadrature uncertainty. The per-view `time_s` is never
rewritten. `timing_report.json` restates each view's offset, uncertainty and
rate deviation and adds the skew it is expected to accumulate over its
recording, which is what tells the reconstruction stage whether one offset per
session is sufficient.

## Match Tab, Overlays and Provenance

The _Match_ tab names a variant, ticks the cameras to use and picks the
observation set and the source (triangulate, or image space with the
cameras of another variant). _Reconstruct_, _Fit model_, _Kinetics_,
_Compare models_ and _Export_ all act on that variant. In the player, the
_Model overlay_ checkboxes draw any registered variants' joints and model on
the current view (views a variant never used are labelled held out); `rig
overlay` writes the same as a clip. Clicking a row in any results table
opens the _Provenance_ tab with the file's lineage down to the recordings
and the detector plug-in (`rig lineage`).

## Annotate and Edit Points

_Annotate / edit points_ opens a dialog on the player's view. The banner
names the frame and joint to click; `S` skips an occluded joint, `B` goes
back, `N` moves to the next frame, `J` jumps, `Q` finishes and saves
`annotations/<view>.json`. With an observation set selected in the player
the same dialog edits that set: the detector's points are drawn, a click
replaces one, `S` rejects it, `A` accepts the frame as detected. `rig
annotations-to-observations` (with `--merge-with SET` for corrections)
turns the file into an observation set the pipeline uses like any other.

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
