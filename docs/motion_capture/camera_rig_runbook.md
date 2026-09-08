# Camera Rig Runbook

Version: 1.0.0

Issues: #9601 (C2 of #9599); #9422

Operator procedure for recording a session on the three-camera markerless
rig, and the acceptance evidence from the first governed recordings. This is
bring-up evidence below the [acceptance program](markerless_mocap_acceptance.md)'s
Camera level: it shows that these units record together on this host, not that
a camera, layout or lab is qualified. Background and root causes are in the
[USB camera rig bring-up](usb_camera_rig_bringup.md); the software is
described in [Camera Rig Capture](capture_rig.md).

## Cabling Rules

1. One ELP AR0234 per USB 2.0 root port. Two on one root port never coexist at
   any resolution (firmware reserves the top isochronous alt-setting).
2. Put the cameras on the Sonnet eGPU's USB jacks, which map to distinct root
   ports; the CalDigit TS4 carries at most one camera and is better left for
   networking and displays.
3. Never chain the TS4 behind the Sonnet. Each dock goes to its own
   Thunderbolt port on the laptop. (The earlier TS4 hub failure was real;
   the topology walk reports `hub_depth` and the root port so a chained
   dock is visible in `plan-check`.)
4. Do not add a hub between a dock and a camera. A 30 ft powered (active)
   cable is itself two hub tiers, which is fine for one camera per cable on
   its own root port and is the planned wiring for all three views; never
   put two cameras on one cable. The 30 ft powered cable
   already spends two of the five allowed hub tiers.
5. After any cabling change, confirm the Sonnet routers are present (a
   Thunderbolt link drop is silent and turns every camera on it into a ghost):

```powershell
Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match 'Sonnet' }
```

## Plan File

The current lab layout is committed as
[`plans/lab_three_view_sonnet.json`](plans/lab_three_view_sonnet.json): three
views bound to `2605160001`, `2601240001` and, as `"unserialized": true`, the
one unit that reports no USB serial. Serial bindings recognise a unit on any
jack; the unserialized binding is resolved by elimination, so it survives a
move too, and is reported ambiguous (never guessed) if a second serial-less
unit is ever attached. Label each body with its serial and view name. Start a
new condition by copying the plan and changing what differs (resolution,
rate, exposure, gain, view names).

## Choosing a Mode

The ELP AR0234 advertises these MJPEG modes over DirectShow (`ffmpeg -f dshow
-list_options true -i video="Global Shutter Camera"`, 2026-09-06):

| Size      | Advertised max fps | Measured, one camera                                           | Compressed rate (dark lab) |
| --------- | ------------------ | -------------------------------------------------------------- | -------------------------- |
| 1920x1200 | 120                | 119.9 fps (555 frames / 4.63 s)                                | 36 MB/s                    |
| 1920x1200 | 60 (plan default)  | 59.8 fps (278 / 4.65 s)                                        | 18 MB/s                    |
| 1920x1080 | 120                | not measured                                                   |                            |
| 1600x1200 | 120                | opened, **zero frames** (I/O error)                            |                            |
| 1280x960  | 120                | not measured                                                   |                            |
| 1280x720  | 200                | 200 fps advertised; opened once, failed once — not yet trusted |                            |
| 1280x720  | 120                | 120.4 fps (620 / 5.15 s)                                       | 16 MB/s                    |
| 640x480   | 200                | 200.0 fps (718 / 3.59 s)                                       | 10 MB/s                    |

Frames and durations come from the decode probe in `recordings.json`. The
compressed rate is scene dependent: the same 1920x1200 @ 60 mode produced
4.8 MB/s per camera in the lit lab earlier in the day and 18 MB/s of noisy
MJPEG with the lights off, so size your disk for the dark case.

Any mode can be requested without editing the plan:

```bash
# one condition, every view
python3 -m motion_capture.rig record --plan P --mode 1280x720@120 --duration 10 --out S
# a subset of views (plan order), e.g. to bring one camera up alone
python3 -m motion_capture.rig record --plan P --views cam_b --duration 3 --out S
```

The derived plan is written to the bundle with its name suffixed by the
overrides (`lab-three-view-sonnet+cam_b+1280x720@120`), so a session always
records what actually ran. Per-view modes and exposure/gain still live in the
plan file.

**Three cameras together.** With all three on distinct root ports the rig
delivers the requested rate on every view: 1920x1200 @ 60 gave 60.2 / 60.0 /
60.1 fps (157 / 156 / 155 frames in 2.6 s) in the same run that the single
camera gave 59.8 fps, i.e. no measurable loss from running three. The loss is
in _bandwidth per root port_, not per camera: one camera per USB 2.0 root port
is the ceiling regardless of resolution. Lower resolutions do not let two
cameras share a port (the firmware reserves the same isochronous budget for
every mode); they buy frame rate and smaller files, not more cameras per port.

A recorder that opens the device but decodes zero frames (1600x1200 @ 120) is
reported `blocked` with ffmpeg's last words in `recorder_note`, never
`degraded`; `recorder_wall_s` gives the host-clock seconds each recorder ran
so throughput can be checked against the file size.

## Procedure

Run from the repository root. Each step exits 0 on success.

```bash
# 1. Is the plan realizable on this host as wired? (enumerates cameras, ~20 s)
python3 -m motion_capture.rig plan-check --plan docs/motion_capture/plans/lab_three_view_sonnet.json

# 2. Optional: observe frame rates and strobe alignment without recording.
python3 -m motion_capture.rig capture --plan docs/motion_capture/plans/lab_three_view_sonnet.json \
  --duration 8 --timing --out sessions/<date>-capture

# 3. Record. Warm-up (default 2 s) lets the devices open before the clock starts.
python3 -m motion_capture.rig record --plan docs/motion_capture/plans/lab_three_view_sonnet.json \
  --duration 10 --out sessions/<date>-record

# 4. Validate the bundle before handing it on.
python3 -m motion_capture.rig session-check --session sessions/<date>-record

# 4b. Once per camera: record the printed 9x6 chessboard moving through the view
#     (any short take), then calibrate; intrinsics.json is reused on every take.
python3 -m motion_capture.rig calibrate-intrinsics --session sessions/<board-take> --board 9x6 --square 0.025

# 5. Detect the golfer in every view (MediaPipe by default).
python3 -m motion_capture.rig ingest --session sessions/<date>-record

# 6. First take of a setup: initialise placement from the golfer, then fit.
python3 -m motion_capture.rig reconstruct --session sessions/<date>-record   --intrinsics docs/motion_capture/plans/intrinsics.json --anchor neck=0.53
# Later takes: start from the previous solution instead.
python3 -m motion_capture.rig reconstruct --session sessions/<date>-record   --cameras sessions/<previous>/reconstruct/reconstruction.json --anchor neck=0.53
```

Step 6 writes `reconstruct/reconstruction.json` (camera placement, learned
bone lengths, every rejected observation), `joints_3d_m.npy` and
`swing_summary.json` (turns, X-factor, hand speed, tempo). The anchor is one
segment measured once on the golfer with a tape; `neck` is hip-to-neck.

Read the outcome, not the exit code alone: `supported` means every view met at
least 90 % of the requested duration and rate; `degraded` names which view fell
short and by how much; `blocked` means a recorder failed or wrote nothing;
`unavailable` means no view recorded. `recordings.json` carries per-view
frames, duration, geometry and bytes from a decode probe of each file.

## Acceptance Evidence

Host: HP laptop, Windows 11, all three cameras on the Sonnet eGPU (root ports
4, 5, 6 of root hub `9&1d291187`). Requested: 1920x1200 MJPEG at 60 fps for
10 s per view.

| Run                              | View  | Frames | Duration | Bytes      | Verdict                    |
| -------------------------------- | ----- | ------ | -------- | ---------- | -------------------------- |
| First recording (before the fix) | cam_a | 472    | 7.91 s   | 38,997,682 | 20 % short, said supported |
|                                  | cam_b | 505    | 8.42 s   | 38,294,287 |                            |
|                                  | cam_c | 537    | 8.95 s   | 42,752,380 |                            |
| Warm-up + simultaneous stop      | cam_a | 616    | 10.30 s  | 50,681,913 | supported                  |
|                                  | cam_b | 619    | 10.31 s  | 46,695,085 |                            |
|                                  | cam_c | 616    | 10.26 s  | 48,976,838 |                            |

The first run is kept in the table on purpose. Every recorder exited 0 and the
bundle reported `supported` while each file was a fifth short, because
DirectShow devices take one to two seconds to open inside ffmpeg and the
recorders were stopped one after another. The fix (issue #9600) starts the
duration clock after a warm-up, signals all recorders before reaping any, and
decode-probes every file so a shortfall becomes `degraded`. After it, the
three views cover the same interval within about 50 ms and each carries about
4.8 MB/s of compressed MJPEG.

`plan-check` on the same wiring matched all three views with no root-port
conflicts; `session-check` on the second bundle reported no problems.

## Stability Across Sessions

Six back-to-back three-camera `record` runs on 2026-09-06 (4 s each; all
three cameras on Sonnet jacks, cam_c through a 30 ft powered cable, which
is two hub tiers):

| Run | Mode           | Result                                                     |
| --- | -------------- | ---------------------------------------------------------- |
| 1   | 1920x1200 @ 60 | supported: 59.8 / 60.0 / 60.1 fps                          |
| 2   | 1920x1200 @ 60 | supported: 59.8 / 60.1 / 60.0 fps                          |
| 3-5 | 60 and 120     | `plan-check` aborted: cam_b not enumerated                 |
| 6   | 640x480 @ 200  | blocked: cam_a 200.5 fps; cam_b, cam_c `I/O error` at open |

One camera's `LastArrivalDate` moved to 20:50:23, so a unit dropped off the bus
and re-enumerated during the series. Solo opens of every camera succeed. The
rig therefore meets the rate on every view when the cameras open, but is not
yet shown to open reliably session after session; #9613 tracks the
soak test that decides whether the active-cable hubs, Sonnet USB power, or
the DirectShow open race is responsible. Until it closes, check `plan-check` and
the bundle outcome before every take rather than trusting the previous one.

## Proxies for Playback

The recordings are MJPEG in Matroska, which OpenCV and ffmpeg read but browsers
do not. The React Video Analyzer (`ui/src/pages/VideoAnalyzer.tsx`) and the
Tools web Video Processor both play through an HTML `<video>` element, so make
H.264 proxies first:

```bash
python3 -m motion_capture.rig proxy --session sessions/<date>-record   # libx264
python3 -m motion_capture.rig proxy --session S --encoder h264_nvenc    # GPU
```

Each `<view>_<identity>.mp4` lands beside its recording and `proxies.json`
records the encoder and ffmpeg's exit per view. Proxies are for viewing only:
ingest reads the original MJPEG, and a failed transcode is listed with its
reason rather than dropped. The PyQt6 MediaPipe/OpenPose GUIs can decode the
`.mkv` directly once their file filter admits it (#9611).

## Import, Analyse, Reliability, Export

Commands added for the guided workflow (#9658); the Capture Rig tile runs
the same ones, and `docs/motion_capture/user_guide.md` (generated from the
tile's step model) walks through them in order.

```bash
python3 -m src.motion_capture.rig import --out S --view face_on=clip.mp4 [--view dtl=clip2.mp4]
python3 -m src.motion_capture.rig ingest --session S --estimator openpose_dnn --option input_height=368 --out S/observations_openpose_dnn
python3 -m src.motion_capture.rig reliability --session S
python3 -m src.motion_capture.rig analyze --session S            # single view: 2-D events and tempo
python3 -m src.motion_capture.rig reconstruct --session S --intrinsics S/intrinsics.json --anchor shank=0.42 --anchor forearm=0.26 --exclude-joints nose
python3 -m src.motion_capture.rig export --session S             # reconstruct/reconstruction.trc + reconstruction_export.json
```

- `import` builds a bundle around existing files (one file is a valid
  single-camera session); nothing is copied and each file's probe becomes
  the recorded capture mode.
- `ingest --option KEY=VALUE` passes typed settings to the estimator and
  records them in the observations' provenance; `--out` keeps one set per
  detector so re-runs never overwrite another detector's output.
- `reliability` grades every shared joint from all observation sets and the
  clean report (`reliability.json` + `.md`) and lists recommended exclusions.
- `analyze` writes `analysis_2d/<view>.json`: address / top / peak / finish,
  tempo and normalised hand speed in subject box heights.
- `reconstruct --exclude-joints` treats the named joints as unobserved;
  `export` writes the fitted joints as a TRC marker file the motion pipeline
  and model-matching tools read directly.

## Boards, Clips and Take Comparison

```bash
python3 -m src.motion_capture.rig board --board charuco:7x5:0.04:0.03 --out board.png   # print at 100 %, measure a square
python3 -m src.motion_capture.rig calibrate-intrinsics --session S --board charuco:7x5:0.04:0.03
python3 -m src.motion_capture.rig clip --session S --view cam_b --from address-30 --to finish+30 --speed 0.25 --out swing.mp4
python3 -m src.motion_capture.rig compare-takes --session S --view cam_b --other-session T --other-view cam_b --align top --out ab.mp4
```

- **ChArUco boards** (#9679) are detected partially, with corner ids, so
  frames at the edge of the field count; the plain `9x6` chessboard still
  works. Print the generated image at 100 % and enter the measured square
  and marker sizes.
- **Clips** (#9680) keep every source frame and play at `fps x speed`, with
  the pose overlay and a frame/time stamp; frame bounds are events
  (`address`, `top`, `peak`, `finish`, with `+N`/`-N`) or numbers.
- **Compare takes** (#9681) renders two views side by side aligned on the
  chosen event (each at its own rate) and writes the metric deltas beside
  the video. The tile offers both as _Export clip_ and _Compare takes_.

## Which Segments to Measure

`--anchor` is repeatable (#9707). The first reading sets the scale; every
further one replaces a 5 cm anthropometric prior with a 3 mm tape reading,
so measure as many as you can. Everyday names constrain both sides at once;
use `left_...`/`right_...` only when the sides really differ.

| Name             | Tape from ... to ...                                       | Why it ranks where it does                              |
| ---------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| `shank`          | lateral knee joint line → lateral ankle bone, leg straight | bony landmarks, long, rigid, knees/ankles detected well |
| `forearm`        | lateral elbow crease → wrist bone, arm straight            | bony landmarks; carries the swing                       |
| `upper_arm`      | acromion → lateral elbow joint line                        | good landmarks; shoulder detection a little softer      |
| `thigh`          | greater trochanter → lateral knee joint line               | long, but the hip centre sits inside the body           |
| `shoulder_width` | acromion → acromion across the back                        | halves onto the two shoulder segments                   |
| `hip_width`      | trochanter → trochanter                                    | halves onto the two hip segments                        |
| `torso`          | mid-hip → base of the neck (C7)                            | both ends are virtual points; measure last              |

The tile's _Measured segments_ field takes the same `name=metres` list and
shows these landmarks as a tooltip.

## Articulated Model Fit

```bash
python3 -m src.motion_capture.rig fit-model --session S [--sigma-accel 300] [--max-velocity 25]
python3 -m src.motion_capture.rig export --session S      # now also model/joint_angles_simscape.csv
```

After a reconstruction, `fit-model` solves the joint angles of the
scapula-capable golfer (spine, axial torso, scapula struts from the hub,
gimbal shoulders, elbows, pronation, wrists, plus legs and head for the
detectors) through every frame at once with an acceleration prior on each
angle, soft joint limits and robust rejection: a point the model cannot reach
by continuous motion is listed in `model/fit_report.json`, never followed.
`--sigma-accel` is the continuity strength in rad/s² (smaller = stiffer);
`--max-velocity` flags joint speeds above it. Design and evidence:
`docs/motion_capture/articulated_model.md`. The tile runs it as _Fit model_
and shows the report in the _Model fit_ tab.

## Capture Rig Tool (Desktop)

The launcher tile **Capture Rig** (`python3 -m src.tools.capture_rig`) is the
desktop front end over the commands above. It does not re-implement any of
them: every button starts `python -m src.motion_capture.rig ...` as a child
process with the same flags, so a bundle made from the tool is
indistinguishable from one made in a terminal.

- **Capture** — plan file, capture mode preset (the advertised ELP modes),
  view subset, exposure / gain / auto-exposure overrides (`--exposure`,
  `--gain`, `--auto-exposure`, applied to every selected view and recorded
  in the derived plan name), duration and dry run. _Plan check_ and _Record_.
- **Process** — _Proxies_, _Ingest_ with any registered estimator (MediaPipe,
  OpenPose, OpenPose BODY_25 DNN; unavailable ones are marked with the
  install hint), _Calibrate intrinsics_ (board, square size) and
  _Reconstruct_ (anchor segment and tape-measured length; a file named
  `intrinsics*.json` starts a first take, anything else is taken as a
  previous `reconstruction.json`).
- **Review** — frame-accurate playback of any view (the H.264 proxy when it
  exists, else the recording) with the ingested pose drawn on the frame it
  came from; joints under the confidence threshold are drawn small and red
  rather than hidden. The swing summary appears as a metric table once
  `reconstruct/swing_summary.json` exists.

The log pane shows the child's output verbatim and _Stop_ kills it. A
session folder can also be loaded directly to review an earlier take.

## What This Does Not Show

No calibration, no timing between cameras beyond arrival clocks (strobe
alignment is `capture --timing`, not yet applied to recordings), no pose
quality, no reconstruction. Those are the remaining children of #9599 and the
acceptance program's Camera and Layout levels.
