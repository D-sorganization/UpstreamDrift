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
   Thunderbolt port on the laptop. (On 2026-09-06 a camera two hub tiers
   deep behind a TS4 on the Sonnet did stream at 60 fps beside the other
   two; the rule stays because the earlier TS4 hub failure was real, and
   the topology walk reports `hub_depth` so the case is visible.)
4. Do not add a hub between a dock and a camera; the 30 ft powered cable
   already spends two of the five allowed hub tiers.
5. After any cabling change, confirm the Sonnet routers are present (a
   Thunderbolt link drop is silent and turns every camera on it into a ghost):

```powershell
Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match 'Sonnet' }
```

## Plan File

The current lab layout is committed as
[`plans/lab_three_view_sonnet.json`](plans/lab_three_view_sonnet.json): three
views bound to `2605160001`, `2601240001` and the serial-less unit's port path
`path_D-D35A8F7-0-0000`. A port-path identity changes if that unit is moved to
another jack; `plan-check` reports it as missing, and the fix is to update the
plan, never to guess. Start a new condition by copying the plan and changing
what differs (resolution, rate, exposure, gain, view names).

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
```

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

Six back-to-back three-camera `record` runs on 2026-09-06 (4 s each, TS4
chained on the Sonnet with cam_c two hub tiers deep, against rule 3):

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
soak test that decides whether the TS4 chain, Sonnet USB power, or the
DirectShow open race is responsible. Until it closes, check `plan-check` and
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

## What This Does Not Show

No calibration, no timing between cameras beyond arrival clocks (strobe
alignment is `capture --timing`, not yet applied to recordings), no pose
quality, no reconstruction. Those are the remaining children of #9599 and the
acceptance program's Camera and Layout levels.
