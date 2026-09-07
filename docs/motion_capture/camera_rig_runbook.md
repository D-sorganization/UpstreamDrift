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
   Thunderbolt port on the laptop.
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

## Procedure

Run from the repository root. Each step exits 0 on success.

```bash
# 1. Is the plan realizable on this host as wired? (enumerates cameras, ~60 s)
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

## What This Does Not Show

No calibration, no timing between cameras beyond arrival clocks (strobe
alignment is `capture --timing`, not yet applied to recordings), no pose
quality, no reconstruction. Those are the remaining children of #9599 and the
acceptance program's Camera and Layout levels.
