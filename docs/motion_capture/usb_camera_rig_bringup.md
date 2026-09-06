# USB Camera Rig Bring-Up

Version: 1.0.0

Issues: #9063, #9069; Tools #4706

Date of evidence: 2026-09-06

This note records the physical bring-up of a three-camera USB rig for the
markerless-mocap program and the constraints it exposed. It is bring-up
evidence for one host and one set of cameras. Under the
[acceptance program](markerless_mocap_acceptance.md) it qualifies neither a
camera, a layout, nor a physical lab; it documents what was measured so the
next person does not re-derive it.

## Hardware

| Item    | Detail                                                                                                               |
| ------- | -------------------------------------------------------------------------------------------------------------------- |
| Cameras | 3 x ELP 2 MP USB 2.0, AR0234 global shutter; enumerate as "Global Shutter Camera", `VID_32E4&PID_5234`               |
| Dock    | CalDigit TS4 (USB4 router `VID_8087&PID_0B26`, firmware 39.1, Microsoft inbox USB4 driver 10.0.26100.8972)           |
| eGPU    | Sonnet eGPU Breakaway Box 750ex; exposes two tunneled xHCIs (Fresco Logic `1B73:1100`, Intel `15C1`) and an I210 NIC |
| Host    | HP laptop, i7-13700H (20 logical cores), Windows 11 26200; xHCIs `DEV_51ED` and `DEV_A71E`; two USB4 host routers    |
| Cable   | one 30 ft powered USB cable (contains a Realtek `0BDA:5411/0411` hub pair, i.e. two hub tiers)                       |
| Serials | `2605160001`, `2601240001`, and one unit that exposes no `iSerial` (identity falls back to its USB port path)        |

## Findings

Each item below was measured, not inferred.

1. **A camera behind the TS4 was invisible.** The TS4 had been daisy-chained
   behind the Sonnet. Its Realtek hub pair reported `CM_PROB_FAILED_START`,
   then disappeared (`present=False`), and its USB 2.0 companion hub
   (`PID_5411`), the only attachment path for a USB 2.0 device, never
   enumerated. The dock's PCIe tunnel (Intel i225 2.5GbE) kept working, so the
   dock looked healthy. Fix: TS4 alone on its own Thunderbolt port. One camera
   through the dock then ran 1920x1200 at 60.0 fps for 30 s with no failed
   reads.
2. **Hub depth reaches the USB limit.** The path TS4 -> 30 ft cable -> camera is
   five cascaded hubs. USB allows five. A hub added downstream of the TS4 fails
   silently: no error, no enumeration.
3. **The 30 ft powered cable is fine.** 60.0 fps, no failed reads, worst frame
   gap 57.8 ms versus 49.3 ms on a short cable.
4. **Only one of these cameras streams per USB 2.0 bus, at any resolution.**
   See the root cause below.
5. **One camera per root port works at full rate.** Moving a camera to a laptop
   root port gave two concurrent streams at 59.7 and 60.2 fps while the camera
   still sharing a root port produced nothing, exactly as predicted.
6. **CPU, not USB, is the next ceiling.** One decoded 1920x1200 at 60 fps MJPEG
   stream costs 1.2 to 1.5 cores; two cost about four (404 %). Requesting NV12
   (`CAP_PROP_CONVERT_RGB = 0`) drops two streams to 94 %: the JPEG decode is
   cheap, the NV12-to-BGR conversion is not. Unrelated CPU load on the host
   (a `pytest` run) pulled streams down to about 43 fps with 100 to 145 ms
   stalls.
7. **The Thunderbolt link to the eGPU can drop silently.** While cameras were
   being plugged into the Sonnet, its routers, both xHCIs, PCIe switch ports,
   and NIC all became `present=False` at once, with no error and no problem
   device. Re-seating the Thunderbolt cable restored everything. Confirm the
   Sonnet routers are present before trusting a capture session.

## Root Cause of the One-Camera-per-Bus Limit

A `usbview` descriptor dump is identical on all three units. Bus speed is High.
The VideoStreaming interface offers alt-settings `0x01` to `0x0B`; the largest is

```text
bAlternateSetting 0x0B   wMaxPacketSize 0x13FC
  = 3 transactions per microframe x 1020 bytes = 3060 bytes per microframe
```

USB 2.0 High-Speed carries 7500 bytes per 125 us microframe and caps periodic
(isochronous) reservation at 80 %, i.e. 6000 bytes. The camera firmware
requests alt-setting `0x0B` for every format, so `usbvideo.sys` reserves 3060
bytes per stream regardless of resolution. Two streams need 6120 bytes. The
second `SET_INTERFACE` is refused, its `read()` fails immediately, and that
capture handle never recovers. xHCI accounts High-Speed periodic bandwidth per
root port, so the rule is one camera per root port. The TS4 is one root port.

## Evidence

| Test                                                   | Result                                                      |
| ------------------------------------------------------ | ----------------------------------------------------------- |
| A: three cameras on the TS4, 1920x1200 at 60           | 1 of 3 streams                                              |
| B: two cameras, 640x480 at 5 fps (under 1 MB/s)        | 1 of 2 streams; rules out throughput                        |
| Sweep 1920x1200 down to 640x480, at 60 and 30 fps      | 1 of 2 at every mode; the winner is whichever commits first |
| C: staggered open, then the winner closes              | the loser never recovers on the same handle                 |
| D: built-in webcam (root `A71E`) plus one ELP (`51ED`) | both stream; the conflict is bus-level, not software        |
| E: ELP on root port 4 plus ELP on root port 13         | 59.7 and 60.2 fps                                           |
| Solo, each unit, fresh open                            | 59.9 to 60.2 fps, no failed reads                           |

## Validated Topologies

Both configurations below streamed all three cameras concurrently at
1920x1200 MJPEG with no failed reads, and the diagnostic script's prediction
matched its measurement in every run.

| Topology                        | Placement                                                                        | Measured fps       |
| ------------------------------- | -------------------------------------------------------------------------------- | ------------------ |
| TS4 plus Sonnet                 | one camera on the TS4; two on Sonnet root ports 1 and 6 of root hub `9&1d291187` | 60.0 / 59.9 / 59.5 |
| All on the Sonnet (recommended) | root ports 4, 5, and 6 of root hub `9&1d291187`; one via the 30 ft cable         | 59.7 / 59.9 / 60.1 |

The second topology leaves the TS4 free for networking and displays, keeps all
camera cabling on one box, and leaves the Sonnet's second controller unused for
a later fourth view or a high-speed club and ball camera. Note the granularity:
three cameras on distinct root ports of one controller coexist.

## Long-Term Options

| Option                                   | Effect                                              | Verdict                                                        |
| ---------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------- |
| One camera per root port (TS4 or laptop) | proven on root port 13                              | works today; ties cable runs to the laptop                     |
| Sonnet on the second Thunderbolt port    | two more controllers, plus a GPU for pose inference | recommended; never chain it behind the TS4                     |
| USB 3.0 variants of the AR0234 module    | many cameras per bus                                | hardware purchase; avoids the reservation model entirely       |
| Linux `uvcvideo quirks=0x80`             | recomputes the reservation from the real format     | not available on Windows; WSL2 cannot pass isochronous traffic |
| Any hub behind the TS4                   | none                                                | fails silently at the five-hub limit                           |
| Lower resolution or frame rate           | none                                                | the reservation is format-independent                          |

## Recording Strategy

Do not decode to BGR during capture. Either take NV12 from Media Foundation or
record the compressed MJPEG stream and decode offline. Measured with
`ffmpeg` from `imageio-ffmpeg` on camera `2601240001` for 5 s:

```text
ffmpeg -f dshow -vcodec mjpeg -video_size 1920x1200 -framerate 60 \
  -i "video=@device_pnp_\\?\usb#vid_32e4&pid_5234&mi_00#<instance>#{...}\global" \
  -c:v copy out.mkv
-> 301 frames in 5.03 s (60 fps), 55.8 MB (11.2 MB/s), decodes back at 136 fps
```

Three cameras are about 34 MB/s to disk with near-zero CPU. Address a camera by
its DirectShow alternative name (the USB instance path); the friendly name
"Global Shutter Camera" is shared by all three units. NV12 to disk would be
about 200 MB/s per camera.

## Rig Requirements

- Bind cameras by USB serial with a port-path fallback. One unit has no serial,
  and OpenCV indices reshuffle on replug, which would silently swap views.
- After opening each camera, verify that frames arrive; recreate any handle
  that does not. A lost reservation is permanent on that handle.
- Record achieved frames per second per camera as capture metadata. The
  failure mode is silent and asymmetric: the winning camera looks perfect and
  the losing camera produces nothing.
- Wait at least two seconds between opens. Media Foundation teardown is
  asynchronous, and back-to-back opens read 43 to 47 fps instead of 60.
- Keep unrelated CPU-heavy work off the capture host during a session.
- Plan the topology before mounting, with the diagnostic script below.

## Diagnostic Script

`scripts/diagnose_mocap_camera_rig.py` enumerates the cameras through Windows
PnP, walks each hub chain to its root port, maps each camera to its capture
index through DirectShow enumeration order (which matches Media Foundation
order), predicts how many cameras can stream, then measures a solo and a
concurrent capture and compares the two. It writes `rig_report_<timestamp>.json`
and one frame per camera. Exit code 0 means every camera streamed at 90 % of
the target rate or better; 1 means a conflict; 2 means no cameras or no OpenCV.

```bash
python3 scripts/diagnose_mocap_camera_rig.py --out-dir /tmp/rig
```

The script is a Windows operator diagnostic. It defines no capture contract:
camera and capture contracts belong to Tools `sidekick.lab.mocap` under
[ADR-0041](../adr/0041-markerless-mocap-consumer-authority.md).

## Relationship to the Acceptance Program

This evidence sits below the Camera level of the acceptance program. It shows
that three specific units stream concurrently on one specific host. It does
not measure blur, exposure, thermal stability, timestamp skew, or trigger
response, and it establishes no shared exposure clock across the three
controllers. Arrival time is not exposure time; a trigger line or a per-session
visual sync event is still required before calibration.

## Open Items

1. Choose the time-sync source: a hardware trigger if the modules expose one,
   otherwise a per-session flash event.
2. Measure the three-stream CPU budget on a quiet host (expected about 1.5
   cores for NV12 and about 6 cores for BGR).
3. `MediaPipeEstimator` is unavailable in this environment: mediapipe 0.10.35
   removed `mp.solutions`, and `engine_availability.py` fails closed with a
   message that says "not installed". Migration to the Tasks API is separate
   work.
