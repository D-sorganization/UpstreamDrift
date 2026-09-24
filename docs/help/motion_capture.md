---
title: Motion Capture
tile_id: motion_capture
status: complete
---

# Motion Capture

## Purpose

Turn a multi-camera video recording of a swing into frame-by-frame 3D
landmark positions, by driving [FreeMoCap](https://freemocap.org/) as an
isolated subprocess. Use this tile when you have already recorded and
calibrated a FreeMoCap session and want its triangulated landmarks on
disk so the rest of UpstreamDrift can consume them. It is the markerless
counterpart to the optical-marker workflow in [C3D Viewer](c3d_viewer.md).

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| `--input` session directory | filesystem path (required) | Directory of the FreeMoCap session recordings (the synchronised camera videos). |
| `--output` directory | filesystem path (required) | Created if absent; receives `landmarks.csv` and `metadata.json`. |
| `--env-python` | filesystem path (optional) | Interpreter inside the separate FreeMoCap virtualenv. Defaults to the current `sys.executable`, which will normally *not* have FreeMoCap installed. |
| `--timeout` | seconds (float, default `1800.0`) | Subprocess wall-clock limit, i.e. 30 minutes. |
| `--dry-run` | flag | Skips the subprocess entirely and writes stub artifacts. |
| `--json` | flag | Prints the result record to stdout as JSON. |

Camera count, capture frame rate (Hz) and calibration are properties of
the FreeMoCap session you recorded, not of this tile; it passes the
directory through untouched.

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| `landmarks.csv` | columns `frame,landmark_id,x,y,z` | `frame` is a 0-based integer frame index; `landmark_id` is a 0-based integer. `x`/`y`/`z` are written by FreeMoCap in whatever length unit it emits (typically metres) - **the sidecar performs no unit conversion and no axis reorientation**. |
| `metadata.json` | JSON object | Session metadata as emitted by FreeMoCap: `freemocap_version`, `n_frames` (frames), `n_landmarks` (count), `fps` (Hz), `duration_s` (seconds). |
| Process exit code | integer | `0` on success; otherwise the subprocess return code, with `-1` for timeout and `127` for a missing interpreter. |
| `FreeMoCapResult` (Python API) | dataclass | `success`, `used_real_freemocap`, `return_code`, `stderr_tail` (last 4 KB of stderr), and the two artifact paths. |

## Method

`run_freemocap_sidecar()` in
[`src/tools/freemocap_sidecar/run_freemocap.py`](../../src/tools/freemocap_sidecar/run_freemocap.py)
shells out to `<env-python> -m freemocap --input ... --output ...` and
then reads the results back with the standard library only. No symbol
from `freemocap` is ever imported into the UpstreamDrift process. This
is a licence boundary, not a performance choice: UpstreamDrift is MIT
and FreeMoCap is AGPL. The rationale and the process diagram are in
[FreeMoCap sidecar pipeline](../motion_capture/freemocap_sidecar.md) and
[FreeMoCap integration](../motion_capture/freemocap.md).

FreeMoCap's own landmark estimation is MediaPipe-based, and its
`n_landmarks` is reported as 33 in the documented output contract.

## Limitations

- **It is a command-line runner, not a GUI.** The module exposes an
  `argparse` CLI and no Qt window, and both `--input` and `--output` are
  required. Invoking it with no arguments exits with an argparse usage
  error.
- **It does not record.** Recording, camera synchronisation and
  checkerboard calibration all happen inside FreeMoCap itself,
  beforehand.
- **It does not install FreeMoCap**, and will not work until you have
  built a separate virtualenv and passed its interpreter via
  `--env-python`. Installing FreeMoCap into the main UpstreamDrift
  environment is explicitly disallowed.
- **A "successful"-looking run can be a stub.** When the interpreter is
  missing, or the module is not installed, or `--dry-run` is set, the
  sidecar still writes `landmarks.csv` and `metadata.json` so downstream
  code has a stable contract. Stub metadata carries `"stub": true`,
  `"n_landmarks": 1` and `"freemocap_version": "stub"`. Check
  `used_real_freemocap` before trusting any numbers.
- **Landmark positions are inference, not measurement.** They are the
  triangulated output of a pose estimator run on video, with no optical
  markers to anchor them. The repo documentation notes plainly that
  MediaPipe Holistic was not trained on high-speed sports motion, so
  tracking dropouts through the downswing are expected.
- **No smoothing, gap filling, unit conversion, retargeting or scaling
  happens here.** Those belong to the motion pipeline downstream.
- The tile is registered at maturity `beta`.

## See Also
- [FreeMoCap sidecar pipeline](../motion_capture/freemocap_sidecar.md)
- [FreeMoCap integration](../motion_capture/freemocap.md)
- [Markerless mocap acceptance criteria](../motion_capture/markerless_mocap_acceptance.md)
- [Motion pipeline workflow guide](../motion_pipeline/README.md)
- [Motion pipeline format matrix](../motion_pipeline/formats.md)
- [Motion pipeline troubleshooting](../motion_pipeline/troubleshooting.md)
- [ADR-0007: motion pipeline architecture](../adr/0007-motion-pipeline-architecture.md)
- [C3D Viewer](c3d_viewer.md), [MediaPipe](mediapipe_analysis.md), [OpenPose](openpose_analysis.md)
