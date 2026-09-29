---
title: Video Processor
tile_id: video_processor
status: complete
---

# Video Processor

## Purpose

Open a swing video in a browser-based editor to step through it frame by
frame, draw annotations on top of it, and run an in-browser pose overlay
and swing-analysis view. It is the general-purpose "look at the video"
tool, as distinct from the estimator tiles that batch a video into a
JSON file of joint angles.

**This tile is not part of UpstreamDrift.** Its registry entry declares
`provider: tools`, and the entry point
`src/media_processing/video_processor/apps/web/launch_platform.py`
resolves against the sibling **Tools** repository, not this one. If you
have not checked Tools out beside UpstreamDrift, the tile has nothing to
launch.

## Inputs

| Input | Unit / type | Notes |
| --- | --- | --- |
| Video file | `video/mp4`, `video/webm`, `video/ogg`, `video/quicktime` (`.mov`), `video/x-msvideo` (`.avi`), `video/x-matroska` (`.mkv`) | Uploaded through the page. Declared MIME type is cross-checked against the file's magic bytes. |
| Maximum file size | 500 MB (`500 * 1024 * 1024` bytes) | Hard limit in the upload validator. Also surfaced as a configurable `maxFileSizeMB`, default 500 MB. |
| Working frame rate | Hz (frames per second), default 30 | Configuration value `defaultFPS`, capped at 120 Hz. Used to convert between frame index and seconds; frame duration is `1 / defaultFPS` seconds. |
| Current frame | integer frame index, 0-based | Clamped to `0 .. totalFrames - 1` by the frame navigator. |
| Annotations | drawing input in canvas pixels | Drawn over the video on a Fabric.js canvas. |
| Node.js | version 18 or newer, plus npm 9 or newer | A prerequisite of the launcher, not a user setting. |

## Outputs

| Output | Unit / type | Notes |
| --- | --- | --- |
| A running local web application | HTTP on the Next.js dev-server port | The tile's job is to start this; the work happens in the page. |
| Frame readout | frame index, time as `frame / fps` in seconds, and the frame rate in Hz | Displayed by the frame navigator. |
| Exported annotations | annotation records | Produced by the app's annotation exporter. |
| On-screen swing analysis | scores, phase timeline, tempo chart, metrics, detected issues, recommendations | Rendered in the browser by the app's swing-analysis dashboard. |

Concrete numeric ranges and units for the swing metrics are defined
inside the Tools repository, not here; read them there rather than
assuming.

## Method

`launch_platform.py` is a thin cross-platform launcher, not the tool. It
checks that `node --version` runs, checks that `package.json` is present
beside itself, runs `npm install` if `node_modules` is absent, then runs
`npm run dev` and stays in the foreground until interrupted. Each step
exits with a specific error message on failure.

What it starts is a Next.js application (`@golf-swing-analyzer/web`)
that does its media work client-side: `@ffmpeg/ffmpeg` (ffmpeg compiled
to WebAssembly) for video handling, `@mediapipe/pose` for the pose
overlay, and `fabric` for the annotation canvas. Its registry entry adds
`src` and `src/shared/python` to `PYTHONPATH` and sets the working
directory to the app folder, and it declares the web route
`/tools/video-processor`.

The tile shares the video-analysis walkthrough at
[Video analysis tutorial](../tutorials/content/04_video_analysis.md) with
the MediaPipe and OpenPose tiles.

## Limitations

- **It is not a PyQt tool and cannot be docked.** There is no
  `get_dockable_ui`; the launcher spawns a web dev server. Nothing
  renders inside the launcher window itself.
- **It needs a Node.js toolchain and, on a cold start, network access**
  for `npm install`. Neither is a Python dependency the UpstreamDrift
  venv can satisfy.
- **It runs a development server** (`npm run dev`), not a production
  build.
- **It lives in another repository.** UpstreamDrift's tests and CI do
  not exercise it, and its behaviour can change without any change here.
- **Its registry description overstates the overlap with this repo.**
  The tile is described as a "shared media processor for video
  conversion, frame extraction, and analysis"; the application it starts
  is a golf-swing video analysis platform for coaches. Format conversion
  and frame extraction are ffmpeg-in-the-browser capabilities of that
  app, not a shared UpstreamDrift service.
- **Its pose overlay is estimated, not measured.** It uses MediaPipe in
  the browser, so the same caveat applies as on the
  [MediaPipe](mediapipe_analysis.md) page: the landmarks are a network's
  inference from pixels.
- **Nothing it produces enters the motion pipeline automatically.**
  There is no wiring from this tile into the canonical observation
  format.

## See Also
- [Video analysis tutorial](../tutorials/content/04_video_analysis.md)
- [MediaPipe](mediapipe_analysis.md), [OpenPose](openpose_analysis.md)
- [Motion pipeline format matrix](../motion_pipeline/formats.md) - what the pipeline will actually ingest
- [Motion Capture (FreeMoCap sidecar)](motion_capture.md)
- [Analysis Tools](analysis_tools.md)
