# ADR-0049: OBS Studio Is Not a Capture Layer for the Markerless Rig

- **Status:** Accepted
- **Date:** 2026-09-07
- **Issue:** #9678 (epic #9677)
- **Deciders:** repository owner, Claude (implementation)

## Context

The markerless motion-capture rig records three USB cameras through
`python3 -m src.motion_capture.rig record`: one ffmpeg DirectShow stream copy
per camera, per-camera probes (frames, duration, achieved rate), UVC control
overrides, a bundle index with typed outcomes, and a strobe-based timing
alignment. The question was whether integrating OBS Studio would add value
for the user, or whether the in-house path is the better foundation.

## Decision

OBS Studio is **not** used as the recorder and is not a dependency of the
rig. The in-house path stays the capture layer. OBS is kept open as an
optional _share_ surface only (streaming, virtual camera), reached through
obs-websocket from the tile if a user ever asks for it.

## Rationale

What the rig needs from a recorder, and what OBS provides:

| Need                                               | `rig record`                           | OBS Studio                                                 |
| -------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------- |
| One raw stream per camera, no re-encode            | MJPEG stream copy, per camera          | Composites sources into one re-encoded output              |
| Per-camera frame count, duration, achieved fps     | Probed and stored in `recordings.json` | Not exposed; one output clock                              |
| Camera controls (mode, exposure, gain) per view    | Plan file + CLI overrides              | Source properties, per scene, not scriptable per take      |
| Frame-accurate alignment across cameras            | Strobe alignment, reference clock      | None across sources                                        |
| Headless, scriptable, identical from CLI and GUI   | Yes (the tile runs the CLI)            | GUI application; scripting through a plugin/websocket only |
| Typed session outcome (supported/degraded/blocked) | Yes                                    | No                                                         |

Where OBS is genuinely better: live composition and overlays for a viewing
audience, streaming, and a virtual camera other applications can consume.
None of those is on the capture-to-reconstruction path. The two pieces that
touch coaching are covered in house and remain customisable: playback with
pose overlay in the tile, and `rig clip` / `rig compare-takes` for annotated,
slowed exports (#9680, #9681).

## Consequences

- No OBS dependency, plugin, or scene template ships with the repository.
- Recording quality work (mode sweeps, bandwidth rules, warm-up handling)
  keeps landing in `motion_capture.rig`.
- If a share/stream workflow is requested, the design is: the tile talks to
  a user-installed OBS over obs-websocket to start/stop a stream of the
  tile's own rendered output; OBS never records the cameras itself.

## Alternatives Considered

- **OBS as recorder with one scene per camera and multi-track output:** OBS
  multi-track applies to audio; video is one composited track. Rejected.
- **OBS virtual camera into `rig record`:** adds a re-encode and a
  compositor between the sensor and the file. Rejected.
