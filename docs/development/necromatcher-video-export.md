# Necromatcher Video Export and Control Procedure

## Scope and Acceptance

Issue #11246 under epic #11232 delivers original-footage model review through
the same native/web library. Export a saved source-bound fit, poll or cancel
owned execution, and download a verified ZIP containing the original-sized
overlay MP4, selected stills and exact-frame/hash manifest. Geometry remains a
native body-origin joint-tree wireframe plus declared finite attachment seeds;
the output is an unqualified monocular research hypothesis.

Scientific acceptance is separate from artifact readiness. Export does not
establish player anatomy, camera calibration, physical swing time, measured
joint effort, continuous grip closure, independent dynamics or impact validity.

## Implementation Plan

1. Reuse `export_fit_video` for original-frame rendering and codec verification.
2. Run it in a clean SDK subprocess owned by the canonical matching-job service.
   Retain execution stamps, source/model/capture identities and cancellation.
3. Expose one shared `NativeVideoSession` for native and API consumers. Reopen
   terminal results from durable manifests; unowned active records remain
   execution-unverified and cannot offer live cancellation or download.
4. Add local-only API submission, polling, cancellation and checked download.
   Accept a fit identity, never a caller-supplied server filesystem path.
5. Bind native/web controls to the selected saved fit. Separate refit version
   creation from video artifact readiness. Hide stale results when ownership
   changes, and retain export-run identity for later recall.
6. Verify with red-first job/transport/UI tests and actual player exports, then
   update turnover, the standalone methods report and GitHub evidence.

## API Contract

| Action   | Local API Route                                            |
| -------- | ---------------------------------------------------------- |
| Submit   | `POST /api/v1/necromatcher/fits/{fit_id}/video-exports`    |
| Poll     | `GET /api/v1/necromatcher/video-exports/{run_id}`          |
| Cancel   | `POST /api/v1/necromatcher/video-exports/{run_id}/cancel`  |
| Download | `GET /api/v1/necromatcher/video-exports/{run_id}/download` |

The response binds `run_id` and `source_fit_id` and separates `status`,
`acceptance`, `execution_started`, `execution_verified`, `control_available`
and `download_available`. `qualification` remains
`monocular_research_hypothesis`. A successful codec export is not an accepted
historical reconstruction. No host filesystem paths are exposed in responses.

## Review and Repeatability

The initial actual Desktop examples are under
`C:/Users/diete/Desktop/Necromatcher Review 2026-10-01/`: Tiger Overlay has 210
frames at 1280×720 and source rate 30000/1001; Hogan Overlay has 750 frames at
320×240 and source rate 30. The local review index and SHA-256 manifest also
identify original portable fit packages, failed interim candidates and the
compiled standalone LaTeX report. Media remains outside Git.

The exact source PTS governs playback only. Irregular timestamps or gaps are
rejected. Original PNGs remain unchanged; blue model geometry, green image
observations and yellow same-identity residuals reveal disagreements. Use
worst-frame, whole-track, midpoint and visibility diagnostics alongside these
videos before considering a scientific acceptance claim.

## Verification Evidence

The original renderer passed eight red-first tests for native geometry, exact
frame/source bindings, finite projection, timing, missing observations,
unchanged source bytes, codec/PNG readback and exclusive destination creation.
The control implementation adds admission, process ownership, cancellation,
durable recall, guarded download, cross-fit UI ownership and stale-response
coverage. Record completed checks and actual UI journeys in turnover rather
than treating this plan as execution evidence.

Actual native/web control verification and the two live downloaded runs are
recorded in `necromatcher-turnover.md` and
`historical_capture/live-web-export-review.json`. Updated Desktop directories
`Tiger Web Overlay` and `Hogan Web Overlay` contain all-frame decoded outputs,
first/middle/last stills and SHA-256 manifests. Completed execution retains
rejected scientific acceptance.
