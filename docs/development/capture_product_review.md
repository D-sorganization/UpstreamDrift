# Capture Product Review and Responsiveness Evidence

Epic #9849; audit #9851; recovery #9857. Reviewed 2026-09-09 against
`072db0891`, with the isolated changes described below. This is a source,
test, and synthetic-performance review, not a new camera-hardware acceptance run.

## Product Assessment

The product has a substantial capture-to-analysis foundation: named camera
plans, recording with preview, reusable lens calibration, single-view analysis,
multi-view reconstruction, estimator comparison, manual corrections, articulated
model fitting, kinetics, synchronized playback, and provenance-bearing export.
The remaining product risk is the operator's ability to understand and recover
the workflow, alongside predictable responsiveness and honest scientific limits.
It should remain qualified by feature and hardware configuration rather than
receive a blanket production-readiness label.

The review covers `src/tools/capture_rig/`, the corresponding tests,
`src/motion_capture/rig/`, `src/motion_capture/reconstruct/`, the motion pipeline,
the launcher/parity registries, and the existing camera evidence. The ongoing
GUI redesign is owned by #9843 and its children; this work does not replace it.

## Findings and Execution Ownership

| Priority | Finding and Evidence                                                                                                                                                                                                           | Execution and Acceptance                                                                                                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1       | The prior action grid and window impose an impractical width, and video loses space to controls. The owning agent measured a 3276 px window floor, 2102 px action grid, and approximately 68 px video panes.                   | #9844, #9846, #9847 own the action grid, central video, control docks and width regression. Independently verify the merged window at 900 px and common laptop sizes.                                                              |
| P1       | Operators need a clear next action, with blocked-stage reasons and distinction between board calibration and swing takes. `workflow.py` already models requirements and readiness, so another workflow definition would drift. | #9845 owns the step rail; #9848 owns aligned guide/parity/evidence. Keep advanced controls discoverable and collapsed until needed.                                                                                                |
| P1       | `playback._source_frame` shares readers by view, but raw/overlay requests for the same index made `VideoReader` seek backwards and decode again.                                                                               | #9851 implements a bounded last-frame cache, isolated returned pixels, and repeatable benchmark. Duplicate-frame median improved 15.6 times on the synthetic fixture below.                                                        |
| P1       | `RigProcessRunner` listened to `finished` but not `FailedToStart`; a missing executable produced neither completion nor an actionable failure. The GUI's completion handler could not recover command state.                   | #9857 handles failed starts exactly once, drains terminal output, and supports retry. A real offscreen QProcess test reproduced the failure before the fix and passes afterward; cancellation is also covered.                     |
| P2       | Feature discoverability and integration limits were spread across static documentation and separate registries.                                                                                                                | #9850 / PR #9856 generates system and capture workflow maps plus a searchable 57-tile/42-contract catalog. Source evidence distinguishes explicit file transfer from runtime connections.                                          |
| P2       | Live recording startup and stop are not instantaneous; existing evidence reports first snapshots after 4.8 seconds and a significant plan-binding cost.                                                                        | Preserve cached device binding and recorder-owned preview. #9848 must explain starting/recording/stopping states and show current-run evidence before release. Do not portray an early preview frame as proof of recording health. |
| P2       | Local media/readiness discovery primarily reports available artifacts; successful fitting and low residuals do not establish physical accuracy.                                                                                | Preserve explicit single-view 2-D limits, calibrated multi-view prerequisites, observability checks, units, and model assumptions in the guide and generated map. Reuse existing analysis-quality mechanisms.                      |

## Measured Responsiveness and Rust Decision

Command: `python3 -m scripts.benchmark_capture_responsiveness --samples 60 --output report.json`.
The retained [before](capture_performance_before.json) and
[after](capture_performance_after.json) reports include interpreter, platform,
OpenCV version, dimensions, random seed, warmup count, and the player's source
digest. Both ran on this Windows 11 host with Python 3.13.5 and OpenCV 4.13.0.
Each uses five warmup operations and sixty measured operations. The generated
MJPEG fixture is intentionally noisy; it is not representative footage or a
portable performance target. Fixture encoding is excluded from timed decoding.

| Path                                       | Before Median / P95 (ms) | After Median / P95 (ms) |
| ------------------------------------------ | ------------------------ | ----------------------- |
| One 1280×720 sequential decode             | 14.001 / 14.935          | 12.207 / 14.602         |
| Raw plus overlay requesting the same index | 196.788 / 226.653        | 12.613 / 14.701         |
| One-view composition to 960×540            | 4.395 / 5.183            | 3.984 / 4.820           |
| Three-view composition to 960×540          | 5.248 / 6.577            | 5.045 / 6.038           |
| Six-view composition to 960×540            | 8.425 / 11.093           | 8.218 / 9.817           |

Composition was unchanged; differences in those rows and sequential decoding
are run-to-run variability, not claimed improvements. The cache removes a
measured duplicate seek/decode. It retains one BGR frame per open reader
(2.64 MiB at 1280×720, 6.59 MiB at 1920×1200), and returns an independent array
on every read. Normal advancement evicts the old frame; close clears it.
It does not accelerate export paths that use different reader instances.

**Decision: retain Python for this path.** OpenCV decoding/resizing already
executes in native code; avoiding redundant native work delivered the measured
gain with a small change and no new runtime/packaging boundary. There is no
evidence here that a Rust rewrite would improve the remaining bottleneck.
Reconsider a narrow native component only after representative profiling shows
substantial Python CPU time that batching, caching, or existing native kernels
cannot remove. Require identical output, bounded cancellation, cross-platform
packaging, memory measurement, and a measured interaction-latency benefit.

This benchmark excludes Qt paint/event latency, camera transport, inference,
hardware synchronization, disk contention, and mixed-resolution multi-camera
playback. It must not be interpreted as a guaranteed GUI frame rate. For a
release run, record input-to-paint median/P95, missed repaint deadlines,
capture loss per camera, and memory over a long session on supported hardware.
Keep time-based thresholds out of shared-runner unit tests.

## Workflow and Interaction Contract

The first visible decision should be **record cameras or import existing video**.
The user should always see the selected plan/session, capture state, current
stage, and next available action. Reuse `workflow.Step` requirements and reasons.
Keep recording, analysis jobs, and playback state visually distinct.

1. Set up named cameras and validate the plan; show unavailable devices with a
   recovery path. Import should work without camera hardware.
2. For multiple views, record/import a separate board take, calibrate lenses,
   and retain calibration provenance for the actual camera mode.
3. Capture or import the swing, keeping video central. Countdown, recording,
   stopping, preview loss and successful output must be distinguishable.
4. Generate proxies when useful, inspect synchronized playback, then detect or
   manually correct observations. Expose confidence and occlusion honestly.
5. Choose 2-D analysis for one view or calibrated reconstruction for multiple
   views. Show prerequisites before expensive work starts.
6. Review reconstructed/model-fitted results and assumptions before kinetics.
   Export with units, source session, settings and provenance so downstream
   imports are understandable and reproducible.

The layout should retain stage/session context while advanced settings move
into docks or drawers. Named layouts, reset/show-all recovery, theme contrast,
focus order, keyboard operation, and visible cancellation must remain usable
after resizing or moving between monitors. Existing annotations already define
keyboard controls; document them where the action is performed. Reuse the
existing compositor and layout schema across preview, playback and export.

## Existing Evidence and Release Gates

The [three-camera hardware report](../motion_capture/evidence/capture_rig_multiview.md)
provides the prior physical-rig baseline.
Its source is `docs/motion_capture/evidence/capture_rig_multiview.md`, dated
2026-09-08. It reports a degraded full-resolution recorder tee and a supported
quarter-resolution tee (493/494/462 camera frames over an approximately 8-second
take). This is prior evidence for that rig; it is not a new no-drop guarantee.

Before calling the redesigned capture experience professional-grade, attach
evidence to #9849 for:

- The merged GUI at 900 px, a typical laptop size, and a large display; verify
  keyboard focus, readable labels, reset/reopen docks, and saved layouts.
- Cold setup, a second take using cached binding, import-only use, single-view
  and calibrated multi-view routes, and failed/missing prerequisites.
- Preview loss, unavailable executable, failed estimator, stop/cancel, retry,
  window close, camera release, and no stale frames after switching sessions.
- Raw/overlay synchronized playback, scrubbing, proxy selection, annotations,
  composite export and provenance round-trip on representative recordings.
- Supported hardware frame counts, synchronization quality, first-preview and
  stop latency, and long-session responsiveness/memory. Record observed limits.

The camera suite passed 241 tests after the cache change. Six focused cache and
process tests passed after the recovery fix, including the previously failing
missing-executable case. Qt/offscreen tests exercise controls and lifecycle,
not visual usability or physical-camera correctness. The GUI owner's remaining
evidence and integrated hardware run remain release requirements.
