# Capture Editing and Reference Integration

Product epic: #9849. Editing #9860, capture library #9861, coaching drawings #9862.
Advanced reference-overlay epic #9863 has import #9864, registration/synchronization
#9865 and comparison/export #9866. These are all part of the active user goal.
Gasification_Model mapping epic #4920 remains planned for future cheaper agents,
as requested; it is not an implementation dependency of this product.

## Existing Capabilities and Integration Decisions

| Source                                                           | Verified capability                                                             | Integration                                                                                                                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UD capture_rig/clips.py                                          | Inclusive frame/event clips, slowdown, pose overlay, aligned side-by-side takes | Keep as coaching export; use the same source-frame semantics in the editor.                                                                                     |
| UD capture_rig/player.py                                         | Frame-accurate OpenCV decode                                                    | Reuse directly for editing original recordings.                                                                                                                 |
| UD capture_rig/annotate_widget.py                                | Zoom/letterbox coordinate inversion, detector-joint correction                  | Reuse ImageCanvas; keep coaching shapes separate from scientific point corrections.                                                                             |
| UD capture_rig/layout_model.py and layout_editor.py              | Display crop, orientation, composited layouts                                   | Preserve presentation controls. Inference crop is a separate saved recipe in original pixels.                                                                   |
| UD capture_rig/overlay_render.py and reconstruct/overlay3d.py    | Projection/rendering of current-session fitted variants                         | Extend with an external reference adapter and saved registration, rather than duplicating projection.                                                           |
| UD launchers/library_widget.py                                   | PDF/LaTeX document index                                                        | Keep document semantics; use session bundles for the capture library.                                                                                           |
| Tools video_processor/apps/web/components/video/VideoEditor.tsx  | FFmpeg trim and rotation export                                                 | Useful workflow reference. Crop setter is unused; do not claim crop UI complete or embed a second browser/transcode stack in Qt.                                |
| Tools video_processor/apps/web/components/video/EditorCanvas.tsx | Fabric line/arrow/text/freehand, selection, colour/width, delete                | Adapt a versioned source-coordinate annotation contract. Circle/ellipse are absent and currentTime is unused; persistent frame visibility needs implementation. |
| Tools video_processor/apps/web/lib/video/annotationExporter.ts   | Fabric annotation JSON export/import                                            | Evaluate explicit interchange mapping; never claim Fabric serialization is already the desktop contract.                                                        |

Tools was inspected read-only. Reusable shared code changes must be implemented
upstream in Tools and consumed through its pinned integration, never edited in
vendor/ud-tools. Native Qt controls remain in Capture Rig.

## Implemented Editing Foundation

`swing_edits.json` stores validated per-view inclusive source-frame bounds and
integer-pixel crop rectangles. Missing means unedited; malformed/unknown versions
fail visibly. Saving uses a same-directory temporary file and atomic replacement.
Once observations or derived results exist, changing the recipe requires an
editable copy so previous analyses remain tied to their inputs.

Ingest estimates only selected frames and the cropped region, then translates
pose pixels back into the full original image. Original time_s, reference-clock
offsets and calibration coordinates survive unchanged. frames_total retains the
original source timeline because downstream reconstruction indexes by source frame;
provenance.frames_processed reports actual inference work. Excluded frames stay
unobserved. Raw media remains byte-identical.

The native SwingEditor reuses VideoReader and ImageCanvas, supports per-view
scrubbing, in/out marks, inclusive numeric bounds, drag or numeric cropping,
selection playback, reset/save and unsaved-change handling. It opens original
recordings rather than resized proxies. Library and Edit swing buttons in the
Capture Rig header open the catalog and editor. Analyzed takes offer an editable
copy before editing, preserving earlier results. Buttons follow command and
recording/countdown state.

## Library Workflow and Recovery

CaptureLibrary indexes registered bundle locations in SQLite; capture_notes.json
owns stable identity, title, multiline notes, archive state and source-copy lineage.
Catalog rebuilding preserves sidecars. Archive/restore changes metadata and reclaims
no disk space. Storage reports logical bytes inside the session separately from
external recordings, without following directory symlinks. The visible library
scans in a cancellable worker and shows entry-specific failures. Operators can
search titles/notes, change catalog location, import videos, add existing sessions,
open/edit captures, browse folders and save notes with unsaved-change protection.

Editable copies retain original file references, plan, manifest/timing, calibration
and edit recipe, plus a new identity and source_capture pointer. Derived analyses
are not copied. Filename changes are limited to owned, unprocessed recordings with
unchanged extensions and collision/reserved-name checks. Index-write failures roll
back. A validated rename journal recovers interrupted file/index changes on the
next open; ambiguous recovery states fail visibly. SQLite connections close after
each transaction, including failures, preventing Windows catalog file locks.

## Qualification and Remaining Work

Initial missing-module tests established RED. A downstream reconstruction regression
exposed why processed-frame count cannot replace source timeline extent; the fix
preserves original indices. The integrated camera package plus editing/ingestion/
timing suite passed 300 tests before three additional focused regressions were
added. Library/dialog/integration qualification now passes 12 tests; editor gesture,
unsaved-change and persistence qualification passes 5. Eight implementation modules
pass mypy and scoped Ruff passes. These are synthetic-media checks, not physical
camera or calibration qualification.

Visual review used 850x650 for the editor (492-px minimum width) and 900x650 for
the library (350-px minimum width). The fontless Qt offscreen backend required
loading installed Segoe UI in the screenshot harness; application fonts were not
changed. A real Windows catalog-handle cleanup failure was reproduced and fixed.

Export swing now reuses clips.export_clip with a cancellable native worker, exact
inclusive selection and source crop. A separate video and source-hash/recipe sidecar
are published without replacing existing paths. Odd dimensions are padded at the
right/bottom instead of losing selected pixels. Cancellation or decode shortfall
publishes neither file. The two-file publication handles ordinary errors but is not
a power-loss transaction. Main multiview playback keeps its original scene; the
editor previews only the selection.

Two downstream regressions were reproduced: initialization failed when selected
frames started after frame ten, and all-zero excluded frames introduced a false
speed peak (21.134 versus 2.059 m/s in the synthetic example). Camera initialization
now reuses observed-frame compaction. Summary computation excludes wholly missing
prefix/suffix frames, restores source event indices and marks excluded angle values
unobserved. A wholly missing interior frame produces a clear continuous-interval
requirement instead of invented motion evidence. Existing analytical formulas remain
unchanged. The focused reconstruction/analytics suite passes 11 tests.

The integrated run passed all behavioral tests and exposed one stale generated user
guide; regeneration repaired that, with 10 guide/synchronization checks passing.
The registry/atlas suite passes 49 tests. Export/editor/coaching clip tests pass 12,
and five affected export/reconstruction modules pass mypy. The final 850x650 export
editor remains at a 492-px minimum width. Generated feature, workflow and atlas
references now include selection/library/export integration.

Remaining: protected CI and merge for editing/library and coaching drawings; advanced reference
import/registration/sync/comparison under #9863. Do not close epics based on this
document or claim these remaining features already work.

## Coaching Drawing Implementation

The native coaching layer (#9862) implements lines, arrows, circles, ellipses and
rectangles as strict versioned source-pixel geometry, stable IDs, style and
inclusive frame visibility. Its immutable history supports move/resize, deletion,
clear, undo/redo and keyboard/numeric edits. The existing ImageCanvas provides
letterbox/zoom inversion; a single OpenCV renderer runs before crop in preview,
PNG export and cancellable swing-video export. Portable sidecars remain independent
of measured landmark corrections. Atomic JSON replacement preserves the prior
document on write failure; output publication shares the existing no-overwrite
media/sidecar publisher. Escape follows unsaved-change/export guards.

Tools reuse is explicit: the Fabric editor informed tool/selection/style workflow;
its JSON is not silently relabeled as the desktop schema. Browser components and
FFmpeg-WASM are not embedded in the native dialog. Text/freehand and general-purpose
timeline compositing are excluded from this bounded reference-drawing feature.
These exclusions avoid two renderers disagreeing about timestamps or source pixels.
Existing decoder, layout, clip exporter and atomic sidecar infrastructure are
reused directly; external-reference imports/projection remain under #9863.

Initial missing-module tests established RED; 18 focused drawing/editor/export
checks passed, and ten source modules passed mypy with the repository's
follow-imports=silent configuration. Integration caught a laptop-width regression
from adding a button to a fixed row; that row now uses the existing wrapping
FlowLayout. Physical camera/coach usability qualification is still required.

Sparse manual observations remain valid reconstruction/model-fit inputs. When their
wholly missing interior frames prevent a trustworthy swing summary, the chain
records swing_summary_unavailable_reason, omits the summary path, removes any
stale summary and continues to produce the reconstruction. It does not invent
speed/event metrics across gaps. The existing every-fifth-frame model-fit accuracy
regression and nine pipeline/analytics checks pass.
