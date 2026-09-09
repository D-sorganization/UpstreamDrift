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

Remaining: selection export/playback integration, generated map updates, protected
CI and merge; coaching shape model/editing/export under #9862 and advanced reference
import/registration/sync/comparison under #9863. Do not close epics based on this
document or claim these remaining features already work.
