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
recordings rather than resized proxies. Its integration button, editable-copy
flow, library and subsequent drawing/reference features are still in progress.

## Qualification and Remaining Work

Initial missing-module tests established RED; first backend suite passed 20 tests
and initial native editor suite passed 3. A downstream reconstruction regression
then exposed why processed-frame count cannot replace the source timeline; the
implementation now preserves that contract. The combined backend/timing/editor suite passes 23 tests and all three modules
pass mypy. At 850x650 the themed editor has readable controls, a 492-px minimum
width and an 828x367-px canvas. The fontless Qt offscreen test backend required
loading the installed Segoe UI font for the screenshot; application code was unchanged. No camera hardware or physical
calibration validation is claimed by synthetic footage.

Remaining: visible app entry points, library/notes/storage/archive/rename and
editable copies; editor failure/unsaved/drag visual checks; coaching shape model,
editing and export; reference import/registration/sync/comparison; generated map
updates; protected CI and merges. Do not close epics based on this document.
