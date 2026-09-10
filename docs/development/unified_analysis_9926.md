# Unified Model and Video Analysis

Tracking: [Epic #9926](https://github.com/D-sorganization/UpstreamDrift/issues/9926).
First bounded delivery: [Story #9929](https://github.com/D-sorganization/UpstreamDrift/issues/9929).
Model-only delivery: [Story #9930](https://github.com/D-sorganization/UpstreamDrift/issues/9930).
This is an implementation ledger; unchecked integration paths are not shipped
capabilities. Reference fitting, club display, handedness, and ellipsoid display
from #9914 remain the foundation.

## Reuse and Surface Audit

| Surface              | Existing Infrastructure                                                                 | Required Integration                                                                      | Status                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Capture Coaching     | `CoachingDialog`, `CoachingCanvas`, immutable `DrawingLayer`, bounded `History`         | Share the drawing editor with model comparison; keep original pixel coordinates           | Pending                                                                                                      |
| Reference Comparison | `ReferenceComparisonDialog`, placement, event pairing, appearance, `ComparisonRenderer` | Editable shared world planes and points; measurement readout and drawing tools            | Plane/point editor and compositor implemented locally; qualification ongoing                                 |
| Comparison Export    | Shared compositor, camera/clock snapshots, input hashes, export sidecar                 | Include exact geometry document and validate source evidence                              | Implemented locally; full regression and visual qualification pending                                        |
| Model-Only Analysis  | `ReferenceMotion`, `ReferenceTimeline`, fit artifacts and model registry                | Use common controls and a clearly virtual camera without inventing capture media          | Shared coaching editor, controls, persistence and verified exports implemented locally; protected CI pending |
| Pose Studio          | Canonical poses, live kinematics, `View3D`, existing club bone                          | Consume shared analysis references and model/trace handoff                                | Pending                                                                                                      |
| Simulation Traces    | `simulation_backends.protocol.Trace` v2 optional metric markers, `trace_io`             | Explicit marker topology and unit/frame adapter into common analysis                      | Pending                                                                                                      |
| Native Viewports     | `Viewport` mesh protocol, `FspRenderer`, provider capability evaluation                 | Render the same reference geometry with explicit unsupported-capability reasons           | Pending                                                                                                      |
| Native Engines       | Pose interchange adapters and engine registry                                           | Qualify actual available model/trace routes; do not claim optional SDK support from mocks | Pending                                                                                                      |
| Functionality Maps   | Capability connections, generated agent-context maps                                    | Record real edges after integration; coordinate with #9907 and #9915 owners               | Pending                                                                                                      |

## Contracts and Coordinate Ownership

- Screen drawings stay in original source-image pixels and retain their existing
  frame intervals, storage and gestures.
- `ReferenceGeometry` uses world metres, the ADR-0041 Y-up right-handed convention,
  and an explicit scene identity. Plane
  anchors define the normal by `(along-origin) × (across-origin)`. Distance is
  signed distance to the infinite plane; displayed extent does not change it.
- Plane geometry reuses `generate_plane_vertices`; editor undo/redo reuses the
  existing drawing `History`, now generic over immutable Pydantic snapshots.
- `coaching/world.json` is shared by capture views. Its scene identity includes
  recordings, manifest, reconstruction, lens and timing evidence hashes using
  relative paths. A byte-identical bundle can move; changed evidence requires
  review. Reconstruction hashing is intentionally conservative.
- Camera overlays use the existing distortion-aware reference projection. A
  missing camera cannot produce metric overlay pixels. Polygons crossing the
  near plane are omitted. Planes are illustrative overlays without model
  occlusion; they do not claim anatomical surfaces or calibration accuracy.
- Geometry is independent of model-layer visibility. Preview and export consume
  the same immutable geometry in `ComparisonRenderContext`; export embeds it in
  the sidecar and monitors the world-reference file for concurrent changes.

## Test-First Evidence

Failing tests were observed before implementing the geometry module, renderer,
editor, capture storage and comparison integration. Tests cover signed distances,
degenerate anchors, NaN gaps, forbidden units, scene mismatch, portable sidecars,
alpha, clock bounds, behind-camera geometry, original-image immutability,
numeric edits, undo/redo, reopening and export recipe reuse.

The existing drawing, appearance, reference-comparison and video-export tests
are regression targets. Full protected CI, broader visual checks,
simulation integration and capability-map updates remain completion gates.

Local qualification to date: 46 combined geometry, coaching, comparison, display
and export tests passed before the additional pending-edit/precision regression
was added. Whole-repository Ruff and format checks passed across 6,914 files;
file budgets passed; seven changed production modules passed mypy. Synthetic
projection and the 410-pixel editor layout were visually inspected. Original
numeric precision is retained when editing another field or axis. Final tests
and protected CI still gate story closure.

## Model Analysis Workflow

In the reference library, select a motion asset and choose **Analyze Model…**.
The existing coaching editor opens with its drawing tools, selection handles,
undo/redo, timeline, and still/video export. Appearance controls share skeleton,
club, ellipsoid opacity and radius settings with comparison playback. Placement
controls share scale, rotation, translation and lateral mirroring. The 3D
References tab edits the same scene-bound plane/point document as comparison.

The library stores `analysis/<asset-id>/model-analysis.json`, containing the
source motion, virtual camera, display recipe, metric references and drawings.
Reopening rejects changed source identity. Model analysis creates no recording
manifest or synthetic capture bundle. The camera frames the entire trajectory;
it is explicitly virtual and does not imply camera calibration. Ellipsoids are
illustrative segment volumes, not anatomical meshes.

`FrameSource` and `CoachingSource` keep the editor independent of media storage.
Original capture exports retain their established path. Model exports reuse the
frame encoder and atomic paired publication, decode every encoded frame before
publication, and embed their exact recipe and source times. PNG export uses the
same drawing renderer as playback. Export jobs snapshot edits before workers run.

Playback retains uniform source cadence between 1 and 360 Hz, otherwise sampling
a uniform display clock at the bounded median cadence. Existing interpolation
gap limits still apply; missing points are not invented. Display timelines are
limited to 100,000 frames. The full Tour Average Driver fit retained all 654
samples at 360 Hz in the inspected model window and exported annotated still.

Test-first evidence includes source-clock bounds, rendering immutability,
handedness/appearance changes, save/reopen, drawing identity, cancellation,
corrupt encoded-output rejection and library launch without capture media.
The full Capture Rig suite plus geometry/storage regressions passed locally,
followed by eight passing model-frame tests after adding display-budget and
minimum-rate contracts. Whole-repository Ruff and format checks passed across
6,923 files. The actual Driver window, ellipsoids, drawing and PNG output were
visually inspected. These checks do not close the parent epic's remaining
comparison drawing, measurement, simulation and functionality-map work.

## Coordination

Owner session: `codex-unified-analysis-20260910`, isolated branch
`feat/9926-unified-analysis`. Repository Management lease and presence cover
the coaching contracts and comparison UI. Equipment/session fitting, capture
goal-planner, Simscape matching and agent-context edits remain owned by their
active peer sessions. Do not edit their worktrees or rewrite their branches.
