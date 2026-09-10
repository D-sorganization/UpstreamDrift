# Unified Model and Video Analysis

Tracking: [Epic #9926](https://github.com/D-sorganization/UpstreamDrift/issues/9926).
Merged Foundation: [PR #9933](https://github.com/D-sorganization/UpstreamDrift/pull/9933),
`f04aa1a570e64c3db0b3d009351ab222656175b2`.
First bounded delivery: [Story #9929](https://github.com/D-sorganization/UpstreamDrift/issues/9929).
Model-only delivery: [Story #9930](https://github.com/D-sorganization/UpstreamDrift/issues/9930).
This is an implementation ledger; unchecked integration paths are not shipped
capabilities. Reference fitting, club display, handedness, and ellipsoid display
from #9914 remain the foundation.

Stories #9929 and #9930 are complete: required protected CI passed at
`ca71d2c1515c9f19b1ab8d851d594974933817dc` before the normal merge. The combined
Capture Rig and geometry regression run passed 497 tests after integrating the
guided capture wizard. Independent queued jobs had no result at merge time;
they are not represented as passing evidence. Historical qualification notes
below retain the checks and limitations observed during implementation.

Story #9932 is implemented locally in `feat/9932-comparison-drawings`.
**Draw on Comparison…** saves the current recipe and opens the common editor
with the current frame, detected pose, drawings, world geometry and reference.
Saving and closing reloads drawings in the parent comparison. The compositor
preserves layer order and selection handles stay outside exported pixels.
PNG export retains the original uncropped grid; video applies the saved crop.
Export workers snapshot drawings and reject changed source or scene evidence.

Local qualification: 504 Capture Rig/geometry tests and 50 atlas/parity tests
passed; Ruff, formatting, architecture budgets, LoD no-growth and mypy on seven
production files passed. Full Tour Average Driver motion was visually inspected
at 0.567 seconds in the common editor and its 960×540 PNG using an explicitly
synthetic background and virtual camera. This is rendering evidence, not a
comparison against measured player motion. Artifacts are in the external
`analysis-9926-artifacts/comparison-visual-9von8rml` directory. Normal push hooks passed. [PR #9943](https://github.com/D-sorganization/UpstreamDrift/pull/9943)
is draft; protected CI and review still gate story closure. Child #9942 now
tracks shared metric readouts and simulation analysis.

The capability registry now records model-only analysis, shared drawing-editor
contracts and metric-reference routes. Existing executable capture-goal metadata
and desktop parity limitations are preserved. Measurement readouts, simulation
integration and reviewed agent-context boundaries remain parent-epic work.

Camera-bound geometry now fingerprints `reconstruct/reconstruction.json`, the
actual projection-camera evidence, in addition to the reconstruction summary.
Existing geometry saved with the older incomplete camera fingerprint requires
explicit review/recreation when rejected; it is not silently rebound. Scenes
without that camera file retain their existing identity.

## Reuse and Surface Audit

| Surface              | Existing Infrastructure                                                                 | Required Integration                                                                      | Status                                                                     |
| -------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Capture Coaching     | `CoachingDialog`, `CoachingCanvas`, immutable `DrawingLayer`, bounded `History`         | Share the drawing editor with model comparison; keep original pixel coordinates           | Pending                                                                    |
| Reference Comparison | `ReferenceComparisonDialog`, placement, event pairing, appearance, `ComparisonRenderer` | Editable shared world planes and points; measurement readout and drawing tools            | Plane/point editor merged in #9933; drawing tools and measurements pending |
| Comparison Export    | Shared compositor, camera/clock snapshots, input hashes, export sidecar                 | Include exact geometry document and validate source evidence                              | Geometry recipe merged in #9933 with regression and visual qualification   |
| Model-Only Analysis  | `ReferenceMotion`, `ReferenceTimeline`, fit artifacts and model registry                | Use common controls and a clearly virtual camera without inventing capture media          | Shared editor, controls, persistence and verified exports merged in #9933  |
| Pose Studio          | Canonical poses, live kinematics, `View3D`, existing club bone                          | Consume shared analysis references and model/trace handoff                                | Pending                                                                    |
| Simulation Traces    | `simulation_backends.protocol.Trace` v2 optional metric markers, `trace_io`             | Explicit marker topology and unit/frame adapter into common analysis                      | Pending                                                                    |
| Native Viewports     | `Viewport` mesh protocol, `FspRenderer`, provider capability evaluation                 | Render the same reference geometry with explicit unsupported-capability reasons           | Pending                                                                    |
| Native Engines       | Pose interchange adapters and engine registry                                           | Qualify actual available model/trace routes; do not claim optional SDK support from mocks | Pending                                                                    |
| Functionality Maps   | Capability connections, generated agent-context maps                                    | Record real edges after integration; coordinate with #9907 and #9915 owners               | Pending                                                                    |

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

After merging current main (including its unchanged Tools provider selection),
25 affected playback, source, drawing and export regressions passed. CI exposed
seven deep model-adapter attribute chains and a nine-parameter encoder helper;
explicit recipe delegation and a validated encoding-options object resolve
these without baseline exemptions. Repo-wide LoD and architecture budgets pass
locally. The required single #9933 SPEC row and development-log state are included.

## Coordination

Owner session: `codex-unified-analysis-20260910`, isolated branch
`feat/9926-unified-analysis`. Repository Management lease and presence cover
the coaching contracts and comparison UI. Active work is on `feat/9932-comparison-drawings`. Equipment/session fitting, capture
goal-planner, Simscape matching and agent-context edits remain owned by their
active peer sessions. Do not edit their worktrees or rewrite their branches.

## Shared Metric Readouts (#9942)

The same Measurements panel now appears in model-only and video-comparison
analysis. Choose a motion landmark and a saved point or plane. Readouts use
registered world metres and the same source timing, coordinate conversion,
scale, translation and handedness as rendering. Missing samples remain
unavailable. Plane values are signed distances to the infinite plane, independent
of its display extent. This is geometric reference analysis, not anatomical
rotation or inverse dynamics.

TDD covers known transforms, point/plane distances, missing samples, wrong-scene
rejection and playback in both windows. The 513-test Capture Rig/geometry run
passed; ten focused regressions passed after the final cache/type corrections.
The full Driver's frame200 at0.556s was visually inspected with the shared panel
and plane. Final mypy is still running. Trace v2 import, native simulation
reference geometry, final capability/context-map updates and protected delivery
remain open for #9942 and the parent epic.
