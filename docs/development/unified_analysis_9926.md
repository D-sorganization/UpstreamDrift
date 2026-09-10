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

Story #9932 shipped in #9943.
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
merged normally at `3fbd4b2da5f8661ec1f1e91b97884f781d83f1fd`; #9932 is closed. Child #9942 now
tracks shared metric readouts and simulation analysis.

The capability registry now records model-only analysis, shared drawing-editor
contracts and metric-reference routes. Existing executable capture-goal metadata
and desktop parity limitations are preserved. Measurement readouts, simulation
integration and reviewed shared contracts shipped in #9945.

Camera-bound geometry now fingerprints `reconstruct/reconstruction.json`, the
actual projection-camera evidence, in addition to the reconstruction summary.
Existing geometry saved with the older incomplete camera fingerprint requires
explicit review/recreation when rejected; it is not silently rebound. Scenes
without that camera file retain their existing identity.

## Reuse and Surface Audit

| Surface              | Existing Infrastructure                                 | Integrated Route                                                        | Qualification                                                                                              |
| -------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Capture Coaching     | CoachingDialog/Canvas, DrawingLayer, History            | Common drawing editor and original-pixel exports                        | Merged #9933/#9943; capture regression coverage                                                            |
| Reference Comparison | Registration, event pairing, ComparisonRenderer         | Common drawings, geometry, appearance and Measurements tab              | Drawing/geometry merged; metric tests pass, #9945 merged; required quality gate passed                     |
| Comparison Export    | Shared compositor and immutable source/camera snapshots | Saved drawings and geometry with clock/crop parity                      | Merged #9943; PNG/video, cancellation and changed-evidence tests                                           |
| Model-Only Analysis  | ReferenceMotion/Timeline and fit artifacts              | Same editor and controls with explicit virtual camera                   | Merged #9933; added measurements/Trace regression tests pass                                               |
| Pose Studio          | Canonical poses/FK, View3D and existing club bone       | Shared reference editor, mesh renderer and save/load                    | Native30 tests and visual plane/editor review; #9945 merged; required quality gate passed                  |
| Simulation Traces    | Trace v2 metric markers and trace_io                    | Explicit axes/names/topology into reference library and common analysis | Trace126 and topology17 tests pass; q-only traces report missing marker channels                           |
| Native Viewports     | Existing Viewport mesh protocol                         | Scene-bound adapter with explicit coordinate frame                      | Pose Studio qualified; other renderer SDKs are not claimed from the protocol alone                         |
| Native Models        | Existing fitted ReferenceMotion assets and catalog      | Analyze Model or Compare Reference regardless of fit adapter            | See model-by-model fitting qualification in reference_model_fitting.md; unavailable model reasons retained |
| Functionality Maps   | Capability registry and generated atlas                 | Measurements, Trace and native-reference contract edges                 | Atlas/parity50 tests pass; shared contracts reviewed; optional catalog indexing remains with #9915         |

The native pose editor is static and uses scene time zero. Dynamic simulation
analysis uses the Trace marker route and the shared playback/export workspace.
Direct control injection into every optional engine's separate native viewer is
not implemented: those viewers differ in coordinate frames, camera contracts and
SDK availability. A `Viewport`-compatible renderer can use the shared adapter only
after its frame and mesh behavior are qualified; unsupported objects fail the
protocol contract instead of silently dropping references.

The reviewed [shared analysis contracts](../architecture/SHARED_ANALYSIS_CONTRACTS.md)
record public boundaries, failure behavior, ownership and executable evidence.

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
are regression targets. Final combined qualification, reviewed context boundaries
and protected delivery remain completion gates.

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
and plane. Readout mypy passes four source files. The initial Trace v2 importer passes
126 reference/import regressions and mypy on five source files. Explicit club
and skeleton topology preservation and native Pose Studio reference geometry are
implemented and tested. Capability maps are updated; #9945 passed protected CI and merged.

## Native Reference Geometry

Pose Studio's **3D References…** button opens the same editor as model/video
analysis. Points and translucent planes render beside the existing canonical
FK skeleton through the shared mesh viewport protocol. Closing and reopening
retains the scene; Save/Load References persists its versioned JSON. Coordinates
remain ADR-0041 Y-up metres in the editor and convert explicitly to the native
Z-up viewport. The static editor evaluates visibility at time zero. Documents
from other scenes are rejected rather than silently rebound.

The native point is a fixed-size location glyph, not a measured anatomical
ellipsoid. The existing animated analysis appearance controls continue to own
club display, model handedness and body ellipsoids. This native qualification
covers Pose Studio, not every optional engine's own graphical application.

Native visual evidence: `analysis-9926-artifacts/native-references-saep77qb`.
The plane and shared controls were inspected on the canonical reference pose.
Partial submission failures roll back new mesh handles and preserve prior
references; replacing references leaves other viewport objects alone.

## Final Runtime Qualification

At b6325dfe8, all20 hash-verified Tour Average display assets passed the actual
shared analysis/save/reopen/export route: nine catalog variants each for Driver
and Iron, plus full-rate golfer fits. Every route retained drawings, planes,
club controls, ellipsoid opacity and handedness and produced metric readouts.
Forty original/mirrored stills and the per-asset qualification report are in the
external `analysis-9926-artifacts/catalog-analysis-zcgqk8ta` directory. The
reproduction script is `analysis-9926-artifacts/qualify_catalog_analysis.py`.
Additional observed-club frames at Driver0.838889s and Iron0.891365s were visually
inspected. Source club gaps and the separate measured shaft connector are retained.

The current context boundary requirement is satisfied by reviewed
SHARED_ANALYSIS_CONTRACTS.md and the authoritative capability registry. The
independently owned #9915 catalog can index these same contracts when its runtime
lands; it is not required to run the implemented analysis routes. No claim of
catalog registration or optional native SDK qualification is made here.

## Protected Delivery

PR #9945 merged at `2d41aba4159162f91c0cd1cc6919d0341755b492` on
2026-09-10T05:20:06Z. [CI Standard run34440058289](https://github.com/D-sorganization/UpstreamDrift/actions/runs/34440058289)
succeeded, including the required quality gate, unit gate and Python3.11/3.12
matrix. Independent optional queued jobs are not represented as passing.
After integrating main300d96a1 and the biomechanics API helper typing fix
f58ee5b3, all88 combined biomechanics/API/display and analysis integration tests
passed. All four implementation children are closed. This final documentation
record reconciles the shipped ledger and preserves tracking records lost in the
concurrent main merge. Its protected merge closes the parent epic.
