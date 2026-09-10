# Unified Model and Video Analysis

Tracking: [Epic #9926](https://github.com/D-sorganization/UpstreamDrift/issues/9926).
First bounded delivery: [Story #9929](https://github.com/D-sorganization/UpstreamDrift/issues/9929).
This is an implementation ledger; unchecked integration paths are not shipped
capabilities. Reference fitting, club display, handedness, and ellipsoid display
from #9914 remain the foundation.

## Reuse and Surface Audit

| Surface              | Existing Infrastructure                                                                 | Required Integration                                                                      | Status                                                                       |
| -------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Capture Coaching     | `CoachingDialog`, `CoachingCanvas`, immutable `DrawingLayer`, bounded `History`         | Share the drawing editor with model comparison; keep original pixel coordinates           | Pending                                                                      |
| Reference Comparison | `ReferenceComparisonDialog`, placement, event pairing, appearance, `ComparisonRenderer` | Editable shared world planes and points; measurement readout and drawing tools            | Plane/point editor and compositor implemented locally; qualification ongoing |
| Comparison Export    | Shared compositor, camera/clock snapshots, input hashes, export sidecar                 | Include exact geometry document and validate source evidence                              | Implemented locally; full regression and visual qualification pending        |
| Model-Only Analysis  | `ReferenceMotion`, `ReferenceTimeline`, fit artifacts and model registry                | Use common controls and a clearly virtual camera without inventing capture media          | Pending                                                                      |
| Pose Studio          | Canonical poses, live kinematics, `View3D`, existing club bone                          | Consume shared analysis references and model/trace handoff                                | Pending                                                                      |
| Simulation Traces    | `simulation_backends.protocol.Trace` v2 optional metric markers, `trace_io`             | Explicit marker topology and unit/frame adapter into common analysis                      | Pending                                                                      |
| Native Viewports     | `Viewport` mesh protocol, `FspRenderer`, provider capability evaluation                 | Render the same reference geometry with explicit unsupported-capability reasons           | Pending                                                                      |
| Native Engines       | Pose interchange adapters and engine registry                                           | Qualify actual available model/trace routes; do not claim optional SDK support from mocks | Pending                                                                      |
| Functionality Maps   | Capability connections, generated agent-context maps                                    | Record real edges after integration; coordinate with #9907 and #9915 owners               | Pending                                                                      |

## Contracts and Coordinate Ownership

- Screen drawings stay in original source-image pixels and retain their existing
  frame intervals, storage and gestures.
- `ReferenceGeometry` uses world metres and an explicit scene identity. Plane
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
are regression targets. Full protected CI, visual checks, model-only analysis,
simulation integration and capability-map updates remain completion gates.

Local qualification to date: 46 combined geometry, coaching, comparison, display
and export tests passed before the additional pending-edit/precision regression
was added. Whole-repository Ruff and format checks passed across 6,914 files;
file budgets passed; seven changed production modules passed mypy. Synthetic
projection and the 410-pixel editor layout were visually inspected. Original
numeric precision is retained when editing another field or axis. Final tests
and protected CI still gate story closure.

## Coordination

Owner session: `codex-unified-analysis-20260910`, isolated branch
`feat/9926-unified-analysis`. Repository Management lease and presence cover
the coaching contracts and comparison UI. Equipment/session fitting, capture
goal-planner, Simscape matching and agent-context edits remain owned by their
active peer sessions. Do not edit their worktrees or rewrite their branches.
