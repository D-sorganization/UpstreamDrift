# Shared Model and Video Analysis Contracts

These reviewed boundaries describe the implemented routes in epic #9926 and
story #9942. The existing capability registry remains the product map authority.
The separately owned #9915 agent-context catalog may index these same contracts;
it must not introduce a second geometry, sampling or readiness implementation.

## Registered Motion to Measurements

Provider: `src/motion_capture/reference/registration.py`, through
`sample_reference_motion`. Consumer:
`src/motion_capture/coaching/measurements.py`, through `reference_distances`.
Both model-only analysis and video comparison use the single
`src/tools/capture_rig/reference_readout.py` widget.

Input is an immutable ReferenceMotion, ReferenceRegistration, ReferenceGeometry,
finite one-dimensional scene times and explicit landmark/reference/scene IDs.
Registration owns source timing, lateral reflection, proper frame conversion,
uniform scale, rotation and translation. Readouts use the same registered samples
as display. Invalid samples and times outside the motion remain unavailable;
masked zero storage must never become a measurement.

Geometry is ADR-0041 right-handed Y-up world metres. Point distance is Euclidean;
plane distance is signed against the infinite plane's oriented normal. Display
extent, opacity and visibility do not change the underlying measurement. Unknown
IDs and a different scene fail explicitly. These are geometric comparisons,
not anatomical rotations, joint centers or inverse dynamics.

Evidence: `tests/motion_capture/test_reference_measurements.py` and
`tests/tools/capture_rig/test_reference_readout.py` exercise transforms,
handedness, gaps, scene rejection and both playback consumers.

## Simulation Trace to Shared Analysis

Provider: `src/shared/python/simulation_backends/trace_io.py` and its Trace v2
schema. Adapter: `src/motion_capture/reference/trace_import.py`. Consumers:
`src/tools/capture_rig/reference_import.py` and the existing reference library.

One rollout must provide metric marker trajectories with increasing finite times.
The adapter checks the source hash, decoded size, dataset links and array bounds.
It delegates decoding to the shared Trace reader. State-only outputs report the
missing marker-kinematics capability; the importer never guesses a model's FK.

Scalar `frame="world_Zup"` declares canonical right-handed Z-up; otherwise the
mapping dialog requires source-axis confirmation. `marker_names_json` declares
column names; absent names expose index labels for explicit mapping, without
invented anatomical semantics. `edges_json` and `club_edges_json` declare valid
marker-index pairs; club pairs must belong to the complete edge set. Removing a
mapped edge also removes its club role. Backend/model identity is retained.

After mapping, the ordinary ReferenceMotion route owns rendering, club controls,
ellipsoid opacity, handedness, drawings and measurements. No separate simulation
player or mutable capture bundle is fabricated. A virtual model camera is not
camera calibration. The source clock and missing samples remain explicit.

Evidence: `tests/motion_capture/test_trace_reference_import.py` covers bounded
inputs, topology and gaps. `tests/tools/capture_rig/test_trace_analysis_import.py`
checks mapping into the actual editor and save/reopen of club visibility,
ellipsoid opacity, handedness, drawings and reference planes.

## Shared Geometry to Native Viewports

Provider: `src/motion_capture/coaching/geometry.py`. Native consumer:
`src/motion_capture/coaching/native_geometry.py`, using the existing Viewport
mesh protocol in `src/shared/python/visualization/fsp_renderer.py`.
`src/tools/pose_studio/reference_geometry.py` reuses Capture Rig's
GeometryControls; Pose Studio's View3D implements mesh submission/removal.

The adapter requires an explicit scene and viewport frame. Plane vertices reuse
the shared plane generator. Y-up to native Z-up conversion is the inverse of
reference registration's proper frame transform. Points are fixed-size location
glyphs, not anatomical ellipsoids. Visibility uses inclusive scene-clock bounds;
Pose Studio's static pose editor uses time zero.

The adapter replaces only its owned mesh handles. Failed partial submissions
remove new handles and retain previous references. Other pose artists remain
untouched. Save/load preserves the versioned scene identity; another scene's
references are rejected rather than silently relocated. Unsupported viewport
objects fail the protocol contract. Protocol membership alone does not qualify
an optional engine SDK or its separate native graphical application.

Evidence: `tests/motion_capture/test_native_reference_geometry.py`,
`tests/tools/pose_studio/test_reference_mesh_viewport.py` and
`tests/tools/pose_studio/test_reference_geometry_dialog.py` cover frame conversion,
visibility, opacity, rollback, ownership and the real editor launch/save/reopen.
Native plane/editor graphics were inspected separately from physical calibration.

## Preview, Persistence and Export

Model-only and comparison analysis share CoachingCanvas, drawing history,
selection/editing and export interfaces. Immutable recipes retain asset identity,
registration, appearance and world geometry. Comparison additionally binds the
actual reconstruction camera evidence. Changed inputs require review.

Model and comparison exports use their existing shared compositor. Selection
handles remain outside exported pixels. Comparison PNGs retain original pixels;
comparison video uses the saved crop and source clock. Cancellation or changed
source evidence must not publish a partial result. Missing observed club samples
remain gaps; the observed shaft connector can be separate from the fitted hands.

The current runtime audit exercised all20 hash-verified Tour Average display
assets: nine catalog model variants for each of Driver and Iron, plus full-rate
golfer fits. Each passed shared-editor measurements, mirrored placement, still
export and save/reopen of drawings, geometry, club and ellipsoid controls.
This qualifies those analysis routes, not new dynamics or anatomical accuracy.
See `docs/development/unified_analysis_9926.md` for the surface matrix and
`docs/motion_capture/reference_model_fitting.md` for model-specific limitations.

## Review and Change Rules

Before changing a boundary, read both provider and consumer and update the cited
behavior tests. Keep the current capability connections and generated atlas in
sync. Coordinate changes to anatomical metrics (#9934), native Simscape fitting
(#9927), calibration and the separately owned context catalog rather than copying
those authorities. A generated graph or a green structural check alone is not
runtime or scientific qualification.
