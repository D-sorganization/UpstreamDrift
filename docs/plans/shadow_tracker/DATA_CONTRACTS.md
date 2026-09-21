# Data Contracts

## Status and Principles

The source/frame, binary-mask and camera-conversion subsets are now frozen for
implementation in [Contract Freeze](CONTRACT_FREEZE.md). That narrower document
takes precedence for its explicit fields. The remaining records below remain
proposals; model qualification and scientific claims stay blocked.

This is the implementation contract proposal for ST-02, not an executable schema.
Freeze schema version `shadow-tracker/1.0.0` only when its tests and compatibility
review pass. Reuse repository validators and immutable camera/state records.
Persisted files use explicit units and schema tags. No untyped catch-all metadata
for quantities used by the solver. Unknown values are null plus a reason, never
zero, identity calibration, fabricated confidence, or silently imputed evidence.

## Records

| Proposed Record       | Required Content                                                                                                                          | Preconditions and Postconditions                                                                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SourceAsset`         | Asset ID, source URI, checksum, retrieval date, rights status, decoder/version, original dimensions, source timebase                      | Nonempty IDs; immutable original; checksum verified before use                                                                                             |
| `Shot`                | Asset ID, frame/PTS interval, subject ID, swing ID, camera ID, cuts, mirror/rotation/crop transforms                                      | Interval in source; one continuous shot; different swings cannot share a synchronized event ID                                                             |
| `FrameObservation`    | Shot/camera IDs, original PTS and frame ID, physical-time mapping, body/club mask refs, valid-pixel mask, confidence provenance           | Strictly increasing usable presentation times per shot; no duplicate IDs; decode order is not presentation order                                           |
| `MaskAsset`           | Version, dimensions, encoding, body/club class, visibility, unknown pixels, provider/checkpoint hash, edits, source frame hash            | Same image plane as camera projection; bool or bounded probability; no NaN; immutable copies; wholly missing is not an empty observed person               |
| `CameraTrack`         | Reused intrinsics/extrinsics, frame convention, distortion, image transforms, track times, measured/estimated/unknown status, uncertainty | Positive focal lengths; finite proper rotations; invertible transforms; no projection behind camera; every frame resolves to a supported camera hypothesis |
| `SubjectModelBinding` | Subject and model hashes, joint/body IDs, geometry binding, visual envelope, SI inertial priors, scale evidence, handedness               | Positive sizes/masses, valid inertia; fixed morphology during a swing; all referenced bodies exist                                                         |
| `FitRequest`          | Evidence/model versions, candidates, objective profile, bounds, seeds, budget, intended time window, engine capability requirement        | All hashes resolve; cameras/times compatible; enough valid evidence for requested tier                                                                     |
| `CandidateResult`     | Canonical trajectory, native mapping, initial state, controls/basis, source evidence refs, diagnostics, assumptions, uncertainty method   | Finite ordered state; declared units and provenance; no acceptance flag without fresh replay audit                                                         |
| `ReplayAudit`         | Initial/reset count, controls applied, integrator/version, full time coverage, contact/grip/actuation residuals, independent mask scores  | One initial reset and no later state injection; full requested interval; reference-free continuous execution                                               |
| `ResultBundle`        | Request, candidates, statuses, metrics, excluded intervals, hashes, environment, render receipts, report                                  | Atomic write; reload preserves values/status; corruption fails clearly; resumable only with compatible inputs                                              |

## Coordinates and Time

Use SI units for world geometry/dynamics. Reuse canonical-v2: quaternion order
`wxyz`, base linear velocity in world frame and base angular velocity in body
frame, with existing manifold operations. See
[Canonical V2](../../conventions/canonical-v2.md). Pose-only v1 degrees must be
converted at the explicit adapter boundary.

The proposed image convention is top-left origin, x right, y down, integer pixel
centers, and mask arrays indexed `[row, column]`. Store original and processed
image dimensions, pixel aspect ratio, and the complete resize/crop/rotation/
mirror transform. Apply intrinsics to the matching image plane exactly once.
Do not treat lens distortion as a homography. Projection parity tests cover both
undistorted images and direct distortion-aware projection.

Preserve original PTS and rational timebase. Store a separate mapping from
presentation time to physical swing time, its evidence and uncertainty. A
constant-rate mapping may be affine; variable-speed replay needs piecewise
mapping with evidence and no artificial continuity across edits. Camera offsets
and drift have uncertainty. Repeated telecine frames remain auditable even if
deduplicated for fitting. Phase normalization is for comparison only, not dynamics.

## Service Protocols to Freeze in ST-02

These are proposed narrow methods; implement only after tests establish their
contract. Use typed request/result records rather than exposing engine objects.

- `Segmenter.segment(request) -> MaskSequence`: no engine dependency; manual
  provider and optional model providers share correction/provenance semantics.
- `SilhouetteRenderer.render(request) -> RenderedMasks`: geometry/state and
  camera in, body/club/visibility masks out; no hidden camera updates.
- `ForwardModel.capabilities() -> ModelCapabilities`: explicit supported bodies,
  state convention, actuator/contact modes and environment availability.
- `ForwardModel.rollout(request) -> Rollout`: initial state, controls and times
  in; trajectory, realized controls and audit out. No access to observed masks.
- `ShadowTrackerService.fit(request) -> ResultBundle`: orchestrates the above;
  cancellation/progress/resume use existing job services, not global mutable state.

## Status and Failure Semantics

Separate execution (`completed`, `cancelled`, `budget_exhausted`, `failed`) from
evidence quality (`unreviewed`, `kinematic_only`, `dynamic_candidate`,
`validated_profile`, `insufficient_evidence`). A converged optimizer may still
produce `insufficient_evidence`. Profiles are versioned and named in the result.

Invalid user data raises existing contract exceptions with a specific field and
reason; missing optional capabilities produce actionable capability errors.
Divergence or contact failure returns failed-candidate diagnostics and cannot
become a successful zero-cost result. Preserve known input defects separately
from internal software faults. Logs include run IDs, not private footage contents.

## Export and Compatibility

Persist raw observations separately from inferred trajectories. Export to
canonical state/pose and existing model formats via adapters. If a consumer
requires synthetic marker trajectories, label them as model-generated, include
candidate/hash/uncertainty, and never overwrite measured `BodyTarget` data.
Keep per-camera masks and uncertainty available even after export.

Unknown major schema versions must fail. Minor additive migrations require
round-trip and legacy-fixture tests; do not silently drop fields. Asset paths
are bundle-relative and resolve within the chosen root; reject traversal and
checksum mismatches. Large footage, weights and results live outside Git;
commit only tiny redistributable fixtures and manifests.

## Manual Revision Store Contract (#10233)

Revision IDs are globally unique within a ManualMaskProvider. A repeated ID
is a no-op only for complete immutable MaskFrame equality; it never changes
history or selection. A correction requires a registered parent with identical
complete FrameIdentity and pixel dimensions. Missing parents, cross-observation
parents, cycles and conflicting IDs fail before mutation. Observation scopes
include asset, shot, swing, camera and frame IDs.

`select_revision` selects an existing revision without appending history.
Consumers must key cached results with `get_cache_key`, which follows the
selected revision and complete observation hash.

Provider JSON schema 1.1.0 contains exactly `schema_version`, `revisions` and
`current_revision_ids`. Revisions retain registration order, with parents before
children, and preserve complete masks and identity/provenance. Selection contains
exactly one existing revision per scope. Schema 1.0.0 is read explicitly with
last-registered selection. Invalid snapshots never produce a partial provider.
The existing atomic sibling-file replacement preserves the previous snapshot
on write/replace failure. These guarantees concern the provider store, separately
from the ST-11 directory-bundle format.
