# F09q Provisional Source-Bound Marker Calibration

Issue #12164 extends the F09 marker map with a versioned calibration artifact.
It reuses the shared `TourCapture`, `static_marker_offsets`, and
`score_frozen_marker_offsets` routines. It adds no second placement estimator,
capture reader, scoring authority, or qualification gate.

## Artifact and Fit

`calibrate_static_marker_attachments` accepts one `NativeMarkerCalibrationRequest`
that groups the ordered capture, label-to-native-frame mapping, native poses and
clock, adapter binding, pose-provider identity, and capture frame/timebase. The
request is supplied again to artifact revalidation so those inputs are rehashed.
The producer accepts one ordered calibration capture,
one exact native frame ID per capture label, native world-pose samples for the
same frame count, and an explicit pose-time vector. The vector must exactly
match the capture clock. For each label it applies the shared body-frame
placement estimate

$$
\hat o_m = \frac{1}{|V_m|}\sum_{f\in V_m}
R_{f,b_m}^{T}(y_{f,m}-t_{f,b_m}),
$$

then computes the existing in-sample frozen-placement residual. Each marker
must have at least one valid observation; rotations and translations must be
finite, and rotations must be proper. Exact label order and frame identity are
retained.

The frozen `NativeMarkerAttachmentCalibrationArtifact` binds:

- inventory-to-native adapter model and provider identities;
- source and loaded model hashes plus the separately named pose-provider hash;
- capture source hash, ordered labels, capture frame/timebase, clock digest,
  and observation-byte digest;
- ordered native frame IDs, local offsets in metres, valid-sample counts,
  in-sample residuals, calibration method and pose-trajectory digest.

`validate` snapshots and revalidates the supplied capture arrays, recomputes
capture/pose digests, estimates and residuals, and requires callers to restate
the expected native engine, pose-provider identity, frame mapping, and capture
frame/timebase. It rejects aliased-clock mutation, source, order, clock,
provider, model or payload changes. It retains no observation arrays or local
source paths. `as_dict`
contains placement metadata and digests only. `to_native_marker_map` carries
the artifact digest into the F09 map identity; the replay output still uses its
own simulation-relative clock.

## Evidence Status and Limits

The result is `provisional_estimated`, `unqualified`, with holdout
`not_evaluated` and physiology `unqualified`. It does not require human approval
to be used as a provisional mapping, but it does not turn an anatomical seed
into measured geometry. The residual is in-sample, not independent prediction.
Artifact digests and Python records are integrity assertions, not signatures or
proof that the named native pose provider executed. A consumer must preserve
the calibration artifact and independently verify its inputs before using the
mapping in a scored result.

Issue #12178 extends that handoff through the existing observation scorer. The
MuJoCo fixture runs a full-state replay to produce calibration poses, builds
and revalidates the provisional artifact against the exact capture clock,
constructs a map carrying the artifact digest, and executes a fresh native
marker replay. The scorer consumes those positions with separate synthetic
observations and records calibration, map, replay, output, observation, and
alignment identities. Negative checks reject a changed pose clock and a
tampered frozen offset. The resulting row remains unqualified and the complete
17-row denominator remains present. This is fixture-level integration only;
it does not establish production marker attachments or qualification.

The focused calibration-to-score integration now also passes against actual
Drake 1.57 and Pinocchio 4.1 SDKs. Pinocchio ran with ordinary source imports
from an isolated copied test module. The minimal Drake environment could not
import the broad Drake package initializer because `structlog` and then
`simpleeval` were absent. Its native test therefore used a test-only namespace
bootstrap for the two Drake package parents and loaded the real provider source
and SDK; no physics/provider implementation was mocked. A temporary 65 KB
`structlog` overlay did not make the full application bootstrap complete.
Consequently these runs verify the native provider and scorer behavior, not the
public deployment package bootstrap. OpenSim's native geometry provider
remains position-level rather than a full-state replay adapter.

No coordinate-retarget map is accepted as marker geometry. In particular,
MyoSuite `coordinate_map_anthro.json` remains a coordinate map. Existing
full-body `marker_attachments` remain seeds until their calibration evidence
is available. The artifact producer is a generic seam; it does not itself
provide full-state replay, independent holdout, uncertainty, lab registration,
native anatomy, contact, physiology, or six-engine qualification.

## Validation

Portable synthetic fixtures recover known local offsets, exercise changed
capture/pose bytes, provider identity and clock rejection, recompute tampered
artifact offsets, verify ordered native-frame mapping and ensure observation
arrays are absent from serialization. An actual OpenSim 4.6 fixture exercises
the existing native `NativeMarkerGeometry` pose provider on a one-coordinate
model and recovers the known offsets. This is a software-contract test, not a
production model or capture qualification.

Run the portable tests with the repository test configuration and the native
test in the reviewed OpenSim 4.6 environment. Keep all manual/calculation
inventory and release gates blocked until the governing review and independent
evidence exist.

For the replay-to-score integration, run the MuJoCo observation-qualification
case and the two native FK cases in
`tests/unit/engines/test_feedback_native_markers.py` with their actual SDKs.
The Drake minimal-runtime namespace bootstrap is described in the turnover;
it must not be presented as proof that the ordinary deployment package import
works. These fixtures use synthetic observations and have no private-capture
dependency.
