# F09q Provisional Source-Bound Marker Calibration

Issue #12164 extends the F09 marker map with a versioned calibration artifact.
It reuses the shared `TourCapture`, `static_marker_offsets`, and
`score_frozen_marker_offsets` routines. It adds no second placement estimator,
capture reader, scoring authority, or qualification gate.

## Artifact and Fit

`calibrate_static_marker_attachments` accepts one ordered calibration capture,
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

`validate` recomputes capture/pose digests, estimates and residuals from the
supplied inputs and requires callers to restate the expected pose-provider
identity and capture frame/timebase. It rejects source, order, clock, provider,
model or payload changes. It retains no observation arrays or local source paths. `as_dict`
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
